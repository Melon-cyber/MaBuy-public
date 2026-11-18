"""
MaBuy V4 — Data Engineering Department
Module: Macro Data Pipeline & Covariance Engine for MaInvest.
Collects daily OHLCV candles (1d, 365+ days) across core crypto assets via Binance API,
caches them reliably in data/cache/macro_candles/, and calculates real macro-indicators:
  - 200-day SMA, 50-day EMA (Golden / Death Cross)
  - Mayer Multiple (Price / 200 SMA) & Valuation Regime
  - Realized Volatilities (30d, 90d, 365d Annualized)
  - Multi-Asset Return Covariance and Correlation Matrices.
"""
import os
import time
import json
import logging
import asyncio
import urllib.request
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.domain.services.symbol_normalizer import SymbolNormalizer
from src.infrastructure.data_fetchers.data_circuit_breaker import DataCircuitBreaker, StaleDataException
from src.infrastructure.data_fetchers.atomic_cache_guard import AtomicCacheGuard

logger = logging.getLogger(__name__)


class MacroDataPipeline:
    """
    Industrial Macro Data Pipeline for MaInvest Investment Hub.
    
    Features:
      - Resilient Binance REST klines ingestion with multi-endpoint failover.
      - Atomic multi-process cache locking with configurable TTL.
      - 100% mathematical calculations of macro indicators and covariance matrices.
      - JSON-serializable output structures ready for API, bot, and ML consumption.
      - Synchronous and asynchronous interfaces.
    """

    DEFAULT_SYMBOLS: List[str] = [
        "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX"
    ]

    ENDPOINTS: List[str] = [
        "https://api.binance.com",
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com"
    ]

    def __init__(
        self,
        cache_dir: str = "data/cache/macro_candles",
        cache_ttl_sec: int = 3600,
        default_limit: int = 500
    ):
        self.cache_dir = cache_dir
        self.cache_ttl_sec = cache_ttl_sec
        self.default_limit = default_limit
        os.makedirs(self.cache_dir, exist_ok=True)
        self.circuit_breaker = DataCircuitBreaker.get_instance(
            service_name="binance_macro",
            failure_threshold=3,
            recovery_timeout_sec=30.0
        )

    def _get_cache_path(self, clean_symbol: str) -> str:
        return os.path.join(self.cache_dir, f"{clean_symbol}_1d.csv")

    def _read_cache(self, cache_path: str, min_rows: int = 1) -> Optional[pd.DataFrame]:
        """Safely reads cached CSV into a clean pandas DataFrame."""
        try:
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path)
                if not df.empty and "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
                    if len(df) >= min_rows:
                        return df
        except Exception as e:
            logger.warning(f"Error reading macro cache {cache_path}: {e}")
        return None

    def _fetch_from_binance_network(self, clean_symbol: str, limit: int = 500) -> Optional[List]:
        """Iterates through Binance failover endpoints to fetch daily klines."""
        if not self.circuit_breaker.allow_request():
            logger.warning(f"[Binance Macro Circuit Breaker OPEN] Skipping live network request for {clean_symbol}")
            return None

        pair = f"{clean_symbol}USDT"
        last_err = None
        for base_url in self.ENDPOINTS:
            url = f"{base_url}/api/v3/klines?symbol={pair}&interval=1d&limit={limit}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "MaBuy-MacroEngine/1.0"})
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                if data and isinstance(data, list) and len(data) > 0:
                    self.circuit_breaker.record_success()
                    return data
            except Exception as e:
                last_err = e
                continue

        self.circuit_breaker.record_failure(last_err)
        logger.warning(f"All Binance API endpoints failed for macro symbol {clean_symbol}: {last_err}")
        return None

    def fetch_daily_candles(
        self,
        symbol: str,
        limit: int = 500,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetches 1d OHLCV candles (default 500 days) for the given asset symbol.
        Uses atomic file locking and cache freshness checking.
        """
        clean_symbol = SymbolNormalizer.normalize(symbol)
        cache_path = self._get_cache_path(clean_symbol)

        # 1. Check local cache freshness (Fast Path)
        if not force_refresh and os.path.exists(cache_path):
            file_age = time.time() - os.path.getmtime(cache_path)
            if file_age < self.cache_ttl_sec:
                df = self._read_cache(cache_path, min_rows=200)
                if df is not None:
                    logger.debug(f"Loaded fresh ({int(file_age)}s old) macro candles for {clean_symbol}")
                    return df

        # 2. Acquire Atomic Lock and fetch fresh data from Binance (Slow Path)
        try:
            with AtomicCacheGuard(cache_path, timeout=5.0):
                # Double check cache after lock acquisition
                if not force_refresh and os.path.exists(cache_path):
                    file_age = time.time() - os.path.getmtime(cache_path)
                    if file_age < self.cache_ttl_sec:
                        df = self._read_cache(cache_path, min_rows=200)
                        if df is not None:
                            return df

                raw_data = self._fetch_from_binance_network(clean_symbol, limit=limit)
                if raw_data:
                    records = []
                    for item in raw_data:
                        records.append({
                            "timestamp": pd.to_datetime(item[0], unit="ms"),
                            "open": float(item[1]),
                            "high": float(item[2]),
                            "low": float(item[3]),
                            "close": float(item[4]),
                            "volume": float(item[5]),
                        })
                    df = pd.DataFrame(records)
                    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
                    df.to_csv(cache_path, index=False)
                    logger.info(f"Fetched & cached {len(df)} live daily candles for {clean_symbol}")
                    return df
        except Exception as lock_err:
            logger.warning(f"Atomic cache guard error for macro {clean_symbol}: {lock_err}")

        # 3. Fallback to existing disk cache if network is unavailable
        cached_df = self._read_cache(cache_path, min_rows=1)
        if cached_df is not None and len(cached_df) > 0:
            logger.warning(f"Serving fallback disk cached candles for macro {clean_symbol} ({len(cached_df)} rows)")
            return cached_df

        raise StaleDataException(
            f"No live or cached macro candle data available for {clean_symbol}."
        )

    def calculate_macro_indicators(self, df: pd.DataFrame, symbol: str = "BTC") -> Dict[str, Any]:
        """
        Calculates 100% real macro-indicators on the provided daily OHLCV DataFrame:
          - 200-day SMA, 50-day EMA
          - Trend Cross status (Golden Cross / Death Cross) & EMA/SMA spread %
          - Mayer Multiple & Valuation Regime classification
          - Realized Volatilities (30d, 90d, 365d annualized)
          - 365d High/Low, Drawdown from 365d High
        """
        if df.empty or len(df) < 5:
            raise ValueError(f"Insufficient candle history for symbol {symbol}: {len(df)} rows")

        clean_symbol = SymbolNormalizer.normalize(symbol)
        df = df.sort_values("timestamp").reset_index(drop=True).copy()

        # Closing prices & Returns
        closes = df["close"].astype(float)
        highs = df["high"].astype(float)
        lows = df["low"].astype(float)
        daily_returns = closes.pct_change().dropna()

        current_price = float(closes.iloc[-1])
        latest_timestamp = df["timestamp"].iloc[-1]

        # 1. 200-day SMA & 50-day EMA
        sma_window = min(200, len(closes))
        ema_span = min(50, len(closes))

        sma_200_series = closes.rolling(window=sma_window, min_periods=min(30, sma_window)).mean()
        ema_50_series = closes.ewm(span=ema_span, adjust=False).mean()

        sma_200 = float(sma_200_series.iloc[-1])
        ema_50 = float(ema_50_series.iloc[-1])

        # Trend & Cross evaluation
        spread_pct = float(((ema_50 - sma_200) / sma_200 * 100.0)) if sma_200 > 0 else 0.0
        if ema_50 >= sma_200:
            trend_cross = "GOLDEN_CROSS"
            trend_cross_label = "Golden Cross (Бычий тренд)"
        else:
            trend_cross = "DEATH_CROSS"
            trend_cross_label = "Death Cross (Медвежий тренд)"

        # Check for recent cross in last 14 days
        recent_cross_detected = False
        if len(ema_50_series) >= 15 and len(sma_200_series) >= 15:
            diff_recent = ema_50_series.iloc[-14:] - sma_200_series.iloc[-14:]
            if (diff_recent > 0).any() and (diff_recent < 0).any():
                recent_cross_detected = True

        # 2. Mayer Multiple (Price / 200 SMA)
        mayer_multiple = float(current_price / sma_200) if sma_200 > 0 else 1.0
        
        # Mayer Multiple Regime
        if mayer_multiple < 0.8:
            mayer_regime = "DEEP_VALUE"
            mayer_regime_desc = "Глубокая недооценка (Зона сверхаккумуляции)"
            mayer_signal = "STRONG_BUY"
        elif mayer_multiple < 1.0:
            mayer_regime = "ACCUMULATION"
            mayer_regime_desc = "Зона накопления (Ниже 200 SMA)"
            mayer_signal = "BUY"
        elif mayer_multiple < 1.4:
            mayer_regime = "BULLISH_EXPANSION"
            mayer_regime_desc = "Справедливая оценка / Здоровый бычий рост"
            mayer_signal = "HOLD"
        elif mayer_multiple < 2.4:
            mayer_regime = "OVERHEATED"
            mayer_regime_desc = "Перегретый рынок / Фиксация прибыли"
            mayer_signal = "TAKE_PROFIT"
        else:
            mayer_regime = "EUPHORIC_BUBBLE"
            mayer_regime_desc = "Экстремальный пузырь / Историческая эйфория"
            mayer_signal = "STRONG_SELL"

        # 3. Realized Volatilities (30d, 90d, 365d annualized)
        # RV = std(daily_returns) * sqrt(365)
        sqrt_365 = float(np.sqrt(365.0))

        # 30-day RV
        n_30 = min(30, len(daily_returns))
        daily_vol_30d = float(daily_returns.tail(n_30).std()) if n_30 > 1 else 0.0
        realized_vol_30d = float(daily_vol_30d * sqrt_365)

        # 90-day RV
        n_90 = min(90, len(daily_returns))
        daily_vol_90d = float(daily_returns.tail(n_90).std()) if n_90 > 1 else daily_vol_30d
        realized_vol_90d = float(daily_vol_90d * sqrt_365)

        # 365-day RV
        n_365 = min(365, len(daily_returns))
        daily_vol_365d = float(daily_returns.tail(n_365).std()) if n_365 > 1 else daily_vol_90d
        realized_vol_365d = float(daily_vol_365d * sqrt_365)

        # 4. Range, Highs, Lows and Drawdown
        lookback_365 = min(365, len(df))
        df_365 = df.tail(lookback_365)
        
        high_365d = float(df_365["high"].max())
        low_365d = float(df_365["low"].min())
        
        drawdown_from_365d_high_pct = float(((current_price - high_365d) / high_365d * 100.0)) if high_365d > 0 else 0.0
        distance_from_365d_low_pct = float(((current_price - low_365d) / low_365d * 100.0)) if low_365d > 0 else 0.0

        # Performance Returns
        price_30d_ago = float(closes.iloc[-min(30, len(closes))])
        price_90d_ago = float(closes.iloc[-min(90, len(closes))])
        price_365d_ago = float(closes.iloc[-min(365, len(closes))])

        ret_30d_pct = float(((current_price - price_30d_ago) / price_30d_ago * 100.0)) if price_30d_ago > 0 else 0.0
        ret_90d_pct = float(((current_price - price_90d_ago) / price_90d_ago * 100.0)) if price_90d_ago > 0 else 0.0
        ret_365d_pct = float(((current_price - price_365d_ago) / price_365d_ago * 100.0)) if price_365d_ago > 0 else 0.0

        return {
            "symbol": clean_symbol,
            "current_price": round(current_price, 4 if current_price < 10 else 2),
            "sma_200": round(sma_200, 4 if sma_200 < 10 else 2),
            "ema_50": round(ema_50, 4 if ema_50 < 10 else 2),
            "trend_cross": trend_cross,
            "trend_cross_label": trend_cross_label,
            "recent_cross_detected": bool(recent_cross_detected),
            "ema_sma_spread_pct": round(spread_pct, 2),
            "mayer_multiple": round(mayer_multiple, 3),
            "mayer_regime": mayer_regime,
            "mayer_regime_desc": mayer_regime_desc,
            "mayer_signal": mayer_signal,
            "daily_volatility_30d_pct": round(daily_vol_30d * 100.0, 2),
            "daily_volatility_90d_pct": round(daily_vol_90d * 100.0, 2),
            "daily_volatility_365d_pct": round(daily_vol_365d * 100.0, 2),
            "realized_volatility_30d": round(realized_vol_30d, 4),
            "realized_volatility_30d_pct": round(realized_vol_30d * 100.0, 2),
            "realized_volatility_90d": round(realized_vol_90d, 4),
            "realized_volatility_90d_pct": round(realized_vol_90d * 100.0, 2),
            "realized_volatility_365d": round(realized_vol_365d, 4),
            "realized_volatility_365d_pct": round(realized_vol_365d * 100.0, 2),
            "high_365d": round(high_365d, 4 if high_365d < 10 else 2),
            "low_365d": round(low_365d, 4 if low_365d < 10 else 2),
            "drawdown_from_365d_high_pct": round(drawdown_from_365d_high_pct, 2),
            "distance_from_365d_low_pct": round(distance_from_365d_low_pct, 2),
            "return_30d_pct": round(ret_30d_pct, 2),
            "return_90d_pct": round(ret_90d_pct, 2),
            "return_365d_pct": round(ret_365d_pct, 2),
            "candles_count": int(len(df)),
            "latest_candle_timestamp": latest_timestamp.isoformat() if hasattr(latest_timestamp, "isoformat") else str(latest_timestamp),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }

    def get_macro_snapshot(self, symbol: str, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Public API: Fetches daily candles and calculates full macro-snapshot for a symbol.
        """
        df = self.fetch_daily_candles(symbol=symbol, limit=self.default_limit, force_refresh=force_refresh)
        return self.calculate_macro_indicators(df=df, symbol=symbol)

    def get_all_macro_snapshots(
        self,
        symbols: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetches macro snapshots across a collection of cryptocurrency symbols.
        """
        target_symbols = symbols if symbols is not None else self.DEFAULT_SYMBOLS
        results = {}
        for sym in target_symbols:
            try:
                results[sym] = self.get_macro_snapshot(sym, force_refresh=force_refresh)
            except Exception as e:
                logger.error(f"Failed to calculate macro snapshot for {sym}: {e}")
        return results

    def _collect_daily_returns(
        self,
        symbols: Optional[List[str]] = None,
        history_days: int = 90,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Gathers daily closing prices and calculates aligned percentage returns across symbols.
        """
        target_symbols = symbols if symbols is not None else self.DEFAULT_SYMBOLS
        clean_symbols = [SymbolNormalizer.normalize(s) for s in target_symbols]
        
        fetch_limit = max(history_days + 30, self.default_limit)
        
        prices_dict: Dict[str, pd.Series] = {}
        for sym in clean_symbols:
            try:
                df = self.fetch_daily_candles(sym, limit=fetch_limit, force_refresh=force_refresh)
                if not df.empty:
                    df = df.sort_values("timestamp").drop_duplicates("timestamp")
                    series = df.set_index("timestamp")["close"].astype(float)
                    prices_dict[sym] = series
            except Exception as e:
                logger.warning(f"Skipping {sym} in returns matrix due to error: {e}")

        if not prices_dict:
            raise ValueError("No valid price series collected for covariance calculation.")

        df_prices = pd.DataFrame(prices_dict).dropna()
        if len(df_prices) < 5:
            df_prices = pd.DataFrame(prices_dict).ffill().bfill().dropna()

        if len(df_prices) > history_days:
            df_prices = df_prices.tail(history_days)

        df_returns = df_prices.pct_change().dropna()
        return df_returns

    def get_assets_returns_dataframe(
        self,
        symbols: Optional[List[str]] = None,
        history_days: int = 90,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Public API: Returns aligned daily percentage returns across symbols.
        """
        return self._collect_daily_returns(
            symbols=symbols,
            history_days=history_days,
            force_refresh=force_refresh
        )

    def get_assets_covariance_matrix(
        self,
        symbols: Optional[List[str]] = None,
        history_days: int = 90,
        annualized: bool = False,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Calculates the real daily return covariance matrix across assets.
        If annualized=True, scales daily covariance by 365 (crypto trading calendar).
        """
        df_returns = self._collect_daily_returns(
            symbols=symbols,
            history_days=history_days,
            force_refresh=force_refresh
        )
        cov_matrix = df_returns.cov()
        if annualized:
            cov_matrix = cov_matrix * 365.0
        return cov_matrix

    def get_assets_correlation_matrix(
        self,
        symbols: Optional[List[str]] = None,
        history_days: int = 90,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Calculates the Pearson correlation matrix of daily returns across assets.
        """
        df_returns = self._collect_daily_returns(
            symbols=symbols,
            history_days=history_days,
            force_refresh=force_refresh
        )
        return df_returns.corr()

    # Async interfaces for non-blocking execution in async callers
    async def fetch_daily_candles_async(
        self,
        symbol: str,
        limit: int = 500,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        return await asyncio.to_thread(
            self.fetch_daily_candles,
            symbol=symbol,
            limit=limit,
            force_refresh=force_refresh
        )

    async def get_macro_snapshot_async(
        self,
        symbol: str,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self.get_macro_snapshot,
            symbol=symbol,
            force_refresh=force_refresh
        )

    async def get_all_macro_snapshots_async(
        self,
        symbols: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        return await asyncio.to_thread(
            self.get_all_macro_snapshots,
            symbols=symbols,
            force_refresh=force_refresh
        )

    async def get_assets_covariance_matrix_async(
        self,
        symbols: Optional[List[str]] = None,
        history_days: int = 90,
        annualized: bool = False,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        return await asyncio.to_thread(
            self.get_assets_covariance_matrix,
            symbols=symbols,
            history_days=history_days,
            annualized=annualized,
            force_refresh=force_refresh
        )

    async def get_assets_correlation_matrix_async(
        self,
        symbols: Optional[List[str]] = None,
        history_days: int = 90,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        return await asyncio.to_thread(
            self.get_assets_correlation_matrix,
            symbols=symbols,
            history_days=history_days,
            force_refresh=force_refresh
        )
