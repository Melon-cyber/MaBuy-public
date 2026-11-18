import os
import time
import logging
import urllib.request
import json
import asyncio
import numpy as np
import pandas as pd
from typing import Optional, List

from src.infrastructure.data_fetchers.base_fetcher import BaseDataFetcher
from src.domain.services.symbol_normalizer import SymbolNormalizer
from src.infrastructure.data_fetchers.data_circuit_breaker import DataCircuitBreaker, StaleDataException
from src.infrastructure.data_fetchers.atomic_cache_guard import AtomicCacheGuard

logger = logging.getLogger(__name__)


class CryptoDataFetcher(BaseDataFetcher):
    """
    Market Data Ingestion implementation for cryptocurrency OHLCV data.
    Features:
      - Multi-domain failover rotation (api.binance.com, api1, api2, api3)
      - Atomic multi-process file locking (Thundering Herd shield)
      - Industrial Circuit Breaker protection for network outages
      - Strict StaleDataException enforcement for live trading execution
    """

    ENDPOINTS: List[str] = [
        "https://api.binance.com",
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com"
    ]

    def __init__(self, cache_dir: str = ".data_cache", max_cache_age_sec: int = 60):
        self.cache_dir = cache_dir
        self.max_cache_age_sec = max_cache_age_sec
        os.makedirs(self.cache_dir, exist_ok=True)
        self.circuit_breaker = DataCircuitBreaker.get_instance("binance", failure_threshold=3, recovery_timeout_sec=30.0)
        self._depth_cache = {}  # In-memory short-lived depth cache with 5s TTL

    def fetch_orderbook_depth(self, symbol: str, limit: int = 20) -> dict:
        """
        Fetches live orderbook depth (bids and asks) from Binance with a dedicated 5-second TTL cache.
        Returns: {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}
        """
        clean_symbol = SymbolNormalizer.normalize(symbol)
        if not clean_symbol.endswith("USDT"):
            clean_symbol += "USDT"

        now = time.time()
        cached_entry = self._depth_cache.get(clean_symbol)
        if cached_entry:
            cached_time, cached_depth = cached_entry
            if (now - cached_time) < 5.0:
                return cached_depth

        if not self.circuit_breaker.allow_request():
            if cached_entry:
                return cached_entry[1]
            return {"bids": [], "asks": []}

        last_err = None
        for base_url in self.ENDPOINTS:
            url = f"{base_url}/api/v3/depth?symbol={clean_symbol}&limit={limit}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "MaBuy-Engine/1.0"})
                with urllib.request.urlopen(req, timeout=3) as response:
                    depth_data = json.loads(response.read().decode())
                if depth_data and "bids" in depth_data and "asks" in depth_data:
                    self.circuit_breaker.record_success()
                    self._depth_cache[clean_symbol] = (now, depth_data)
                    return depth_data
            except Exception as e:
                last_err = e
                continue

        self.circuit_breaker.record_failure(last_err)
        if cached_entry:
            return cached_entry[1]
        return {"bids": [], "asks": []}

    def _get_cache_path(self, symbol: str, timeframe: str) -> str:
        clean_symbol = SymbolNormalizer.normalize(symbol)
        return os.path.join(self.cache_dir, f"{clean_symbol}_{timeframe}.csv")

    def _read_cache(self, cache_path: str, limit: int) -> Optional[pd.DataFrame]:
        """Safely reads dataframe from disk cache."""
        try:
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                if len(df) >= limit:
                    return df.iloc[-limit:].reset_index(drop=True)
                return df.reset_index(drop=True)
        except Exception as e:
            logger.warning(f"Cache read error for {cache_path}: {e}")
        return None

    def _fetch_from_network(self, clean_symbol: str, timeframe: str, limit: int, end_time: Optional[int]) -> Optional[List]:
        """Iterates through Binance endpoints to fetch raw kline data."""
        if not self.circuit_breaker.allow_request():
            logger.warning(f"[Binance Circuit Breaker OPEN] Skipping live network request for {clean_symbol}")
            return None

        last_err = None
        for base_url in self.ENDPOINTS:
            url = f"{base_url}/api/v3/klines?symbol={clean_symbol}&interval={timeframe}&limit={limit}"
            if end_time:
                url += f"&endTime={end_time}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "MaBuy-Engine/1.0"})
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode())
                if data:
                    self.circuit_breaker.record_success()
                    return data
            except Exception as e:
                last_err = e
                continue

        self.circuit_breaker.record_failure(last_err)
        logger.warning(f"All Binance API endpoints failed for {clean_symbol}: {last_err}")
        return None

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 1000,
        end_time: Optional[int] = None,
        require_fresh: bool = False
    ) -> pd.DataFrame:
        clean_symbol = SymbolNormalizer.normalize(symbol)
        if not clean_symbol.endswith("USDT"):
            clean_symbol += "USDT"

        cache_path = self._get_cache_path(clean_symbol, timeframe)

        # 1. Fast Path: Check local cache freshness without acquiring lock
        if not end_time and os.path.exists(cache_path):
            file_age = time.time() - os.path.getmtime(cache_path)
            if file_age < self.max_cache_age_sec:
                df = self._read_cache(cache_path, limit)
                if df is not None:
                    logger.debug(f"Loaded fresh ({int(file_age)}s old) candles from cache for {clean_symbol}")
                    return df

        # 2. Slow Path: Multi-Process Atomic Lock for fetching fresh data
        try:
            with AtomicCacheGuard(cache_path, timeout=5.0):
                # Double-check cache after acquiring lock (another process might have just updated it)
                if not end_time and os.path.exists(cache_path):
                    file_age = time.time() - os.path.getmtime(cache_path)
                    if file_age < self.max_cache_age_sec:
                        df = self._read_cache(cache_path, limit)
                        if df is not None:
                            return df

                # Fetch fresh from Binance REST API
                raw_data = self._fetch_from_network(clean_symbol, timeframe, limit, end_time)
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
                    df.to_csv(cache_path, index=False)
                    logger.debug(f"Fetched fresh {len(df)} live candles from Binance API for {clean_symbol}")
                    return df
        except Exception as lock_err:
            logger.warning(f"Atomic lock/fetch error for {clean_symbol}: {lock_err}")

        # 3. Handle Fallbacks and Live-Trading Safety Guard
        cached_df = self._read_cache(cache_path, limit)
        if cached_df is not None and len(cached_df) > 0:
            if require_fresh:
                # Enforce freshness for live trading execution
                file_age = time.time() - os.path.getmtime(cache_path)
                if file_age > (15 * 60):  # Stale if older than 15 minutes
                    raise StaleDataException(
                        f"Live trading halted: candle data for {clean_symbol} is stale ({int(file_age)}s old) "
                        f"and Circuit Breaker is active/network unreachable."
                    )
            logger.warning(f"Serving fallback cached candles for {clean_symbol}")
            return cached_df

        if require_fresh:
            raise StaleDataException(
                f"Live trading halted: no live or cached data available for {clean_symbol}."
            )

        # 4. Fail-Fast (Rule of 3 Whales & Astra Directive: Zero Stubs / Zero Synthetic Mocks)
        logger.error(f"Failed to retrieve genuine candles for {clean_symbol} and no fresh cache exists. Refusing synthetic data.")
        raise StaleDataException(
            f"Market data unavailable: network failed and no valid cache exists for {clean_symbol}. "
            f"Halting operation to prevent synthetic data poisoning."
        )

    async def fetch_ohlcv_async(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 1000,
        end_time: Optional[int] = None,
        require_fresh: bool = False
    ) -> pd.DataFrame:
        """Non-blocking async interface using threadpool offloading."""
        return await asyncio.to_thread(
            self.fetch_ohlcv,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            end_time=end_time,
            require_fresh=require_fresh
        )
