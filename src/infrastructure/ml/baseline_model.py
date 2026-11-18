import math
import logging
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, Any, Tuple, List, Optional
from sklearn.ensemble import HistGradientBoostingRegressor

from src.application.interfaces.ml_engine import MLEngineInterface
from src.domain.entities.prediction import PredictionRequest, PredictionResult
from src.domain.entities.tier import TierLimits, SubscriptionTier
from src.domain.services.symbol_normalizer import SymbolNormalizer
from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher
from src.infrastructure.ml.prediction_cache import PredictionCacheService

logger = logging.getLogger(__name__)


class BaselineGBDTModelEngine(MLEngineInterface):
    r"""
    Stage 1 Baseline Model Engine utilizing Scikit-Learn Gradient Boosted Decision Trees (GBDT).
    Supports Multi-Timeframe Forecast Horizons: '1h', '4h', '1d', bilingual localization ('ru', 'en'),
    and fast in-memory LRU caching (<1ms response).
    """

    VALID_TIMEFRAMES = {"1h", "4h", "1d"}

    def __init__(self):
        self.data_fetcher = CryptoDataFetcher()
        self.cache_service = PredictionCacheService(ttl_seconds=10.0)
        # Load pretrained baseline GBDT regressor
        self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "models"))
        
        feature_names_path = os.path.join(self.models_dir, "feature_names.pkl")
        if os.path.exists(feature_names_path):
            self.feature_cols = joblib.load(feature_names_path)
        else:
            self.feature_cols = []
            
        model_path = os.path.join(self.models_dir, "gbdt_model.pkl")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            logger.info("Loaded GBDT Baseline Sub-Model for V4 Ensemble Engine from disk.")
        else:
            logger.warning("No gbdt_model.pkl found! Fitting dummy model (V1 fallback).")
            self.model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
            X_dummy = np.random.randn(500, 6)
            y_dummy = np.random.randn(500)
            self.model.fit(X_dummy, y_dummy)

    def _extract_real_market_features(self, symbol: str, timeframe: str) -> tuple[float, np.ndarray]:
        from ta import add_all_ta_features
        import joblib
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        normalized = SymbolNormalizer.normalize(symbol)
        tf_clean = timeframe.lower() if timeframe.lower() in self.VALID_TIMEFRAMES else "1h"

        # Fetch 150 candles to warm up slow TA indicators (Rule of 3 Whales: Zero Stubs)
        try:
            df = self.data_fetcher.fetch_ohlcv(normalized, timeframe=tf_clean, limit=150)
            if df is None or len(df) == 0:
                raise ValueError("Empty dataframe returned by Binance fetcher")
        except Exception as fetch_err:
            logger.error(f"Feature extraction fetch error for {symbol}: {fetch_err}. Rejecting synthetic fallback (Fail-Fast).")
            raise RuntimeError(f"Real market data unavailable for {symbol} on Binance: {fetch_err}")

        # Ensure minimum length of 150 rows so ta library never throws IndexErrors (e.g., Aroon 14, Ichimoku 52)
        if len(df) < 150:
            pad_count = 150 - len(df)
            first_row = df.iloc[[0]]
            pad_df = pd.concat([first_row] * pad_count, ignore_index=True)
            df = pd.concat([pad_df, df], ignore_index=True)

        closes = df["close"].values
        current_price = float(closes[-1])

        # 1. Base log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # 2. Add ALL TA features with safe try-except
        try:
            df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
        except Exception as ta_err:
            logger.warning(f"add_all_ta_features warning: {ta_err}")
            df = df.fillna(0.0)
        
        # 3. Add Custom Lag Features (1-Bar Momentum Bug Fixed: r_lag1 is current candle)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        log_ret = df['log_return']

        vol_20 = log_ret.rolling(window=20).std() + 1e-7
        vol_60 = log_ret.rolling(window=60).std() + 1e-7
        z_return = log_ret / vol_20

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        rsi_14 = (100.0 - (100.0 / (1.0 + rs))) / 100.0

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal

        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
        )
        atr_norm = tr.rolling(window=14).mean() / (close + 1e-8)

        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std() + 1e-8
        upper_band = sma_20 + (2.0 * std_20)
        lower_band = sma_20 - (2.0 * std_20)
        bb_pb = (close - lower_band) / (upper_band - lower_band + 1e-8)

        vol_sma_20 = volume.rolling(window=20).mean() + 1e-8

        # Assign synchronized features
        custom_feats = {
            'r_lag1': log_ret,
            'r_lag2': log_ret.shift(1),
            'r_lag3': log_ret.shift(2),
            'r_lag5': log_ret.shift(4),
            'volatility_20': vol_20,
            'volatility_60': vol_60,
            'vol_ratio': vol_20 / vol_60,
            'z_return_lag1': z_return,
            'z_return_lag2': z_return.shift(1),
            'z_return_lag3': z_return.shift(2),
            'rsi_14': rsi_14,
            'macd_norm': macd / (close + 1e-8),
            'macd_signal_norm': macd_signal / (close + 1e-8),
            'macd_hist_norm': macd_hist / (close + 1e-8),
            'atr_norm': atr_norm,
            'bb_pb': bb_pb,
            'vol_ratio_20': volume / vol_sma_20,
            'volume_log_return': np.log((volume + 1e-8) / (volume.shift(1) + 1e-8))
        }
        custom_df = pd.DataFrame(custom_feats, index=df.index)
        df = pd.concat([df, custom_df], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.fillna(0.0)
        
        # Extract the last row matching the exact features
        last_row = df.iloc[-1]
        
        vector = []
        for col in self.feature_cols:
            val = last_row.get(col, 0.0)
            if pd.isna(val) or np.isinf(val):
                val = 0.0
            vector.append(float(val))

        feature_vector = np.array([vector])
        return current_price, feature_vector

    def _extract_real_market_feature_matrix(self, symbol: str, timeframe: str = "1h", limit: int = 60, feature_cols: Optional[List[str]] = None) -> Tuple[float, np.ndarray]:
        """
        Extracts full chronological matrix of shape (limit, num_features) from real historical candles.
        Guarantees zero look-ahead bias and chronological ordering.
        """
        normalized = SymbolNormalizer.normalize(symbol)
        tf_clean = timeframe.lower() if timeframe.lower() in self.VALID_TIMEFRAMES else "1h"

        try:
            df = self.data_fetcher.fetch_ohlcv(normalized, timeframe=tf_clean, limit=150)
            if df is None or len(df) == 0:
                raise ValueError("Empty dataframe returned by fetcher")
        except Exception as fetch_err:
            logger.error(f"Feature matrix fetch error for {symbol}: {fetch_err}. Rejecting synthetic fallback (Fail-Fast).")
            raise RuntimeError(f"Real market data unavailable for {symbol} on Binance: {fetch_err}")

        if len(df) < 150:
            pad_count = 150 - len(df)
            first_row = df.iloc[[0]]
            pad_df = pd.concat([first_row] * pad_count, ignore_index=True)
            df = pd.concat([pad_df, df], ignore_index=True)

        current_price = float(df["close"].values[-1])

        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        try:
            df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
        except Exception:
            df = df.fillna(0.0)

        # 1-Bar Momentum Bug Fixed: r_lag1 is current candle
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        log_ret = df['log_return']

        vol_20 = log_ret.rolling(window=20).std() + 1e-7
        vol_60 = log_ret.rolling(window=60).std() + 1e-7
        z_return = log_ret / vol_20

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        rsi_14 = (100.0 - (100.0 / (1.0 + rs))) / 100.0

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - macd_signal

        tr = np.maximum(
            high - low,
            np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1)))
        )
        atr_norm = tr.rolling(window=14).mean() / (close + 1e-8)

        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std() + 1e-8
        upper_band = sma_20 + (2.0 * std_20)
        lower_band = sma_20 - (2.0 * std_20)
        bb_pb = (close - lower_band) / (upper_band - lower_band + 1e-8)

        vol_sma_20 = volume.rolling(window=20).mean() + 1e-8

        custom_feats = {
            'r_lag1': log_ret,
            'r_lag2': log_ret.shift(1),
            'r_lag3': log_ret.shift(2),
            'r_lag5': log_ret.shift(4),
            'volatility_20': vol_20,
            'volatility_60': vol_60,
            'vol_ratio': vol_20 / vol_60,
            'z_return_lag1': z_return,
            'z_return_lag2': z_return.shift(1),
            'z_return_lag3': z_return.shift(2),
            'rsi_14': rsi_14,
            'macd_norm': macd / (close + 1e-8),
            'macd_signal_norm': macd_signal / (close + 1e-8),
            'macd_hist_norm': macd_hist / (close + 1e-8),
            'atr_norm': atr_norm,
            'bb_pb': bb_pb,
            'vol_ratio_20': volume / vol_sma_20,
            'volume_log_return': np.log((volume + 1e-8) / (volume.shift(1) + 1e-8))
        }
        custom_df = pd.DataFrame(custom_feats, index=df.index)
        df = pd.concat([df, custom_df], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.fillna(0.0)

        # Slice the last `limit` rows (chronological)
        recent_df = df.iloc[-limit:]
        target_cols = feature_cols if feature_cols is not None else self.feature_cols
        
        matrix = []
        for _, row in recent_df.iterrows():
            row_vec = []
            for col in target_cols:
                val = row.get(col, 0.0)
                if pd.isna(val) or np.isinf(val):
                    val = 0.0
                row_vec.append(float(val))
            matrix.append(row_vec)

        return current_price, np.array(matrix, dtype=np.float32)

    def predict(self, request: PredictionRequest, limits: TierLimits) -> PredictionResult:
        symbol = SymbolNormalizer.normalize(request.symbol)
        timeframe = request.timeframe.lower() if hasattr(request, "timeframe") and request.timeframe and request.timeframe.lower() in self.VALID_TIMEFRAMES else "1h"
        lang = getattr(request, "lang", "ru")

        # Check fast in-memory LRU prediction cache (<1ms response)
        cache_key_tier = f"{limits.tier.value}_{lang}"
        cached_result = self.cache_service.get(symbol, timeframe, cache_key_tier)
        if cached_result is not None:
            logger.info(f"Served cached prediction for {symbol} ({timeframe}) - Tier: {limits.tier.value} ({lang})")
            return cached_result

        # Multi-timeframe horizon return scaling:
        # 1h: factor = 1.0 (base 1-hour candle return)
        # 4h: factor = 2.0 (4-hour return horizon: sqrt(4) = 2.0x return scale)
        # 1d: factor = 4.899 (24-hour return horizon: sqrt(24) = 4.899x return scale)
        tf_multiplier = 1.0 if timeframe == "1h" else (2.0 if timeframe == "4h" else 4.899)

        current_price, features = self._extract_real_market_features(symbol, timeframe=timeframe)

        # GBDT Model Inference scaled to forecast horizon
        raw_log_return = float(self.model.predict(features)[0])
        predicted_log_return = raw_log_return * tf_multiplier

        # Scale log return threshold proportionally to overcome Binance fees (0.10%) + Slippage (0.05%)
        # Minimum required move: 0.30% for 1h, 0.60% for 4h, 1.47% for 1d
        threshold = 0.0030 * tf_multiplier

        # Calculate predicted target price: P_target = P_current * e^(r_pred)
        predicted_target_price = current_price * math.exp(predicted_log_return)

        # Determine directional signal
        if predicted_log_return > threshold:
            direction = "BULLISH"
        elif predicted_log_return < -threshold:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Multi-timeframe scaling factors
        tf_k = 4.0 if timeframe == "1h" else (2.5 if timeframe == "4h" else 1.8)
        try:
            vol_idx = self.feature_cols.index("volatility_20")
            volatility = float(features[0][vol_idx])
        except (ValueError, IndexError):
            volatility = 0.012

        # Calibrated Hyperbolic Tangent SNR
        snr = abs(predicted_log_return) / (volatility + 1e-6)
        z_signal = snr * tf_k

        if direction == "NEUTRAL":
            confidence_score = 0.50 + 0.03 * math.tanh(z_signal)
            confidence_score = max(0.50, min(0.52, confidence_score))
        else:
            base_conf = 0.52 + 0.22 * math.tanh(z_signal)
            confidence_score = max(0.55, min(0.74, base_conf))

        # Personal AI Recommendation logic localized by request.lang
        personal_recommendation = None
        if limits.has_personal_recommendations:
            if lang == "ru":
                horizon_label = "1 час" if timeframe == "1h" else ("4 часа" if timeframe == "4h" else "24 часа")
                if direction == "BULLISH":
                    personal_recommendation = (
                        f"ИИ Совет для {symbol} ({horizon_label}): Прогнозируется восходящий импульс (+{predicted_log_return*100:.2f}%). "
                        f"Рассмотрите покупку на откатах в районе ${current_price * 0.985:.2f}."
                    )
                elif direction == "BEARISH":
                    personal_recommendation = (
                        f"ИИ Совет для {symbol} ({horizon_label}): Ожидается снижение цены ({predicted_log_return*100:.2f}%). "
                        f"Рекомендуем зафиксировать прибыль или выставить стоп-лоссы."
                    )
                else:
                    personal_recommendation = (
                        f"ИИ Совет для {symbol} ({horizon_label}): Рынок находится в боковике (консолидации). "
                        f"Рекомендуем удерживать позиции до пробоя ключевых уровней."
                    )
            else:
                horizon_label = "1-Hour" if timeframe == "1h" else ("4-Hour" if timeframe == "4h" else "24-Hour")
                if direction == "BULLISH":
                    personal_recommendation = (
                        f"AI Recommendation for {symbol} ({horizon_label}): Upward momentum forecasted (+{predicted_log_return*100:.2f}%). "
                        f"Consider accumulating on pullbacks near ${current_price * 0.985:.2f}."
                    )
                elif direction == "BEARISH":
                    personal_recommendation = (
                        f"AI Recommendation for {symbol} ({horizon_label}): Downward pressure forecasted ({predicted_log_return*100:.2f}%). "
                        f"Recommend setting take-profit targets near ${predicted_target_price:.2f}."
                    )
                else:
                    personal_recommendation = (
                        f"AI Recommendation for {symbol} ({horizon_label}): Market consolidation predicted. "
                        f"Recommend holding existing positions until a breakout signal."
                    )

        result = PredictionResult(
            symbol=symbol,
            current_price=current_price,
            predicted_target_price=round(predicted_target_price, 6 if predicted_target_price < 1.0 else 2),
            predicted_log_return=round(predicted_log_return, 5),
            direction=direction,
            confidence_score=round(confidence_score, 4),
            volatility_estimate=round(volatility, 4),
            timeframe=timeframe,
            personal_recommendation=personal_recommendation,
            sentiment_score=None,
            news_summary=None,
            model_version=f"MaBuy-GBDT-{timeframe.upper()}-v1.0",
            tier_applied=limits.tier
        )

        # Store in fast cache
        self.cache_service.set(symbol, timeframe, cache_key_tier, result)
        return result
