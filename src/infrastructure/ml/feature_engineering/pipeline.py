import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.base import BaseEstimator, TransformerMixin


class CryptoFeaturePipeline(BaseEstimator, TransformerMixin):
    r"""
    Enhanced Financial Time Series Feature Engineering Pipeline for MaBuy.

    Fixes for Market Noise & Regime Shifts:
    --------------------------------------
    1. Volatility Standardization: z_t = r_t / \sigma_{t, 20}
    2. Volatility Ratio (Regime Detector): \sigma_{short} / \sigma_{long}
    3. Normalized Average True Range (ATR_norm): ATR_14 / Close
    4. Trend Velocity: Slope of EMA_12 / EMA_26
    5. Bollinger Band Position & Squeeze Indicator
    """

    def __init__(self, target_horizon: int = 1, lag_periods: List[int] = None):
        self.target_horizon = target_horizon
        self.lag_periods = lag_periods or [1, 2, 3, 5, 10]
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes stationary, regime-normalized features and target variable."""
        data = df.copy().sort_values("timestamp").reset_index(drop=True)

        close = data["close"]
        high = data["high"]
        low = data["low"]
        volume = data["volume"]

        # 1. Primary Log Returns
        log_return = np.log(close / close.shift(1))
        data["log_return"] = log_return

        # 2. Rolling Volatility over window 20
        volatility_20 = log_return.rolling(window=20).std() + 1e-6
        data["volatility_20"] = volatility_20

        # 3. Volatility-Standardized Returns (Z-score normalization against regime noise)
        z_return = log_return / volatility_20
        data["z_return"] = z_return

        # 4. Autoregressive Lags on Volatility-Standardized Returns
        for lag in self.lag_periods:
            data[f"z_return_lag_{lag}"] = z_return.shift(lag)

        # 5. Volatility Regime Ratios
        vol_5 = log_return.rolling(window=5).std() + 1e-6
        vol_60 = log_return.rolling(window=60).std() + 1e-6
        data["volatility_ratio_5_60"] = vol_5 / vol_60

        # 6. Normalized Average True Range (ATR_norm)
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - close.shift(1)),
                np.abs(low - close.shift(1))
            )
        )
        atr_14 = tr.rolling(window=14).mean()
        data["atr_norm"] = atr_14 / (close + 1e-8)

        # 7. Technical Momentum & Oscillators
        data["rsi_14"] = self._compute_rsi(close, period=14) / 100.0 # Scale [0, 1]

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        data["macd_norm"] = (ema_12 - ema_26) / (close + 1e-8)

        sma_20 = close.rolling(window=20).mean()
        data["bollinger_b"] = (close - sma_20) / (2 * volatility_20 * close + 1e-8)

        # 8. Volume Log Return
        data["volume_log_return"] = np.log((volume + 1e-8) / (volume.shift(1) + 1e-8))

        # 9. Multi-Horizon Target Variables:
        # Horizon 1h: r_{t+1} = ln(P_{t+1} / P_t)
        # Horizon 4h: r_{t+4} = ln(P_{t+4} / P_t)
        # Horizon 1d: r_{t+24} = ln(P_{t+24} / P_t)
        data["target_1h"] = np.log(close.shift(-1) / close)
        data["target_4h"] = np.log(close.shift(-4) / close)
        data["target_1d"] = np.log(close.shift(-24) / close)
        
        # Primary target for backward compatibility
        data["target_log_return"] = np.log(close.shift(-self.target_horizon) / close)

        # Drop initial NaN rows created by rolling windows and future target shifts
        clean_df = data.dropna().reset_index(drop=True)

        target_col_set = {"timestamp", "open", "high", "low", "close", "volume", 
                          "target_log_return", "target_1h", "target_4h", "target_1d"}
        feature_cols = [c for c in clean_df.columns if c not in target_col_set]
        self.feature_names = feature_cols
        return clean_df

    def transform(self, X: pd.DataFrame, y=None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Transforms raw OHLCV DataFrame into numpy (X_features, y_target, feature_names).
        """
        processed_df = self.transform_df(X)
        X_matrix = processed_df[self.feature_names].values
        y_target = processed_df["target_log_return"].values
        return X_matrix, y_target, self.feature_names

    def transform_multi_horizon(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Transforms raw OHLCV DataFrame into multi-horizon numpy matrices.
        Returns:
            X_matrix: (N, num_features)
            Y_multi: (N, 3) where columns are [target_1h, target_4h, target_1d]
            feature_names: List of feature names
        """
        processed_df = self.transform_df(X)
        X_matrix = processed_df[self.feature_names].values
        Y_multi = processed_df[["target_1h", "target_4h", "target_1d"]].values
        return X_matrix, Y_multi, self.feature_names
