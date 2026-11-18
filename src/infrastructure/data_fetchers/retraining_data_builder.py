"""
MaBuy V4 — Data Engineering Department
Module: Retraining Dataset Builder & Feature Engineering Engine for MaPredict & MaInvest.

Responsible for:
1. Automated ingestion & preparation of multi-timeframe training datasets (1h, 4h for MaPredict, 1d for MaInvest)
   across top crypto assets: BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX.
2. Mathematical feature engineering & strict validation:
   - Lagged returns (log returns, autoregressive lags, rolling volatility standardized returns)
   - Realized Garman-Klass (GK) & Parkinson volatility estimators
   - Order Flow Imbalance (OFI) proxy & microstructure volume delta
   - SMA/EMA spreads (EMA 12/26, SMA 20/50, EMA 50/SMA 200) & trend velocity
   - Mayer Multiple (Close / 200-period SMA) & valuation regime categorization
3. Strict No-Lookahead Bias Guarantee:
   - 100% causal feature computation (no future data leakage, no center-window rolling)
   - Automated invariant audit (truncated history equivalence test)
4. Versioned Snapshot Storage in data/training_snapshots/ with timestamping & metadata sidecars.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.domain.services.symbol_normalizer import SymbolNormalizer
from src.infrastructure.data_fetchers.base_fetcher import BaseDataFetcher
from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher
from src.infrastructure.data_fetchers.macro_data_pipeline import MacroDataPipeline

logger = logging.getLogger("RetrainingDataBuilder")


# =====================================================================
# Custom Exceptions
# =====================================================================

class RetrainingDataBuilderError(Exception):
    """Base exception for RetrainingDataBuilder module."""
    pass


class InsufficientDataError(RetrainingDataBuilderError):
    """Raised when candle history is shorter than the required minimum rows."""
    pass


class LookaheadBiasError(RetrainingDataBuilderError):
    """Raised when feature computation exhibits lookahead bias or future data leakage."""
    pass


class DataValidationError(RetrainingDataBuilderError):
    """Raised when raw OHLCV data fails schema, timestamp, or validity checks."""
    pass


# =====================================================================
# Retraining Data Builder & Feature Engine
# =====================================================================

class RetrainingDataBuilder:
    """
    Automated Dataset Builder for continuous retraining of MaPredict (1h, 4h)
    and MaInvest (1d) machine learning & macro forecasting models.
    """

    TOP_ASSETS: List[str] = [
        "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX"
    ]

    SUPPORTED_TIMEFRAMES: List[str] = ["1h", "4h", "1d"]

    REQUIRED_RAW_COLUMNS: List[str] = [
        "timestamp", "open", "high", "low", "close", "volume"
    ]

    METADATA_COLUMNS: List[str] = [
        "timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume"
    ]

    TARGET_COLUMNS: List[str] = [
        "target_return",
        "target_direction",
        "target_1h",
        "target_4h",
        "target_1d",
        "target_dir_1h",
        "target_dir_4h",
        "target_dir_1d",
    ]

    def __init__(
        self,
        snapshot_dir: str = "data/training_snapshots",
        cache_dir: str = ".data_cache",
        fetcher: Optional[BaseDataFetcher] = None,
        macro_pipeline: Optional[MacroDataPipeline] = None,
    ):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        self.fetcher = fetcher or CryptoDataFetcher(cache_dir=cache_dir)
        self.macro_pipeline = macro_pipeline or MacroDataPipeline()

    # =================================================================
    # Validation & Data Cleansing (Garbage In = Garbage Out Guard)
    # =================================================================

    @classmethod
    def validate_raw_ohlcv(cls, df: pd.DataFrame, min_rows: int = 200) -> pd.DataFrame:
        """
        Validates and cleans raw OHLCV candle DataFrame:
        - Ensures presence of required columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        - Converts and sorts timestamps chronologically (strictly ascending)
        - Removes duplicate timestamps
        - Ensures positive numeric prices and volumes
        - Enforces minimum row count
        """
        if df is None or df.empty:
            raise DataValidationError("Input OHLCV DataFrame is empty or None.")

        clean = df.copy()
        clean.columns = [str(c).strip().lower() for c in clean.columns]

        missing_cols = [c for c in cls.REQUIRED_RAW_COLUMNS if c not in clean.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required OHLCV columns: {missing_cols}")

        # Ensure datetime format for timestamps
        clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce")
        if clean["timestamp"].isna().any():
            clean = clean.dropna(subset=["timestamp"])

        # Chronological sort and deduplication
        clean = clean.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

        # Numeric conversions
        for col in ["open", "high", "low", "close", "volume"]:
            clean[col] = pd.to_numeric(clean[col], errors="coerce")

        clean = clean.dropna(subset=["open", "high", "low", "close", "volume"])

        # Sanity check: prices must be positive and high >= low
        valid_prices = (
            (clean["open"] > 0) &
            (clean["high"] > 0) &
            (clean["low"] > 0) &
            (clean["close"] > 0) &
            (clean["volume"] >= 0) &
            (clean["high"] >= clean["low"])
        )
        clean = clean[valid_prices].reset_index(drop=True)

        if len(clean) < min_rows:
            raise InsufficientDataError(
                f"Insufficient candle count after cleaning: found {len(clean)} rows, "
                f"minimum {min_rows} required."
            )

        return clean

    # =================================================================
    # Feature Engineering (Strictly Causal / No Lookahead Bias)
    # =================================================================

    @classmethod
    def calculate_features(
        cls,
        df: pd.DataFrame,
        symbol: str = "BTC",
        timeframe: str = "1h",
        lag_periods: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Calculates stationary, causal financial features on validated OHLCV data.

        Features computed:
        1. Log Returns & Autoregressive Lags (t-1, t-2, t-3, t-5, t-10)
        2. Volatility Standardized Returns (Z-scores)
        3. Realized Garman-Klass (GK) and Parkinson Volatility Estimators
        4. Order Flow Imbalance (OFI) proxy & Volume Delta Dynamics
        5. Fast/Slow Moving Average Spreads (SMA 20/50, EMA 12/26, EMA 50/SMA 200)
        6. Mayer Multiple (Close / 200-period SMA) & Valuation Regime Codes
        7. Multi-Horizon Forward Return Targets (t+1, t+4, t+24) and Binary Direction
        """
        clean_symbol = SymbolNormalizer.normalize(symbol)
        tf = timeframe.lower()
        if lag_periods is None:
            lag_periods = [1, 2, 3, 5, 10]

        data = df.copy().sort_values("timestamp").reset_index(drop=True)

        data["symbol"] = clean_symbol
        data["timeframe"] = tf

        close = data["close"].astype(np.float64)
        open_p = data["open"].astype(np.float64)
        high = data["high"].astype(np.float64)
        low = data["low"].astype(np.float64)
        volume = data["volume"].astype(np.float64)

        eps = 1e-8

        # -------------------------------------------------------------
        # 1. Log Returns & Lagged Returns
        # -------------------------------------------------------------
        # Causal: r_t = ln(Close_t / Close_{t-1})
        log_return = np.log((close + eps) / (close.shift(1) + eps))
        data["log_return"] = log_return

        # Multi-period rolling log returns
        data["cum_return_5"] = np.log((close + eps) / (close.shift(5) + eps))
        data["cum_return_20"] = np.log((close + eps) / (close.shift(20) + eps))

        # Rolling standard deviation of returns (20 periods with graceful min_periods)
        volatility_20 = log_return.rolling(window=20, min_periods=5).std() + eps
        data["volatility_20"] = volatility_20

        # Volatility-standardized return (Z-score)
        z_return = log_return / volatility_20
        data["z_return"] = z_return

        # Autoregressive Lags (strictly shifted forward in time, i.e., past info at t)
        for lag in lag_periods:
            data[f"return_lag_{lag}"] = log_return.shift(lag)
            data[f"z_return_lag_{lag}"] = z_return.shift(lag)

        # -------------------------------------------------------------
        # 2. Garman-Klass (GK) & Parkinson Realized Volatility Estimators
        # -------------------------------------------------------------
        # GK single-candle variance:
        # sigma_GK^2 = 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
        log_hl = np.log((high + eps) / (low + eps))
        log_co = np.log((close + eps) / (open_p + eps))
        c2 = 2.0 * np.log(2.0) - 1.0  # ~0.386294

        gk_variance = 0.5 * (log_hl ** 2) - c2 * (log_co ** 2)
        gk_variance = np.maximum(0.0, gk_variance)

        data["gk_variance_instant"] = gk_variance
        data["volatility_gk_14"] = np.sqrt(gk_variance.rolling(window=14, min_periods=5).mean() + eps)
        data["volatility_gk_30"] = np.sqrt(gk_variance.rolling(window=30, min_periods=5).mean() + eps)

        # Annualization factor depending on timeframe
        if tf == "1h":
            annual_factor = 365.0 * 24.0
        elif tf == "4h":
            annual_factor = 365.0 * 6.0
        elif tf == "1d":
            annual_factor = 365.0
        else:
            annual_factor = 365.0 * 24.0

        data["volatility_gk_annualized"] = data["volatility_gk_14"] * np.sqrt(annual_factor)

        # Parkinson (1980) High/Low Volatility: (ln(H/L)^2) / (4*ln(2))
        parkinson_var = (log_hl ** 2) / (4.0 * np.log(2.0))
        data["volatility_parkinson_14"] = np.sqrt(parkinson_var.rolling(window=14, min_periods=5).mean() + eps)

        # Normalized Average True Range (ATR_norm)
        tr = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - close.shift(1)),
                np.abs(low - close.shift(1))
            )
        )
        atr_14 = tr.rolling(window=14, min_periods=5).mean()
        data["atr_norm_14"] = atr_14 / (close + eps)

        # -------------------------------------------------------------
        # 3. Order Flow Imbalance (OFI) Proxy & Microstructure Dynamics
        # -------------------------------------------------------------
        # Close Location Value (CLV): CLV = ((C - L) - (H - C)) / (H - L) in [-1, +1]
        candle_range = high - low
        safe_range = np.where(candle_range > eps, candle_range, eps)
        clv = ((close - low) - (high - close)) / safe_range
        clv = np.clip(clv, -1.0, 1.0)
        data["clv"] = clv

        # Candle OFI Proxy: OFI = CLV * Volume
        ofi_raw = clv * volume
        data["ofi_raw"] = ofi_raw

        # Volume-normalized OFI
        vol_mean_20 = volume.rolling(window=20, min_periods=5).mean() + eps
        data["ofi_normalized_20"] = ofi_raw / vol_mean_20
        data["ofi_proxy_14"] = data["ofi_normalized_20"].rolling(window=14, min_periods=5).mean()

        # Signed Volume Delta based on price movement direction
        price_delta_sign = np.sign(close - close.shift(1))
        volume_delta = price_delta_sign * volume
        data["volume_delta"] = volume_delta
        data["volume_delta_ratio"] = volume_delta / (volume + eps)
        data["volume_log_return"] = np.log((volume + eps) / (volume.shift(1) + eps))

        # -------------------------------------------------------------
        # 4. SMA / EMA Spreads & Trend Velocities
        # -------------------------------------------------------------
        sma_20 = close.rolling(window=20, min_periods=5).mean()
        sma_50 = close.rolling(window=50, min_periods=10).mean()
        sma_200 = close.rolling(window=200, min_periods=30).mean()

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()

        # Stationary percentage spreads
        data["sma_spread_20_50"] = (sma_20 - sma_50) / (sma_50 + eps)
        data["ema_spread_12_26"] = (ema_12 - ema_26) / (ema_26 + eps)
        data["ema_spread_50_200"] = (ema_50 - sma_200) / (sma_200 + eps)

        # Trend slopes / velocities
        data["ema_slope_12"] = (ema_12 - ema_12.shift(1)) / (ema_12.shift(1) + eps)
        data["sma_slope_50"] = (sma_50 - sma_50.shift(1)) / (sma_50.shift(1) + eps)

        # Golden / Death Cross state: +1 for Golden Cross (EMA50 >= SMA200), -1 for Death Cross
        data["trend_cross_state"] = np.where(ema_50 >= sma_200, 1.0, -1.0)

        # Bollinger Band Position
        data["bollinger_b_20"] = (close - sma_20) / (2.0 * volatility_20 * close + eps)

        # -------------------------------------------------------------
        # 5. Mayer Multiple & Valuation Regime
        # -------------------------------------------------------------
        # Mayer Multiple: Close / 200-period SMA
        mayer_multiple = close / (sma_200 + eps)
        data["mayer_multiple"] = mayer_multiple
        data["mayer_log_ratio"] = np.log((close + eps) / (sma_200 + eps))

        # Mayer Rolling Z-score (60 periods)
        mayer_mean_60 = mayer_multiple.rolling(window=60, min_periods=10).mean()
        mayer_std_60 = mayer_multiple.rolling(window=60, min_periods=10).std() + eps
        data["mayer_zscore_60"] = (mayer_multiple - mayer_mean_60) / mayer_std_60

        # Mayer Regime classification codes:
        # 0: Deep Value (< 0.8)
        # 1: Accumulation (0.8 - 1.0)
        # 2: Bullish Expansion (1.0 - 1.4)
        # 3: Overheated (1.4 - 2.4)
        # 4: Euphoric Bubble (>= 2.4)
        mayer_regime_codes = np.select(
            [
                mayer_multiple < 0.8,
                mayer_multiple < 1.0,
                mayer_multiple < 1.4,
                mayer_multiple < 2.4,
            ],
            [0, 1, 2, 3],
            default=4
        )
        data["mayer_regime_code"] = mayer_regime_codes.astype(int)

        # -------------------------------------------------------------
        # 6. Multi-Horizon Future Targets (Shifted Causally for Supervised Retraining)
        # -------------------------------------------------------------
        # Target returns:
        # 1-step target: ln(Close_{t+1} / Close_t)
        # 4-step target: ln(Close_{t+4} / Close_t)
        # 24-step target (or 1-day for 1d timeframe): ln(Close_{t+24} / Close_t)
        data["target_1h"] = np.log((close.shift(-1) + eps) / (close + eps))
        data["target_4h"] = np.log((close.shift(-4) + eps) / (close + eps))
        data["target_1d"] = np.log((close.shift(-24 if tf == "1h" else -1) + eps) / (close + eps))

        # Binary directional targets
        data["target_dir_1h"] = (data["target_1h"] > 0).astype(int)
        data["target_dir_4h"] = (data["target_4h"] > 0).astype(int)
        data["target_dir_1d"] = (data["target_1d"] > 0).astype(int)

        # Primary target aligned to requested timeframe:
        if tf == "1h":
            data["target_return"] = data["target_1h"]
            data["target_direction"] = data["target_dir_1h"]
        elif tf == "4h":
            # For 4h timeframe, 1-step forward is a 4-hour horizon
            data["target_return"] = data["target_1h"]
            data["target_direction"] = data["target_dir_1h"]
        elif tf == "1d":
            # For 1d timeframe, 1-step forward is a 1-day horizon
            data["target_return"] = data["target_1d"]
            data["target_direction"] = data["target_dir_1d"]
        else:
            data["target_return"] = data["target_1h"]
            data["target_direction"] = data["target_dir_1h"]

        return data

    @classmethod
    def clean_dataset_for_training(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops warmup NaN rows created by rolling windows (e.g. 200 SMA)
        and tail NaN rows created by future target shifts.
        Returns a clean, fully populated dataset ready for ML model ingestion.
        """
        # All columns except non-critical metadata must be non-NaN
        clean = df.dropna().reset_index(drop=True)
        return clean

    @classmethod
    def get_feature_column_names(cls, df: pd.DataFrame) -> List[str]:
        """Returns the list of purely causal feature column names (excluding metadata & targets)."""
        excluded = set(cls.METADATA_COLUMNS) | set(cls.TARGET_COLUMNS)
        return [c for c in df.columns if c not in excluded]

    # =================================================================
    # Strict No-Lookahead Bias Guarantee & Audit
    # =================================================================

    @classmethod
    def audit_no_lookahead_bias(
        cls,
        raw_df: pd.DataFrame,
        symbol: str = "BTC",
        timeframe: str = "1h",
        check_points: int = 5,
        tolerance: float = 1e-7,
    ) -> Dict[str, Any]:
        """
        Rigorous mathematical audit for Strict No-Lookahead Bias Guarantee.

        How it works:
        1. Computes features on the FULL historical dataset [0 .. N].
        2. Selects multiple random cutoff timestamps K (where K >= 210 to satisfy 200 SMA warmup).
        3. Computes features on the TRUNCATED dataset [0 .. K].
        4. Verifies that the feature row computed at index K on the truncated dataset
           is EXACTLY IDENTICAL to the feature row at index K on the full dataset.
        5. If any feature differs by > tolerance, raises LookaheadBiasError!
        """
        validated_raw = cls.validate_raw_ohlcv(raw_df, min_rows=220)
        n_rows = len(validated_raw)

        # Full features
        full_featured = cls.calculate_features(validated_raw, symbol=symbol, timeframe=timeframe)
        feature_cols = cls.get_feature_column_names(full_featured)

        if not feature_cols:
            raise LookaheadBiasError("No feature columns identified for lookahead bias audit.")

        # Pick evenly spaced audit checkpoints between 210 and n_rows - 5
        min_k = 210
        max_k = n_rows - 5
        if max_k <= min_k:
            cutoffs = [min_k]
        else:
            step = max(1, (max_k - min_k) // check_points)
            cutoffs = list(range(min_k, max_k, step))[:check_points]
            if max_k not in cutoffs:
                cutoffs.append(max_k)

        max_discrepancy = 0.0
        discrepancy_details = []

        for k in cutoffs:
            truncated_raw = validated_raw.iloc[:k + 1].copy()
            truncated_featured = cls.calculate_features(truncated_raw, symbol=symbol, timeframe=timeframe)

            full_row = full_featured.iloc[k][feature_cols].astype(float).to_numpy()
            trunc_row = truncated_featured.iloc[k][feature_cols].astype(float).to_numpy()

            # 1. NaN mismatch check: a causal feature must be populated identically
            nan_mismatch = np.isnan(full_row) != np.isnan(trunc_row)
            if np.any(nan_mismatch):
                bad_idx = np.where(nan_mismatch)[0]
                bad_features = [feature_cols[i] for i in bad_idx]
                err_msg = (
                    f"Lookahead bias detected at candle {k}: NaN presence mismatch on features {bad_features} "
                    f"between truncated and full dataset!"
                )
                logger.error(err_msg)
                raise LookaheadBiasError(err_msg)

            # 2. Numerical equality check on valid values
            valid_mask = ~(np.isnan(full_row) | np.isnan(trunc_row))
            if np.any(valid_mask):
                diff = np.abs(full_row[valid_mask] - trunc_row[valid_mask])
                max_diff = float(np.max(diff))
                if max_diff > max_discrepancy:
                    max_discrepancy = max_diff

                if max_diff > tolerance:
                    valid_feat_names = [feature_cols[i] for i in np.where(valid_mask)[0]]
                    bad_idx = np.where(diff > tolerance)[0]
                    bad_features = [valid_feat_names[i] for i in bad_idx]
                    err_msg = (
                        f"Lookahead bias detected at candle {k} for features {bad_features}! "
                        f"Max discrepancy: {max_diff} > tolerance {tolerance}"
                    )
                    discrepancy_details.append(err_msg)
                    logger.error(err_msg)
                    raise LookaheadBiasError(err_msg)

        # Audit monotonic timestamp causality
        ts_diffs = validated_raw["timestamp"].diff().dropna()
        if (ts_diffs <= pd.Timedelta(0)).any():
            raise LookaheadBiasError("Timestamps are not strictly monotonically increasing!")

        return {
            "passed": True,
            "audited_cutoffs": cutoffs,
            "audited_feature_count": len(feature_cols),
            "max_discrepancy": max_discrepancy,
            "tolerance": tolerance,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # =================================================================
    # Snapshot Storage & Versioning
    # =================================================================

    def _generate_snapshot_filename(
        self,
        symbol: str,
        timeframe: str,
        timestamp_str: Optional[str] = None
    ) -> Tuple[str, str]:
        """Generates versioned snapshot filename and metadata filename."""
        clean_sym = SymbolNormalizer.normalize(symbol)
        ts = timestamp_str or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_name = f"{clean_sym}_{timeframe.lower()}_{ts}"
        return f"{base_name}.csv", f"{base_name}_meta.json"

    def save_snapshot(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        version_tag: Optional[str] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Saves a clean, versioned dataset snapshot to disk (CSV) alongside a JSON metadata sidecar.
        Returns the absolute path to the saved dataset file.
        """
        clean_sym = SymbolNormalizer.normalize(symbol)
        csv_filename, meta_filename = self._generate_snapshot_filename(clean_sym, timeframe, version_tag)

        csv_path = self.snapshot_dir / csv_filename
        meta_path = self.snapshot_dir / meta_filename

        # Write clean DataFrame to CSV
        df.to_csv(csv_path, index=False)

        # Compute SHA256 data hash for integrity tracking
        data_bytes = df.to_csv(index=False).encode("utf-8")
        data_hash = hashlib.sha256(data_bytes).hexdigest()

        feature_names = self.get_feature_column_names(df)
        target_names = [c for c in df.columns if c in self.TARGET_COLUMNS]

        start_ts = str(df["timestamp"].iloc[0]) if "timestamp" in df.columns and len(df) > 0 else None
        end_ts = str(df["timestamp"].iloc[-1]) if "timestamp" in df.columns and len(df) > 0 else None

        metadata = {
            "symbol": clean_sym,
            "timeframe": timeframe.lower(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "snapshot_file": csv_filename,
            "row_count": int(len(df)),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "target_names": target_names,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "data_hash": data_hash,
            "file_size_bytes": os.path.getsize(csv_path),
            "no_lookahead_verified": True,
        }

        if custom_metadata:
            metadata.update(custom_metadata)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved retraining snapshot for {clean_sym} ({timeframe}) to {csv_path} ({len(df)} rows)")
        return str(csv_path)

    def load_latest_snapshot(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Finds and loads the most recent training snapshot for a given symbol and timeframe.
        """
        clean_sym = SymbolNormalizer.normalize(symbol)
        tf = timeframe.lower()
        pattern = f"{clean_sym}_{tf}_*.csv"

        matching_files = sorted(self.snapshot_dir.glob(pattern), reverse=True)
        if not matching_files:
            return None

        latest_path = matching_files[0]
        try:
            df = pd.read_csv(latest_path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
        except Exception as e:
            logger.error(f"Failed to read snapshot {latest_path}: {e}")
            return None

    def list_snapshots(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Lists all available snapshots with their metadata, sorted newest to oldest.
        """
        meta_files = sorted(self.snapshot_dir.glob("*_meta.json"), reverse=True)
        results = []

        clean_sym = SymbolNormalizer.normalize(symbol) if symbol else None
        tf = timeframe.lower() if timeframe else None

        for meta_file in meta_files:
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                if clean_sym and meta.get("symbol") != clean_sym:
                    continue
                if tf and meta.get("timeframe") != tf:
                    continue

                results.append(meta)
            except Exception as e:
                logger.warning(f"Could not parse snapshot metadata {meta_file}: {e}")

        return results

    # =================================================================
    # Primary API: Build Training Dataset
    # =================================================================

    def fetch_raw_candles(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 1000,
    ) -> pd.DataFrame:
        """
        Fetches raw OHLCV candles using appropriate infrastructure fetcher.
        For '1d' timeframe, routes through MacroDataPipeline; for intraday ('1h', '4h'), routes through CryptoDataFetcher.
        """
        clean_sym = SymbolNormalizer.normalize(symbol)
        tf = timeframe.lower()

        if tf == "1d":
            return self.macro_pipeline.fetch_daily_candles(symbol=clean_sym, limit=limit)
        else:
            return self.fetcher.fetch_ohlcv(symbol=clean_sym, timeframe=tf, limit=limit)

    def build_training_dataset(
        self,
        symbol: str,
        timeframe: str = "1h",
        min_rows: int = 200,
        raw_df: Optional[pd.DataFrame] = None,
        save_snapshot: bool = True,
        version_tag: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Main public interface for generating fully validated, stationary,
        and causal retraining datasets for MaPredict (1h, 4h) and MaInvest (1d).

        Args:
            symbol: Ticker symbol (e.g. 'BTC', 'ETH', 'SOL', 'BTCUSDT').
            timeframe: Timeframe code ('1h', '4h', '1d').
            min_rows: Minimum clean historical rows required (default 200).
            raw_df: Optional explicit OHLCV DataFrame (if provided, bypasses network fetch).
            save_snapshot: Whether to automatically save versioned snapshot to data/training_snapshots/.
            version_tag: Optional explicit version tag/timestamp for snapshot file.

        Returns:
            Clean, stationary pandas DataFrame with all features and targets ready for model training.
        """
        clean_sym = SymbolNormalizer.normalize(symbol)
        tf = timeframe.lower()

        # 1. Acquire raw OHLCV candles
        if raw_df is not None:
            raw_candles = raw_df.copy()
        else:
            # Request enough candles to satisfy 200 SMA lookback plus targets
            fetch_limit = max(min_rows + 250, 1000)
            raw_candles = self.fetch_raw_candles(clean_sym, timeframe=tf, limit=fetch_limit)

        # 2. Validate raw data integrity
        validated = self.validate_raw_ohlcv(raw_candles, min_rows=min_rows)

        # 3. Calculate all features causally
        featured = self.calculate_features(validated, symbol=clean_sym, timeframe=tf)

        # 4. Clean NaNs from rolling window warmup & target lookahead shifts
        clean_dataset = self.clean_dataset_for_training(featured)

        if len(clean_dataset) < 10:
            raise InsufficientDataError(
                f"Dataset for {clean_sym} ({tf}) has only {len(clean_dataset)} rows after dropna. "
                f"Provide more historical candles."
            )

        # 5. Strictly audit for no-lookahead bias
        audit_res = self.audit_no_lookahead_bias(validated, symbol=clean_sym, timeframe=tf)
        logger.debug(f"Lookahead audit passed for {clean_sym} ({tf}): max discrepancy = {audit_res['max_discrepancy']}")

        # 6. Save snapshot if requested
        if save_snapshot:
            self.save_snapshot(
                df=clean_dataset,
                symbol=clean_sym,
                timeframe=tf,
                version_tag=version_tag,
                custom_metadata={"audit_result": audit_res},
            )

        return clean_dataset

    def build_multi_asset_dataset(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        min_rows: int = 200,
        save_snapshot: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        Batch builder: Generates retraining datasets across multiple assets and timeframes.
        Returns: Dict with keys '{symbol}_{timeframe}' mapping to clean DataFrames.
        """
        target_symbols = symbols or self.TOP_ASSETS
        target_tfs = timeframes or self.SUPPORTED_TIMEFRAMES

        datasets: Dict[str, pd.DataFrame] = {}

        for sym in target_symbols:
            clean_sym = SymbolNormalizer.normalize(sym)
            for tf in target_tfs:
                key = f"{clean_sym}_{tf.lower()}"
                try:
                    df = self.build_training_dataset(
                        symbol=clean_sym,
                        timeframe=tf,
                        min_rows=min_rows,
                        save_snapshot=save_snapshot,
                    )
                    datasets[key] = df
                    logger.info(f"Built retraining dataset for {key}: {len(df)} rows")
                except Exception as e:
                    logger.error(f"Failed to build retraining dataset for {key}: {e}")

        return datasets

    # =================================================================
    # ML Matrix Extraction Helpers (PyTorch & GBDT Ready)
    # =================================================================

    def extract_feature_matrix(
        self,
        df: pd.DataFrame,
        target_col: str = "target_return"
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Transforms clean retraining DataFrame into numpy (X_features, y_target, feature_names).
        Ready for Scikit-Learn GBDT, LightGBM, CatBoost, or PyTorch Datasets.
        """
        feature_names = self.get_feature_column_names(df)
        if not feature_names:
            raise RetrainingDataBuilderError("No feature columns available in dataset.")

        if target_col not in df.columns:
            raise RetrainingDataBuilderError(f"Target column '{target_col}' not found in DataFrame.")

        X_mat = df[feature_names].to_numpy(dtype=np.float32)
        y_vec = df[target_col].to_numpy(dtype=np.float32)

        return X_mat, y_vec, feature_names

    def extract_multi_horizon_matrices(
        self,
        df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Extracts multi-horizon targets [1h, 4h, 1d] for Multi-Horizon attention models.
        Returns: (X_mat, Y_returns, Y_directions, feature_names)
        """
        feature_names = self.get_feature_column_names(df)
        X_mat = df[feature_names].to_numpy(dtype=np.float32)

        Y_returns = df[["target_1h", "target_4h", "target_1d"]].to_numpy(dtype=np.float32)
        Y_directions = df[["target_dir_1h", "target_dir_4h", "target_dir_1d"]].to_numpy(dtype=np.float32)

        return X_mat, Y_returns, Y_directions, feature_names
