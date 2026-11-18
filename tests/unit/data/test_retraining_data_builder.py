"""
MaBuy V4 — Data Engineering Department Unit Test Suite.
Tests RetrainingDataBuilder:
  1. OHLCV Data Validation & Cleansing (Garbage In = Garbage Out Guard).
  2. Mathematical correctness of feature computations:
     - Log Returns & Autoregressive Lags (t-1, t-2, t-3, t-5, t-10)
     - Garman-Klass (GK) and Parkinson Realized Volatilities
     - Order Flow Imbalance (OFI) proxy & Volume Delta Dynamics
     - SMA / EMA spreads (SMA 20/50, EMA 12/26, EMA 50/SMA 200)
     - Mayer Multiple (Close / 200 SMA) & Valuation Regime Codes
  3. Strict No-Lookahead Bias Guarantee (truncated history invariance test).
  4. Snapshot Versioning & Storage in data/training_snapshots/ with metadata sidecars.
  5. Multi-Timeframe (1h, 4h, 1d) and Multi-Asset (BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX) snapshot builds.
  6. ML Matrix extraction (GBDT & PyTorch compatible).
"""
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src.infrastructure.data_fetchers.retraining_data_builder import (
    RetrainingDataBuilder,
    InsufficientDataError,
    LookaheadBiasError,
    DataValidationError,
)


class TestRetrainingDataBuilder(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="snapshot_test_")
        self.temp_cache = tempfile.mkdtemp(prefix="cache_test_")
        self.builder = RetrainingDataBuilder(
            snapshot_dir=self.temp_dir,
            cache_dir=self.temp_cache,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        shutil.rmtree(self.temp_cache, ignore_errors=True)

    def _generate_synthetic_ohlcv(
        self,
        n_candles: int = 300,
        base_price: float = 65000.0,
        freq: str = "1h",
        seed: int = 42
    ) -> pd.DataFrame:
        """Generates synthetic, deterministic OHLCV candles without lookahead bias."""
        np.random.seed(seed)
        start_ts = pd.Timestamp("2026-01-01 00:00:00", tz="UTC")
        dates = pd.date_range(start=start_ts, periods=n_candles, freq=freq)

        records = []
        price = base_price
        for ts in dates:
            ret = np.random.normal(0.0002, 0.015)
            open_p = price
            close_p = max(1.0, price * (1.0 + ret))
            high_p = max(open_p, close_p) * (1.0 + abs(np.random.normal(0, 0.005)))
            low_p = min(open_p, close_p) * (1.0 - abs(np.random.normal(0, 0.005)))
            volume = float(np.random.uniform(50.0, 500.0))

            records.append({
                "timestamp": ts,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": volume,
            })
            price = close_p

        return pd.DataFrame(records)

    def test_top_assets_and_timeframes_configuration(self):
        """Verifies configured assets and supported timeframes match MaBuy standards."""
        expected_assets = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX"]
        for asset in expected_assets:
            self.assertIn(asset, self.builder.TOP_ASSETS)

        self.assertIn("1h", self.builder.SUPPORTED_TIMEFRAMES)
        self.assertIn("4h", self.builder.SUPPORTED_TIMEFRAMES)
        self.assertIn("1d", self.builder.SUPPORTED_TIMEFRAMES)

    def test_raw_ohlcv_validation_and_cleaning(self):
        """Tests data cleaning: sorting, duplicate removal, invalid price handling, and minimum rows."""
        # 1. Valid data passing
        raw = self._generate_synthetic_ohlcv(n_candles=250)
        clean = self.builder.validate_raw_ohlcv(raw, min_rows=200)
        self.assertEqual(len(clean), 250)

        # 2. Out of order timestamps & duplicates -> successfully sorted & deduped
        shuffled = pd.concat([raw.iloc[50:], raw.iloc[:50], raw.iloc[10:15]], ignore_index=True)
        clean_shuffled = self.builder.validate_raw_ohlcv(shuffled, min_rows=200)
        self.assertEqual(len(clean_shuffled), 250)
        # Check strict monotonicity
        self.assertTrue((clean_shuffled["timestamp"].diff().dropna() > pd.Timedelta(0)).all())

        # 3. Missing required columns
        invalid_cols_df = raw.drop(columns=["close"])
        with self.assertRaises(DataValidationError):
            self.builder.validate_raw_ohlcv(invalid_cols_df, min_rows=200)

        # 4. Insufficient rows
        short_df = raw.iloc[:100]
        with self.assertRaises(InsufficientDataError):
            self.builder.validate_raw_ohlcv(short_df, min_rows=200)

        # 5. Invalid negative price rows get filtered out
        bad_price_df = raw.copy()
        bad_price_df.loc[10, "open"] = -50.0
        clean_bad_price = self.builder.validate_raw_ohlcv(bad_price_df, min_rows=200)
        self.assertEqual(len(clean_bad_price), 249)

    def test_lagged_returns_and_zscore_features(self):
        """Verifies calculation of log returns, autoregressive lags, and volatility Z-scores."""
        raw = self._generate_synthetic_ohlcv(n_candles=300)
        featured = self.builder.calculate_features(raw, symbol="BTC", timeframe="1h")

        # 1. Log return calculation
        expected_ret = np.log(raw["close"] / raw["close"].shift(1))
        np.testing.assert_allclose(
            featured["log_return"].iloc[1:50],
            expected_ret.iloc[1:50],
            rtol=1e-5
        )

        # 2. Autoregressive Lags
        for lag in [1, 2, 3, 5, 10]:
            col_name = f"return_lag_{lag}"
            self.assertIn(col_name, featured.columns)
            np.testing.assert_allclose(
                featured[col_name].iloc[lag + 1 : lag + 20],
                featured["log_return"].shift(lag).iloc[lag + 1 : lag + 20],
                rtol=1e-5
            )

        # 3. Volatility standardized returns (Z-scores)
        self.assertIn("z_return", featured.columns)
        self.assertIn("z_return_lag_1", featured.columns)
        self.assertIn("z_return_lag_3", featured.columns)
        self.assertTrue(featured["volatility_20"].dropna().gt(0).all())

    def test_garman_klass_and_volatility_features(self):
        """Verifies mathematical correctness of Garman-Klass, Parkinson, and ATR volatility."""
        raw = self._generate_synthetic_ohlcv(n_candles=300)
        featured = self.builder.calculate_features(raw, symbol="ETH", timeframe="1h")

        self.assertIn("volatility_gk_14", featured.columns)
        self.assertIn("volatility_gk_30", featured.columns)
        self.assertIn("volatility_gk_annualized", featured.columns)
        self.assertIn("volatility_parkinson_14", featured.columns)
        self.assertIn("atr_norm_14", featured.columns)

        # Manual calculation of single-step GK variance at index 50
        c2 = 2.0 * np.log(2.0) - 1.0
        h_l = np.log(raw["high"].iloc[50] / raw["low"].iloc[50])
        c_o = np.log(raw["close"].iloc[50] / raw["open"].iloc[50])
        manual_gk = 0.5 * (h_l ** 2) - c2 * (c_o ** 2)

        np.testing.assert_almost_equal(
            featured["gk_variance_instant"].iloc[50],
            max(0.0, manual_gk),
            decimal=5
        )

        # Ensure all rolling volatilities are positive
        valid_gk = featured["volatility_gk_14"].dropna()
        self.assertTrue((valid_gk > 0).all())

    def test_order_flow_imbalance_proxy_and_volume_delta(self):
        """Verifies OFI proxy, CLV bounds, and signed volume delta."""
        raw = self._generate_synthetic_ohlcv(n_candles=300)
        featured = self.builder.calculate_features(raw, symbol="SOL", timeframe="1h")

        # 1. Close Location Value (CLV) must be in [-1.0, 1.0]
        clv_vals = featured["clv"].dropna()
        self.assertTrue((clv_vals >= -1.0 - 1e-6).all())
        self.assertTrue((clv_vals <= 1.0 + 1e-6).all())

        # 2. OFI proxy columns exist and are finite
        self.assertIn("ofi_raw", featured.columns)
        self.assertIn("ofi_normalized_20", featured.columns)
        self.assertIn("ofi_proxy_14", featured.columns)
        self.assertIn("volume_delta", featured.columns)
        self.assertIn("volume_delta_ratio", featured.columns)

        # 3. Check volume delta sign matches price change
        price_diff = raw["close"].diff().iloc[10:30]
        v_delta = featured["volume_delta"].iloc[10:30]
        for p_d, v_d in zip(price_diff, v_delta):
            if p_d > 0:
                self.assertGreater(v_d, 0)
            elif p_d < 0:
                self.assertLess(v_d, 0)

    def test_sma_ema_spreads_and_mayer_multiple(self):
        """Verifies Moving Average spreads, trend crosses, and Mayer Multiple regime classifications."""
        raw = self._generate_synthetic_ohlcv(n_candles=350, base_price=50000.0)
        featured = self.builder.calculate_features(raw, symbol="BTC", timeframe="1h")

        # 1. Moving average spreads
        self.assertIn("sma_spread_20_50", featured.columns)
        self.assertIn("ema_spread_12_26", featured.columns)
        self.assertIn("ema_spread_50_200", featured.columns)
        self.assertIn("trend_cross_state", featured.columns)

        # 2. Mayer Multiple: Close / 200-SMA
        expected_sma_200 = raw["close"].rolling(200).mean().iloc[250]
        expected_mayer = raw["close"].iloc[250] / expected_sma_200
        actual_mayer = featured["mayer_multiple"].iloc[250]
        self.assertAlmostEqual(actual_mayer, expected_mayer, places=4)

        # 3. Mayer regime codes in {0, 1, 2, 3, 4}
        regimes = featured["mayer_regime_code"].dropna()
        self.assertTrue(regimes.isin([0, 1, 2, 3, 4]).all())

    def test_strict_no_lookahead_bias_guarantee(self):
        """
        Validates that features calculated at candle K are invariant
        to whether future candles K+1 .. N are present in the dataset.
        """
        raw = self._generate_synthetic_ohlcv(n_candles=350)
        
        # Run audit on clean data -> must pass with zero discrepancy
        audit_result = self.builder.audit_no_lookahead_bias(
            raw_df=raw,
            symbol="BTC",
            timeframe="1h",
            check_points=5,
            tolerance=1e-7,
        )

        self.assertTrue(audit_result["passed"])
        self.assertLess(audit_result["max_discrepancy"], 1e-7)
        self.assertGreater(audit_result["audited_feature_count"], 15)

    def test_detect_lookahead_bias_violation(self):
        """Verifies that non-causal operations trigger LookaheadBiasError."""
        raw = self._generate_synthetic_ohlcv(n_candles=300)

        class LeakyDataBuilder(RetrainingDataBuilder):
            @classmethod
            def calculate_features(cls, df: pd.DataFrame, symbol: str = "BTC", timeframe: str = "1h", lag_periods=None):
                data = super().calculate_features(df, symbol=symbol, timeframe=timeframe, lag_periods=lag_periods)
                # Intentionally inject forward lookahead leakage (future price at t+3)
                data["future_leaked_indicator"] = data["close"].shift(-3)
                return data

        with self.assertRaises(LookaheadBiasError):
            LeakyDataBuilder.audit_no_lookahead_bias(raw, symbol="BTC", timeframe="1h")

    def test_snapshot_saving_versioning_and_metadata(self):
        """Tests snapshot disk persistence, timestamp versioning, and JSON metadata sidecars."""
        raw = self._generate_synthetic_ohlcv(n_candles=280)
        df_clean = self.builder.build_training_dataset(
            symbol="BTC",
            timeframe="1h",
            min_rows=200,
            raw_df=raw,
            save_snapshot=True,
            version_tag="20260823_120000"
        )

        self.assertGreater(len(df_clean), 0)

        # Check saved snapshot files
        expected_csv = os.path.join(self.temp_dir, "BTC_1h_20260823_120000.csv")
        expected_meta = os.path.join(self.temp_dir, "BTC_1h_20260823_120000_meta.json")

        self.assertTrue(os.path.exists(expected_csv))
        self.assertTrue(os.path.exists(expected_meta))

        # Validate JSON metadata content
        with open(expected_meta, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(meta["symbol"], "BTC")
        self.assertEqual(meta["timeframe"], "1h")
        self.assertEqual(meta["row_count"], len(df_clean))
        self.assertTrue(meta["no_lookahead_verified"])
        self.assertIn("data_hash", meta)
        self.assertIn("mayer_multiple", meta["feature_names"])
        self.assertIn("volatility_gk_14", meta["feature_names"])

        # Test load_latest_snapshot
        loaded_df = self.builder.load_latest_snapshot(symbol="BTC", timeframe="1h")
        self.assertIsNotNone(loaded_df)
        self.assertEqual(len(loaded_df), len(df_clean))

        # Test list_snapshots
        snapshots_list = self.builder.list_snapshots(symbol="BTC", timeframe="1h")
        self.assertEqual(len(snapshots_list), 1)
        self.assertEqual(snapshots_list[0]["symbol"], "BTC")

    def test_multi_timeframe_dataset_building(self):
        """Tests building datasets for 1h, 4h, and 1d timeframes."""
        for tf in ["1h", "4h", "1d"]:
            raw = self._generate_synthetic_ohlcv(n_candles=260, freq="1h" if tf == "1h" else ("4h" if tf == "4h" else "1D"))
            dataset = self.builder.build_training_dataset(
                symbol="ETH",
                timeframe=tf,
                min_rows=200,
                raw_df=raw,
                save_snapshot=False,
            )

            self.assertFalse(dataset.empty)
            self.assertEqual(dataset["timeframe"].iloc[0], tf)
            self.assertIn("target_return", dataset.columns)
            self.assertIn("target_direction", dataset.columns)

    def test_multi_asset_batch_building(self):
        """Tests batch building across top assets."""
        symbols = ["BTC", "SOL", "DOGE"]
        timeframes = ["1h", "1d"]

        # Patch or provide raw dataframe via helper dictionary
        datasets = {}
        for sym in symbols:
            for tf in timeframes:
                raw = self._generate_synthetic_ohlcv(n_candles=260)
                df = self.builder.build_training_dataset(
                    symbol=sym,
                    timeframe=tf,
                    min_rows=200,
                    raw_df=raw,
                    save_snapshot=True,
                )
                datasets[f"{sym}_{tf}"] = df

        self.assertEqual(len(datasets), 6)
        for k, df in datasets.items():
            self.assertGreater(len(df), 20)
            self.assertFalse(df.isna().any().any())

    def test_ml_matrix_extraction(self):
        """Tests extraction of feature matrices and targets for PyTorch/GBDT training."""
        raw = self._generate_synthetic_ohlcv(n_candles=260)
        df_clean = self.builder.build_training_dataset(
            symbol="AVAX",
            timeframe="1h",
            min_rows=200,
            raw_df=raw,
            save_snapshot=False,
        )

        # 1. Single target extraction
        X, y, feature_names = self.builder.extract_feature_matrix(df_clean, target_col="target_return")
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(X.shape[0], len(df_clean))
        self.assertEqual(X.shape[1], len(feature_names))
        self.assertEqual(len(y), len(df_clean))
        self.assertFalse(np.isnan(X).any())
        self.assertFalse(np.isnan(y).any())

        # 2. Multi-horizon extraction
        X_multi, Y_rets, Y_dirs, feat_names = self.builder.extract_multi_horizon_matrices(df_clean)
        self.assertEqual(X_multi.shape[0], len(df_clean))
        self.assertEqual(Y_rets.shape, (len(df_clean), 3))
        self.assertEqual(Y_dirs.shape, (len(df_clean), 3))


if __name__ == "__main__":
    unittest.main()
