"""
Unit Tests for Continuous Retrainer and MLOps Health Auditor.
R&D Quantitative Laboratory — MaBuy Project.

Strict Mathematical Guarantees Tested:
1. Feature Drift Evaluation (PSI & KS-Test).
2. Champion vs Challenger Promotion Rules.
3. Merton Jump Diffusion (MJD) Exact Mathematical Calibration (lambda, mu_J, sigma_J, sigma_c, compensator k).
4. Positive Semi-Definite (PSD) Covariance & Correlation Matrix with Eigenvalue Spectrum.
5. Snapshot Persistence to JSON Cache.
6. Unified Ecosystem Health & Retrain Cycle.
"""

import os
import json
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.infrastructure.mlops.continuous_retrainer import (
    ContinuousRetrainer,
    ModelAuditReport,
    ChampionVsChallengerResult,
    MertonAssetCalibration,
    MaInvestCalibrationSnapshot,
    EcosystemHealthReport,
)
from src.infrastructure.mlops.model_registry import ModelRegistry


class TestContinuousRetrainer(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="mabuy_test_mlops_")
        self.models_dir = os.path.join(self.test_dir, "models")
        self.cache_dir = os.path.join(self.test_dir, ".data_cache")
        self.snapshot_path = os.path.join(self.test_dir, "data", "cache", "mainvest_calibration.json")

        self.retrainer = ContinuousRetrainer(
            models_dir=self.models_dir,
            cache_dir=self.cache_dir,
            mainvest_snapshot_path=self.snapshot_path,
            p_value_threshold=0.05,
            psi_threshold=0.25,
            min_directional_accuracy=0.505,
            min_sharpe_ratio=0.0,
            drift_ratio_threshold=0.30,
            enable_catboost=False  # Deterministic HistGradientBoosting in tests
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _generate_stationary_ohlcv(self, n_candles: int = 600, seed: int = 42) -> pd.DataFrame:
        """Generates stationary returns OHLCV DataFrame."""
        np.random.seed(seed)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n_candles, freq="1h")
        rets = np.random.normal(loc=0.0, scale=0.005, size=n_candles)
        prices = 50000.0 * np.exp(np.cumsum(rets))

        highs = prices * (1.0 + np.abs(np.random.normal(0, 0.002, size=n_candles)))
        lows = prices * (1.0 - np.abs(np.random.normal(0, 0.002, size=n_candles)))
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        volume = np.random.lognormal(mean=4.0, sigma=0.2, size=n_candles) * 10.0

        return pd.DataFrame({
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volume
        })

    def _generate_momentum_signal_ohlcv(self, n_candles: int = 800, seed: int = 42) -> pd.DataFrame:
        """Generates AR(1) momentum process with real predictive feature structure."""
        np.random.seed(seed)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n_candles, freq="1h")
        rets = np.zeros(n_candles)
        noise = np.random.normal(0, 0.004, size=n_candles)
        for t in range(1, n_candles):
            rets[t] = 0.45 * rets[t-1] + noise[t]
        
        prices = 50000.0 * np.exp(np.cumsum(rets))
        highs = prices * (1.0 + np.abs(np.random.normal(0, 0.001, size=n_candles)))
        lows = prices * (1.0 - np.abs(np.random.normal(0, 0.001, size=n_candles)))
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        volume = np.random.lognormal(mean=4.0, sigma=0.2, size=n_candles) * 10.0

        return pd.DataFrame({
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volume
        })

    def _generate_daily_candles(self, n_days: int = 365, seed: int = 100, jump_prob: float = 0.03) -> pd.DataFrame:
        """Generates 365 daily candles with compound Poisson jump diffusion."""
        np.random.seed(seed)
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n_days, freq="D")
        
        diff_rets = np.random.normal(loc=0.001, scale=0.03, size=n_days)
        jump_occurrences = np.random.binomial(1, jump_prob, size=n_days)
        jump_sizes = np.random.normal(loc=-0.04, scale=0.08, size=n_days) * jump_occurrences
        total_rets = diff_rets + jump_sizes
        
        prices = 100.0 * np.exp(np.cumsum(total_rets))
        highs = prices * (1.0 + np.abs(np.random.normal(0, 0.015, size=n_days)))
        lows = prices * (1.0 - np.abs(np.random.normal(0, 0.015, size=n_days)))
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        volume = np.random.lognormal(mean=6.0, sigma=0.8, size=n_days)

        return pd.DataFrame({
            "timestamp": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": prices,
            "volume": volume
        })

    # ========================================================================
    # TEST 1: MAPREDICT FEATURE DRIFT DETECTION
    # ========================================================================

    def test_audit_mapredict_drift_detection(self):
        """Tests that stationary market data passes drift check and shifted market triggers retrain."""
        ref_df = self._generate_stationary_ohlcv(n_candles=500, seed=42)
        curr_df_normal = self._generate_stationary_ohlcv(n_candles=500, seed=43)

        report_normal = self.retrainer.audit_mapredict_model(
            symbol="BTCUSDT",
            raw_df=curr_df_normal,
            reference_df=ref_df
        )

        self.assertIsInstance(report_normal, ModelAuditReport)
        self.assertEqual(report_normal.symbol, "BTCUSDT")
        self.assertGreater(report_normal.total_features_count, 0)
        self.assertFalse(report_normal.drift_detected)

        # 2. Shifted data with substantial variance / mean shift in current distribution
        np.random.seed(999)
        shifted_rets = np.random.normal(loc=0.08, scale=0.09, size=500)
        shifted_prices = 50000.0 * np.exp(np.cumsum(shifted_rets))
        shifted_df = pd.DataFrame({
            "timestamp": pd.date_range(end=pd.Timestamp.now(), periods=500, freq="1h"),
            "open": shifted_prices * 0.99,
            "high": shifted_prices * 1.10,
            "low": shifted_prices * 0.90,
            "close": shifted_prices,
            "volume": np.random.lognormal(mean=9.0, sigma=1.5, size=500)
        })

        report_shifted = self.retrainer.audit_mapredict_model(
            symbol="BTCUSDT",
            raw_df=shifted_df,
            reference_df=ref_df
        )

        self.assertTrue(report_shifted.retraining_recommended)
        self.assertTrue(report_shifted.drift_detected)
        self.assertGreater(report_shifted.drifted_features_count, 0)

    # ========================================================================
    # TEST 2: CHAMPION VS CHALLENGER PROMOTION GOVERNANCE
    # ========================================================================

    def test_champion_vs_challenger_promotion_logic(self):
        """Tests strict promotion governance: challenger replaces champion only when superior."""
        # 1. Establish initial Champion with strong momentum signal
        train_df = self._generate_momentum_signal_ohlcv(n_candles=800, seed=42)
        res_first = self.retrainer.retrain_mapredict_model(symbol="BTCUSDT", raw_df=train_df, model_type="GBDT")

        self.assertTrue(res_first.promoted_to_active)
        active_champ = self.retrainer.registry.get_active_model_info("BTCUSDT")
        self.assertIsNotNone(active_champ)
        self.assertEqual(active_champ["version"], res_first.challenger_version)

        # 2. Test Inferior Challenger (artificially inferior metrics)
        inferior_metrics = {
            "mean_directional_accuracy": 0.48,  # Below minimum 0.505
            "mean_sharpe_ratio": -0.5,
            "mean_mae": 0.05
        }
        res_inferior = self.retrainer.evaluate_champion_vs_challenger(
            symbol="BTCUSDT",
            timeframe="1h",
            challenger_metrics=inferior_metrics,
            challenger_version="v_inferior_test",
            artifact_path="dummy_path.joblib",
            model_type="GBDT"
        )
        self.assertFalse(res_inferior.promoted_to_active)
        # Champion must remain untouched
        active_champ_after = self.retrainer.registry.get_active_model_info("BTCUSDT")
        self.assertEqual(active_champ_after["version"], res_first.challenger_version)

        # 3. Test Superior Challenger
        superior_metrics = {
            "mean_directional_accuracy": 0.75,
            "mean_sharpe_ratio": 30.0,
            "mean_mae": 0.001
        }
        res_superior = self.retrainer.evaluate_champion_vs_challenger(
            symbol="BTCUSDT",
            timeframe="1h",
            challenger_metrics=superior_metrics,
            challenger_version="v_superior_test",
            artifact_path="dummy_superior_path.joblib",
            model_type="GBDT"
        )
        self.assertTrue(res_superior.promoted_to_active)
        # New active champion promoted
        active_champ_promoted = self.retrainer.registry.get_active_model_info("BTCUSDT")
        self.assertEqual(active_champ_promoted["version"], "v_superior_test")

    # ========================================================================
    # TEST 3: MERTON JUMP DIFFUSION MATHEMATICAL RECALIBRATION
    # ========================================================================

    def test_merton_jump_diffusion_calibration_math(self):
        """Verifies exact Merton Jump Diffusion mathematical formulas."""
        df_daily = self._generate_daily_candles(n_days=365, seed=2024, jump_prob=0.05)
        calib = self.retrainer.calibrate_merton_jump_diffusion("BTC", df_daily)

        self.assertIsInstance(calib, MertonAssetCalibration)
        self.assertEqual(calib.symbol, "BTC")
        self.assertEqual(calib.sample_days, 364)  # 365 candles -> 364 log returns

        # Mathematical constraints:
        # 1. Lambda > 0 (jump intensity)
        self.assertGreater(calib.jump_intensity_lambda, 0.0)

        # 2. Continuous volatility > 0
        self.assertGreater(calib.continuous_vol_sigma_c, 0.0)
        self.assertLess(calib.continuous_vol_sigma_c, 3.0)

        # 3. Jump std sigma > 0
        self.assertGreater(calib.jump_std_sigma, 0.0)

        # 4. Compensator k exact identity: k = exp(mu_J + 0.5 * sigma_J^2) - 1
        expected_k = np.exp(calib.jump_mean_mu + 0.5 * (calib.jump_std_sigma ** 2)) - 1.0
        self.assertAlmostEqual(calib.jump_compensator_k, round(float(expected_k), 4), places=3)

    # ========================================================================
    # TEST 4: POSITIVE SEMI-DEFINITE (PSD) COVARIANCE & CORRELATION MATRIX
    # ========================================================================

    def test_covariance_and_correlation_psd_guarantee(self):
        """Verifies covariance matrix is symmetric, strictly positive definite, and correlation valid."""
        assets = ["BTC", "ETH", "SOL", "BNB", "TON", "XRP", "AVAX"]
        custom_candles = {
            a: self._generate_daily_candles(n_days=365, seed=100 + i)
            for i, a in enumerate(assets)
        }

        snapshot = self.retrainer.recalibrate_mainvest(
            assets=assets,
            lookback_days=365,
            custom_candles=custom_candles
        )

        self.assertIsInstance(snapshot, MaInvestCalibrationSnapshot)
        self.assertEqual(snapshot.assets, assets)

        cov_matrix = np.array(snapshot.covariance_matrix)
        corr_matrix = np.array(snapshot.correlation_matrix)
        k = len(assets)

        # 1. Dimension checks
        self.assertEqual(cov_matrix.shape, (k, k))
        self.assertEqual(corr_matrix.shape, (k, k))

        # 2. Symmetry check
        np.testing.assert_allclose(cov_matrix, cov_matrix.T, atol=1e-5)
        np.testing.assert_allclose(corr_matrix, corr_matrix.T, atol=1e-5)

        # 3. Positive Definite Guarantee (all eigenvalues > 0)
        eigvals = np.linalg.eigvalsh(cov_matrix)
        self.assertTrue(np.all(eigvals > 0.0), f"Covariance eigenvalues must be strictly positive: {eigvals}")

        # 4. Correlation diagonal is strictly 1.0
        for i in range(k):
            self.assertAlmostEqual(corr_matrix[i, i], 1.0, places=3)

        # 5. Correlation off-diagonals in [-1.0, 1.0]
        self.assertTrue(np.all(corr_matrix >= -1.0) and np.all(corr_matrix <= 1.0))

        # 6. Systemic Absorption Ratio
        self.assertGreater(snapshot.top_eigenvalue_pct, 0.0)
        self.assertLessEqual(snapshot.top_eigenvalue_pct, 100.0)

    # ========================================================================
    # TEST 5: MAINVEST SNAPSHOT JSON PERSISTENCE
    # ========================================================================

    def test_mainvest_snapshot_persistence(self):
        """Verifies serialization and disk retrieval of MaInvest calibration snapshot."""
        assets = ["BTC", "ETH", "SOL", "BNB"]
        custom_candles = {
            a: self._generate_daily_candles(n_days=180, seed=50 + i)
            for i, a in enumerate(assets)
        }

        snapshot = self.retrainer.recalibrate_mainvest(
            assets=assets,
            lookback_days=180,
            custom_candles=custom_candles
        )

        self.assertTrue(os.path.exists(self.snapshot_path))

        with open(self.snapshot_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("timestamp", data)
        self.assertIn("assets", data)
        self.assertIn("asset_calibrations", data)
        self.assertIn("covariance_matrix", data)
        self.assertIn("correlation_matrix", data)
        self.assertIn("top_eigenvalue_pct", data)
        self.assertEqual(data["assets"], assets)
        self.assertIn("BTC", data["asset_calibrations"])
        self.assertIn("jump_compensator_k", data["asset_calibrations"]["BTC"])

    # ========================================================================
    # TEST 6: FULL ECOSYSTEM HEALTH AUDIT & RETRAINING RUN
    # ========================================================================

    def test_run_ecosystem_health_and_retrain_cycle(self):
        """Tests the master unified audit and retrain interface run_ecosystem_health_and_retrain."""
        # Pre-populate synthetic cached data in test cache dir
        btc_df = self._generate_momentum_signal_ohlcv(n_candles=700, seed=77)
        eth_df = self._generate_momentum_signal_ohlcv(n_candles=700, seed=88)
        btc_df.to_csv(os.path.join(self.cache_dir, "BTCUSDT_1h.csv"), index=False)
        eth_df.to_csv(os.path.join(self.cache_dir, "ETHUSDT_1h.csv"), index=False)

        # 1. Run full cycle with force=True
        report_dict = self.retrainer.run_ecosystem_health_and_retrain(
            force=True,
            symbols=["BTCUSDT", "ETHUSDT"],
            mainvest_assets=["BTC", "ETH", "SOL"]
        )

        self.assertIsInstance(report_dict, dict)
        self.assertIn("timestamp", report_dict)
        self.assertIn("status", report_dict)
        self.assertIn("mapredict_audit", report_dict)
        self.assertIn("mapredict_retraining", report_dict)
        self.assertIn("mainvest_calibration", report_dict)
        self.assertIn("execution_time_sec", report_dict)
        self.assertIn("summary_ru", report_dict)
        self.assertIn("summary_en", report_dict)

        # Check MaPredict Retraining results
        self.assertIn("BTCUSDT", report_dict["mapredict_retraining"])
        self.assertIn("ETHUSDT", report_dict["mapredict_retraining"])

        # Check MaInvest Calibration results
        mainvest_res = report_dict["mainvest_calibration"]
        self.assertEqual(mainvest_res["status"], "SUCCESS")
        self.assertIn("snapshot", mainvest_res)
        self.assertEqual(mainvest_res["snapshot"]["assets"], ["BTC", "ETH", "SOL"])

        # Check JSON serializability
        json_str = json.dumps(report_dict)
        self.assertGreater(len(json_str), 100)


if __name__ == "__main__":
    unittest.main()
