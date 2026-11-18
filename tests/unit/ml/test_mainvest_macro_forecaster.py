"""
Unit tests for MaInvest Macro Investment Forecaster Engine.
Rigorous mathematical validation without mocks.
"""

import datetime
import unittest
import numpy as np
import pandas as pd

from src.infrastructure.ml.mainvest_macro_forecaster import (
    MaInvestMacroForecaster,
    RealizedVolatilityEngine,
    HalvingMacroCycleClassifier,
    MertonJumpDiffusionEngine,
    SUPPORTED_TOP_ASSETS,
)


def generate_synthetic_ohlcv(
    num_days: int = 365,
    start_price: float = 50000.0,
    daily_drift: float = 0.0005,
    daily_vol: float = 0.03,
    seed: int = 42,
) -> pd.DataFrame:
    """Generates synthetic daily OHLCV dataframe for quant mathematical testing."""
    np.random.seed(seed)
    dates = [datetime.date(2025, 1, 1) + datetime.timedelta(days=i) for i in range(num_days)]
    
    closes = [start_price]
    for _ in range(1, num_days):
        ret = np.random.normal(daily_drift, daily_vol)
        closes.append(closes[-1] * np.exp(ret))
    
    closes = np.array(closes)
    # Generate realistic High, Low, Open from Close
    intraday_noise = np.abs(np.random.normal(0, daily_vol * 0.6, num_days))
    highs = closes * (1.0 + intraday_noise)
    lows = closes * (1.0 - intraday_noise * 0.9)
    opens = np.roll(closes, 1)
    opens[0] = start_price * 0.99
    volumes = np.random.uniform(1000, 50000, num_days)

    return pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


class TestRealizedVolatilityEngine(unittest.TestCase):
    """Mathematical validation of realized volatility estimators and cones."""

    def setUp(self):
        self.df = generate_synthetic_ohlcv(num_days=200, seed=123)

    def test_parkinson_volatility_positive_and_finite(self):
        highs = self.df["high"].to_numpy()
        lows = self.df["low"].to_numpy()
        pk_vol = RealizedVolatilityEngine.parkinson_volatility(highs, lows)
        
        self.assertGreater(pk_vol, 0.0)
        self.assertLess(pk_vol, 5.0)
        self.assertFalse(np.isnan(pk_vol))

    def test_garman_klass_volatility_positive_and_finite(self):
        opens = self.df["open"].to_numpy()
        highs = self.df["high"].to_numpy()
        lows = self.df["low"].to_numpy()
        closes = self.df["close"].to_numpy()
        
        gk_vol = RealizedVolatilityEngine.garman_klass_volatility(opens, highs, lows, closes)
        self.assertGreater(gk_vol, 0.0)
        self.assertLess(gk_vol, 5.0)
        self.assertFalse(np.isnan(gk_vol))

    def test_close_to_close_volatility(self):
        closes = self.df["close"].to_numpy()
        c2c_vol = RealizedVolatilityEngine.close_to_close_volatility(closes)
        self.assertGreater(c2c_vol, 0.0)
        self.assertLess(c2c_vol, 5.0)

    def test_volatility_cones_structure_and_monotonicity(self):
        cones = RealizedVolatilityEngine.compute_volatility_cones(self.df, windows=[14, 30, 60, 90])
        self.assertGreaterEqual(len(cones), 3)

        for cone in cones:
            self.assertIn(cone.window_days, [14, 30, 60, 90])
            self.assertLessEqual(cone.min_vol, cone.p25_vol)
            self.assertLessEqual(cone.p25_vol, cone.median_vol)
            self.assertLessEqual(cone.median_vol, cone.p75_vol)
            self.assertLessEqual(cone.p75_vol, cone.max_vol)
            self.assertGreaterEqual(cone.percentile_rank, 0.0)
            self.assertLessEqual(cone.percentile_rank, 100.0)
            self.assertIn(cone.regime, ["COMPRESSED", "NORMAL", "EXPANDED"])

    def test_analyze_realized_volatility_complete(self):
        res = RealizedVolatilityEngine.analyze_realized_volatility(self.df)
        self.assertGreater(res.composite_annualized, 0.0)
        self.assertGreater(res.garman_klass_annualized, 0.0)
        self.assertGreater(res.parkinson_annualized, 0.0)
        self.assertIn(res.dominant_vol_regime, ["COMPRESSED", "NORMAL", "EXPANDED"])


class TestHalvingMacroCycleClassifier(unittest.TestCase):
    """Validation of Bitcoin Halving Macro-Cycle relative stage logic."""

    def test_post_halving_early_accumulation_2024(self):
        # 1 month post-2024 halving
        test_date = datetime.date(2024, 5, 20)
        state = HalvingMacroCycleClassifier.get_macro_cycle_state(test_date)

        self.assertEqual(state.cycle_index, 4)
        self.assertEqual(state.prev_halving_date, "2024-04-19")
        self.assertGreater(state.days_since_halving, 25)
        self.assertLess(state.cycle_progress_pct, 15.0)
        self.assertEqual(state.stage_code, "ACCUMULATION_EARLY")
        self.assertEqual(state.regime_bias, "ACCUMULATION")

    def test_bull_parabolic_expansion_phase(self):
        # ~1 year post-2024 halving (~25% cycle progress)
        test_date = datetime.date(2025, 4, 19)
        state = HalvingMacroCycleClassifier.get_macro_cycle_state(test_date)

        self.assertEqual(state.cycle_index, 4)
        self.assertGreater(state.cycle_progress_pct, 20.0)
        self.assertLess(state.cycle_progress_pct, 42.0)
        self.assertEqual(state.stage_code, "BULL_PARABOLIC_EXPANSION")
        self.assertEqual(state.regime_bias, "STRONG_BULL")
        self.assertGreater(state.cycle_momentum_multiplier, 1.2)

    def test_pre_halving_recovery_phase(self):
        # Late 2027 (~90% cycle progress towards 2028 halving)
        test_date = datetime.date(2027, 11, 15)
        state = HalvingMacroCycleClassifier.get_macro_cycle_state(test_date)

        self.assertEqual(state.cycle_index, 4)
        self.assertGreater(state.cycle_progress_pct, 82.0)
        self.assertEqual(state.stage_code, "PRE_HALVING_RECOVERY")


class TestMertonJumpDiffusionEngine(unittest.TestCase):
    """Validation of MJD calibration and Monte Carlo simulation."""

    def setUp(self):
        self.df = generate_synthetic_ohlcv(num_days=300, seed=777)
        self.vol_analysis = RealizedVolatilityEngine.analyze_realized_volatility(self.df)
        self.macro_cycle = HalvingMacroCycleClassifier.get_macro_cycle_state()

    def test_mjd_parameter_calibration(self):
        asset_info = SUPPORTED_TOP_ASSETS["BTC"]
        params = MertonJumpDiffusionEngine.calibrate_mjd_parameters(
            df=self.df,
            asset_info=asset_info,
            macro_cycle=self.macro_cycle,
            vol_analysis=self.vol_analysis,
        )

        self.assertGreater(params.continuous_vol_annualized, 0.10)
        self.assertGreater(params.jump_intensity_lambda, 0.0)
        self.assertGreater(params.jump_std_sigma, 0.0)
        # Check compensator formula: k = exp(mu_J + 0.5 * sigma_J^2) - 1
        expected_k = np.exp(params.jump_mean_mu + 0.5 * (params.jump_std_sigma ** 2)) - 1.0
        self.assertAlmostEqual(params.jump_compensator_k, expected_k, places=3)

    def test_mjd_monte_carlo_trajectories_reproducibility(self):
        s0 = 60000.0
        asset_info = SUPPORTED_TOP_ASSETS["BTC"]
        params = MertonJumpDiffusionEngine.calibrate_mjd_parameters(
            df=self.df,
            asset_info=asset_info,
            macro_cycle=self.macro_cycle,
            vol_analysis=self.vol_analysis,
        )

        term_1, paths_1 = MertonJumpDiffusionEngine.simulate_trajectories(
            s0=s0,
            params=params,
            horizons=[30, 90, 365],
            num_simulations=600,
            random_seed=42,
        )

        term_2, paths_2 = MertonJumpDiffusionEngine.simulate_trajectories(
            s0=s0,
            params=params,
            horizons=[30, 90, 365],
            num_simulations=600,
            random_seed=42,
        )

        np.testing.assert_array_almost_equal(term_1[30], term_2[30])
        np.testing.assert_array_almost_equal(term_1[365], term_2[365])
        self.assertEqual(len(term_1[30]), 600)
        self.assertEqual(len(term_1[90]), 600)
        self.assertEqual(len(term_1[365]), 600)


class TestMaInvestMacroForecasterFullPipeline(unittest.TestCase):
    """End-to-end testing of MaInvestMacroForecaster on top assets."""

    def setUp(self):
        self.forecaster = MaInvestMacroForecaster(default_simulations=800)
        self.df = generate_synthetic_ohlcv(num_days=250, start_price=64000.0, seed=999)

    def test_predict_investment_horizon_btc(self):
        result = self.forecaster.predict_investment_horizon(
            symbol="BTCUSDT",
            daily_candles_df=self.df,
            num_simulations=800,
            random_seed=101,
        )

        self.assertEqual(result["symbol"], "BTCUSDT")
        self.assertEqual(result["normalized_symbol"], "BTC")
        self.assertEqual(result["asset_name"], "Bitcoin")
        self.assertGreater(result["current_price"], 0)

        # Check Macro Cycle
        self.assertIn("stage_code", result["macro_cycle"])
        self.assertIn("cycle_progress_pct", result["macro_cycle"])
        self.assertIn("stage_name_ru", result["macro_cycle"])

        # Check Horizons (30d, 90d, 365d)
        horizons = result["horizons"]
        self.assertIn("30d", horizons)
        self.assertIn("90d", horizons)
        self.assertIn("365d", horizons)

        for h_code in ["30d", "90d", "365d"]:
            h = horizons[h_code]
            self.assertGreater(h["target_p10"], 0)
            self.assertGreaterEqual(h["target_p25"], h["target_p10"])
            self.assertGreaterEqual(h["target_p50"], h["target_p25"])
            self.assertGreaterEqual(h["target_p75"], h["target_p50"])
            self.assertGreaterEqual(h["target_p90"], h["target_p75"])

            self.assertGreaterEqual(h["prob_growth_up"], 0.0)
            self.assertLessEqual(h["prob_growth_up"], 1.0)
            self.assertGreaterEqual(h["var_95_pct"], 0.0)
            self.assertGreaterEqual(h["cvar_95_pct"], h["var_95_pct"] * 0.95)

        # Check Investment Recommendation
        rec = result["investment_recommendation"]
        self.assertIn(rec["dca_suitability"], ["HIGH", "MODERATE", "LOW"])
        self.assertIn("recommendation_ru", rec)
        self.assertIn("risk_reward_score", rec)

        # Check sample trajectories
        self.assertIsNotNone(result["simulated_trajectories_sample"])
        self.assertGreater(len(result["simulated_trajectories_sample"]), 0)

    def test_all_supported_top_assets(self):
        """Validates all 8 top assets specified in requirements."""
        assets = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX"]
        for sym in assets:
            df = generate_synthetic_ohlcv(num_days=100, start_price=100.0, seed=42)
            res = self.forecaster.predict_investment_horizon(
                symbol=sym,
                daily_candles_df=df,
                num_simulations=500,
                random_seed=42,
            )
            self.assertEqual(res["symbol"], sym)
            self.assertEqual(res["normalized_symbol"], sym)
            self.assertIn("30d", res["horizons"])
            self.assertIn("90d", res["horizons"])
            self.assertIn("365d", res["horizons"])

    def test_generic_symbol_fallback(self):
        """Validates handling of unlisted/generic asset."""
        df = generate_synthetic_ohlcv(num_days=90, start_price=5.0, seed=55)
        res = self.forecaster.predict_investment_horizon(
            symbol="NEAR",
            daily_candles_df=df,
            num_simulations=500,
            random_seed=55,
        )
        self.assertEqual(res["symbol"], "NEAR")
        self.assertEqual(res["normalized_symbol"], "NEAR")
        self.assertIn("365d", res["horizons"])

    def test_input_validation_errors(self):
        # Empty DataFrame
        with self.assertRaises(ValueError):
            self.forecaster.predict_investment_horizon("BTC", pd.DataFrame())

        # Missing required columns
        with self.assertRaises(ValueError):
            bad_df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
            self.forecaster.predict_investment_horizon("BTC", bad_df)

        # Insufficient candles (< 5)
        with self.assertRaises(ValueError):
            short_df = pd.DataFrame({
                "open": [10, 11],
                "high": [12, 13],
                "low": [9, 10],
                "close": [11, 12]
            })
            self.forecaster.predict_investment_horizon("BTC", short_df)


if __name__ == "__main__":
    unittest.main()
