import unittest
import numpy as np
from src.domain.services.portfolio_optimizer import (
    PortfolioOptimizerService,
    PortfolioOptimizationResult,
    EfficientFrontierResult,
    DCASimulationResult,
    PortfolioStressTestReport,
    CompletePortfolioAnalysis,
)


class TestPortfolioOptimizerService(unittest.TestCase):
    def setUp(self):
        self.service = PortfolioOptimizerService
        self.assets_classic = ["BTC", "ETH", "SOL"]
        self.assets_diversified = ["BTC", "ETH", "SOL", "BNB", "USDT"]
        self.assets_alts_only = ["SOL", "NEAR", "AVAX", "TON"]

    # =========================================================================
    # 1. NORMALIZATION & PREPARATION
    # =========================================================================

    def test_normalize_weights(self):
        """Verifies weight normalization to strict 1.0 sum and uppercase keys."""
        # Unnormalized input 50, 30, 20
        raw = {"btc": 50.0, "Eth": 30.0, "sol": 20.0}
        norm = self.service.normalize_weights(raw)
        self.assertAlmostEqual(sum(norm.values()), 1.0, places=5)
        self.assertAlmostEqual(norm["BTC"], 0.50, places=4)
        self.assertAlmostEqual(norm["ETH"], 0.30, places=4)
        self.assertAlmostEqual(norm["SOL"], 0.20, places=4)

        # Empty fallback
        empty_res = self.service.normalize_weights({})
        self.assertEqual(empty_res, {"BTC": 1.0})

        # Zero values
        zero_res = self.service.normalize_weights({"BTC": 0.0, "ETH": 0.0})
        self.assertEqual(zero_res, {"BTC": 1.0})

    # =========================================================================
    # 2. MARKOWITZ MPT & EFFICIENT FRONTIER OPTIMIZATION
    # =========================================================================

    def test_max_sharpe_optimization_constraints(self):
        """Verifies Max Sharpe optimization satisfies bounds and altcoin <= 35% cap."""
        assets = ["BTC", "ETH", "SOL", "AVAX", "USDT"]
        res = self.service.optimize_max_sharpe(
            assets=assets,
            risk_free_rate=0.035,
            max_alt_weight=0.35
        )

        self.assertIsInstance(res, PortfolioOptimizationResult)
        self.assertEqual(res.optimization_type, "max_sharpe")
        
        # 1. Sum of weights must equal 1.0
        total_w = sum(res.weights.values())
        self.assertAlmostEqual(total_w, 1.0, places=3)

        # 2. Long only: w_i >= 0
        for a, w in res.weights.items():
            self.assertGreaterEqual(w, -1e-4)

        # 3. Altcoin cap <= 35% (SOL, AVAX are altcoins, BTC/USDT are core)
        for alt in ["SOL", "AVAX"]:
            if alt in res.weights:
                self.assertLessEqual(res.weights[alt], 0.35 + 1e-3)

        # 4. Sharpe ratio and volatility must be positive
        self.assertGreater(res.sharpe_ratio, 0.0)
        self.assertGreater(res.volatility_annual, 0.0)
        self.assertGreater(res.expected_return_annual, 0.0)
        self.assertGreaterEqual(res.diversification_score, 0)
        self.assertLessEqual(res.diversification_score, 100)

    def test_min_variance_optimization_constraints(self):
        """Verifies Global Minimum Variance (GMV) portfolio minimizes portfolio volatility."""
        assets = ["BTC", "ETH", "SOL", "USDT"]
        res_min_var = self.service.optimize_min_variance(
            assets=assets,
            risk_free_rate=0.035,
            max_alt_weight=0.35
        )
        res_max_sharpe = self.service.optimize_max_sharpe(
            assets=assets,
            risk_free_rate=0.035,
            max_alt_weight=0.35
        )

        self.assertIsInstance(res_min_var, PortfolioOptimizationResult)
        self.assertEqual(res_min_var.optimization_type, "min_variance")

        # Sum of weights equals 1.0
        self.assertAlmostEqual(sum(res_min_var.weights.values()), 1.0, places=3)

        # Min variance volatility must be <= Max Sharpe volatility
        self.assertLessEqual(res_min_var.volatility_annual, res_max_sharpe.volatility_annual + 1e-4)

        # For a portfolio with USDT (low vol), min variance will heavily allocate to USDT
        self.assertGreater(res_min_var.weights.get("USDT", 0.0), 0.50)

    def test_single_asset_optimization_edge_case(self):
        """Verifies single asset portfolio behaves correctly without optimization failure."""
        res = self.service.optimize_max_sharpe(["BTC"])
        self.assertEqual(res.weights, {"BTC": 1.0})
        self.assertAlmostEqual(res.volatility_annual, self.service.ASSET_ANNUAL_VOLATILITIES["BTC"], places=2)

    def test_all_altcoins_portfolio_constraint(self):
        """Verifies optimization when all assets are altcoins (sum of 4 alts at 0.35 cap = 1.4 >= 1.0)."""
        assets = ["SOL", "NEAR", "AVAX", "TON"]
        res = self.service.optimize_max_sharpe(
            assets=assets,
            max_alt_weight=0.35
        )
        self.assertAlmostEqual(sum(res.weights.values()), 1.0, places=3)
        for a, w in res.weights.items():
            self.assertLessEqual(w, 0.35 + 1e-3)

    def test_efficient_frontier_generation(self):
        """Verifies calculation of continuous Efficient Frontier points."""
        assets = ["BTC", "ETH", "SOL", "USDT"]
        ef = self.service.compute_efficient_frontier(
            assets=assets,
            n_points=15,
            risk_free_rate=0.035,
            max_alt_weight=0.35
        )

        self.assertIsInstance(ef, EfficientFrontierResult)
        self.assertGreaterEqual(len(ef.frontier_points), 2)
        
        # Verify frontier monotonicity of target returns
        returns = [pt.target_return_annual for pt in ef.frontier_points]
        for i in range(len(returns) - 1):
            self.assertLessEqual(returns[i], returns[i + 1] + 1e-4)

        # Verify weights on every point sum to 1.0
        for pt in ef.frontier_points:
            self.assertAlmostEqual(sum(pt.weights.values()), 1.0, places=3)
            self.assertGreater(pt.volatility_annual, 0.0)

    # =========================================================================
    # 3. REALISTIC DCA SIMULATOR (BLOCK BOOTSTRAP)
    # =========================================================================

    def test_dca_simulation_bootstrap_fan(self):
        """Verifies DCA simulator executes Historical Block Bootstrap and produces P10 <= P50 <= P90."""
        weights = {"BTC": 0.50, "ETH": 0.30, "SOL": 0.20}
        weekly_inv = 100.0
        horizon_yrs = 2.0  # 104 weeks
        initial_dep = 500.0

        dca = self.service.simulate_dca(
            weekly_investment_usd=weekly_inv,
            horizon_years=horizon_yrs,
            portfolio_weights=weights,
            n_simulations=1500,
            block_size_weeks=4,
            initial_deposit_usd=initial_dep,
            seed=42
        )

        self.assertIsInstance(dca, DCASimulationResult)
        self.assertEqual(dca.total_weeks, 104)

        # Total invested math: 500 + 104 * 100 = 10,900
        expected_invested = initial_dep + 104 * weekly_inv
        self.assertAlmostEqual(dca.total_invested_usd, expected_invested, places=1)

        # Strict quantile ordering: P10 <= P50 <= P90
        self.assertLessEqual(dca.p10_final_value_usd, dca.p50_final_value_usd)
        self.assertLessEqual(dca.p50_final_value_usd, dca.p90_final_value_usd)

        # ROI ordering: P10_ROI <= P50_ROI <= P90_ROI
        self.assertLessEqual(dca.p10_roi_pct, dca.p50_roi_pct)
        self.assertLessEqual(dca.p50_roi_pct, dca.p90_roi_pct)

        # Probability of profit in [0, 1]
        self.assertGreaterEqual(dca.prob_profit, 0.0)
        self.assertLessEqual(dca.prob_profit, 1.0)
        self.assertGreaterEqual(dca.prob_double, 0.0)
        self.assertLessEqual(dca.prob_double, 1.0)

        # Trajectories length matches total_weeks
        self.assertEqual(len(dca.weekly_trajectory_p10), 104)
        self.assertEqual(len(dca.weekly_trajectory_p50), 104)
        self.assertEqual(len(dca.weekly_trajectory_p90), 104)
        self.assertEqual(len(dca.weekly_invested_curve), 104)

        # Final trajectory point matches final value
        self.assertAlmostEqual(dca.weekly_trajectory_p10[-1], dca.p10_final_value_usd, delta=1.0)
        self.assertAlmostEqual(dca.weekly_trajectory_p50[-1], dca.p50_final_value_usd, delta=1.0)
        self.assertAlmostEqual(dca.weekly_trajectory_p90[-1], dca.p90_final_value_usd, delta=1.0)

    def test_dca_custom_returns_matrix(self):
        """Verifies DCA simulator with custom user-provided historical return matrix."""
        # 50 weeks of deterministic positive returns (1% per week) for BTC and ETH
        T = 60
        custom_R = np.full((T, 2), 0.01)
        weights = {"BTC": 0.60, "ETH": 0.40}

        dca = self.service.simulate_dca(
            weekly_investment_usd=50.0,
            horizon_years=1.0,
            portfolio_weights=weights,
            historical_returns=custom_R,
            n_simulations=500,
            block_size_weeks=4,
            seed=123
        )

        self.assertEqual(dca.total_weeks, 52)
        # In a deterministic positive return environment, prob_profit must be 1.0
        self.assertEqual(dca.prob_profit, 1.0)
        self.assertGreater(dca.p50_roi_pct, 0.0)

    def test_dca_invalid_inputs_raise(self):
        """Verifies DCA simulator raises ValueError on invalid parameters."""
        with self.assertRaises(ValueError):
            self.service.simulate_dca(weekly_investment_usd=-10.0, horizon_years=2.0, portfolio_weights={"BTC": 1.0})

        with self.assertRaises(ValueError):
            self.service.simulate_dca(weekly_investment_usd=100.0, horizon_years=0.1, portfolio_weights={"BTC": 1.0})

    # =========================================================================
    # 4. HISTORICAL CRASH STRESS TESTS
    # =========================================================================

    def test_stress_test_historical_scenarios(self):
        """Verifies all 5 historical crash stress tests are evaluated with accurate math."""
        # Mixed portfolio: 50% BTC, 30% ETH, 20% SOL
        weights = {"BTC": 0.50, "ETH": 0.30, "SOL": 0.20}
        cap = 10000.0

        report = self.service.run_stress_tests(portfolio_weights=weights, initial_capital_usd=cap)

        self.assertIsInstance(report, PortfolioStressTestReport)
        self.assertEqual(len(report.scenarios), 5)
        self.assertIn("ftx_collapse_2022", report.scenarios)
        self.assertIn("covid_flash_crash_2020", report.scenarios)
        self.assertIn("luna_ust_crash_2022", report.scenarios)
        self.assertIn("china_mining_ban_2021", report.scenarios)
        self.assertIn("crypto_winter_bear_2022", report.scenarios)

        # 1. Verify FTX collapse math:
        # BTC: -30%, ETH: -35%, SOL: -65%
        # Expected port drop: 0.50 * (-0.30) + 0.30 * (-0.35) + 0.20 * (-0.65) = -0.15 - 0.105 - 0.13 = -0.385 (-38.5%)
        ftx = report.scenarios["ftx_collapse_2022"]
        self.assertAlmostEqual(ftx.portfolio_drawdown_pct, -38.5, places=1)
        self.assertAlmostEqual(ftx.loss_usd_on_10k, 3850.0, places=1)
        self.assertAlmostEqual(ftx.remaining_value_usd_on_10k, 6150.0, places=1)
        self.assertEqual(ftx.worst_hit_asset, "SOL")
        self.assertEqual(ftx.worst_hit_asset_drop_pct, -65.0)

        # 2. Verify COVID Dump math:
        # BTC: -50%, ETH: -55%, SOL: -60%
        # Expected: 0.5*(-0.5) + 0.3*(-0.55) + 0.2*(-0.6) = -0.25 - 0.165 - 0.12 = -0.535 (-53.5%)
        covid = report.scenarios["covid_flash_crash_2020"]
        self.assertAlmostEqual(covid.portfolio_drawdown_pct, -53.5, places=1)

        # 3. Overall report integrity
        self.assertGreater(report.worst_case_loss_pct, 0.0)
        self.assertGreaterEqual(report.stress_survival_score, 0)
        self.assertLessEqual(report.stress_survival_score, 100)
        self.assertIsInstance(report.recommendations, list)
        self.assertGreater(len(report.recommendations), 0)

    def test_stress_test_stablecoin_cushion(self):
        """Verifies stablecoins provide defensive cushion during systemic crashes."""
        # 100% SOL vs 50% SOL + 50% USDT
        pure_sol = {"SOL": 1.0}
        shielded = {"SOL": 0.50, "USDT": 0.50}

        rep_pure = self.service.run_stress_tests(pure_sol, initial_capital_usd=10000.0)
        rep_shielded = self.service.run_stress_tests(shielded, initial_capital_usd=10000.0)

        # Shielded portfolio must have lower loss and higher survival score
        self.assertLess(rep_shielded.worst_case_loss_pct, rep_pure.worst_case_loss_pct)
        self.assertGreater(rep_shielded.stress_survival_score, rep_pure.stress_survival_score)

    # =========================================================================
    # 5. FULL END-TO-END AUDIT & SERIALIZATION
    # =========================================================================

    def test_full_portfolio_analysis_and_dict_serialization(self):
        """Verifies end-to-end full portfolio diagnostic and JSON/Dict serialization."""
        weights = {"BTC": 0.40, "ETH": 0.30, "SOL": 0.20, "USDT": 0.10}
        
        full_analysis = self.service.full_portfolio_analysis(
            portfolio_weights=weights,
            weekly_investment_usd=100.0,
            horizon_years=3.0,
            initial_capital_usd=10000.0
        )

        self.assertIsInstance(full_analysis, CompletePortfolioAnalysis)
        self.assertIn("BTC", full_analysis.current_weights)
        self.assertIsInstance(full_analysis.max_sharpe_optimal, PortfolioOptimizationResult)
        self.assertIsInstance(full_analysis.min_variance_optimal, PortfolioOptimizationResult)
        self.assertIsInstance(full_analysis.dca_simulation, DCASimulationResult)
        self.assertIsInstance(full_analysis.stress_test_report, PortfolioStressTestReport)
        self.assertIsInstance(full_analysis.efficient_frontier_sample, list)

        # Verify to_dict() serialization works without exception
        analysis_dict = full_analysis.to_dict()
        self.assertIn("max_sharpe_optimal", analysis_dict)
        self.assertIn("dca_simulation", analysis_dict)
        self.assertIn("stress_test_report", analysis_dict)
        self.assertIn("efficient_frontier_sample", analysis_dict)


if __name__ == "__main__":
    unittest.main()
