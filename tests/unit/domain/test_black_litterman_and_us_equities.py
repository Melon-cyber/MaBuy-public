"""
Unit test suite for MaInvest US Equities Calibration & Black-Litterman Asset Allocation Model.
Rigorous mathematical validation without mocks.
"""

import datetime
import unittest
import numpy as np
import pandas as pd

from src.domain.services.symbol_normalizer import SymbolNormalizer
from src.infrastructure.ml.mainvest_macro_forecaster import (
    MaInvestMacroForecaster,
    RealizedVolatilityEngine,
    HalvingMacroCycleClassifier,
    MertonJumpDiffusionEngine,
    SUPPORTED_TOP_ASSETS,
)
from src.domain.services.portfolio_optimizer import (
    PortfolioOptimizerService,
    BlackLittermanView,
    BlackLittermanResult,
    PortfolioOptimizationResult,
)


def generate_synthetic_stock_ohlcv(
    num_days: int = 252,
    start_price: float = 150.0,
    daily_drift: float = 0.0008,
    daily_vol: float = 0.015,
    seed: int = 101,
) -> pd.DataFrame:
    """Generates synthetic daily OHLCV for US Equities (N=252 trading days/year)."""
    np.random.seed(seed)
    dates = [datetime.date(2025, 1, 1) + datetime.timedelta(days=int(i * 1.45)) for i in range(num_days)]

    closes = [start_price]
    for _ in range(1, num_days):
        ret = np.random.normal(daily_drift, daily_vol)
        closes.append(closes[-1] * np.exp(ret))

    closes = np.array(closes)
    intraday_noise = np.abs(np.random.normal(0, daily_vol * 0.5, num_days))
    highs = closes * (1.0 + intraday_noise)
    lows = closes * (1.0 - intraday_noise * 0.85)
    opens = np.roll(closes, 1)
    opens[0] = start_price * 0.995
    volumes = np.random.uniform(500000, 5000000, num_days)

    return pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })


class TestSymbolNormalizerCrossMarket(unittest.TestCase):
    """Verifies that cross-market symbols are accurately normalized."""

    def test_crypto_symbols(self):
        self.assertEqual(SymbolNormalizer.normalize("bitcoin"), "BTC")
        self.assertEqual(SymbolNormalizer.normalize("BTCUSDT"), "BTC")
        self.assertEqual(SymbolNormalizer.normalize("solana"), "SOL")
        self.assertEqual(SymbolNormalizer.normalize("TONCOIN"), "TON")

    def test_us_equities_and_etfs(self):
        self.assertEqual(SymbolNormalizer.normalize("NVIDIA"), "NVDA")
        self.assertEqual(SymbolNormalizer.normalize("apple"), "AAPL")
        self.assertEqual(SymbolNormalizer.normalize("TESLA"), "TSLA")
        self.assertEqual(SymbolNormalizer.normalize("microsoft"), "MSFT")
        self.assertEqual(SymbolNormalizer.normalize("GOOGLE"), "GOOGL")
        self.assertEqual(SymbolNormalizer.normalize("SP500"), "SPY")
        self.assertEqual(SymbolNormalizer.normalize("NASDAQ"), "QQQ")
        self.assertEqual(SymbolNormalizer.normalize("GOLD"), "GLD")


class TestUSEquitiesMacroForecasterCalibration(unittest.TestCase):
    """Verifies N=252 trading day calendar handling and asset class logic."""

    def setUp(self):
        self.df_nvda = generate_synthetic_stock_ohlcv(num_days=252, start_price=120.0, seed=42)
        self.forecaster = MaInvestMacroForecaster(default_simulations=500)

    def test_supported_equity_assets_metadata(self):
        for sym in ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "SPY", "QQQ", "GLD"]:
            self.assertIn(sym, SUPPORTED_TOP_ASSETS)
            spec = SUPPORTED_TOP_ASSETS[sym]
            self.assertEqual(spec["trading_days_per_year"], 252)
            self.assertIn(spec["asset_class"], ["equity", "macro_etf", "commodity_etf"])

    def test_realized_volatility_annualized_252(self):
        vol_res = RealizedVolatilityEngine.analyze_realized_volatility(self.df_nvda, annualized_factor=252.0)
        self.assertGreater(vol_res.composite_annualized, 0.05)
        self.assertLess(vol_res.composite_annualized, 1.50)
        self.assertGreater(vol_res.garman_klass_annualized, 0.0)

    def test_equity_macro_regime_classification(self):
        cycle_state = HalvingMacroCycleClassifier.get_macro_cycle_state(
            current_date=datetime.date(2025, 6, 1),
            asset_class="equity"
        )
        self.assertEqual(cycle_state.stage_code, "EQUITY_SECULAR_REGIME")
        self.assertEqual(cycle_state.cycle_momentum_multiplier, 1.00)
        self.assertEqual(cycle_state.regime_bias, "MACRO_EQUITY")

    def test_predict_horizon_nvda(self):
        report = self.forecaster.predict_investment_horizon("NVDA", self.df_nvda, num_simulations=500, random_seed=42)
        self.assertEqual(report["symbol"], "NVDA")
        self.assertEqual(report["normalized_symbol"], "NVDA")
        self.assertEqual(report["merton_params"]["trading_days_per_year"], 252)

        horizons = report["horizons"]
        self.assertIn("30d", horizons)
        self.assertIn("90d", horizons)
        self.assertIn("365d", horizons)
        h365 = horizons["365d"]
        self.assertGreater(h365["target_p10"], 0.0)
        self.assertLess(h365["target_p10"], h365["target_p50"])
        self.assertLess(h365["target_p50"], h365["target_p90"])


class TestBlackLittermanModel(unittest.TestCase):
    """Deep mathematical verification of the Black-Litterman Asset Allocation Engine."""

    def setUp(self):
        self.assets = ["BTC", "ETH", "NVDA", "SPY", "GLD"]
        self.cov = PortfolioOptimizerService.build_covariance_matrix(self.assets)

    def test_implied_equilibrium_returns_calculation(self):
        """Validates Pi = delta * Sigma * w_mkt + r_f."""
        delta = 2.5
        rf = 0.035
        pi = PortfolioOptimizerService.calculate_implied_equilibrium_returns(
            assets=self.assets,
            cov_matrix=self.cov,
            risk_aversion_delta=delta,
            risk_free_rate=rf
        )
        self.assertEqual(len(pi), len(self.assets))
        # Higher volatility assets (BTC, ETH, NVDA) must have higher implied equilibrium returns than SPY and GLD
        btc_idx = self.assets.index("BTC")
        spy_idx = self.assets.index("SPY")
        gld_idx = self.assets.index("GLD")
        self.assertGreater(pi[btc_idx], pi[spy_idx])
        self.assertGreater(pi[spy_idx], pi[gld_idx])

    def test_build_view_matrices_structure(self):
        views = [
            BlackLittermanView(view_type="absolute", assets=["NVDA"], weights=[1.0], expected_return=0.40, confidence=0.75),
            BlackLittermanView(view_type="relative", assets=["NVDA", "SPY"], weights=[1.0, -1.0], expected_return=0.15, confidence=0.60),
        ]
        P, Q, Omega = PortfolioOptimizerService.build_view_matrices(self.assets, views, tau=0.05, cov_matrix=self.cov)
        self.assertEqual(P.shape, (2, 5))
        self.assertEqual(Q.shape, (2,))
        self.assertEqual(Omega.shape, (2, 2))

        # Check P weights
        nvda_idx = self.assets.index("NVDA")
        spy_idx = self.assets.index("SPY")
        self.assertEqual(P[0, nvda_idx], 1.0)
        self.assertEqual(P[1, nvda_idx], 1.0)
        self.assertEqual(P[1, spy_idx], -1.0)
        self.assertEqual(Q[0], 0.40)
        self.assertEqual(Q[1], 0.15)

        # Omega must be diagonal positive
        self.assertGreater(Omega[0, 0], 0.0)
        self.assertGreater(Omega[1, 1], 0.0)
        self.assertEqual(Omega[0, 1], 0.0)

    def test_black_litterman_posterior_with_bullish_view(self):
        """Verifies that a bullish view on NVDA increases its posterior return and allocation."""
        # 1. Baseline equilibrium (no views)
        res_baseline = PortfolioOptimizerService.compute_black_litterman_posterior(
            assets=self.assets,
            views=[],
            tau=0.05
        )

        # 2. Strong Bullish view on NVDA (expected 60% return with 90% confidence)
        bullish_views = [
            BlackLittermanView(view_type="absolute", assets=["NVDA"], weights=[1.0], expected_return=0.60, confidence=0.90)
        ]
        res_bullish = PortfolioOptimizerService.compute_black_litterman_posterior(
            assets=self.assets,
            views=bullish_views,
            tau=0.05
        )

        # NVDA posterior return must be noticeably higher
        nvda_baseline_ret = res_baseline.posterior_expected_returns["NVDA"]
        nvda_bullish_ret = res_bullish.posterior_expected_returns["NVDA"]
        self.assertGreater(nvda_bullish_ret, nvda_baseline_ret)

        # Max Sharpe weight of NVDA should increase or stay maximal
        w_nvda_base = res_baseline.optimized_max_sharpe.weights["NVDA"]
        w_nvda_bull = res_bullish.optimized_max_sharpe.weights["NVDA"]
        self.assertGreaterEqual(w_nvda_bull, w_nvda_base)

    def test_black_litterman_empty_views_fallback(self):
        """Verifies smooth fallback to equilibrium Pi when views are empty."""
        res = PortfolioOptimizerService.compute_black_litterman_posterior(assets=self.assets, views=None)
        for a in self.assets:
            self.assertAlmostEqual(res.implied_equilibrium_returns[a], res.posterior_expected_returns[a], places=3)

    def test_optimize_black_litterman_direct_api(self):
        views = [
            BlackLittermanView(view_type="absolute", assets=["BTC"], weights=[1.0], expected_return=0.70, confidence=0.80),
            BlackLittermanView(view_type="absolute", assets=["GLD"], weights=[1.0], expected_return=0.12, confidence=0.60),
        ]
        opt_res = PortfolioOptimizerService.optimize_black_litterman_portfolio(
            assets=self.assets,
            views=views,
            optimization_type="max_sharpe"
        )
        self.assertIsInstance(opt_res, PortfolioOptimizationResult)
        self.assertAlmostEqual(sum(opt_res.weights.values()), 1.0, places=3)
        self.assertGreater(opt_res.sharpe_ratio, 0.0)

    def test_model_presets_all_weather_and_tech_crypto(self):
        presets = PortfolioOptimizerService.MODEL_PRESETS
        self.assertIn("all_weather_2", presets)
        self.assertIn("tech_crypto_alpha", presets)
        self.assertIn("buffett_hedge", presets)

        for p_name, p_data in presets.items():
            weights = p_data["weights"]
            self.assertAlmostEqual(sum(weights.values()), 1.0, places=4)
            for asset, w in weights.items():
                self.assertGreater(w, 0.0)
                self.assertLessEqual(w, 1.0)

    def test_generate_ai_views_from_forecaster(self):
        """Tests integration between MaInvestMacroForecaster output and BlackLittermanView generation."""
        mock_forecasts = {
            "BTC": {
                "horizons": {
                    "365d": {"p50_return_pct": 55.0, "prob_growth_up": 0.72}
                }
            },
            "NVDA": {
                "horizons": {
                    "365d": {"p50_return_pct": 38.0, "prob_growth_up": 0.68}
                }
            }
        }
        views = PortfolioOptimizerService.generate_ai_views_from_forecaster(mock_forecasts)
        self.assertEqual(len(views), 2)
        btc_view = next(v for v in views if v.assets == ["BTC"])
        self.assertEqual(btc_view.expected_return, 0.55)
        self.assertGreater(btc_view.confidence, 0.50)


class TestCrossAssetStressTests(unittest.TestCase):
    """Verifies that systemic crash stress tests handle mixed Crypto / US Stock / ETF portfolios."""

    def test_all_weather_stress_survival(self):
        preset_weights = PortfolioOptimizerService.MODEL_PRESETS["all_weather_2"]["weights"]
        report = PortfolioOptimizerService.run_stress_tests(preset_weights, initial_capital_usd=10000.0)

        # All-weather portfolio contains SPY, GLD, QQQ, BTC, USDT -> must have strong resilience
        self.assertGreaterEqual(report.stress_survival_score, 50)
        self.assertIn(report.worst_case_scenario_id, PortfolioOptimizerService.HISTORICAL_CRASH_SCENARIOS)

        # Verify COVID-19 scenario output
        covid_res = report.scenarios["covid_flash_crash_2020"]
        self.assertLess(covid_res.portfolio_drawdown_pct, 0.0)
        self.assertGreater(covid_res.remaining_value_usd_on_10k, 0.0)


class TestPortfolioRiskAuditorCrossMarket(unittest.TestCase):
    """Verifies PortfolioRiskAuditorService with US stocks, ETFs and cross-market presets."""

    def test_audit_cross_market_presets(self):
        from src.domain.services.portfolio_auditor import PortfolioRiskAuditorService

        weights = {"NVDA": 25.0, "MSFT": 20.0, "BTC": 25.0, "SOL": 15.0, "ETH": 15.0}
        report = PortfolioRiskAuditorService.audit_portfolio(weights)

        self.assertIn("diversification_score", report)
        self.assertIn("portfolio_volatility_pct", report)
        self.assertIn("annual_volatility_pct", report)
        self.assertIn("daily_var_95_pct", report)
        self.assertIn("cvar_95_pct", report)
        self.assertEqual(report["diversification_score"], 99)
        self.assertGreater(report["portfolio_volatility_pct"], 0.0)
        self.assertGreater(report["annual_volatility_pct"], 0.0)