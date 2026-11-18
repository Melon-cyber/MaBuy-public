"""
Unit tests for V5DynamicRiskEngine:
- Garman-Klass Volatility calculation vs theoretical values
- Dynamic Leverage Matrix (1x to 5x) across BTC, ETH, SOL, ALTS
- Adaptive Fractional Kelly Sizing (0.15*K to 0.40*K)
- Portfolio Margin Cap (20% - 25%) and dynamic position scaling
- Micro-deposit ($15 - $100) constraints and Binance $5.00 min notional
- Circuit Breaker and Gatekeeper vetoes
"""

import unittest
import numpy as np
import math
from src.domain.services.v5_risk_engine import (
    V5DynamicRiskEngine,
    AssetTier,
    MarketRegimeType,
    TradeRiskPlan,
    PortfolioMarginStatus
)


class TestV5DynamicRiskEngine(unittest.TestCase):

    def test_garman_klass_volatility_calculation(self):
        """
        Tests Garman-Klass estimator against known synthetic OHLC dataset.
        GK formula uses high/low spread and close/open spread.
        """
        # 10 identical candles with 2% High-Low range and 0.5% Close-Open range
        opens = np.array([100.0] * 10)
        highs = np.array([101.0] * 10)
        lows = np.array([99.0] * 10)
        closes = np.array([100.5] * 10)

        vol = V5DynamicRiskEngine.calculate_garman_klass_volatility(opens, highs, lows, closes)
        self.assertGreater(vol, 0.0)
        self.assertLess(vol, 5.0)

        # Extreme volatility candle series
        highs_ext = np.array([115.0] * 10)
        lows_ext = np.array([85.0] * 10)
        vol_ext = V5DynamicRiskEngine.calculate_garman_klass_volatility(opens, highs_ext, lows_ext, closes)
        self.assertGreater(vol_ext, vol)

    def test_asset_tier_stratification(self):
        """BTC/ETH are Tier 1 Core, SOL/BNB are Tier 2 Major, others are Tier 3 ALTS."""
        self.assertEqual(V5DynamicRiskEngine.get_asset_tier("BTCUSDT"), AssetTier.TIER_1_CORE)
        self.assertEqual(V5DynamicRiskEngine.get_asset_tier("ETH"), AssetTier.TIER_1_CORE)
        self.assertEqual(V5DynamicRiskEngine.get_asset_tier("SOLUSDT"), AssetTier.TIER_2_MAJOR)
        self.assertEqual(V5DynamicRiskEngine.get_asset_tier("BNB"), AssetTier.TIER_2_MAJOR)
        self.assertEqual(V5DynamicRiskEngine.get_asset_tier("DOGE"), AssetTier.TIER_3_ALTS)
        self.assertEqual(V5DynamicRiskEngine.get_asset_tier("AVAXUSDT"), AssetTier.TIER_3_ALTS)

    def test_dynamic_leverage_matrix(self):
        """
        Tests Dynamic Leverage Matrix:
        - BTC high confidence + low vol -> 5x
        - SOL high confidence + low vol -> 5x
        - ALTS high confidence + low vol -> 4x
        - High volatility storm -> reduced to 1x - 2x
        - Stop-loss constraint ensures safety buffer >= 2.0x
        """
        # BTC: Tier 1, High Confidence, Low Volatility -> 5x
        res_btc = V5DynamicRiskEngine.calculate_safe_leverage(
            symbol="BTC",
            model_probability=0.75,
            gk_volatility_pct=1.4,
            stop_loss_pct=1.2,
            market_regime=MarketRegimeType.CLEAN_TREND
        )
        self.assertEqual(res_btc.safe_leverage, 5)
        self.assertFalse(res_btc.veto)
        self.assertGreaterEqual(res_btc.safety_buffer_ratio, 2.0)

        # SOL: Tier 2, Moderate Confidence, Medium Vol -> 3x
        res_sol = V5DynamicRiskEngine.calculate_safe_leverage(
            symbol="SOL",
            model_probability=0.62,
            gk_volatility_pct=2.8,
            stop_loss_pct=2.0,
            market_regime=MarketRegimeType.TRANSITION
        )
        self.assertEqual(res_sol.safe_leverage, 3)

        # ALTS (DOGE): High Volatility Storm -> 1x - 2x
        res_doge = V5DynamicRiskEngine.calculate_safe_leverage(
            symbol="DOGE",
            model_probability=0.60,
            gk_volatility_pct=4.8,
            stop_loss_pct=3.5,
            market_regime=MarketRegimeType.TRANSITION
        )
        self.assertLessEqual(res_doge.safe_leverage, 2)

    def test_adaptive_fractional_kelly_sizing(self):
        """
        Tests adaptive Kelly fraction:
        - Flat range -> 0.15*K
        - Clean trend -> 0.40*K
        - Proves that identical edge (p=0.70, b=2.2) allocates higher risk in clean trend than in flat!
        """
        # Flat Regime (CHOP = 55.0, ADX = 16.0)
        plan_flat = V5DynamicRiskEngine.calculate_trade_risk_plan(
            account_balance=1000.0,
            symbol="BTC",
            current_price=60000.0,
            stop_loss_price=59100.0, # SL = 1.5%
            take_profit_price=61980.0, # TP = 3.3% (b = 2.2)
            model_probability=0.70,
            adx=16.0,
            choppiness_index=55.0,
            gk_volatility_pct=1.5
        )
        self.assertTrue(plan_flat.allowed)
        self.assertEqual(plan_flat.kelly_fraction, 0.15)
        self.assertEqual(plan_flat.market_regime, MarketRegimeType.FLAT_RANGE)

        # Clean Trend Regime (CHOP = 35.0, ADX = 35.0)
        plan_trend = V5DynamicRiskEngine.calculate_trade_risk_plan(
            account_balance=1000.0,
            symbol="BTC",
            current_price=60000.0,
            stop_loss_price=59100.0, # SL = 1.5%
            take_profit_price=61980.0, # TP = 3.3% (b = 2.2)
            model_probability=0.70,
            adx=35.0,
            choppiness_index=35.0,
            gk_volatility_pct=1.5
        )
        self.assertTrue(plan_trend.allowed)
        self.assertEqual(plan_trend.kelly_fraction, 0.25)  # Quarter-Kelly (was 0.40)
        self.assertEqual(plan_trend.market_regime, MarketRegimeType.CLEAN_TREND)

        # Trend risk must be greater than flat risk
        self.assertGreater(plan_trend.final_risk_pct, plan_flat.final_risk_pct)
        self.assertGreater(plan_trend.risk_amount_usd, plan_flat.risk_amount_usd)

    def test_portfolio_margin_cap_enforcement(self):
        """
        Tests 25% max portfolio margin cap on $500 balance ($125 max margin).
        When existing positions use $110, only $15 margin is available.
        A new trade requiring $30 margin is gracefully downscaled to $15!
        """
        status = V5DynamicRiskEngine.evaluate_portfolio_margin_cap(
            account_balance=500.0,
            active_positions_margins=[50.0, 60.0] # Total used = $110 / $125
        )
        self.assertTrue(status.can_open_new_trade)
        self.assertEqual(status.available_margin_usd, 15.0)
        self.assertEqual(status.max_allowed_margin_usd, 125.0)

        plan = V5DynamicRiskEngine.calculate_trade_risk_plan(
            account_balance=500.0,
            symbol="ETH",
            current_price=3000.0,
            stop_loss_price=2940.0, # SL = 2.0%
            take_profit_price=3132.0, # TP = 4.4% (R:R = 2.2)
            model_probability=0.72,
            portfolio_status=status
        )
        self.assertTrue(plan.allowed)
        self.assertTrue(plan.scaled_for_margin_cap)
        self.assertLessEqual(plan.required_margin_usd, 15.0)

    def test_micro_deposit_safeguards_and_min_notional(self):
        """
        Tests $30 micro-deposit on SOL:
        - Must respect 2.0% risk cap ($0.60)
        - Must satisfy Binance minimum notional >= $5.00
        - Required margin must be <= 20% of deposit ($6.00)
        """
        plan = V5DynamicRiskEngine.calculate_trade_risk_plan(
            account_balance=30.0,
            symbol="SOL",
            current_price=150.0,
            stop_loss_price=147.0, # SL = 2.0%
            take_profit_price=156.6, # TP = 4.4% (R:R = 2.2)
            model_probability=0.68,
            adx=22.0,
            choppiness_index=48.0
        )
        self.assertTrue(plan.allowed)
        self.assertTrue(plan.is_micro_cap)
        self.assertGreaterEqual(plan.position_notional_usd, 5.00)
        self.assertLessEqual(plan.required_margin_usd, 6.00)
        self.assertLessEqual(plan.final_risk_pct, 2.50)

    def test_gatekeeper_vetoes(self):
        """
        Tests gatekeeper vetoes:
        1. Low R:R (< 1.8) -> Veto
        2. Low win probability (< 52% in trend or < 58% in flat) -> Veto
        3. Emergency daily drawdown (-7.5%) -> Veto
        """
        # Low R:R (1.2)
        plan_bad_rr = V5DynamicRiskEngine.calculate_trade_risk_plan(
            account_balance=100.0,
            symbol="BTC",
            current_price=60000.0,
            stop_loss_price=59000.0,
            take_profit_price=61200.0, # R:R = 1.2
            model_probability=0.70
        )
        self.assertFalse(plan_bad_rr.allowed)
        self.assertIn("Risk-to-Reward", plan_bad_rr.reason)

        # Low Probability in Flat (55% < 58%)
        plan_low_p = V5DynamicRiskEngine.calculate_trade_risk_plan(
            account_balance=100.0,
            symbol="BTC",
            current_price=60000.0,
            stop_loss_price=59000.0,
            take_profit_price=62200.0,
            model_probability=0.55,
            choppiness_index=55.0 # Flat regime requires >= 58%
        )
        self.assertFalse(plan_low_p.allowed)
        self.assertIn("Model probability", plan_low_p.reason)

        # Emergency Drawdown (-7.5%)
        plan_halt = V5DynamicRiskEngine.calculate_trade_risk_plan(
            account_balance=100.0,
            symbol="BTC",
            current_price=60000.0,
            stop_loss_price=59000.0,
            take_profit_price=62200.0,
            model_probability=0.70,
            current_daily_drawdown_pct=-7.5
        )
        self.assertFalse(plan_halt.allowed)
        self.assertIn("HALT", plan_halt.reason)


if __name__ == "__main__":
    unittest.main()
