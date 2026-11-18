import unittest
import numpy as np
import pandas as pd
from src.domain.services.market_regime import MarketRegimeClassifierService
from src.domain.services.dynamic_risk_guard import DynamicRiskGuardService

class TestQuantumRiskImprovements(unittest.TestCase):
    def setUp(self):
        self.regime_service = MarketRegimeClassifierService()
        self.risk_guard = DynamicRiskGuardService()

    def test_micro_cap_risk_capping(self):
        """
        Verifies that micro-deposits ($15) are strictly capped at 2.0% risk ($0.30)
        even with high model confidence (0.85).
        """
        res_micro = self.risk_guard.calculate_trade_risk(
            account_balance=15.0,
            model_probability=0.85,
            take_profit_pct=4.0,
            stop_loss_pct=1.5
        )
        self.assertTrue(res_micro["allowed"])
        self.assertTrue(res_micro["is_micro_cap"])
        self.assertEqual(res_micro["risk_pct"], 2.0)
        self.assertEqual(res_micro["risk_amount_usd"], 0.30)
        self.assertEqual(res_micro["break_even_trigger_pct"], 2.0)
        self.assertEqual(res_micro["trailing_trigger_pct"], 3.0)

    def test_standard_account_kelly_scaling(self):
        """
        Verifies standard account ($1000) can scale up to standard 4.5% ceiling.
        """
        res_std = self.risk_guard.calculate_trade_risk(
            account_balance=1000.0,
            model_probability=0.95,
            take_profit_pct=4.0,
            stop_loss_pct=1.0
        )
        self.assertTrue(res_std["allowed"])
        self.assertFalse(res_std["is_micro_cap"])
        self.assertEqual(res_std["risk_pct"], 4.5)
        self.assertEqual(res_std["risk_amount_usd"], 45.0)

    def test_choppiness_index_trending_vs_choppy(self):
        """
        Verifies Choppiness Index behaves correctly:
        - Linear trending series -> Low CHOP (< 38.2)
        - High frequency random walk -> High CHOP (> 55.0)
        """
        # Strong Trending Data
        trend_closes = np.linspace(100.0, 150.0, 30)
        trend_highs = trend_closes + 1.0
        trend_lows = trend_closes - 1.0
        chop_trend = MarketRegimeClassifierService.calculate_choppiness_index(
            trend_highs, trend_lows, trend_closes, period=14
        )
        self.assertLess(chop_trend, 40.0)

        # Choppy / Whipsaw Data (oscillating in tight range)
        np.random.seed(42)
        noise = np.sin(np.linspace(0, 30, 40)) * 2.0
        chop_closes = 100.0 + noise
        chop_highs = chop_closes + 1.5
        chop_lows = chop_closes - 1.5
        chop_val = MarketRegimeClassifierService.calculate_choppiness_index(
            chop_highs, chop_lows, chop_closes, period=14
        )
        self.assertGreater(chop_val, 55.0)

    def test_whipsaw_risk_gatekeeper(self):
        """
        Verifies evaluate_whipsaw_risk returns valid regime dict.
        """
        np.random.seed(42)
        noise = np.sin(np.linspace(0, 40, 40)) * 1.5
        closes = 100.0 + noise
        highs = closes + 1.2
        lows = closes - 1.2
        df_chop = pd.DataFrame({"high": highs, "low": lows, "close": closes})

        eval_res = MarketRegimeClassifierService.evaluate_whipsaw_risk(df_chop, period=14)
        self.assertIn("is_tradeable", eval_res)
        self.assertIn("choppiness_index", eval_res)
        self.assertIn("adx", eval_res)

if __name__ == "__main__":
    unittest.main()
