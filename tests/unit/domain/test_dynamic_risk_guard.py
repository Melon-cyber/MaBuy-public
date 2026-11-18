"""
Unit tests for DynamicRiskGuardService (Quarter-Kelly & Circuit Breaker)
"""
import unittest
from src.domain.services.dynamic_risk_guard import DynamicRiskGuardService

class TestDynamicRiskGuardService(unittest.TestCase):

    def setUp(self):
        self.guard = DynamicRiskGuardService()

    def test_pure_kelly_b_ratio_modulation(self):
        """
        PROOFS THAT b (Reward-to-Risk ratio) MODULATES POSITION SIZE AT SAME PROBABILITY (p=0.75):
        Trade A: p=0.75, b=1.2 -> Full Kelly K = 0.5417 -> Risk = 2.71%
        Trade B: p=0.75, b=3.0 -> Full Kelly K = 0.6667 -> Risk = 3.33%
        """
        # Trade A: Low R:R (b=1.2)
        res_a = self.guard.calculate_trade_risk(
            account_balance=1000.0,
            model_probability=0.75,
            take_profit_pct=1.2,
            stop_loss_pct=1.0
        )
        # Trade B: High R:R (b=3.0)
        res_b = self.guard.calculate_trade_risk(
            account_balance=1000.0,
            model_probability=0.75,
            take_profit_pct=3.0,
            stop_loss_pct=1.0
        )
        
        self.assertTrue(res_a["allowed"])
        self.assertTrue(res_b["allowed"])
        # Prove that higher R:R (b) at same probability p=0.75 yields higher position risk!
        self.assertNotEqual(res_a["risk_pct"], res_b["risk_pct"])
        self.assertGreater(res_b["risk_pct"], res_a["risk_pct"])
        self.assertEqual(res_a["risk_pct"], 2.71)
        self.assertEqual(res_b["risk_pct"], 3.33)

    def test_max_risk_bound_ceiling(self):
        """Ultra-high Kelly edge should still be capped at max ceiling 4.5%."""
        res = self.guard.calculate_trade_risk(
            account_balance=1000.0,
            model_probability=0.95,
            take_profit_pct=4.0,
            stop_loss_pct=1.0
        )
        self.assertTrue(res["allowed"])
        self.assertEqual(res["risk_pct"], 4.5)
        self.assertEqual(res["risk_amount_usd"], 45.0)

    def test_circuit_breaker_boundary_conditions(self):
        """
        Boundary tests:
        -7.4% drawdown -> ALLOWED (below limit)
        -7.5% drawdown -> HALTED (exactly on limit)
        -7.6% drawdown -> HALTED (past limit)
        """
        # -7.4% -> Allowed
        res_74 = self.guard.calculate_trade_risk(
            account_balance=1000.0,
            model_probability=0.75,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            current_daily_drawdown_pct=-7.4
        )
        self.assertTrue(res_74["allowed"])
        self.assertFalse(res_74["circuit_breaker_triggered"])

        # -7.5% -> Halted (Exactly on limit)
        res_75 = self.guard.calculate_trade_risk(
            account_balance=1000.0,
            model_probability=0.75,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            current_daily_drawdown_pct=-7.5
        )
        self.assertFalse(res_75["allowed"])
        self.assertTrue(res_75["circuit_breaker_triggered"])

        # -7.6% -> Halted (Past limit)
        res_76 = self.guard.calculate_trade_risk(
            account_balance=1000.0,
            model_probability=0.75,
            take_profit_pct=2.0,
            stop_loss_pct=1.0,
            current_daily_drawdown_pct=-7.6
        )
        self.assertFalse(res_76["allowed"])
        self.assertTrue(res_76["circuit_breaker_triggered"])

    def test_invalid_balance_throws(self):
        """Zero or negative balance must raise ValueError."""
        with self.assertRaises(ValueError):
            self.guard.calculate_trade_risk(
                account_balance=0.0,
                model_probability=0.6,
                take_profit_pct=1.5,
                stop_loss_pct=1.0
            )

if __name__ == "__main__":
    unittest.main()
