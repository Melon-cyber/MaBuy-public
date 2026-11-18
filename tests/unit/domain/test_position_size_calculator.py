import unittest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.domain.services.position_size_calculator import PositionSizeCalculatorService
from src.domain.services.dynamic_risk_guard import DynamicRiskGuardService

class TestPositionSizeCalculator(unittest.TestCase):
    def setUp(self):
        self.mock_fetcher = MagicMock()
        # Create a mock 1h OHLCV dataframe with known ATR
        dates = pd.date_range("2026-08-16", periods=20, freq="h")
        data = {
            "open": [100.0 + i for i in range(20)],
            "high": [105.0 + i for i in range(20)],
            "low": [95.0 + i for i in range(20)],
            "close": [102.0 + i for i in range(20)],
            "volume": [1000.0] * 20
        }
        self.mock_df = pd.DataFrame(data, index=dates)
        self.mock_fetcher.fetch_ohlcv.return_value = self.mock_df
        
        self.risk_guard = DynamicRiskGuardService()
        self.calculator = PositionSizeCalculatorService(
            risk_guard=self.risk_guard,
            data_fetcher=self.mock_fetcher
        )

    def test_calculate_position_plan_balanced(self):
        plan = self.calculator.calculate_position_plan(
            account_balance=1000.0,
            symbol="BTC",
            direction="BULLISH",
            risk_level="balanced"
        )
        
        self.assertEqual(plan["symbol"], "BTC")
        self.assertEqual(plan["direction"], "BULLISH")
        self.assertGreater(plan["current_price"], 0)
        self.assertGreater(plan["take_profit_price"], plan["current_price"])
        self.assertLess(plan["stop_loss_price"], plan["current_price"])
        self.assertGreaterEqual(plan["reward_risk_ratio"], 1.8)
        self.assertGreaterEqual(plan["risk_pct"], 1.0)
        self.assertLessEqual(plan["risk_pct"], 4.5)
        self.assertEqual(plan["kelly_fraction"], 0.05)
        self.assertGreater(plan["position_size_usd"], 0)
        self.assertGreater(plan["position_size_coins"], 0)
        self.assertGreaterEqual(plan["max_safe_leverage"], 1)

    def test_calculate_position_plan_bearish(self):
        plan = self.calculator.calculate_position_plan(
            account_balance=5000.0,
            symbol="ETH",
            direction="BEARISH",
            risk_level="conservative"
        )
        self.assertEqual(plan["direction"], "BEARISH")
        self.assertLess(plan["take_profit_price"], plan["current_price"])
        self.assertGreater(plan["stop_loss_price"], plan["current_price"])

    def test_invalid_balance_raises(self):
        with self.assertRaises(ValueError):
            self.calculator.calculate_position_plan(account_balance=-100.0, symbol="SOL")

if __name__ == "__main__":
    unittest.main()
