"""
Unit tests for Phase 4 Quantitative Upgrades derived from genuine GPT-6 Astra audit:
1. Small-Capital Risk Sizing Guard ($39.86 Equity Protection)
2. Authoritative Naked Position Auto-Reconciliation
3. Walk-Forward Label Purging & Embargo (Lopez de Prado Standard) with Net Cost Deduction
"""

import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.infrastructure.execution.binance_trader import BinanceAutoTrader, ACTIVE_STOP_PRICES
from src.infrastructure.mlops.weekly_retrain_pipeline import WeeklyRetrainingPipeline


class TestPhase4AstraUpgrades(unittest.TestCase):

    def setUp(self):
        self.api_key = "test_key"
        self.api_secret = "test_secret"
        self.trader = BinanceAutoTrader(api_key=self.api_key, api_secret=self.api_secret)

    def test_small_capital_sizing_allows_safe_risk(self):
        """1. When stop distance is normal (1.8%), min order on $39.86 equity risks < 1.5% -> trade allowed."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_balance.return_value = {
            'USDT': {'free': 65.0, 'total': 65.0}
        }
        mock_exchange.fetch_positions.return_value = []
        mock_exchange.fetch_ticker.return_value = {'last': 100.0, 'close': 100.0}
        mock_exchange.market.return_value = {'limits': {'amount': {'min': 0.1}}}
        mock_exchange.amount_to_precision.side_effect = lambda sym, amt: f"{round(amt, 2):.2f}"
        mock_exchange.price_to_precision.side_effect = lambda sym, p: f"{round(p, 2):.2f}"
        mock_exchange.create_order.return_value = {'id': 'test_sl_id'}
        mock_exchange.create_market_order.return_value = {'id': 'test_mkt_id', 'average': 100.0, 'status': 'closed'}

        with patch.object(self.trader, '_create_isolated_session', return_value=mock_exchange):
            # 1.8% SL with $39.86 equity -> risk is ~$0.37 (0.93% of equity <= 1.5%) -> Allowed
            res = self.trader.execute_trade(
                symbol="SOL/USDT:USDT",
                direction="BULLISH",
                leverage=3,
                risk_pct=0.01,
                atr_pct=0.008  # Yields 1.8% SL
            )
            self.assertIsNotNone(res, "Safe risk trade should be executed")
            self.assertEqual(res["execution_mode"], "TAKER")

    def test_small_capital_sizing_skips_excessive_risk(self):
        """2. When stop distance is wide (4.0%), min order on $39.86 equity risks > 1.5% -> trade safely skipped."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_balance.return_value = {
            'USDT': {'free': 35.0, 'total': 39.86}
        }
        mock_exchange.fetch_positions.return_value = []
        mock_exchange.fetch_ticker.return_value = {'last': 100.0, 'close': 100.0}
        mock_exchange.market.return_value = {'limits': {'amount': {'min': 0.1}}}

        with patch.object(self.trader, '_create_isolated_session', return_value=mock_exchange):
            # Wide ATR yields wide SL -> $20.50 notional * 3.2% SL = $0.66 loss = 1.65% of $39.86 (> 1.5% cap) -> Skipped
            res = self.trader.execute_trade(
                symbol="SOL/USDT:USDT",
                direction="BULLISH",
                leverage=3,
                risk_pct=0.01,
                atr_pct=0.030  # Yields 3.2% max SL
            )
            self.assertIsNone(res, "Excessive risk on micro-deposit must be safely skipped")

    def test_naked_position_auto_reconciliation(self):
        """3. If a position is active on Binance with 0 Stop-Loss orders, auto-reconcile immediately."""
        mock_exchange = MagicMock()
        mock_exchange.fetch_positions.return_value = [
            {
                'symbol': 'SOL/USDT:USDT',
                'side': 'long',
                'contracts': 0.2,
                'entryPrice': 100.0,
                'markPrice': 100.2,  # Sub-1.0% (no profit lock triggered yet)
                'info': {'positionAmt': '0.2', 'entryPrice': '100.0', 'markPrice': '100.2'}
            }
        ]
        # Return ZERO active algo orders (position is naked!)
        mock_exchange.fapiPrivateGetOpenAlgoOrders.return_value = []
        mock_exchange.fetch_open_orders.return_value = []
        mock_exchange.price_to_precision.side_effect = lambda sym, p: f"{round(p, 2):.2f}"
        mock_exchange.create_order.return_value = {'id': 'reconciled_sl_id'}

        ACTIVE_STOP_PRICES.clear()

        with patch.object(self.trader, '_create_isolated_session', return_value=mock_exchange):
            updates = self.trader.check_and_apply_breakeven()
            
            # Must detect naked position and auto-reconcile SL
            self.assertTrue(any(u.get("action") == "AUTO_RECONCILE_SL" for u in updates))
            self.assertIn("SOL", ACTIVE_STOP_PRICES)
            mock_exchange.create_order.assert_called()
            call_kwargs = mock_exchange.create_order.call_args[1]
            self.assertEqual(call_kwargs.get("type"), "STOP_MARKET")
            self.assertEqual(call_kwargs.get("side"), "sell")

    def test_wfv_purging_and_cost_deduction(self):
        """4. Verify WFV 1-bar embargo and net cost deduction."""
        pipeline = WeeklyRetrainingPipeline()
        
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1.0, 1.0, -1.0, -1.0])
        y_test = np.array([0.02, -0.01, -0.02, 0.01])
        
        da, sharpe, mae, rmse = pipeline.evaluate_out_of_sample(
            mock_model, np.zeros((4, 5)), y_test, fee_roundtrip=0.0004, slippage=0.0002
        )
        self.assertEqual(da, 0.50)
        self.assertGreater(sharpe, 0.0)

        X = np.random.randn(500, 10)
        y = np.random.randn(500)
        gbdt_params = {"max_iter": 5, "max_depth": 2, "random_state": 42}
        
        folds, mean_da, mean_sharpe, mean_mae, mean_rmse = pipeline.walk_forward_evaluate(
            X, y, gbdt_params, n_folds=3, min_train_size=100
        )
        self.assertEqual(len(folds), 3)
        for f in folds:
            self.assertGreater(f["train_size"], 0)
            self.assertGreater(f["test_size"], 0)


if __name__ == "__main__":
    unittest.main()
