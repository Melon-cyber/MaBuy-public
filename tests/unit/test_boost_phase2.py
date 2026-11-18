import unittest
from unittest.mock import MagicMock, patch
import os
import sys
import numpy as np
import pandas as pd
import tempfile
import shutil
import gc
import torch

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.infrastructure.mlops.weekly_retrain_pipeline import FeatureEngineer, WeeklyRetrainingPipeline
from src.infrastructure.ml.baseline_model import BaselineGBDTModelEngine
from src.infrastructure.execution.binance_trader import BinanceAutoTrader, ACTIVE_STOP_PRICES


class TestBoostPhase2Upgrades(unittest.TestCase):
    def setUp(self):
        # Generate clean synthetic candles
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2026-01-01", periods=n, freq="1h")
        self.df = pd.DataFrame({
            "timestamp": dates,
            "open": 100.0 + np.random.randn(n).cumsum(),
            "high": 102.0 + np.random.randn(n).cumsum(),
            "low": 98.0 + np.random.randn(n).cumsum(),
            "close": 100.0 + np.random.randn(n).cumsum(),
            "volume": np.random.uniform(10, 100, n)
        })

    def test_feature_engineer_lag_fix(self):
        """1. Verify 1-bar momentum blind spot fix in FeatureEngineer."""
        clean_df, feature_cols = FeatureEngineer.compute_features(self.df)
        
        # Check columns exist
        for col in ["r_lag1", "r_lag2", "r_lag3", "r_lag5", "z_return_lag1", "z_return_lag2", "z_return_lag3"]:
            self.assertIn(col, clean_df.columns)
            self.assertIn(col, feature_cols)

        # Re-compute expected log return
        close = clean_df["close"]
        expected_log_ret = clean_df["log_return"]
        
        # r_lag1 MUST equal log_return of the candle that just closed (NOT shifted by an extra bar)
        np.testing.assert_allclose(clean_df["r_lag1"].values, expected_log_ret.values, rtol=1e-5)
        
        # r_lag2 is shift(1)
        np.testing.assert_allclose(clean_df["r_lag2"].iloc[1:].values, expected_log_ret.shift(1).iloc[1:].values, rtol=1e-5)
        
        # z_return_lag1 MUST equal z_return of the candle that just closed
        vol_20 = clean_df["volatility_20"]
        expected_z = expected_log_ret / vol_20
        np.testing.assert_allclose(clean_df["z_return_lag1"].values, expected_z.values, rtol=1e-5)

    def test_baseline_model_lag_alignment(self):
        """1b. Verify baseline_model.py lag features alignment with weekly retrain."""
        engine = BaselineGBDTModelEngine()
        
        # Test feature extraction on synthetic data
        with patch.object(engine.data_fetcher, 'fetch_ohlcv', return_value=self.df.copy()):
            price, matrix = engine._extract_real_market_feature_matrix("BTC", timeframe="1h", limit=50)
            self.assertEqual(matrix.shape[0], 50)
            self.assertGreater(price, 0)

    def test_atomic_stop_replacement_emergency_restore(self):
        """2. Verify atomic stop-loss replacement restores protective SL if API fails."""
        trader = BinanceAutoTrader(api_key="mock_key", api_secret="mock_sec")
        
        mock_exchange = MagicMock()
        # Mock active long position on BTC in profit (+1.5%)
        mock_exchange.fetch_positions.return_value = [
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "contracts": 0.01,
                "entryPrice": 60000.0,
                "markPrice": 60900.0, # +1.5% profit -> Tier 1 (+1.0% break-even)
                "info": {"positionAmt": "0.01", "entryPrice": "60000.0", "markPrice": "60900.0"}
            }
        ]
        mock_exchange.fapiPrivateGetOpenAlgoOrders.return_value = [
            {
                "orderType": "STOP_MARKET",
                "triggerPrice": "59000.0",
                "algoId": 12345
            }
        ]
        mock_exchange.price_to_precision.side_effect = lambda sym, val: f"{val:.2f}"
        
        mock_exchange.fetch_open_orders.return_value = []
        mock_exchange.cancel_order.return_value = {}
        
        # Simulate: delete order succeeds, but create_order for new stop FAILS
        create_order_calls = []
        def mock_create_order(symbol, type, side, amount, price=None, params=None):
            create_order_calls.append({"symbol": symbol, "type": type, "params": params})
            if len(create_order_calls) == 1:
                # First call (the new trailing SL) fails due to rate limit/glitch
                raise Exception("Binance API 429: Too Many Requests")
            # Second call (emergency restore) succeeds
            return {"id": "emergency_stop_id", "status": "NEW"}

        mock_exchange.create_order.side_effect = mock_create_order
        
        with patch.object(trader, '_create_isolated_session', return_value=mock_exchange):
            ACTIVE_STOP_PRICES["BTC"] = 59000.0
            updates = trader.check_and_apply_breakeven()
            
            # In C3 Create-Before-Cancel: old order is NEVER deleted if new order creation fails
            # This guarantees zero naked exposure during transit!
            mock_exchange.fapiPrivateDeleteAlgoOrder.assert_not_called()
            
            # Must have attempted to create new order, caught the error, and placed emergency restore
            self.assertEqual(len(create_order_calls), 2)
            self.assertEqual(create_order_calls[0]["params"]["stopPrice"], 60150.0) # Break-even target (+0.25% Net-of-Fees)
            self.assertEqual(create_order_calls[1]["params"]["stopPrice"], 59000.0) # Restored previous stop!
            self.assertEqual(ACTIVE_STOP_PRICES.get("BTC"), 59000.0)

    def test_atomic_stop_disaster_clears_cache(self):
        """2b. Verify that if both new SL and emergency restore fail, stale cache is purged."""
        trader = BinanceAutoTrader(api_key="mock_key", api_secret="mock_sec")
        mock_exchange = MagicMock()
        mock_exchange.fetch_positions.return_value = [
            {
                "symbol": "BTC/USDT:USDT",
                "side": "long",
                "contracts": 0.01,
                "entryPrice": 60000.0,
                "markPrice": 60900.0,
                "info": {"positionAmt": "0.01", "entryPrice": "60000.0", "markPrice": "60900.0"}
            }
        ]
        mock_exchange.fetch_open_orders.return_value = []
        mock_exchange.fapiPrivateGetOpenAlgoOrders.return_value = []
        mock_exchange.price_to_precision.side_effect = lambda sym, val: f"{val:.2f}"
        # Both attempts fail (Binance total outage)
        mock_exchange.create_order.side_effect = Exception("Binance Connection Refused")

        with patch.object(trader, '_create_isolated_session', return_value=mock_exchange):
            ACTIVE_STOP_PRICES["BTC"] = 59000.0
            trader.check_and_apply_breakeven()
            # Cache must be evicted to avoid skipping on next cycle
            self.assertNotIn("BTC", ACTIVE_STOP_PRICES)

    def test_daemon_interval(self):
        """3. Verify 10-second trailing stop resolution interval in daemon script."""
        daemon_path = os.path.join(BASE_DIR, "scripts", "auto_trader_daemon.py")
        with open(daemon_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertTrue("FastPositionEscortWorker" in content or "interval_sec: float = 10.0" in content or "now - last_verify_time >= 10" in content)

    def test_pytorch_finetuning_loop(self):
        """4. Verify PyTorch Attention LSTM warm-start fine-tuning loop."""
        tmpdir = tempfile.mkdtemp()
        try:
            pipeline = WeeklyRetrainingPipeline()
            pipeline.models_dir = tmpdir
            
            # Fine-tune on 200 synthetic candles
            res = pipeline.finetune_pytorch_lstm("BTC", self.df, epochs=2, lr=1e-4)
            self.assertTrue(res.get("success"), f"Fine-tuning failed: {res}")
            
            expected_file = os.path.join(tmpdir, "pytorch_attention_lstm_BTCUSDT_1h.pt")
            self.assertTrue(os.path.exists(expected_file))
            
            # Check checkpoint contents (must contain both state_dict keys for v4 ensemble)
            ckpt = torch.load(expected_file, map_location="cpu", weights_only=False)
            self.assertIn("model_state_dict", ckpt)
            self.assertIn("state_dict", ckpt)
            self.assertIn("feature_names", ckpt)
            self.assertEqual(ckpt["seq_len"], 60)
            
            # Check no NaN weights in saved checkpoint
            for k, param in ckpt["model_state_dict"].items():
                self.assertFalse(torch.isnan(param).any(), f"NaN in saved parameter {k}")
            
            del ckpt
            gc.collect()

            # Second run should warm-start from previous checkpoint
            res2 = pipeline.finetune_pytorch_lstm("BTC", self.df, epochs=1, lr=1e-4)
            self.assertTrue(res2.get("warm_started"))
        finally:
            gc.collect()
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_pytorch_nan_immunity_on_remainder_one(self):
        """4b. Verify fine-tuning loop is completely immune to NaN weight corruption on remainder-1 batch."""
        from src.infrastructure.ml.pytorch.financial_loss import AsymmetricFinancialLoss
        crit = AsymmetricFinancialLoss()
        
        # Single sample input (where std() formerly yielded NaN)
        pred_ret = torch.tensor([[0.05]], requires_grad=True)
        pred_prob = torch.tensor([[0.6]], requires_grad=True)
        true_ret = torch.tensor([[0.03]])
        true_dir = torch.tensor([[1.0]])
        
        loss = crit(pred_ret, pred_prob, true_ret, true_dir)
        self.assertFalse(torch.isnan(loss), "Loss on single sample must not be NaN")
        loss.backward()
        self.assertFalse(torch.isnan(pred_ret.grad), "Gradients on single sample must not be NaN")


if __name__ == "__main__":
    unittest.main()
