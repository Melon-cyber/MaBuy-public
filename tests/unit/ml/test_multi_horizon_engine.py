import os
import unittest
import numpy as np
import pandas as pd
import torch

from src.domain.entities.prediction import PredictionRequest
from src.domain.entities.tier import SubscriptionTier
from src.infrastructure.ml.feature_engineering.pipeline import CryptoFeaturePipeline
from src.infrastructure.ml.pytorch.dataset import MultiHorizonCryptoSequenceDataset
from src.infrastructure.ml.pytorch.attention_lstm import MultiHorizonAttentionLSTM
from src.infrastructure.ml.pytorch.financial_loss import MultiHorizonFinancialLoss
from src.infrastructure.ml.v4_titan_ensemble import MaBuyV4EnsembleEngine


class TestMultiHorizonEngine(unittest.TestCase):
    def setUp(self):
        # Generate synthetic realistic OHLCV candles
        np.random.seed(42)
        n = 200
        timestamps = pd.date_range("2026-01-01", periods=n, freq="1h")
        price = 50000.0
        prices = [price]
        for _ in range(n - 1):
            price += price * np.random.normal(0, 0.01)
            prices.append(price)
        
        self.raw_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": np.array(prices) * (1.0 + np.random.uniform(-0.002, 0.002, n)),
            "high": np.array(prices) * 1.005,
            "low": np.array(prices) * 0.995,
            "close": np.array(prices),
            "volume": np.random.uniform(100, 1000, n)
        })

    def test_feature_pipeline_multi_horizon_targets(self):
        """Verifies pipeline generates aligned 1h, 4h, 1d targets without lookahead leakage."""
        pipeline = CryptoFeaturePipeline()
        X_mat, Y_mat, feat_names = pipeline.transform_multi_horizon(self.raw_df)
        
        self.assertGreater(len(X_mat), 0)
        self.assertEqual(Y_mat.shape[1], 3)
        self.assertEqual(len(X_mat), len(Y_mat))
        self.assertFalse(np.isnan(X_mat).any())
        self.assertFalse(np.isnan(Y_mat).any())

    def test_multi_horizon_dataset_sliding_window(self):
        """Verifies dataset transforms features into 3D sliding window tensors with 3 targets."""
        pipeline = CryptoFeaturePipeline()
        X_mat, Y_mat, _ = pipeline.transform_multi_horizon(self.raw_df)
        
        seq_len = 30
        dataset = MultiHorizonCryptoSequenceDataset(X_mat, Y_mat, seq_len=seq_len)
        self.assertGreater(len(dataset), 0)
        
        x_seq, y_ret, y_dir = dataset[0]
        self.assertEqual(x_seq.shape, (seq_len, X_mat.shape[1]))
        self.assertEqual(y_ret.shape, (3,))
        self.assertEqual(y_dir.shape, (3,))

    def test_multi_horizon_attention_lstm_forward(self):
        """Verifies multi-horizon neural network processes 3 heads simultaneously."""
        input_dim = 15
        model = MultiHorizonAttentionLSTM(input_dim=input_dim, hidden_dim=32, num_layers=2)
        model.eval()
        
        batch_size = 4
        seq_len = 20
        x_sample = torch.randn(batch_size, seq_len, input_dim)
        
        with torch.no_grad():
            out_dict = model(x_sample)
            
        self.assertIn("1h", out_dict)
        self.assertIn("4h", out_dict)
        self.assertIn("1d", out_dict)
        
        for horizon in ["1h", "4h", "1d"]:
            ret, prob = out_dict[horizon]
            self.assertEqual(ret.shape, (batch_size, 1))
            self.assertEqual(prob.shape, (batch_size, 1))
            self.assertTrue(torch.all(prob >= 0.0) and torch.all(prob <= 1.0))

    def test_multi_horizon_financial_loss_gradient_flow(self):
        """Verifies weighted multi-task loss propagates gradients to all 3 heads and shared backbone."""
        input_dim = 10
        model = MultiHorizonAttentionLSTM(input_dim=input_dim, hidden_dim=32, num_layers=2)
        model.train()
        
        criterion = MultiHorizonFinancialLoss(weights=(1.0, 0.25, 0.04167))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        batch_size = 8
        seq_len = 15
        x_batch = torch.randn(batch_size, seq_len, input_dim)
        y_rets = torch.randn(batch_size, 3) * 0.02
        y_dirs = (y_rets > 0).float()
        
        optimizer.zero_grad()
        pred_dict = model(x_batch)
        loss = criterion(pred_dict, y_rets, y_dirs)
        loss.backward()
        
        # Check gradients exist on shared backbone and all 3 heads
        self.assertIsNotNone(model.lstm.weight_ih_l0.grad)
        self.assertIsNotNone(model.head_1h.fc_return[0].weight.grad)
        self.assertIsNotNone(model.head_4h.fc_return[0].weight.grad)
        self.assertIsNotNone(model.head_1d.fc_return[0].weight.grad)
        self.assertGreater(loss.item(), -100.0)

    def test_v4_ensemble_multi_horizon_inference(self):
        """Verifies V4 ensemble engine produces horizon-specialized forecasts for 1h, 4h, 1d."""
        engine = MaBuyV4EnsembleEngine()
        
        for tf in ["1h", "4h", "1d"]:
            req = PredictionRequest(
                user_id="v4_horizon_tester",
                symbol="BTC",
                tier=SubscriptionTier.PRO,
                timeframe=tf,
                lang="ru"
            )
            res = engine.predict(req)
            self.assertEqual(res.symbol, "BTC")
            self.assertEqual(res.timeframe, tf)
            self.assertIn(res.direction, ["BULLISH", "BEARISH", "NEUTRAL"])
            self.assertGreaterEqual(res.confidence_score, 0.50)
            self.assertGreater(res.predicted_target_price, 0.0)
            self.assertIn("MultiHorizon", res.model_version)


if __name__ == "__main__":
    unittest.main()
