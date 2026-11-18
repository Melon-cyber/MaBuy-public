import unittest
import torch
from src.infrastructure.ml.pytorch.attention_lstm import LSTMWithAttention


class TestLSTMWithAttention(unittest.TestCase):

    def test_forward_pass(self):
        batch_size = 16
        seq_len = 60
        input_dim = 12

        model = LSTMWithAttention(input_dim=input_dim, hidden_dim=32, num_layers=2)
        x_dummy = torch.randn(batch_size, seq_len, input_dim)

        pred_return, pred_prob = model(x_dummy)

        self.assertEqual(pred_return.shape, (batch_size, 1))
        self.assertEqual(pred_prob.shape, (batch_size, 1))
        self.assertTrue(torch.all(pred_prob >= 0.0) and torch.all(pred_prob <= 1.0))


if __name__ == "__main__":
    unittest.main()
