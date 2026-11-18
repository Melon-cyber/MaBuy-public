import unittest
import torch
from src.infrastructure.ml.pytorch.financial_loss import AsymmetricFinancialLoss


class TestAsymmetricFinancialLoss(unittest.TestCase):

    def test_gradient_flow_with_tanh_proxy(self):
        loss_fn = AsymmetricFinancialLoss()

        pred_returns = torch.randn(32, 1, requires_grad=True)
        pred_probs = torch.sigmoid(torch.randn(32, 1)).requires_grad_()
        true_returns = torch.randn(32, 1)
        true_dirs = (true_returns > 0).float()

        loss = loss_fn(pred_returns, pred_probs, true_returns, true_dirs)
        loss.backward()

        self.assertIsNotNone(pred_returns.grad)
        self.assertIsNotNone(pred_probs.grad)
        # Ensure non-zero gradients flow through tanh proxy
        self.assertTrue(torch.any(pred_returns.grad != 0.0))


if __name__ == "__main__":
    unittest.main()
