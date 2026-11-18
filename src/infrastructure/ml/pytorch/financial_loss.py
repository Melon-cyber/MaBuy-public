import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


class AsymmetricFinancialLoss(nn.Module):
    r"""
    Differentiable Financial Sharpe & Profit-Aware Loss Function for PyTorch.

    Mathematical Breakthrough:
    --------------------------
    Replaces step non-differentiable sign() function with smooth proxy:
    smooth_signal = torch.tanh(pred_returns * 10.0)

    Since d/dx [tanh(10x)] = 10 * (1 - tanh^2(10x)), non-zero gradients flow
    backward through chain-rule into LSTM weights, maximizing risk-adjusted Sharpe Ratio.
    """

    def __init__(self, lambda_sharpe: float = 1.0, lambda_bce: float = 0.5, lambda_mse: float = 1.0):
        super().__init__()
        self.lambda_sharpe = lambda_sharpe
        self.lambda_bce = lambda_bce
        self.lambda_mse = lambda_mse
        self.bce_fn = nn.BCELoss()
        self.mse_fn = nn.MSELoss()

    def forward(
        self,
        pred_returns: torch.Tensor,
        pred_probs: torch.Tensor,
        true_returns: torch.Tensor,
        true_dirs: torch.Tensor
    ) -> torch.Tensor:
        # 1. Differentiable Soft Signal via tanh(10 * pred)
        smooth_signals = torch.tanh(pred_returns * 10.0)

        # 2. Strategy Returns: smooth_signals * true_returns
        strategy_returns = smooth_signals * true_returns

        if strategy_returns.numel() > 1:
            mean_ret = torch.mean(strategy_returns)
            std_ret = torch.std(strategy_returns) + 1e-6
            # Negative Sharpe Ratio to minimize
            sharpe_loss = - (mean_ret / std_ret)
        else:
            sharpe_loss = torch.tensor(0.0, device=strategy_returns.device, dtype=strategy_returns.dtype)

        # 3. Binary Direction Cross-Entropy Loss
        bce_loss = self.bce_fn(pred_probs, true_dirs)

        # 4. Regression Error Loss
        mse_loss = self.mse_fn(pred_returns, true_returns)

        total_loss = (
            self.lambda_sharpe * sharpe_loss +
            self.lambda_bce * bce_loss +
            self.lambda_mse * mse_loss
        )
        return total_loss


class MultiHorizonFinancialLoss(nn.Module):
    r"""
    Multi-Task Multi-Horizon Financial Sharpe & Loss Balancing Function.
    
    Gradient Balancing Solution:
    ----------------------------
    Since daily variance is 24x higher than hourly variance (\sigma_{1d}^2 \approx 24 \sigma_{1h}^2),
    raw unweighted multi-task losses would allow the 1d head to dominate gradients and suppress
    1h feature learning. We weight each horizon loss inversely to its variance:
    \alpha_{1h} = 1.0, \alpha_{4h} = 0.25, \alpha_{1d} = 1 / 24 \approx 0.04167.
    """

    def __init__(
        self,
        weights: Tuple[float, float, float] = (1.0, 0.25, 0.04167),
        lambda_sharpe: float = 1.0,
        lambda_bce: float = 0.5,
        lambda_mse: float = 1.0
    ):
        super().__init__()
        self.weights = weights
        self.sub_loss_1h = AsymmetricFinancialLoss(lambda_sharpe, lambda_bce, lambda_mse)
        self.sub_loss_4h = AsymmetricFinancialLoss(lambda_sharpe, lambda_bce, lambda_mse)
        self.sub_loss_1d = AsymmetricFinancialLoss(lambda_sharpe, lambda_bce, lambda_mse)

    def forward(
        self,
        pred_dict: dict,
        true_returns_matrix: torch.Tensor,
        true_dirs_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred_dict: Dict with keys '1h', '4h', '1d', each containing (pred_return, pred_prob)
            true_returns_matrix: Tensor of shape (Batch, 3) for [1h, 4h, 1d]
            true_dirs_matrix: Tensor of shape (Batch, 3) for [1h, 4h, 1d]
        """
        w_1h, w_4h, w_1d = self.weights

        ret_1h, prob_1h = pred_dict["1h"]
        ret_4h, prob_4h = pred_dict["4h"]
        ret_1d, prob_1d = pred_dict["1d"]

        loss_1h = self.sub_loss_1h(ret_1h, prob_1h, true_returns_matrix[:, 0:1], true_dirs_matrix[:, 0:1])
        loss_4h = self.sub_loss_4h(ret_4h, prob_4h, true_returns_matrix[:, 1:2], true_dirs_matrix[:, 1:2])
        loss_1d = self.sub_loss_1d(ret_1d, prob_1d, true_returns_matrix[:, 2:3], true_dirs_matrix[:, 2:3])

        total_multi_loss = (w_1h * loss_1h) + (w_4h * loss_4h) + (w_1d * loss_1d)
        return total_multi_loss

