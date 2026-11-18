import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List

FEATURE_COLS: List[str] = [
    "r_lag1", "r_lag2", "r_lag3", "r_lag5",
    "volatility_20", "volatility_60", "vol_ratio",
    "z_return_lag1", "z_return_lag2", "z_return_lag3",
    "rsi_14", "macd_norm", "macd_signal_norm", "macd_hist_norm",
    "atr_norm", "bb_pb", "vol_ratio_20", "volume_log_return"
]
DEFAULT_INPUT_DIM: int = len(FEATURE_COLS)  # 18 stationary features


class TemporalAttention(nn.Module):
    """
    Temporal Self-Attention mechanism over sequence steps L.
    Computes dynamic attention weights alpha_t over historical context.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w_a = nn.Linear(hidden_dim, hidden_dim)
        self.v_a = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_outputs: (B, L, H)
        energy = torch.tanh(self.w_a(lstm_outputs)) # (B, L, H)
        scores = self.v_a(energy)                    # (B, L, 1)
        weights = F.softmax(scores, dim=1)           # (B, L, 1)

        # Context vector: Weighted sum over time dimension
        context = torch.sum(weights * lstm_outputs, dim=1) # (B, H)
        return context, weights


class LSTMWithAttention(nn.Module):
    """
    Multi-Head Temporal Attention LSTM Deep Neural Network for Cryptocurrency Forecasting.
    
    Inputs:
        X: (Batch, Sequence_Length=60, Feature_Dim=18)
    Outputs:
        pred_return: Expected log-return target (Batch, 1)
        pred_prob: BULLISH direction probability (Batch, 1)
    """

    def __init__(self, input_dim: int = DEFAULT_INPUT_DIM, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = TemporalAttention(hidden_dim)

        # Multi-Head Output Layers
        self.fc_return = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

        self.fc_prob = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: (B, L, F)
        lstm_out, _ = self.lstm(x)                  # (B, L, H)
        context, _ = self.attention(lstm_out)       # (B, H)

        pred_return = self.fc_return(context)        # (B, 1)
        pred_prob = self.fc_prob(context)            # (B, 1)

        return pred_return, pred_prob


class HorizonHead(nn.Module):
    """
    Specialized prediction head for a specific temporal horizon (1h, 4h, or 1d).
    Produces expected log return and direction win-rate probability.
    """

    def __init__(self, hidden_dim: int = 64, intermediate_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.fc_return = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1)
        )
        self.fc_prob = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.fc_return(context), self.fc_prob(context)


class MultiHorizonAttentionLSTM(nn.Module):
    """
    Multi-Task Multi-Horizon Temporal Attention LSTM Deep Neural Network.
    
    Shared Feature Representation:
    - Multi-layer LSTM + Temporal Self-Attention over sequential history
    
    Dedicated Temporal Heads:
    - 1h Head: Microstructure & high-frequency momentum
    - 4h Head: Swing trend continuation & breakout
    - 1d Head: Macro regime & multi-day directional bias
    """

    def __init__(self, input_dim: int = DEFAULT_INPUT_DIM, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = TemporalAttention(hidden_dim)

        self.head_1h = HorizonHead(hidden_dim, 32, dropout)
        self.head_4h = HorizonHead(hidden_dim, 32, dropout)
        self.head_1d = HorizonHead(hidden_dim, 32, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through shared backbone and all 3 horizon heads simultaneously.
        """
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)

        ret_1h, prob_1h = self.head_1h(context)
        ret_4h, prob_4h = self.head_4h(context)
        ret_1d, prob_1d = self.head_1d(context)

        return {
            "1h": (ret_1h, prob_1h),
            "4h": (ret_4h, prob_4h),
            "1d": (ret_1d, prob_1d)
        }

    def predict_horizon(self, x: torch.Tensor, horizon: str = "1h") -> Tuple[torch.Tensor, torch.Tensor]:
        """Convenience method for single horizon inference."""
        res_dict = self.forward(x)
        clean_h = horizon.lower()
        if clean_h not in res_dict:
            clean_h = "1h"
        return res_dict[clean_h]

