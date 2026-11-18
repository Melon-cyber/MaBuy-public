import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class CryptoSequenceDataset(Dataset):
    """
    PyTorch Dataset transforming 2D feature matrices into 3D sliding window tensors.
    
    Tensor Shapes:
    --------------
    Input  X: (N_samples - seq_len, seq_len, num_features)
    Target y: (N_samples - seq_len, 1)  [Target log return at t+1]
    Target y_dir: (N_samples - seq_len, 1) [Binary direction: 1 for positive return, 0 otherwise]
    """

    def __init__(self, X_features: np.ndarray, y_targets: np.ndarray, seq_len: int = 60):
        self.seq_len = seq_len
        
        # Convert to torch FloatTensors
        X_tensor = torch.tensor(X_features, dtype=torch.float32)
        y_tensor = torch.tensor(y_targets, dtype=torch.float32).unsqueeze(1)

        num_samples = len(X_features) - seq_len
        if num_samples <= 0:
            raise ValueError(f"Insufficient dataset length {len(X_features)} for sequence length {seq_len}")

        X_sequences = []
        y_sequences = []
        y_dir_sequences = []

        for i in range(num_samples):
            # Window [i : i + seq_len] -> Predict target at i + seq_len
            x_seq = X_tensor[i : i + seq_len]
            y_target = y_tensor[i + seq_len]
            y_dir = (y_target > 0).float()

            X_sequences.append(x_seq)
            y_sequences.append(y_target)
            y_dir_sequences.append(y_dir)

        self.X_data = torch.stack(X_sequences)
        self.y_data = torch.stack(y_sequences)
        self.y_dir_data = torch.stack(y_dir_sequences)

    def __len__(self) -> int:
        return len(self.X_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_data[idx], self.y_data[idx], self.y_dir_data[idx]


class MultiHorizonCryptoSequenceDataset(Dataset):
    """
    PyTorch Dataset transforming 2D feature matrices into 3D sliding window tensors
    with Multi-Horizon targets [1h, 4h, 1d].

    Tensor Shapes:
    --------------
    Input  X: (N_samples - seq_len, seq_len, num_features)
    Target Y_rets: (N_samples - seq_len, 3) [Log returns for 1h, 4h, 1d]
    Target Y_dirs: (N_samples - seq_len, 3) [Binary directions for 1h, 4h, 1d]
    """

    def __init__(self, X_features: np.ndarray, Y_targets: np.ndarray, seq_len: int = 60):
        self.seq_len = seq_len

        # Convert to torch FloatTensors
        X_tensor = torch.tensor(X_features, dtype=torch.float32)
        if Y_targets.ndim == 1:
            Y_targets = Y_targets.reshape(-1, 1)
        Y_tensor = torch.tensor(Y_targets, dtype=torch.float32)

        num_samples = len(X_features) - seq_len
        if num_samples <= 0:
            raise ValueError(f"Insufficient dataset length {len(X_features)} for sequence length {seq_len}")

        X_sequences = []
        Y_sequences = []
        Y_dir_sequences = []

        for i in range(num_samples):
            # Window [i : i + seq_len] -> Targets at i + seq_len (representing t)
            x_seq = X_tensor[i : i + seq_len]
            y_tgt = Y_tensor[i + seq_len]
            y_dir = (y_tgt > 0).float()

            X_sequences.append(x_seq)
            Y_sequences.append(y_tgt)
            Y_dir_sequences.append(y_dir)

        self.X_data = torch.stack(X_sequences)
        self.Y_data = torch.stack(Y_sequences)
        self.Y_dir_data = torch.stack(Y_dir_sequences)

    def __len__(self) -> int:
        return len(self.X_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X_data[idx], self.Y_data[idx], self.Y_dir_data[idx]
