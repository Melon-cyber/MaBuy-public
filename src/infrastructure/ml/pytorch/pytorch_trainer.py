import os
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader

from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher
from src.infrastructure.ml.feature_engineering.pipeline import CryptoFeaturePipeline
from src.infrastructure.ml.pytorch.dataset import CryptoSequenceDataset
from src.infrastructure.ml.pytorch.attention_lstm import LSTMWithAttention
from src.infrastructure.ml.pytorch.financial_loss import AsymmetricFinancialLoss

logger = logging.getLogger(__name__)


class PyTorchModelTrainer:
    """
    Stage 2 PyTorch Deep Learning Model Trainer.
    Orchestrates 3D Sequence Dataset generation, LSTMWithAttention model training,
    AsymmetricFinancialLoss optimization, and Walk-Forward evaluation.
    """

    def __init__(self, artifacts_dir: str = "models", seq_len: int = 60, batch_size: int = 32):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.fetcher = CryptoDataFetcher()
        self.feature_pipeline = CryptoFeaturePipeline()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _train_epoch(self, model, dataloader, optimizer, criterion) -> float:
        model.train()
        total_loss = 0.0

        for X_b, y_b, y_dir_b in dataloader:
            X_b, y_b, y_dir_b = X_b.to(self.device), y_b.to(self.device), y_dir_b.to(self.device)

            optimizer.zero_grad()
            pred_return, pred_prob = model(X_b)

            loss = criterion(pred_return, pred_prob, y_b, y_dir_b)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item()) * len(X_b)

        return total_loss / len(dataloader.dataset)

    def train_and_evaluate(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        candle_limit: int = 1500,
        epochs: int = 15
    ) -> Tuple[Dict[str, float], pd.DataFrame, str]:
        logger.info(f"Starting Stage 2 PyTorch Training on Device [{self.device}] for {symbol}...")

        # 1. Fetch OHLCV Market Data
        raw_df = self.fetcher.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=candle_limit)

        # 2. Extract Features
        X_matrix, y_target, feature_names = self.feature_pipeline.transform(raw_df)

        # 3. Create PyTorch 3D Sequence Dataset
        dataset = CryptoSequenceDataset(X_matrix, y_target, seq_len=self.seq_len)

        # 4. Perform Walk-Forward Cross-Validation
        n_splits = 4
        total_samples = len(dataset)
        fold_size = total_samples // (n_splits + 1)

        fold_results = []

        for fold in range(1, n_splits + 1):
            train_end = fold * fold_size
            test_end = min(train_end + fold_size, total_samples)

            train_subset = torch.utils.data.Subset(dataset, range(0, train_end))
            test_subset = torch.utils.data.Subset(dataset, range(train_end, test_end))

            train_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_subset, batch_size=self.batch_size, shuffle=False)

            # Initialize PyTorch Deep Model & Optimizer
            model = LSTMWithAttention(input_dim=len(feature_names), hidden_dim=64, num_layers=2).to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = AsymmetricFinancialLoss(lambda_sharpe=1.5, lambda_bce=0.5, lambda_mse=1.0)

            # Training Loop
            for epoch in range(epochs):
                _ = self._train_epoch(model, train_loader, optimizer, criterion)
                scheduler.step()

            # Evaluation on Test Fold
            model.eval()
            all_preds, all_probs, all_targets = [], [], []

            with torch.no_grad():
                for X_b, y_b, _ in test_loader:
                    X_b = X_b.to(self.device)
                    pred_ret, pred_prob = model(X_b)
                    all_preds.extend(pred_ret.cpu().numpy().flatten())
                    all_probs.extend(pred_prob.cpu().numpy().flatten())
                    all_targets.extend(y_b.numpy().flatten())

            preds_arr = np.array(all_preds)
            targets_arr = np.array(all_targets)

            # Calculate Active Directional Accuracy (where probability > 0.52 or < 0.48)
            signals = np.zeros_like(preds_arr)
            signals[preds_arr > 0] = 1.0
            signals[preds_arr < 0] = -1.0

            active_mask = np.abs(preds_arr) > np.percentile(np.abs(preds_arr), 50)
            if np.sum(active_mask) > 0:
                directional_acc = float(np.mean(np.sign(signals[active_mask]) == np.sign(targets_arr[active_mask])))
            else:
                directional_acc = 0.50

            strategy_returns = signals * targets_arr
            mean_ret = np.mean(strategy_returns)
            std_ret = np.std(strategy_returns) + 1e-8
            sharpe_ratio = float((mean_ret / std_ret) * np.sqrt(365 * 24))

            fold_results.append({
                "fold": fold,
                "train_size": len(train_subset),
                "test_size": len(test_subset),
                "directional_accuracy": round(directional_acc, 4),
                "mae": round(float(np.mean(np.abs(targets_arr - preds_arr))), 6),
                "sharpe_ratio": round(sharpe_ratio, 2)
            })

        df_metrics = pd.DataFrame(fold_results)

        summary_metrics = {
            "mean_directional_accuracy": float(df_metrics["directional_accuracy"].mean()),
            "mean_mae": float(df_metrics["mae"].mean()),
            "mean_sharpe_ratio": float(df_metrics["sharpe_ratio"].mean()),
        }

        # Train Final Model on Full Dataset
        full_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        final_model = LSTMWithAttention(input_dim=len(feature_names), hidden_dim=64, num_layers=2).to(self.device)
        optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-4)

        for _ in range(epochs):
            self._train_epoch(final_model, full_loader, optimizer, criterion)

        clean_symbol = symbol.replace("/", "").upper()
        artifact_path = os.path.join(self.artifacts_dir, f"pytorch_attention_lstm_{clean_symbol}_{timeframe}.pt")

        torch.save({
            "model_state_dict": final_model.state_dict(),
            "feature_names": feature_names,
            "seq_len": self.seq_len,
            "symbol": symbol,
            "timeframe": timeframe,
            "summary_metrics": summary_metrics
        }, artifact_path)

        logger.info(f"Saved PyTorch Stage 2 artifact to: {artifact_path}")

        return summary_metrics, df_metrics, artifact_path
