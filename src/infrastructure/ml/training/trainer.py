import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import HistGradientBoostingRegressor

from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher
from src.infrastructure.ml.feature_engineering.pipeline import CryptoFeaturePipeline
from src.infrastructure.ml.training.validation import WalkForwardValidator

logger = logging.getLogger(__name__)


class BaselineModelTrainer:
    """
    Enhanced Model Trainer utilizing Volatility Normalization, Exponential Sample Recency Weighting,
    and Signal Thresholding.
    """

    def __init__(self, artifacts_dir: str = "models"):
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.fetcher = CryptoDataFetcher()
        self.feature_pipeline = CryptoFeaturePipeline()
        self.validator = WalkForwardValidator(n_splits=5, confidence_percentile=60.0)

    def _compute_recency_weights(self, n_samples: int, decay_factor: float = 0.002) -> np.ndarray:
        """Computes exponential sample weights prioritizing recent market observations."""
        time_steps = np.arange(n_samples)
        weights = np.exp(decay_factor * (time_steps - n_samples))
        return weights / np.mean(weights)

    def train_and_evaluate(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        candle_limit: int = 1500
    ) -> Tuple[Dict[str, float], pd.DataFrame, str]:
        logger.info(f"Starting Enhanced Baseline Training for {symbol} ({timeframe})...")

        # 1. Ingest Market Data
        raw_df = self.fetcher.fetch_ohlcv(symbol=symbol, timeframe=timeframe, limit=candle_limit)

        # 2. Extract Stationary Features & Target Matrix
        X, y, feature_names = self.feature_pipeline.transform(raw_df)

        # 3. Compute Exponential Recency Weights
        sample_weights = self._compute_recency_weights(len(y))

        # 4. Initialize Regularized GBDT Model
        model = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.03,
            max_depth=4,
            min_samples_leaf=20,
            l2_regularization=1.5,
            random_state=42
        )

        # 5. Perform Walk-Forward Cross-Validation with Active Trade Thresholding
        summary_metrics, fold_df = self.validator.evaluate(model, X, y, sample_weights=sample_weights)

        # 6. Fit Final Model on Full Dataset
        model.fit(X, y, sample_weight=sample_weights)

        # 7. Serialize Artifacts to Disk
        clean_symbol = symbol.replace("/", "").upper()
        artifact_path = os.path.join(self.artifacts_dir, f"baseline_gbdt_{clean_symbol}_{timeframe}.joblib")

        artifact_payload = {
            "model": model,
            "feature_pipeline": self.feature_pipeline,
            "feature_names": feature_names,
            "symbol": symbol,
            "timeframe": timeframe,
            "summary_metrics": summary_metrics,
        }
        joblib.dump(artifact_payload, artifact_path)
        logger.info(f"Saved enhanced baseline model artifact to: {artifact_path}")

        return summary_metrics, fold_df, artifact_path
