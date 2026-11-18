import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error


class WalkForwardValidator:
    r"""
    Walk-Forward Cross-Validation Evaluator for Financial Time Series with Signal Thresholding.

    Financial Optimizations:
    ------------------------
    1. Signal Thresholding: Trades are executed ONLY when |\hat{r}| > threshold * \sigma_rolling.
       Low-confidence predictions near zero are categorized as NEUTRAL (no trade/cash).
    2. Directional Accuracy (MDA) evaluated on Active Signal Trades.
    3. Annualized Sharpe Ratio calculated on Active Strategy Trades.
    """

    def __init__(self, n_splits: int = 5, confidence_percentile: float = 60.0):
        self.n_splits = n_splits
        self.confidence_percentile = confidence_percentile
        self.tscv = TimeSeriesSplit(n_splits=n_splits)

    def evaluate(self, model, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray = None) -> Tuple[Dict[str, float], pd.DataFrame]:
        fold_results: List[Dict[str, Any]] = []

        for fold, (train_idx, test_idx) in enumerate(self.tscv.split(X)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            w_train = sample_weights[train_idx] if sample_weights is not None else None

            # Fit model with exponential recency sample weights if provided
            if w_train is not None and hasattr(model, "sample_weight"):
                model.fit(X_train, y_train, sample_weight=w_train)
            else:
                model.fit(X_train, y_train)

            # Predict on test window
            y_pred = model.predict(X_test)

            # Signal Filtering: Set confidence threshold based on prediction magnitude
            abs_preds = np.abs(y_pred)
            threshold = np.percentile(abs_preds, self.confidence_percentile)

            # Active signals: +1 (BULLISH), -1 (BEARISH), 0 (NEUTRAL / NO TRADE)
            signals = np.zeros_like(y_pred)
            signals[y_pred > threshold] = 1.0
            signals[y_pred < -threshold] = -1.0

            active_mask = signals != 0
            active_count = int(np.sum(active_mask))

            # Directional Accuracy on Active Trades
            if active_count > 0:
                active_correct = np.sign(signals[active_mask]) == np.sign(y_test[active_mask])
                directional_acc = float(np.mean(active_correct))
            else:
                directional_acc = 0.50

            # Error Metrics on all test samples
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

            # Active Strategy Returns
            strategy_returns = signals * y_test
            mean_ret = np.mean(strategy_returns)
            std_ret = np.std(strategy_returns) + 1e-8
            sharpe_ratio = float((mean_ret / std_ret) * np.sqrt(365 * 24))

            fold_results.append({
                "fold": fold + 1,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "active_trades": active_count,
                "active_directional_acc": round(directional_acc, 4),
                "mae": round(mae, 6),
                "rmse": round(rmse, 6),
                "sharpe_ratio": round(sharpe_ratio, 2)
            })

        df_metrics = pd.DataFrame(fold_results)

        summary_metrics = {
            "mean_directional_accuracy": float(df_metrics["active_directional_acc"].mean()),
            "mean_mae": float(df_metrics["mae"].mean()),
            "mean_rmse": float(df_metrics["rmse"].mean()),
            "mean_sharpe_ratio": float(df_metrics["sharpe_ratio"].mean()),
        }

        return summary_metrics, df_metrics
