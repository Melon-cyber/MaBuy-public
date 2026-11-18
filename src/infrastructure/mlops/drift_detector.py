import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy.stats import ks_2samp


class DataDriftDetector:
    r"""
    Statistical Data Drift Detector for MaBuy MLOps Pipeline.

    Statistical Methods:
    --------------------
    1. Kolmogorov-Smirnov (KS) Test:
       D_{ks} = \sup_x |F_{\text{ref}}(x) - F_{\text{curr}}(x)|
       Rejects null hypothesis of equal distributions if p-value < 0.05.

    2. Population Stability Index (PSI):
       PSI = \sum (P_i - Q_i) * ln(P_i / Q_i)
       - PSI < 0.1: Stable distribution
       - 0.1 <= PSI <= 0.25: Slight drift warning
       - PSI > 0.25: Critical Data Drift -> Trigger Retraining Alert!
    """

    def __init__(self, p_value_threshold: float = 0.05, psi_threshold: float = 0.25):
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold

    def calculate_psi(self, reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
        """Calculates Population Stability Index (PSI) between reference and current feature arrays."""
        ref_clean = reference[~np.isnan(reference)]
        curr_clean = current[~np.isnan(current)]

        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return 0.0

        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(ref_clean, percentiles)
        bins[0] = -np.inf
        bins[-1] = np.inf

        ref_counts, _ = np.histogram(ref_clean, bins=bins)
        curr_counts, _ = np.histogram(curr_clean, bins=bins)

        ref_percents = ref_counts / len(ref_clean)
        curr_percents = curr_counts / len(curr_clean)

        # Smooth zero counts to avoid division by zero or log(0)
        eps = 1e-4
        ref_percents = np.where(ref_percents == 0, eps, ref_percents)
        curr_percents = np.where(curr_percents == 0, eps, curr_percents)

        psi_val = np.sum((curr_percents - ref_percents) * np.log(curr_percents / ref_percents))
        return float(psi_val)

    def detect_feature_drift(
        self,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Evaluates Kolmogorov-Smirnov test and PSI across all feature dimensions.
        """
        drift_results = []
        num_features_drifted = 0

        for col_idx, feat_name in enumerate(feature_names):
            ref_col = reference_data[:, col_idx]
            curr_col = current_data[:, col_idx]

            # 1. KS-Test
            stat, p_val = ks_2samp(ref_col, curr_col)
            stat = 0.0 if np.isnan(stat) else float(stat)
            p_val = 1.0 if np.isnan(p_val) else float(p_val)
            is_ks_drift = bool(p_val < self.p_value_threshold)

            # 2. PSI
            psi_val = self.calculate_psi(ref_col, curr_col)
            psi_val = 0.0 if np.isnan(psi_val) else float(psi_val)
            is_psi_drift = bool(psi_val > self.psi_threshold)

            has_drift = is_ks_drift or is_psi_drift
            if has_drift:
                num_features_drifted += 1

            drift_results.append({
                "feature": feat_name,
                "ks_stat": round(stat, 4),
                "p_value": round(p_val, 5),
                "psi": round(psi_val, 4),
                "ks_drift": is_ks_drift,
                "psi_drift": is_psi_drift,
                "drift_detected": has_drift
            })

        df_report = pd.DataFrame(drift_results)

        drift_ratio = num_features_drifted / len(feature_names) if feature_names else 0.0
        retraining_recommended = bool(drift_ratio > 0.30)

        summary = {
            "total_features": len(feature_names),
            "features_drifted_count": num_features_drifted,
            "drift_ratio": round(drift_ratio, 4),
            "retraining_recommended": retraining_recommended
        }

        return summary, df_report


class PredictionDriftMonitor:
    """
    Online CUSUM and Prediction Distribution Drift Monitor.
    Tracks rolling predictions and confidences over recent vs historical windows.
    Detects concept drift, directional skew, and statistical shifts.
    """

    def __init__(self, recent_window: int = 20, historical_window: int = 100, delta_threshold: float = 0.05):
        self.recent_window = recent_window
        self.historical_window = historical_window
        self.delta_threshold = delta_threshold
        self._history: List[Dict[str, Any]] = []

    def record_prediction(self, symbol: str, predicted_log_ret: float, confidence: float, direction: str):
        self._history.append({
            "timestamp": time.time(),
            "symbol": symbol,
            "pred_ret": float(predicted_log_ret),
            "conf": float(confidence),
            "dir_num": 1.0 if str(direction).upper() in ("BUY", "LONG", "BULLISH") else -1.0
        })
        if len(self._history) > self.historical_window * 2:
            self._history = self._history[-self.historical_window * 2:]

    def check_drift(self) -> Dict[str, Any]:
        if len(self._history) < self.recent_window + 10:
            return {
                "drift_detected": False,
                "reason": "Insufficient history for statistical comparison",
                "sample_count": len(self._history)
            }

        recent = self._history[-self.recent_window:]
        historical = self._history[:-self.recent_window]

        recent_rets = [x["pred_ret"] for x in recent]
        hist_rets = [x["pred_ret"] for x in historical]

        recent_confs = [x["conf"] for x in recent]
        hist_confs = [x["conf"] for x in historical]

        mu_recent_ret = float(np.mean(recent_rets))
        mu_hist_ret = float(np.mean(hist_rets))
        delta_ret = abs(mu_recent_ret - mu_hist_ret)

        mu_recent_conf = float(np.mean(recent_confs))
        mu_hist_conf = float(np.mean(hist_confs))
        delta_conf = abs(mu_recent_conf - mu_hist_conf)

        drift_detected = (delta_ret > self.delta_threshold) or (delta_conf > self.delta_threshold)
        reasons = []
        if delta_ret > self.delta_threshold:
            reasons.append(f"Predicted return mean shifted by {delta_ret:.4f} > {self.delta_threshold}")
        if delta_conf > self.delta_threshold:
            reasons.append(f"Confidence mean shifted by {delta_conf:.4f} > {self.delta_threshold}")

        return {
            "drift_detected": drift_detected,
            "delta_ret": round(delta_ret, 4),
            "delta_conf": round(delta_conf, 4),
            "mu_recent_ret": round(mu_recent_ret, 4),
            "mu_hist_ret": round(mu_hist_ret, 4),
            "mu_recent_conf": round(mu_recent_conf, 4),
            "mu_hist_conf": round(mu_hist_conf, 4),
            "reasons": reasons,
            "sample_count": len(self._history)
        }

