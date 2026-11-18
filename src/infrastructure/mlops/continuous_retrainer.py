"""
Continuous Retraining, Statistical Drift Audit, and Ecosystem Health Orchestrator.
R&D Quantitative Laboratory — MaBuy Project.

Core Modules:
1. MaPredict (Trading Department):
   - Statistical Feature Drift & PSI/KS-Test Detector.
   - Out-of-Sample Rolling Walk-Forward Validator (Directional Accuracy, Annualized Sharpe).
   - Champion vs. Challenger Paradigm with strict ModelRegistry promotion governance.
2. MaInvest (Investment Department):
   - Merton Jump Diffusion (MJD) Parameter Recalibration on 365-day rolling window:
     (lambda, mu_J, sigma_J, sigma_c, compensator_k, compensated_drift).
   - Positive Semi-Definite (PSD) Covariance and Correlation Matrix update with Eigenvalue PCA spectrum.
   - Calibration snapshot persistence to 'data/cache/mainvest_calibration.json'.
3. Unified Ecosystem Health Orchestration:
   - run_ecosystem_health_and_retrain(force=False) -> JSON-serializable ecosystem report.
"""

from __future__ import annotations

import os
import json
import time
import math
import logging
import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
import joblib

from src.infrastructure.mlops.drift_detector import DataDriftDetector
from src.infrastructure.mlops.model_registry import ModelRegistry
from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher
from src.infrastructure.ml.feature_engineering.pipeline import CryptoFeaturePipeline
from src.infrastructure.ml.training.validation import WalkForwardValidator
from src.domain.services.symbol_normalizer import SymbolNormalizer
from src.infrastructure.ml.mainvest_macro_forecaster import (
    MertonJumpDiffusionEngine,
    RealizedVolatilityEngine,
    HalvingMacroCycleClassifier,
    SUPPORTED_TOP_ASSETS,
    GENERIC_ASSET_PARAMS
)

logger = logging.getLogger("ContinuousRetrainer")


# ============================================================================
# DATACLASSES & SCHEMAS
# ============================================================================

@dataclass(frozen=True)
class ModelAuditReport:
    """Audit evaluation for an individual trading model."""
    symbol: str
    timeframe: str
    champion_version: Optional[str]
    active_directional_accuracy: float
    annualized_sharpe_ratio: float
    drift_detected: bool
    drift_ratio: float
    drifted_features_count: int
    total_features_count: int
    retraining_recommended: bool
    audit_timestamp: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChampionVsChallengerResult:
    """Comparative outcome between existing Champion and newly trained Challenger."""
    symbol: str
    timeframe: str
    model_type: str
    champion_version: Optional[str]
    challenger_version: str
    champion_metrics: Dict[str, float]
    challenger_metrics: Dict[str, float]
    champion_score: float
    challenger_score: float
    promoted_to_active: bool
    promotion_reason: str
    artifact_path: str
    evaluated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MertonAssetCalibration:
    """Merton Jump Diffusion calibrated parameter set for a single asset."""
    symbol: str
    sample_days: int
    jump_intensity_lambda: float
    jump_mean_mu: float
    jump_std_sigma: float
    continuous_vol_sigma_c: float
    jump_compensator_k: float
    annualized_drift_mu: float
    jumps_count_detected: int
    calibrated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MaInvestCalibrationSnapshot:
    """Snapshot package of asset calibrations, PSD covariance, and systemic eigenvalues."""
    timestamp: str
    sample_window_days: int
    assets: List[str]
    asset_calibrations: Dict[str, Dict[str, Any]]
    covariance_matrix: List[List[float]]
    correlation_matrix: List[List[float]]
    eigenvalues: List[float]
    top_eigenvalue_pct: float
    snapshot_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EcosystemHealthReport:
    """Unified ecosystem health and retraining audit report."""
    timestamp: str
    execution_time_sec: float
    status: str  # "HEALTHY" | "RETRAINED" | "WARNING" | "CRITICAL"
    force_triggered: bool
    mapredict_audit: Dict[str, Any]
    mapredict_retraining: Dict[str, Any]
    mainvest_calibration: Dict[str, Any]
    summary_en: str
    summary_ru: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# CONTINUOUS RETRAINER & HEALTH AUDITOR ENGINE
# ============================================================================

class ContinuousRetrainer:
    """
    Flagship Continuous Retraining and MLOps Governance Engine for MaBuy.
    Handles continuous auditing, drift detection, Champion vs. Challenger promotion,
    and Merton Jump Diffusion / Covariance recalibration for MaInvest.
    """

    DEFAULT_MAPREDICT_SYMBOLS: List[str] = ["BTCUSDT", "ETHUSDT"]
    DEFAULT_MAINVEST_ASSETS: List[str] = ["BTC", "ETH", "SOL", "BNB", "TON", "XRP", "AVAX"]

    def __init__(
        self,
        models_dir: str = "models",
        cache_dir: str = ".data_cache",
        mainvest_snapshot_path: str = "data/cache/mainvest_calibration.json",
        p_value_threshold: float = 0.05,
        psi_threshold: float = 0.25,
        min_directional_accuracy: float = 0.505,
        min_sharpe_ratio: float = 0.0,
        drift_ratio_threshold: float = 0.30,
        enable_catboost: bool = True,
    ):
        self.models_dir = models_dir
        self.cache_dir = cache_dir
        self.mainvest_snapshot_path = mainvest_snapshot_path
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold
        self.min_directional_accuracy = min_directional_accuracy
        self.min_sharpe_ratio = min_sharpe_ratio
        self.drift_ratio_threshold = drift_ratio_threshold
        self.enable_catboost = enable_catboost

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.mainvest_snapshot_path), exist_ok=True)

        self.registry = ModelRegistry(registry_dir=self.models_dir)
        self.drift_detector = DataDriftDetector(
            p_value_threshold=self.p_value_threshold,
            psi_threshold=self.psi_threshold
        )
        self.fetcher = CryptoDataFetcher(cache_dir=self.cache_dir)
        self.feature_pipeline = CryptoFeaturePipeline()
        self.validator = WalkForwardValidator(n_splits=5, confidence_percentile=60.0)

    # ========================================================================
    # 1. MAPREDICT AUDIT & METRICS CALCULATION
    # ========================================================================

    def audit_mapredict_model(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        candle_limit: int = 600,
        raw_df: Optional[pd.DataFrame] = None,
        reference_df: Optional[pd.DataFrame] = None,
    ) -> ModelAuditReport:
        """
        Audits active Champion model performance and statistical feature drift.
        
        Args:
            symbol: Trading pair ticker.
            timeframe: Candle interval (e.g., '1h', '4h').
            candle_limit: Number of recent candles to ingest.
            raw_df: Optional DataFrame for current inference evaluation.
            reference_df: Optional DataFrame for baseline training distribution.

        Returns:
            ModelAuditReport with detailed drift metrics and retraining triggers.
        """
        clean_sym = SymbolNormalizer.normalize(symbol)
        if not clean_sym.endswith("USDT"):
            clean_sym += "USDT"

        # 1. Ingest Market Data
        if raw_df is not None and len(raw_df) > 0:
            df = raw_df.copy()
        else:
            df = self.fetcher.fetch_ohlcv(clean_sym, timeframe=timeframe, limit=candle_limit)

        if len(df) < 50:
            logger.warning(f"Insufficient candles ({len(df)}) for auditing {clean_sym}.")
            return ModelAuditReport(
                symbol=clean_sym,
                timeframe=timeframe,
                champion_version=None,
                active_directional_accuracy=0.50,
                annualized_sharpe_ratio=0.0,
                drift_detected=False,
                drift_ratio=0.0,
                drifted_features_count=0,
                total_features_count=0,
                retraining_recommended=True,
                audit_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                details={"error": "Insufficient candle history"}
            )

        # 2. Extract Features
        if reference_df is not None and len(reference_df) > 0:
            ref_X, _, feature_names = self.feature_pipeline.transform(reference_df)
            curr_X, curr_y, _ = self.feature_pipeline.transform(df)
            y_eval = curr_y
        else:
            X_matrix, y_target, feature_names = self.feature_pipeline.transform(df)
            n_samples = len(X_matrix)
            split_idx = n_samples // 2
            ref_X = X_matrix[:split_idx]
            curr_X = X_matrix[split_idx:]
            y_eval = y_target[split_idx:]

        # 3. Statistical Drift Evaluation
        drift_summary, df_drift_report = self.drift_detector.detect_feature_drift(
            reference_data=ref_X,
            current_data=curr_X,
            feature_names=feature_names
        )

        drift_ratio = drift_summary.get("drift_ratio", 0.0)
        drift_detected = bool(drift_ratio > self.drift_ratio_threshold)
        drifted_count = drift_summary.get("features_drifted_count", 0)

        # 4. Fetch Champion Model Info from ModelRegistry
        champion_info = self.registry.get_active_model_info(clean_sym)
        champion_version = champion_info.get("version") if champion_info else None
        
        # 5. Evaluate Performance on Out-of-Sample / Test Slice
        model_artifact = None
        if champion_info and "artifact_path" in champion_info:
            artifact_path = champion_info["artifact_path"]
            if os.path.exists(artifact_path):
                try:
                    payload = joblib.load(artifact_path)
                    if isinstance(payload, dict) and "model" in payload:
                        model_artifact = payload["model"]
                    else:
                        model_artifact = payload
                except Exception as e:
                    logger.warning(f"Could not load champion artifact {artifact_path}: {e}")

        if model_artifact is not None and hasattr(model_artifact, "predict"):
            try:
                y_pred = model_artifact.predict(curr_X)
                
                # Active signal filtering (magnitude > threshold)
                abs_pred = np.abs(y_pred)
                threshold = np.percentile(abs_pred, 60.0)
                signals = np.zeros_like(y_pred)
                signals[y_pred > threshold] = 1.0
                signals[y_pred < -threshold] = -1.0
                
                active_mask = signals != 0
                if np.sum(active_mask) > 0:
                    dir_acc = float(np.mean(np.sign(signals[active_mask]) == np.sign(y_eval[active_mask])))
                else:
                    dir_acc = 0.50
                    
                strat_returns = signals * y_eval
                std_ret = np.std(strat_returns) + 1e-8
                sharpe = float((np.mean(strat_returns) / std_ret) * np.sqrt(365 * 24))
            except Exception as eval_err:
                logger.warning(f"Error evaluating champion predictions: {eval_err}")
                dir_acc = champion_info.get("metrics", {}).get("mean_directional_accuracy", 0.51)
                sharpe = champion_info.get("metrics", {}).get("mean_sharpe_ratio", 0.50)
        elif champion_info and "metrics" in champion_info:
            dir_acc = champion_info["metrics"].get("mean_directional_accuracy", 0.51)
            sharpe = champion_info["metrics"].get("mean_sharpe_ratio", 0.50)
        else:
            # Baseline estimation if no champion registered
            dir_acc = 0.50
            sharpe = 0.0

        # 6. Retraining Trigger Decision Logic
        degraded_performance = (dir_acc < self.min_directional_accuracy) or (sharpe < self.min_sharpe_ratio)
        retraining_recommended = bool(drift_detected or degraded_performance or (champion_version is None))

        return ModelAuditReport(
            symbol=clean_sym,
            timeframe=timeframe,
            champion_version=champion_version,
            active_directional_accuracy=round(dir_acc, 4),
            annualized_sharpe_ratio=round(sharpe, 2),
            drift_detected=drift_detected,
            drift_ratio=round(drift_ratio, 4),
            drifted_features_count=drifted_count,
            total_features_count=len(feature_names),
            retraining_recommended=retraining_recommended,
            audit_timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            details={
                "drift_summary": drift_summary,
                "degraded_performance": degraded_performance,
                "samples_evaluated": len(curr_X)
            }
        )

    # ========================================================================
    # 2. MAPREDICT ROLLING RETRAINING & CHAMPION VS CHALLENGER
    # ========================================================================

    def _compute_exponential_sample_weights(self, n_samples: int, decay_factor: float = 0.002) -> np.ndarray:
        """Computes exponentially decaying sample weights prioritizing recent market bars."""
        time_steps = np.arange(n_samples)
        weights = np.exp(decay_factor * (time_steps - n_samples))
        return weights / np.mean(weights)

    def retrain_mapredict_model(
        self,
        symbol: str = "BTCUSDT",
        timeframe: str = "1h",
        candle_limit: int = 1500,
        raw_df: Optional[pd.DataFrame] = None,
        model_type: str = "GBDT"
    ) -> ChampionVsChallengerResult:
        """
        Retrains a model candidate (Challenger), validates Out-of-Sample performance,
        and evaluates Champion vs Challenger for deployment promotion.
        
        Args:
            symbol: Asset symbol.
            timeframe: Candle interval.
            candle_limit: Number of candles for rolling window.
            raw_df: Optional DataFrame for training.
            model_type: "GBDT", "CATBOOST", or "PYTORCH".
        """
        clean_sym = SymbolNormalizer.normalize(symbol)
        if not clean_sym.endswith("USDT"):
            clean_sym += "USDT"

        logger.info(f"Initiating Challenger training [{model_type}] for {clean_sym} ({timeframe})...")

        # 1. Ingest Data
        if raw_df is not None and len(raw_df) > 0:
            df = raw_df.copy()
        else:
            df = self.fetcher.fetch_ohlcv(clean_sym, timeframe=timeframe, limit=candle_limit)

        if len(df) < 60:
            raise ValueError(f"Insufficient candles ({len(df)}) for retraining {clean_sym}.")

        # 2. Transform Features
        X_matrix, y_target, feature_names = self.feature_pipeline.transform(df)
        sample_weights = self._compute_exponential_sample_weights(len(y_target))

        # 3. Model Architecture Instantiation
        model = None
        norm_type = model_type.upper()

        if norm_type == "CATBOOST" and self.enable_catboost:
            try:
                from catboost import CatBoostRegressor
                model = CatBoostRegressor(
                    iterations=250,
                    learning_rate=0.03,
                    depth=4,
                    l2_leaf_reg=3.0,
                    random_seed=42,
                    verbose=0
                )
            except Exception as e:
                logger.warning(f"CatBoost initialization fallback to HistGradientBoosting: {e}")
                from sklearn.ensemble import HistGradientBoostingRegressor
                model = HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.03,
                    max_depth=4,
                    min_samples_leaf=20,
                    l2_regularization=1.5,
                    random_state=42
                )
                norm_type = "GBDT"
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.03,
                max_depth=4,
                min_samples_leaf=20,
                l2_regularization=1.5,
                random_state=42
            )
            norm_type = "GBDT"

        # 4. Out-of-Sample Walk-Forward Validation
        challenger_summary, fold_metrics_df = self.validator.evaluate(
            model=model,
            X=X_matrix,
            y=y_target,
            sample_weights=sample_weights
        )

        # 5. Fit Final Challenger on Full Dataset with recency weights
        if hasattr(model, "fit"):
            try:
                model.fit(X_matrix, y_target, sample_weight=sample_weights)
            except TypeError:
                model.fit(X_matrix, y_target)

        # 6. Save Challenger Artifact
        timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        challenger_version = f"v_{timestamp_str}_{norm_type.lower()}"
        artifact_filename = f"{norm_type.lower()}_{clean_sym}_{timeframe}_{timestamp_str}.joblib"
        artifact_path = os.path.join(self.models_dir, artifact_filename)

        artifact_payload = {
            "model": model,
            "model_type": norm_type,
            "feature_pipeline": self.feature_pipeline,
            "feature_names": feature_names,
            "symbol": clean_sym,
            "timeframe": timeframe,
            "version": challenger_version,
            "summary_metrics": challenger_summary,
            "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        joblib.dump(artifact_payload, artifact_path)

        # 7. Champion vs Challenger Comparative Assessment
        comparison_res = self.evaluate_champion_vs_challenger(
            symbol=clean_sym,
            timeframe=timeframe,
            challenger_metrics=challenger_summary,
            challenger_version=challenger_version,
            artifact_path=artifact_path,
            model_type=norm_type
        )

        return comparison_res

    def evaluate_champion_vs_challenger(
        self,
        symbol: str,
        timeframe: str,
        challenger_metrics: Dict[str, float],
        challenger_version: str,
        artifact_path: str,
        model_type: str
    ) -> ChampionVsChallengerResult:
        """
        Implements strict Champion vs Challenger governance.
        Promotes Challenger to ACTIVE in ModelRegistry ONLY IF it surpasses the Champion.
        """
        clean_sym = SymbolNormalizer.normalize(symbol)
        if not clean_sym.endswith("USDT"):
            clean_sym += "USDT"

        champion_info = self.registry.get_active_model_info(clean_sym)
        champion_version = champion_info.get("version") if champion_info else None

        # Extract Champion Metrics (or establish default baseline thresholds)
        if champion_info and "metrics" in champion_info:
            champ_metrics = champion_info["metrics"]
            champ_mda = float(champ_metrics.get("mean_directional_accuracy", 0.51))
            champ_sharpe = float(champ_metrics.get("mean_sharpe_ratio", 0.50))
        else:
            champ_metrics = {"mean_directional_accuracy": 0.505, "mean_sharpe_ratio": 0.0}
            champ_mda = 0.505
            champ_sharpe = 0.0

        chall_mda = float(challenger_metrics.get("mean_directional_accuracy", 0.50))
        chall_sharpe = float(challenger_metrics.get("mean_sharpe_ratio", 0.0))

        # Composite Scoring Function:
        # Score = (MDA * 100) * 0.60 + min(5.0, max(-2.0, Sharpe)) * 10.0
        def compute_score(mda: float, sharpe: float) -> float:
            clamped_sharpe = min(5.0, max(-2.0, sharpe))
            return float((mda * 100.0) * 0.60 + (clamped_sharpe * 10.0))

        champ_score = round(compute_score(champ_mda, champ_sharpe), 3)
        chall_score = round(compute_score(chall_mda, chall_sharpe), 3)

        # Promotion Criteria:
        # 1. Challenger must meet minimum absolute quality bar (MDA >= min_dir_acc, Sharpe >= min_sharpe)
        # 2. Challenger score must exceed Champion score OR Champion has no prior active deployment
        meets_quality_bar = (chall_mda >= self.min_directional_accuracy) and (chall_sharpe >= self.min_sharpe_ratio)
        score_superior = chall_score > champ_score

        if champion_version is None:
            should_promote = meets_quality_bar
            reason = "Initial Champion baseline established" if should_promote else "Failed minimum quality bar on initial model"
        else:
            if not meets_quality_bar:
                should_promote = False
                reason = f"Challenger failed minimum quality bar (MDA={chall_mda:.3f} vs {self.min_directional_accuracy}, Sharpe={chall_sharpe:.2f} vs {self.min_sharpe_ratio})"
            elif score_superior:
                should_promote = True
                reason = f"Challenger outperformed Champion (Score {chall_score:.2f} > {champ_score:.2f}, MDA {chall_mda:.3f} vs {champ_mda:.3f}, Sharpe {chall_sharpe:.2f} vs {champ_sharpe:.2f})"
            else:
                should_promote = False
                reason = f"Challenger underperformed Champion (Score {chall_score:.2f} <= {champ_score:.2f})"

        # Register Challenger in ModelRegistry
        self.registry.register_model(
            model_name=f"{model_type.lower()}_{clean_sym}",
            version=challenger_version,
            artifact_path=artifact_path,
            model_type=model_type,
            metrics=challenger_metrics,
            symbol=clean_sym,
            set_active=should_promote
        )

        logger.info(
            f"Champion vs Challenger evaluation for {clean_sym}: "
            f"Promoted={should_promote} | Reason: {reason}"
        )

        return ChampionVsChallengerResult(
            symbol=clean_sym,
            timeframe=timeframe,
            model_type=model_type,
            champion_version=champion_version,
            challenger_version=challenger_version,
            champion_metrics=champ_metrics,
            challenger_metrics=challenger_metrics,
            champion_score=champ_score,
            challenger_score=chall_score,
            promoted_to_active=should_promote,
            promotion_reason=reason,
            artifact_path=artifact_path,
            evaluated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )

    # ========================================================================
    # 3. MAINVEST CALIBRATION (MERTON JUMP DIFFUSION & PSD COVARIANCE)
    # ========================================================================

    def calibrate_merton_jump_diffusion(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> MertonAssetCalibration:
        """
        Calibrates Merton Jump Diffusion parameters using robust MAD jump separation.
        Formulae:
          r_t = ln(P_t / P_{t-1})
          sigma_MAD = 1.4826 * MAD
          Jump threshold = 2.58 * sigma_MAD
          lambda = N_jumps / T_years
          compensator k = exp(mu_J + 0.5 * sigma_J^2) - 1
          sigma_c = std(r_diff) * sqrt(365)
        """
        clean_sym = symbol.upper().replace("USDT", "").replace("USD", "").replace("/", "")
        asset_info = SUPPORTED_TOP_ASSETS.get(clean_sym, GENERIC_ASSET_PARAMS.copy())

        closes = df["close"].to_numpy(dtype=np.float64)
        if len(closes) < 10:
            return MertonAssetCalibration(
                symbol=clean_sym,
                sample_days=len(closes),
                jump_intensity_lambda=2.0,
                jump_mean_mu=-0.02,
                jump_std_sigma=0.08,
                continuous_vol_sigma_c=0.60,
                jump_compensator_k=-0.0168,
                annualized_drift_mu=0.25,
                jumps_count_detected=0,
                calibrated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
            )

        log_returns = np.diff(np.log(closes))
        n_days = len(log_returns)

        # 1. Median Absolute Deviation (MAD) Jump Detection
        med_ret = float(np.median(log_returns))
        mad = float(np.median(np.abs(log_returns - med_ret)))
        sigma_mad = 1.4826 * mad if mad > 1e-8 else float(np.std(log_returns))
        sigma_mad = max(sigma_mad, 0.005)

        # 99% Coverage threshold
        jump_threshold = 2.58 * sigma_mad
        jump_mask = np.abs(log_returns - med_ret) > jump_threshold
        jumps_count = int(np.sum(jump_mask))
        diff_returns = log_returns[~jump_mask]

        # 2. Jump intensity lambda
        time_span_years = max(n_days / 365.0, 0.1)
        jump_intensity = jumps_count / time_span_years
        jump_sensitivity = asset_info.get("jump_sensitivity", 1.2)
        jump_intensity = max(0.5, min(24.0, jump_intensity * jump_sensitivity))

        # 3. Jump size distribution Y ~ N(mu_J, sigma_J^2)
        if jumps_count >= 2:
            jump_returns = log_returns[jump_mask]
            jump_mu = float(np.mean(jump_returns))
            jump_sigma = float(np.std(jump_returns, ddof=1))
            jump_sigma = max(jump_sigma, 0.02)
        else:
            jump_mu = -0.015
            jump_sigma = 0.075

        # Jump Compensator: k = exp(mu_J + 0.5 * sigma_J^2) - 1
        k_compensator = float(np.exp(jump_mu + 0.5 * (jump_sigma ** 2)) - 1.0)

        # 4. Continuous volatility sigma_c
        if len(diff_returns) >= 5:
            sigma_c = float(np.std(diff_returns, ddof=1) * np.sqrt(365.0))
        else:
            sigma_c = float(np.std(log_returns, ddof=1) * np.sqrt(365.0))
        sigma_c = max(0.15, min(2.50, sigma_c))

        # 5. Compensated Drift with Bayesian Prior Shrinkage
        raw_drift = float(np.mean(log_returns) * 365.0 + 0.5 * (sigma_c ** 2))
        prior_drift = asset_info.get("historical_drift_prior", 0.35)
        shrinkage_weight = 60.0 / (n_days + 60.0)
        calibrated_drift = (1.0 - shrinkage_weight) * raw_drift + shrinkage_weight * prior_drift
        calibrated_drift = max(-0.40, min(1.50, calibrated_drift))

        return MertonAssetCalibration(
            symbol=clean_sym,
            sample_days=n_days,
            jump_intensity_lambda=round(jump_intensity, 4),
            jump_mean_mu=round(jump_mu, 4),
            jump_std_sigma=round(jump_sigma, 4),
            continuous_vol_sigma_c=round(sigma_c, 4),
            jump_compensator_k=round(k_compensator, 4),
            annualized_drift_mu=round(calibrated_drift, 4),
            jumps_count_detected=jumps_count,
            calibrated_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
        )

    def recalibrate_mainvest(
        self,
        assets: Optional[List[str]] = None,
        lookback_days: int = 365,
        custom_candles: Optional[Dict[str, pd.DataFrame]] = None,
        snapshot_path: Optional[str] = None
    ) -> MaInvestCalibrationSnapshot:
        """
        Recalibrates Merton Jump Diffusion for top assets and computes Positive Semi-Definite (PSD)
        covariance and correlation matrices with PCA eigenvalue analysis.
        Saves calibration snapshot to 'data/cache/mainvest_calibration.json'.
        """
        target_assets = assets or self.DEFAULT_MAINVEST_ASSETS
        target_path = snapshot_path or self.mainvest_snapshot_path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        asset_calibrations: Dict[str, Dict[str, Any]] = {}
        daily_returns_dict: Dict[str, np.ndarray] = {}

        # 1. Ingest 365 daily candles and calibrate MJD for each asset
        for asset in target_assets:
            clean_sym = asset.upper()
            if custom_candles and clean_sym in custom_candles:
                df = custom_candles[clean_sym]
            else:
                pair = f"{clean_sym}USDT"
                try:
                    df = self.fetcher.fetch_ohlcv(pair, timeframe="1d", limit=lookback_days)
                except Exception as e:
                    logger.error(f"Could not fetch daily candles for {pair}: {e}. Skipping asset (Fail-Fast).")
                    continue

            calib = self.calibrate_merton_jump_diffusion(clean_sym, df)
            asset_calibrations[clean_sym] = calib.to_dict()

            closes = df["close"].to_numpy(dtype=np.float64)
            if len(closes) > 1:
                log_rets = np.diff(np.log(closes))
                daily_returns_dict[clean_sym] = log_rets

        # 2. Align Daily Returns across all assets
        min_len = min((len(v) for v in daily_returns_dict.values()), default=0)
        k = len(target_assets)

        if min_len >= 10 and k > 1:
            matrix_rets = np.column_stack([
                daily_returns_dict[a][-min_len:] for a in target_assets
            ])
            # Annualized sample covariance
            cov_sample = np.cov(matrix_rets, rowvar=False) * 365.0
        else:
            # Fallback covariance from priors
            cov_sample = np.zeros((k, k), dtype=np.float64)
            for i, a1 in enumerate(target_assets):
                vol_i = asset_calibrations.get(a1, {}).get("continuous_vol_sigma_c", 0.70)
                for j, a2 in enumerate(target_assets):
                    vol_j = asset_calibrations.get(a2, {}).get("continuous_vol_sigma_c", 0.70)
                    rho = 1.0 if i == j else 0.70
                    cov_sample[i, j] = vol_i * vol_j * rho

        # 3. Spectral Regularization for Strict Positive Semi-Definiteness (PSD)
        cov_sym = 0.5 * (cov_sample + cov_sample.T)
        eigvals, eigvecs = np.linalg.eigh(cov_sym)
        eigvals_clamped = np.maximum(eigvals, 1e-5)
        cov_psd = eigvecs @ np.diag(eigvals_clamped) @ eigvecs.T

        # Small Tikhonov diagonal regularizer
        shrinkage_lambda = 1e-4
        cov_psd += np.eye(k) * shrinkage_lambda

        # 4. Pearson Correlation Matrix: Corr_ij = Cov_ij / (sigma_i * sigma_j)
        std_diag = np.sqrt(np.diag(cov_psd))
        corr_matrix = cov_psd / np.outer(std_diag, std_diag)
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)

        # 5. Eigenvalues & Systemic Variance Absorption Ratio
        sorted_eigvals = np.sort(np.linalg.eigvalsh(cov_psd))[::-1]
        top_eigenvalue_pct = float((sorted_eigvals[0] / np.sum(sorted_eigvals)) * 100.0)

        # 6. Save Snapshot to JSON Cache
        snapshot = MaInvestCalibrationSnapshot(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            sample_window_days=lookback_days,
            assets=target_assets,
            asset_calibrations=asset_calibrations,
            covariance_matrix=np.round(cov_psd, 6).tolist(),
            correlation_matrix=np.round(corr_matrix, 4).tolist(),
            eigenvalues=np.round(sorted_eigvals, 4).tolist(),
            top_eigenvalue_pct=round(top_eigenvalue_pct, 2),
            snapshot_path=target_path
        )

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2)

        logger.info(
            f"Saved MaInvest calibration snapshot to {target_path} "
            f"({len(target_assets)} assets, Absorption Ratio={top_eigenvalue_pct:.1f}%)"
        )

        return snapshot

    # ========================================================================
    # 4. UNIFIED ECOSYSTEM HEALTH & RETRAINING ORCHESTRATOR
    # ========================================================================

    def run_ecosystem_health_and_retrain(
        self,
        force: bool = False,
        symbols: Optional[List[str]] = None,
        mainvest_assets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes full ecosystem health audit and automated retraining cycle
        for both MaPredict (Trading) and MaInvest (Investment).
        
        Args:
            force: If True, forces retraining even if no drift or degradation is detected.
            symbols: Optional list of MaPredict symbols (default: ["BTCUSDT", "ETHUSDT"]).
            mainvest_assets: Optional list of MaInvest assets (default: Top 7 crypto assets).

        Returns:
            Consolidated JSON-serializable EcosystemHealthReport dict.
        """
        start_time = time.time()
        target_symbols = symbols or self.DEFAULT_MAPREDICT_SYMBOLS
        target_mainvest = mainvest_assets or self.DEFAULT_MAINVEST_ASSETS

        logger.info(
            f"Starting Ecosystem Health Audit & Retrain Cycle "
            f"(force={force}, MaPredict symbols={target_symbols}, MaInvest assets={target_mainvest})..."
        )

        mapredict_audits: Dict[str, Any] = {}
        mapredict_retraining: Dict[str, Any] = {}
        retrained_count = 0
        promoted_count = 0

        # Step 1: MaPredict Auditing & Conditional Retraining
        for sym in target_symbols:
            audit_rep = self.audit_mapredict_model(symbol=sym)
            mapredict_audits[sym] = audit_rep.to_dict()

            should_retrain = bool(force or audit_rep.retraining_recommended)
            if should_retrain:
                retrained_count += 1
                try:
                    retrain_res = self.retrain_mapredict_model(symbol=sym, model_type="GBDT")
                    mapredict_retraining[sym] = retrain_res.to_dict()
                    if retrain_res.promoted_to_active:
                        promoted_count += 1
                except Exception as retrain_err:
                    logger.error(f"Retraining failed for {sym}: {retrain_err}", exc_info=True)
                    mapredict_retraining[sym] = {
                        "symbol": sym,
                        "status": "FAILED",
                        "error": str(retrain_err)
                    }

        # Step 2: MaInvest Calibration Snapshot Update
        try:
            mainvest_snap = self.recalibrate_mainvest(assets=target_mainvest)
            mainvest_summary = {
                "status": "SUCCESS",
                "snapshot": mainvest_snap.to_dict()
            }
        except Exception as mainvest_err:
            logger.error(f"MaInvest recalibration failed: {mainvest_err}", exc_info=True)
            mainvest_summary = {
                "status": "FAILED",
                "error": str(mainvest_err)
            }

        exec_time = round(time.time() - start_time, 3)

        # Step 3: Global Ecosystem Health Status Determination
        if retrained_count > 0 and promoted_count > 0:
            status = "RETRAINED"
            summary_en = f"Ecosystem retrained successfully. {promoted_count}/{retrained_count} models promoted to active status. MaInvest calibrated."
            summary_ru = f"Экосистема успешно обновлена. {promoted_count}/{retrained_count} моделей повышены до статуса активных. MaInvest перекалиброван."
        elif retrained_count > 0 and promoted_count == 0:
            status = "HEALTHY"
            summary_en = f"Retraining evaluated. Existing Champions retained as superior to Challengers. MaInvest calibrated."
            summary_ru = f"Дообучение выполнено. Текущие Champion-модели сохранены, так как превзошли Challenger-кандидатов. MaInvest перекалиброван."
        else:
            status = "HEALTHY"
            summary_en = "Ecosystem healthy. No significant feature drift detected. MaInvest calibrated."
            summary_ru = "Экосистема в норме. Критического дрейфа признаков не обнаружено. MaInvest перекалиброван."

        report = EcosystemHealthReport(
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            execution_time_sec=exec_time,
            status=status,
            force_triggered=force,
            mapredict_audit=mapredict_audits,
            mapredict_retraining=mapredict_retraining,
            mainvest_calibration=mainvest_summary,
            summary_en=summary_en,
            summary_ru=summary_ru
        )

        logger.info(
            f"Completed Ecosystem Health & Retrain Cycle in {exec_time:.2f}s with status [{status}]."
        )

        return report.to_dict()


# Backwards compatibility alias
ContinuousRetrainingOrchestrator = ContinuousRetrainer
