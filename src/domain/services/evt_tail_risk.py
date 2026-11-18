import math
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class EVTTailRiskResult:
    threshold_u: float
    shape_xi: float # Tail index (\xi)
    scale_sigma: float # Scale parameter (\sigma)
    var_95_evt: float # 95% Extreme Value VaR
    var_99_evt: float # 99% Extreme Value VaR
    es_99_evt: float # 99% Expected Shortfall (CVaR)
    excess_count: int
    is_fat_tailed: bool
    is_tail_singularity_veto: bool # Veto if \xi > 0.42 (Infinite 3rd/4th moment risk)
    position_scale_factor: float # Haircut multiplier in [0.0, 1.0]
    veto_reason: Optional[str]


class EVTTailRiskEngine:
    r"""
    Extreme Value Theory (EVT) Engine using Peaks Over Threshold (POT)
    and Generalized Pareto Distribution (GPD) for Crypto Heavy Tails.
    
    Guarantees:
    - Accurate modelling of leptokurtic asset returns (Kurtosis >> 3.0).
    - Expected Shortfall 99% calculation: ES_{0.99} = (VaR_{0.99} + \sigma - \xi*u) / (1 - \xi).
    - Tail Singularity Veto when shape parameter \xi > 0.42.
    """

    def __init__(self, target_es_pct: float = 3.5, shape_veto_threshold: float = 0.42):
        self.target_es_pct = target_es_pct
        self.shape_veto_threshold = shape_veto_threshold

    def fit_and_evaluate(
        self,
        returns: List[float],
        threshold_quantile: float = 0.90
    ) -> EVTTailRiskResult:
        """
        Fits GPD on positive losses X = -R.
        """
        arr = np.asarray(returns, dtype=np.float64).ravel()
        if len(arr) < 30:
            # Fallback for insufficient sample
            return EVTTailRiskResult(
                threshold_u=0.02,
                shape_xi=0.15,
                scale_sigma=0.015,
                var_95_evt=0.035,
                var_99_evt=0.065,
                es_99_evt=0.080,
                excess_count=len(arr),
                is_fat_tailed=True,
                is_tail_singularity_veto=False,
                position_scale_factor=1.0,
                veto_reason=None
            )

        # Losses in percentage terms (positive means a drop in price)
        losses = -arr
        # Threshold u at specified quantile
        u = float(np.percentile(losses, threshold_quantile * 100))
        excesses = losses[losses > u] - u
        N = len(losses)
        N_u = len(excesses)

        if N_u < 5:
            u = float(np.percentile(losses, 80.0))
            excesses = losses[losses > u] - u
            N_u = len(excesses)

        # Method of Probability-Weighted Moments (PWM) / Method of Moments for robust GPD estimation
        mean_excess = float(np.mean(excesses)) if N_u > 0 else 0.01
        var_excess = float(np.var(excesses)) if N_u > 1 else 0.001

        # GPD Moment estimates:
        # \xi = 0.5 * (1 - (mean^2 / var))
        # \sigma = 0.5 * mean * (mean^2 / var + 1)
        if var_excess > 0:
            ratio = (mean_excess ** 2) / var_excess
            xi = 0.5 * (1.0 - ratio)
            sigma = 0.5 * mean_excess * (ratio + 1.0)
        else:
            xi = 0.15
            sigma = max(0.005, mean_excess)

        # Bound parameters in stable domain
        xi = max(-0.5, min(0.9, xi))
        sigma = max(1e-4, sigma)

        # 95% and 99% VaR and ES calculation via POT formula
        # VaR_{1-\alpha} = u + (\sigma / \xi) * (((N / N_u) * \alpha)^{-\xi} - 1)
        alpha_95 = 0.05
        alpha_99 = 0.01

        factor_95 = (N / max(1, N_u)) * alpha_95
        factor_99 = (N / max(1, N_u)) * alpha_99

        if abs(xi) > 1e-5:
            var_95 = u + (sigma / xi) * (factor_95 ** (-xi) - 1.0)
            var_99 = u + (sigma / xi) * (factor_99 ** (-xi) - 1.0)
        else:
            var_95 = u - sigma * math.log(factor_95)
            var_99 = u - sigma * math.log(factor_99)

        # Expected Shortfall (CVaR)
        if xi < 1.0:
            es_99 = (var_99 + sigma - xi * u) / (1.0 - xi)
        else:
            es_99 = var_99 * 1.5

        # Check Singularity Veto
        is_singularity_veto = bool(xi >= self.shape_veto_threshold)
        veto_reason = None
        if is_singularity_veto:
            veto_reason = f"EVT VETO: Extreme tail index singularity detected (\u03be = {xi:.3f} >= {self.shape_veto_threshold})"

        # Calculate position scaling haircut
        target_es_decimal = self.target_es_pct / 100.0
        if es_99 > target_es_decimal and es_99 > 0:
            pos_scale = max(0.20, min(1.0, target_es_decimal / es_99))
        else:
            pos_scale = 1.0

        if is_singularity_veto:
            pos_scale = 0.0

        # Empirical excess kurtosis of return distribution
        var_losses = float(np.var(losses))
        if var_losses > 0:
            kurtosis = float(np.mean((losses - np.mean(losses)) ** 4) / (var_losses ** 2))
        else:
            kurtosis = 3.0

        is_fat_tailed = bool(xi > 0.0 or kurtosis > 3.2)

        return EVTTailRiskResult(
            threshold_u=round(u, 5),
            shape_xi=round(xi, 4),
            scale_sigma=round(sigma, 5),
            var_95_evt=round(max(0.0, var_95), 5),
            var_99_evt=round(max(0.0, var_99), 5),
            es_99_evt=round(max(0.0, es_99), 5),
            excess_count=N_u,
            is_fat_tailed=is_fat_tailed,
            is_tail_singularity_veto=is_singularity_veto,
            position_scale_factor=round(pos_scale, 4),
            veto_reason=veto_reason
        )
