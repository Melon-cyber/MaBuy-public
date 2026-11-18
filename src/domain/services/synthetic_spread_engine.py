# src/domain/services/synthetic_spread_engine.py
"""
Synthetic SOL/BTC market-neutral statistical arbitrage engine (Astra Step 3).

Core model:
    y_t = alpha_t + beta_t * x_t + residual_t

where:
    y_t = log(SOL mid price)
    x_t = log(BTC mid price)

alpha_t and beta_t are estimated online with an adaptive 2D Kalman Filter:
    theta_t = [alpha_t, beta_t]^T
    theta_t = theta_{t-1} + w_t,  w_t ~ N(0, Q * dt)
    y_t = H_t * theta_t + v_t,    v_t ~ N(0, R)

The prior Kalman residual (S_t^{prior} = y_t - (alpha_{t|t-1} + beta_{t|t-1} * x_t))
is modeled as an Ornstein-Uhlenbeck (OU) mean-reverting process through an exponentially
weighted AR(1) regression:
    residual_t = c + phi * residual_{t-1} + epsilon_t

No external heavy dependencies (pure Python standard library).
O(1) memory and O(1) CPU per quote update.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


EPS = 1e-15
BPS = 10_000.0


class SpreadAction(str, Enum):
    HOLD = "HOLD"
    ENTER_LONG_SPREAD = "ENTER_LONG_SPREAD"
    ENTER_SHORT_SPREAD = "ENTER_SHORT_SPREAD"
    EXIT_MEAN_REVERSION = "EXIT_MEAN_REVERSION"
    EXIT_ZERO_CROSS = "EXIT_ZERO_CROSS"
    EXIT_STRUCTURAL_STOP = "EXIT_STRUCTURAL_STOP"


class SpreadPosition(str, Enum):
    FLAT = "FLAT"
    LONG_SPREAD = "LONG_SPREAD"
    SHORT_SPREAD = "SHORT_SPREAD"


@dataclass(frozen=True)
class SyntheticSpreadConfig:
    # -------------------------------------------------------------------------
    # Kalman filter parameters
    # Q is specified per second because state covariance is scaled by dt.
    # -------------------------------------------------------------------------

    # alpha random-walk variance per second (~1% per day)
    q_alpha_per_second: float = 1.20e-9

    # beta random-walk variance per second (~2% per day)
    q_beta_per_second: float = 4.70e-9

    # Measurement noise variance in log-price space (sqrt(R) = 3 bps => 0.0003)
    measurement_r: float = 9.0e-8

    # Initial Kalman state uncertainty
    initial_alpha_variance: float = 0.05 ** 2
    initial_beta_variance: float = 0.50 ** 2
    initial_beta: float = 1.0

    # Do not trust a quote if data is too old
    max_quote_age_seconds: float = 3.0

    # Reject extremely long update gaps; prevents huge Q*dt jumps
    max_kalman_dt_seconds: float = 10.0

    # -------------------------------------------------------------------------
    # OU estimation parameters
    # -------------------------------------------------------------------------

    # OU regression is sampled at this cadence, rather than every market tick
    ou_sample_interval_seconds: float = 5.0

    # Allowed timing deviation around ou_sample_interval_seconds
    ou_sample_tolerance_ratio: float = 0.50

    # Exponentially weighted half-life of OU estimation history (30 minutes)
    ou_estimation_half_life_seconds: float = 1800.0

    # Minimum effective EW samples before enabling trading (~80 samples * 5s = 400s)
    min_ou_effective_samples: float = 80.0

    # Numerical ridge for AR(1) normal-equation matrix
    ou_regression_ridge: float = 1.0e-12

    # -------------------------------------------------------------------------
    # Trade rules & institutional gates
    # -------------------------------------------------------------------------

    entry_z: float = 2.0
    exit_z: float = 0.3
    catastrophic_z: float = 3.8

    min_half_life_seconds: float = 15.0 * 60.0   # 15 minutes minimum
    max_half_life_seconds: float = 6.0 * 3600.0  # 6 hours maximum

    # beta must be positive in the intended SOL long / BTC short hedge model
    min_abs_beta: float = 0.05
    max_abs_beta: float = 5.00

    # -------------------------------------------------------------------------
    # Estimated all-in execution costs, one-way, in bps
    # -------------------------------------------------------------------------

    sol_one_way_fee_bps: float = 4.0
    btc_one_way_fee_bps: float = 4.0
    sol_one_way_slippage_bps: float = 1.5
    btc_one_way_slippage_bps: float = 1.0

    # Required positive net edge beyond estimated trading cost
    edge_safety_buffer_bps: float = 3.0

    # In shadow mode no orders are sent. State transitions are still tracked.
    shadow_mode: bool = True


@dataclass(frozen=True)
class SpreadTelemetry:
    timestamp: float
    sol_mid: float
    btc_mid: float

    alpha: Optional[float]
    beta: Optional[float]
    residual: Optional[float]

    ou_mu: Optional[float]
    sigma_eq: Optional[float]
    kappa: Optional[float]
    half_life_seconds: Optional[float]
    zscore: Optional[float]

    position: SpreadPosition
    action: SpreadAction

    expected_gross_edge_bps: Optional[float]
    estimated_round_trip_cost_bps: Optional[float]
    expected_net_edge_bps: Optional[float]

    tradable: bool
    status: str
    reason: str

    effective_samples: float = 0.0
    min_samples: float = 80.0

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "event": "synthetic_spread_telemetry",
            "timestamp": self.timestamp,
            "sol_mid": self.sol_mid,
            "btc_mid": self.btc_mid,
            "alpha": round(self.alpha, 6) if self.alpha is not None else None,
            "beta": round(self.beta, 4) if self.beta is not None else None,
            "residual": round(self.residual, 6) if self.residual is not None else None,
            "ou_mu": round(self.ou_mu, 6) if self.ou_mu is not None else None,
            "sigma_eq": round(self.sigma_eq, 6) if self.sigma_eq is not None else None,
            "kappa": round(self.kappa, 6) if self.kappa is not None else None,
            "half_life_seconds": round(self.half_life_seconds, 1) if self.half_life_seconds is not None else None,
            "zscore": round(self.zscore, 2) if self.zscore is not None else None,
            "position": self.position.value,
            "action": self.action.value,
            "expected_gross_edge_bps": round(self.expected_gross_edge_bps, 1) if self.expected_gross_edge_bps is not None else None,
            "estimated_round_trip_cost_bps": round(self.estimated_round_trip_cost_bps, 1) if self.estimated_round_trip_cost_bps is not None else None,
            "expected_net_edge_bps": round(self.expected_net_edge_bps, 1) if self.expected_net_edge_bps is not None else None,
            "tradable": self.tradable,
            "status": self.status,
            "reason": self.reason,
            "effective_samples": round(self.effective_samples, 1),
            "min_samples": round(self.min_samples, 1),
        }

    def format_russian_summary(self) -> str:
        """Human-readable Russian summary for daemon console and logs."""
        beta_str = f"{self.beta:.2f}" if self.beta is not None else "1.00"

        if self.status == "WARMUP" or self.zscore is None:
            pct = min(100.0, (self.effective_samples / max(1.0, self.min_samples)) * 100.0)
            return (
                f"Z=N/A | Beta={beta_str} | Half-Life=N/A | "
                f"Прогрев: {self.effective_samples:.0f}/{self.min_samples:.0f} выборок ({pct:.0f}%) | "
                f"Статус: ПРОГРЕВ МОДЕЛИ [ЖДАТЬ]"
            )

        z_str = f"{self.zscore:+.2f}"
        hl_str = f"{self.half_life_seconds / 60.0:.0f}м" if self.half_life_seconds is not None else "N/A"

        status_ru_map = {
            "READY": "НЕЙТРАЛЬНО (НАБЛЮДЕНИЕ)",
            "ENTRY": "ТОЧКА ВХОДА (СИГНАЛ)",
            "IN_POSITION": "УДЕРЖАНИЕ СПРЕДА",
            "EXIT": "ЗАКРЫТИЕ (РЕВЕРСИЯ К СРЕДНЕМУ)",
            "BLOCKED": "БЛОКИРОВКА (ФИЛЬТР)",
        }
        status_ru = status_ru_map.get(self.status, self.status)

        action_ru_map = {
            SpreadAction.HOLD: "ЖДАТЬ",
            SpreadAction.ENTER_LONG_SPREAD: "ВХОД LONG SPREAD (Купить SOL / Продать BTC)",
            SpreadAction.ENTER_SHORT_SPREAD: "ВХОД SHORT SPREAD (Продать SOL / Купить BTC)",
            SpreadAction.EXIT_MEAN_REVERSION: "ВЫХОД (Возврат к среднему Z <= 0.3)",
            SpreadAction.EXIT_ZERO_CROSS: "ВЫХОД (Пересечение нуля Z)",
            SpreadAction.EXIT_STRUCTURAL_STOP: "АВАРИЙНЫЙ СТОП (Слом коинтеграции |Z| >= 3.8)",
        }
        action_ru = action_ru_map.get(self.action, self.action.value)

        net_edge_str = f"{self.expected_net_edge_bps:+.1f} bps" if self.expected_net_edge_bps is not None else "N/A"

        return (
            f"Z={z_str} | Beta={beta_str} | Half-Life={hl_str} | "
            f"Чистый эдж: {net_edge_str} | Статус: {status_ru} [{action_ru}]"
        )


class SyntheticSpreadEngine:
    """
    Stateful O(1) SOL/BTC synthetic spread statistical arbitrage engine.

    Usage:
        engine = SyntheticSpreadEngine()

        telemetry = engine.update(
            sol_mid=145.12,
            btc_mid=68420.10,
            timestamp=time.time(),
        )

        if telemetry.action != SpreadAction.HOLD:
            signal_log.write(telemetry.to_log_dict())
    """

    def __init__(self, config: Optional[SyntheticSpreadConfig] = None) -> None:
        self.cfg = config or SyntheticSpreadConfig()

        # Kalman state theta = [alpha, beta]
        self._initialized = False
        self._alpha = 0.0
        self._beta = self.cfg.initial_beta

        # Symmetric 2x2 covariance matrix:
        # P = [[p00, p01], [p01, p11]]
        self._p00 = self.cfg.initial_alpha_variance
        self._p01 = 0.0
        self._p11 = self.cfg.initial_beta_variance

        self._last_kalman_ts: Optional[float] = None

        # OU sampler state
        self._last_ou_ts: Optional[float] = None
        self._prev_ou_residual: Optional[float] = None

        # EW normal equations for AR(1):
        # residual_t = c + phi * residual_{t-1} + epsilon_t
        self._s00 = 0.0
        self._s01 = 0.0
        self._s11 = 0.0
        self._sy0 = 0.0
        self._sy1 = 0.0

        # EW residual innovation variance
        self._err_sq_sum = 0.0
        self._err_weight = 0.0

        # Last valid OU values
        self._ou_mu: Optional[float] = None
        self._sigma_eq: Optional[float] = None
        self._kappa: Optional[float] = None
        self._half_life_seconds: Optional[float] = None
        self._phi: Optional[float] = None

        # Position state is virtual in shadow/paper mode
        self._position = SpreadPosition.FLAT
        self._entry_z: Optional[float] = None
        self._last_z: Optional[float] = None

    @property
    def position(self) -> SpreadPosition:
        return self._position

    @property
    def beta(self) -> float:
        return self._beta

    @property
    def alpha(self) -> float:
        return self._alpha

    def reset(self) -> None:
        """Reset all model and shadow-position state."""
        self.__init__(self.cfg)

    def update(
        self,
        sol_mid: float,
        btc_mid: float,
        timestamp: Optional[float] = None,
        sol_quote_timestamp: Optional[float] = None,
        btc_quote_timestamp: Optional[float] = None,
    ) -> SpreadTelemetry:
        """
        Process a synchronized SOL and BTC midpoint pair.

        Parameters
        ----------
        sol_mid, btc_mid:
            Best midpoint prices: mid = (best_bid + best_ask) / 2.0
        timestamp:
            Event time in UNIX seconds.
        sol_quote_timestamp, btc_quote_timestamp:
            Optional source quote timestamps. If supplied, stale quotes are
            rejected before model update.
        """
        now = time.time() if timestamp is None else float(timestamp)

        if not self._valid_price(sol_mid) or not self._valid_price(btc_mid):
            return self._telemetry(
                timestamp=now,
                sol_mid=sol_mid,
                btc_mid=btc_mid,
                action=SpreadAction.HOLD,
                tradable=False,
                status="BLOCKED",
                reason="invalid_non_positive_price",
            )

        stale_reason = self._quote_staleness_reason(
            now=now,
            sol_quote_timestamp=sol_quote_timestamp,
            btc_quote_timestamp=btc_quote_timestamp,
        )
        if stale_reason is not None:
            return self._telemetry(
                timestamp=now,
                sol_mid=sol_mid,
                btc_mid=btc_mid,
                action=SpreadAction.HOLD,
                tradable=False,
                status="BLOCKED",
                reason=stale_reason,
            )

        y = math.log(sol_mid)
        x = math.log(btc_mid)

        if not self._initialized:
            # Initial alpha selected such that first residual is zero: y - alpha - beta*x = 0
            self._beta = self.cfg.initial_beta
            self._alpha = y - self._beta * x
            self._initialized = True
            self._last_kalman_ts = now

            return self._telemetry(
                timestamp=now,
                sol_mid=sol_mid,
                btc_mid=btc_mid,
                action=SpreadAction.HOLD,
                tradable=False,
                status="WARMUP",
                reason="kalman_initialized",
            )

        dt = self._safe_kalman_dt(now)

        # ---------------------------------------------------------------------
        # 1. Kalman prediction
        # ---------------------------------------------------------------------
        p00 = self._p00 + self.cfg.q_alpha_per_second * dt
        p01 = self._p01
        p11 = self._p11 + self.cfg.q_beta_per_second * dt

        alpha_prior = self._alpha
        beta_prior = self._beta

        # Observation matrix: H = [1, x]
        predicted_y = alpha_prior + beta_prior * x
        residual_prior = y - predicted_y

        # Innovation variance: S = H P H' + R
        innovation_variance = (
            p00
            + 2.0 * x * p01
            + x * x * p11
            + self.cfg.measurement_r
        )

        if innovation_variance <= EPS or not math.isfinite(innovation_variance):
            return self._telemetry(
                timestamp=now,
                sol_mid=sol_mid,
                btc_mid=btc_mid,
                action=SpreadAction.HOLD,
                tradable=False,
                status="BLOCKED",
                reason="invalid_kalman_innovation_variance",
                residual=residual_prior,
            )

        # ---------------------------------------------------------------------
        # 2. Kalman update
        # ---------------------------------------------------------------------
        # A = P H'
        a0 = p00 + p01 * x
        a1 = p01 + p11 * x

        k0 = a0 / innovation_variance
        k1 = a1 / innovation_variance

        self._alpha = alpha_prior + k0 * residual_prior
        self._beta = beta_prior + k1 * residual_prior

        # Numerically stable symmetric covariance update:
        # P_new = P - (P H' H P) / S
        self._p00 = max(EPS, p00 - (a0 * a0) / innovation_variance)
        self._p01 = p01 - (a0 * a1) / innovation_variance
        self._p11 = max(EPS, p11 - (a1 * a1) / innovation_variance)

        self._last_kalman_ts = now

        # ---------------------------------------------------------------------
        # 3. Update OU estimator from prior residual
        # ---------------------------------------------------------------------
        self._maybe_update_ou(
            residual=residual_prior,
            timestamp=now,
        )

        # ---------------------------------------------------------------------
        # 4. Build Z-score and apply trading gates
        # ---------------------------------------------------------------------
        zscore = self._calculate_zscore(residual_prior)
        ou_reason = self._ou_validity_reason()

        if zscore is None:
            self._last_z = None
            return self._telemetry(
                timestamp=now,
                sol_mid=sol_mid,
                btc_mid=btc_mid,
                action=SpreadAction.HOLD,
                tradable=False,
                status="WARMUP",
                reason=ou_reason or "ou_not_ready",
                residual=residual_prior,
            )

        beta = self._beta
        cost_bps = self._round_trip_cost_bps(beta)
        gross_edge_bps = self._expected_gross_edge_bps(zscore, beta)
        net_edge_bps = gross_edge_bps - cost_bps - self.cfg.edge_safety_buffer_bps

        # Catastrophic stop: cointegration breakdown
        if (
            self._position != SpreadPosition.FLAT
            and abs(zscore) >= self.cfg.catastrophic_z
        ):
            self._position = SpreadPosition.FLAT
            self._entry_z = None
            self._last_z = zscore

            return self._telemetry(
                timestamp=now,
                sol_mid=sol_mid,
                btc_mid=btc_mid,
                action=SpreadAction.EXIT_STRUCTURAL_STOP,
                tradable=False,
                status="EXIT",
                reason="structural_breakdown_zscore",
                residual=residual_prior,
                zscore=zscore,
                gross_edge_bps=gross_edge_bps,
                cost_bps=cost_bps,
                net_edge_bps=net_edge_bps,
            )

        # Standard exit logic for an active position
        if self._position != SpreadPosition.FLAT:
            crossed_zero = (
                self._last_z is not None
                and self._last_z * zscore <= 0.0
            )

            if abs(zscore) <= self.cfg.exit_z:
                self._position = SpreadPosition.FLAT
                self._entry_z = None
                self._last_z = zscore

                return self._telemetry(
                    timestamp=now,
                    sol_mid=sol_mid,
                    btc_mid=btc_mid,
                    action=SpreadAction.EXIT_MEAN_REVERSION,
                    tradable=True,
                    status="EXIT",
                    reason="mean_reversion_exit_band",
                    residual=residual_prior,
                    zscore=zscore,
                    gross_edge_bps=gross_edge_bps,
                    cost_bps=cost_bps,
                    net_edge_bps=net_edge_bps,
                )

            if crossed_zero:
                self._position = SpreadPosition.FLAT
                self._entry_z = None
                self._last_z = zscore

                return self._telemetry(
                    timestamp=now,
                    sol_mid=sol_mid,
                    btc_mid=btc_mid,
                    action=SpreadAction.EXIT_ZERO_CROSS,
                    tradable=True,
                    status="EXIT",
                    reason="zscore_zero_cross",
                    residual=residual_prior,
                    zscore=zscore,
                    gross_edge_bps=gross_edge_bps,
                    cost_bps=cost_bps,
                    net_edge_bps=net_edge_bps,
                )

            self._last_z = zscore
            return self._telemetry(
                timestamp=now,
                sol_mid=sol_mid,
                btc_mid=btc_mid,
                action=SpreadAction.HOLD,
                tradable=True,
                status="IN_POSITION",
                reason="position_open",
                residual=residual_prior,
                zscore=zscore,
                gross_edge_bps=gross_edge_bps,
                cost_bps=cost_bps,
                net_edge_bps=net_edge_bps,
            )

        # Flat state: validate model before opening a new shadow trade
        if ou_reason is not None:
            self._last_z = zscore
            return self._telemetry(
                timestamp=now,
                sol_mid=sol_mid,
                btc_mid=btc_mid,
                action=SpreadAction.HOLD,
                tradable=False,
                status="BLOCKED",
                reason=ou_reason,
                residual=residual_prior,
                zscore=zscore,
                gross_edge_bps=gross_edge_bps,
                cost_bps=cost_bps,
                net_edge_bps=net_edge_bps,
            )

        if net_edge_bps < 0.0:
            self._last_z = zscore
            return self._telemetry(
                timestamp=now,
                sol_mid=sol_mid,
                btc_mid=btc_mid,
                action=SpreadAction.HOLD,
                tradable=False,
                status="BLOCKED",
                reason="insufficient_edge_after_fees_and_slippage",
                residual=residual_prior,
                zscore=zscore,
                gross_edge_bps=gross_edge_bps,
                cost_bps=cost_bps,
                net_edge_bps=net_edge_bps,
            )

        if zscore <= -self.cfg.entry_z:
            self._position = SpreadPosition.LONG_SPREAD
            self._entry_z = zscore
            self._last_z = zscore

            return self._telemetry(
                timestamp=now,
                sol_mid=sol_mid,
                btc_mid=btc_mid,
                action=SpreadAction.ENTER_LONG_SPREAD,
                tradable=True,
                status="ENTRY",
                reason="zscore_below_negative_entry_threshold",
                residual=residual_prior,
                zscore=zscore,
                gross_edge_bps=gross_edge_bps,
                cost_bps=cost_bps,
                net_edge_bps=net_edge_bps,
            )

        if zscore >= self.cfg.entry_z:
            self._position = SpreadPosition.SHORT_SPREAD
            self._entry_z = zscore
            self._last_z = zscore

            return self._telemetry(
                timestamp=now,
                sol_mid=sol_mid,
                btc_mid=btc_mid,
                action=SpreadAction.ENTER_SHORT_SPREAD,
                tradable=True,
                status="ENTRY",
                reason="zscore_above_positive_entry_threshold",
                residual=residual_prior,
                zscore=zscore,
                gross_edge_bps=gross_edge_bps,
                cost_bps=cost_bps,
                net_edge_bps=net_edge_bps,
            )

        self._last_z = zscore
        return self._telemetry(
            timestamp=now,
            sol_mid=sol_mid,
            btc_mid=btc_mid,
            action=SpreadAction.HOLD,
            tradable=True,
            status="READY",
            reason="no_entry_signal",
            residual=residual_prior,
            zscore=zscore,
            gross_edge_bps=gross_edge_bps,
            cost_bps=cost_bps,
            net_edge_bps=net_edge_bps,
        )

    # -------------------------------------------------------------------------
    # OU estimation
    # -------------------------------------------------------------------------

    def _maybe_update_ou(self, residual: float, timestamp: float) -> None:
        if self._last_ou_ts is None:
            self._last_ou_ts = timestamp
            self._prev_ou_residual = residual
            return

        dt = timestamp - self._last_ou_ts
        target_dt = self.cfg.ou_sample_interval_seconds
        tolerance = target_dt * self.cfg.ou_sample_tolerance_ratio

        # We only regress samples close to selected sampling interval.
        # If a large data gap happened, discard the pair to avoid distortion.
        if dt < target_dt - tolerance:
            return

        if abs(dt - target_dt) > tolerance:
            self._last_ou_ts = timestamp
            self._prev_ou_residual = residual
            return

        previous = self._prev_ou_residual
        if previous is None:
            self._last_ou_ts = timestamp
            self._prev_ou_residual = residual
            return

        # EW decay factor: lambda = exp(-ln(2) * dt / EW_half_life)
        decay = math.exp(
            -math.log(2.0)
            * dt
            / max(self.cfg.ou_estimation_half_life_seconds, EPS)
        )

        # Update weighted normal equations for:
        # current = c + phi * previous + error
        self._s00 = decay * self._s00 + 1.0
        self._s01 = decay * self._s01 + previous
        self._s11 = decay * self._s11 + previous * previous
        self._sy0 = decay * self._sy0 + residual
        self._sy1 = decay * self._sy1 + previous * residual

        det = (
            self._s00 * self._s11
            - self._s01 * self._s01
            + self.cfg.ou_regression_ridge
        )

        if det > EPS:
            c = (
                self._sy0 * self._s11
                - self._s01 * self._sy1
            ) / det

            phi = (
                self._s00 * self._sy1
                - self._s01 * self._sy0
            ) / det

            # Innovation estimated using current EW AR(1) coefficients
            error = residual - c - phi * previous
            self._err_sq_sum = decay * self._err_sq_sum + error * error
            self._err_weight = decay * self._err_weight + 1.0

            if self._err_weight > EPS and 0.0 < phi < 1.0:
                innovation_variance = max(
                    EPS,
                    self._err_sq_sum / self._err_weight,
                )

                sigma_eps = math.sqrt(innovation_variance)
                kappa = -math.log(phi) / dt
                sigma_eq = sigma_eps / math.sqrt(max(EPS, 1.0 - phi * phi))
                mu = c / max(EPS, 1.0 - phi)
                half_life = math.log(2.0) / max(EPS, kappa)

                if (
                    self._s00 >= self.cfg.min_ou_effective_samples
                    and math.isfinite(mu)
                    and math.isfinite(sigma_eq)
                    and math.isfinite(kappa)
                    and math.isfinite(half_life)
                    and sigma_eq > EPS
                    and kappa > EPS
                ):
                    self._phi = phi
                    self._ou_mu = mu
                    self._sigma_eq = sigma_eq
                    self._kappa = kappa
                    self._half_life_seconds = half_life

        self._last_ou_ts = timestamp
        self._prev_ou_residual = residual

    # -------------------------------------------------------------------------
    # Trading and validation
    # -------------------------------------------------------------------------

    def _calculate_zscore(self, residual: float) -> Optional[float]:
        if (
            self._s00 < self.cfg.min_ou_effective_samples
            or self._ou_mu is None
            or self._sigma_eq is None
            or self._sigma_eq <= EPS
        ):
            return None

        z = (residual - self._ou_mu) / self._sigma_eq
        return z if math.isfinite(z) else None

    def _ou_validity_reason(self) -> Optional[str]:
        if self._s00 < self.cfg.min_ou_effective_samples:
            return "ou_warmup_insufficient_effective_samples"

        if self._phi is None or not (0.0 < self._phi < 1.0):
            return "ou_phi_not_in_stationary_range"

        if self._kappa is None or self._kappa <= 0.0:
            return "ou_kappa_not_positive"

        if self._sigma_eq is None or self._sigma_eq <= EPS:
            return "ou_sigma_eq_invalid"

        if self._half_life_seconds is None:
            return "ou_half_life_unavailable"

        if self._half_life_seconds < self.cfg.min_half_life_seconds:
            return "half_life_too_short"

        if self._half_life_seconds > self.cfg.max_half_life_seconds:
            return "half_life_too_long"

        beta_abs = abs(self._beta)
        if beta_abs < self.cfg.min_abs_beta:
            return "beta_too_small"

        if beta_abs > self.cfg.max_abs_beta:
            return "beta_too_large"

        # Intended hedge construction assumes SOL/BTC positive relation
        if self._beta <= 0.0:
            return "beta_non_positive"

        return None

    def _round_trip_cost_bps(self, beta: float) -> float:
        """
        Full estimated cost for entry + exit, normalized by gross notional.

        Let SOL leg notional be N.
        BTC hedge notional is abs(beta) * N.

        Gross capital: N * (1 + abs(beta))
        Entry + exit cost: 2 * [N * sol_cost + abs(beta) * N * btc_cost]
        Result in bps of gross capital.
        """
        beta_abs = abs(beta)

        sol_one_way = (
            self.cfg.sol_one_way_fee_bps
            + self.cfg.sol_one_way_slippage_bps
        )
        btc_one_way = (
            self.cfg.btc_one_way_fee_bps
            + self.cfg.btc_one_way_slippage_bps
        )

        return (
            2.0
            * (sol_one_way + beta_abs * btc_one_way)
            / (1.0 + beta_abs)
        )

    def _expected_gross_edge_bps(self, zscore: float, beta: float) -> float:
        """
        Expected gross return if Z moves from abs(z) to exit_z.

        The spread log-return is normalized by gross capital:
            gross capital = N * (1 + abs(beta))
        """
        if self._sigma_eq is None:
            return 0.0

        expected_spread_move = max(
            0.0,
            abs(zscore) - self.cfg.exit_z,
        ) * self._sigma_eq

        return (
            expected_spread_move
            / (1.0 + abs(beta))
            * BPS
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _safe_kalman_dt(self, timestamp: float) -> float:
        if self._last_kalman_ts is None:
            return 1.0

        raw_dt = timestamp - self._last_kalman_ts
        if raw_dt <= 0.0:
            return 0.001

        return min(raw_dt, self.cfg.max_kalman_dt_seconds)

    def _quote_staleness_reason(
        self,
        now: float,
        sol_quote_timestamp: Optional[float],
        btc_quote_timestamp: Optional[float],
    ) -> Optional[str]:
        max_age = self.cfg.max_quote_age_seconds

        if sol_quote_timestamp is not None:
            if now - sol_quote_timestamp > max_age:
                return "sol_quote_stale"

        if btc_quote_timestamp is not None:
            if now - btc_quote_timestamp > max_age:
                return "btc_quote_stale"

        if (
            sol_quote_timestamp is not None
            and btc_quote_timestamp is not None
            and abs(sol_quote_timestamp - btc_quote_timestamp) > max_age
        ):
            return "sol_btc_quote_timestamp_skew"

        return None

    @staticmethod
    def _valid_price(price: float) -> bool:
        return (
            isinstance(price, (int, float))
            and math.isfinite(price)
            and price > 0.0
        )

    def _telemetry(
        self,
        timestamp: float,
        sol_mid: float,
        btc_mid: float,
        action: SpreadAction,
        tradable: bool,
        status: str,
        reason: str,
        residual: Optional[float] = None,
        zscore: Optional[float] = None,
        gross_edge_bps: Optional[float] = None,
        cost_bps: Optional[float] = None,
        net_edge_bps: Optional[float] = None,
    ) -> SpreadTelemetry:
        return SpreadTelemetry(
            timestamp=timestamp,
            sol_mid=sol_mid,
            btc_mid=btc_mid,
            alpha=self._alpha if self._initialized else None,
            beta=self._beta if self._initialized else None,
            residual=residual,
            ou_mu=self._ou_mu,
            sigma_eq=self._sigma_eq,
            kappa=self._kappa,
            half_life_seconds=self._half_life_seconds,
            zscore=zscore,
            position=self._position,
            action=action,
            expected_gross_edge_bps=gross_edge_bps,
            estimated_round_trip_cost_bps=cost_bps,
            expected_net_edge_bps=net_edge_bps,
            tradable=tradable,
            status=status,
            reason=reason,
            effective_samples=self._s00,
            min_samples=self.cfg.min_ou_effective_samples,
        )
