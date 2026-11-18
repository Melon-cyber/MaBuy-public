"""Cross-exchange perpetual futures statistical arbitrage domain service.

The engine evaluates Binance USDT-M Futures, Bybit Linear, and OKX Swap
perpetual markets for cointegration, dynamic hedge-ratio stability, funding
carry, basis divergence, transaction-cost hurdles, margin availability, and
net-delta neutrality.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from math import isfinite, log
from threading import RLock
from typing import Deque, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


class ArbitrageConfigurationError(ValueError):
    """Raised when engine configuration or pair registration is invalid."""


class MarketDataValidationError(ValueError):
    """Raised when a market snapshot contains invalid or unsafe data."""


class Venue(str, Enum):
    """Supported perpetual futures venues."""

    BINANCE_FUTURES = "binance_futures"
    BYBIT_LINEAR = "bybit_linear"
    OKX_SWAP = "okx_swap"


class ExecutionStyle(str, Enum):
    """Execution aggressiveness used for fee-hurdle calculations."""

    MAKER = "maker"
    TAKER = "taker"


class OrderSide(str, Enum):
    """Directional order side."""

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class FeeSchedule:
    """Venue fee schedule expressed as decimal notional rates."""

    maker_rate: float = 0.0002
    taker_rate: float = 0.0005

    def __post_init__(self) -> None:
        for name, value in (
            ("maker_rate", self.maker_rate),
            ("taker_rate", self.taker_rate),
        ):
            if not isfinite(value) or value < 0.0 or value >= 1.0:
                raise ArbitrageConfigurationError(
                    f"{name} must be a finite decimal rate in [0, 1)."
                )

    def rate_for(self, execution_style: ExecutionStyle) -> float:
        """Return the applicable one-way fee rate."""
        return (
            self.maker_rate
            if execution_style == ExecutionStyle.MAKER
            else self.taker_rate
        )


@dataclass(frozen=True)
class MarketSnapshot:
    """Normalized market state for one perpetual instrument at one venue.

    Funding rates must be supplied as decimal rates per stated funding interval.
    For example, a funding rate of 0.01% every eight hours is represented as
    ``funding_rate=0.0001`` and ``funding_interval_hours=8.0``.

    ``contract_multiplier`` is the amount of base asset represented by one
    contract. Binance USDT-M, Bybit Linear, and many OKX USDT swaps can use
    ``1.0`` when quantities are already represented in base-asset units.
    """

    venue: Union[Venue, str]
    symbol: str
    timestamp: Union[datetime, float, int]
    mark_price: float
    funding_rate: float
    funding_interval_hours: float
    available_margin: float
    max_leverage: float
    index_price: Optional[float] = None
    contract_multiplier: float = 1.0

    @property
    def basis(self) -> Optional[float]:
        """Return mark-to-index basis, or None when no index price is supplied."""
        if self.index_price is None:
            return None
        return self.mark_price / self.index_price - 1.0


@dataclass(frozen=True)
class EngineConfig:
    """Risk, estimation, execution, and signal-generation parameters."""

    history_size: int = 1_000
    analysis_window: int = 300
    min_observations: int = 80
    max_snapshot_skew_seconds: float = 2.0
    max_snapshot_age_seconds: float = 20.0

    adf_lags: int = 1
    engle_granger_critical_value_5pct: float = -3.34

    kalman_process_noise: float = 1.0e-6
    kalman_measurement_noise: float = 1.0e-5
    max_abs_hedge_ratio: float = 3.0

    entry_z_score: float = 2.0
    min_half_life_seconds: float = 30.0
    max_half_life_seconds: float = 86_400.0
    max_holding_hours: float = 24.0

    max_gross_notional: float = 100_000.0
    margin_utilization: float = 0.80
    max_delta_fraction: float = 0.01

    min_expected_net_return: float = 0.0002
    allow_funding_only: bool = True
    funding_only_min_return: float = 0.0001

    def __post_init__(self) -> None:
        if self.history_size < 20:
            raise ArbitrageConfigurationError("history_size must be at least 20.")
        if self.analysis_window < 20:
            raise ArbitrageConfigurationError("analysis_window must be at least 20.")
        if self.analysis_window > self.history_size:
            raise ArbitrageConfigurationError(
                "analysis_window cannot exceed history_size."
            )
        if self.min_observations < 20:
            raise ArbitrageConfigurationError(
                "min_observations must be at least 20."
            )
        if self.min_observations > self.analysis_window:
            raise ArbitrageConfigurationError(
                "min_observations cannot exceed analysis_window."
            )
        if self.max_snapshot_skew_seconds < 0.0:
            raise ArbitrageConfigurationError(
                "max_snapshot_skew_seconds cannot be negative."
            )
        if self.max_snapshot_age_seconds <= 0.0:
            raise ArbitrageConfigurationError(
                "max_snapshot_age_seconds must be positive."
            )
        if self.adf_lags < 0:
            raise ArbitrageConfigurationError("adf_lags cannot be negative.")
        if self.kalman_process_noise <= 0.0:
            raise ArbitrageConfigurationError(
                "kalman_process_noise must be positive."
            )
        if self.kalman_measurement_noise <= 0.0:
            raise ArbitrageConfigurationError(
                "kalman_measurement_noise must be positive."
            )
        if self.max_abs_hedge_ratio <= 0.0:
            raise ArbitrageConfigurationError(
                "max_abs_hedge_ratio must be positive."
            )
        if self.entry_z_score <= 0.0:
            raise ArbitrageConfigurationError("entry_z_score must be positive.")
        if self.min_half_life_seconds <= 0.0:
            raise ArbitrageConfigurationError(
                "min_half_life_seconds must be positive."
            )
        if self.max_half_life_seconds < self.min_half_life_seconds:
            raise ArbitrageConfigurationError(
                "max_half_life_seconds must be >= min_half_life_seconds."
            )
        if self.max_holding_hours <= 0.0:
            raise ArbitrageConfigurationError("max_holding_hours must be positive.")
        if self.max_gross_notional <= 0.0:
            raise ArbitrageConfigurationError(
                "max_gross_notional must be positive."
            )
        if not 0.0 < self.margin_utilization <= 1.0:
            raise ArbitrageConfigurationError(
                "margin_utilization must be in (0, 1]."
            )
        if not 0.0 <= self.max_delta_fraction < 1.0:
            raise ArbitrageConfigurationError(
                "max_delta_fraction must be in [0, 1)."
            )
        if self.min_expected_net_return < 0.0:
            raise ArbitrageConfigurationError(
                "min_expected_net_return cannot be negative."
            )
        if self.funding_only_min_return < 0.0:
            raise ArbitrageConfigurationError(
                "funding_only_min_return cannot be negative."
            )


@dataclass(frozen=True)
class PairRegistration:
    """Immutable definition of an exchange-pair arbitrage relationship."""

    pair_id: str
    symbol: str
    venue_a: Venue
    venue_b: Venue


@dataclass(frozen=True)
class CointegrationResult:
    """Engle-Granger two-step and OU mean-reversion estimation output."""

    pair_id: str
    tested: bool
    cointegrated: bool
    reason: str
    sample_size: int
    alpha: Optional[float] = None
    beta: Optional[float] = None
    adf_statistic: Optional[float] = None
    adf_critical_value: Optional[float] = None
    residual_mean: Optional[float] = None
    residual_std: Optional[float] = None
    latest_residual: Optional[float] = None
    latest_z_score: Optional[float] = None
    half_life_seconds: Optional[float] = None
    kalman_hedge_ratio: Optional[float] = None
    basis_divergence: Optional[float] = None
    funding_rate_divergence: Optional[float] = None


@dataclass(frozen=True)
class OrderLeg:
    """A venue-specific leg required to establish the proposed hedge."""

    venue: Venue
    symbol: str
    side: OrderSide
    base_quantity: float
    contract_quantity: float
    mark_price: float
    notional: float
    initial_margin: float
    fee_rate: float


@dataclass(frozen=True)
class ArbitrageSignal:
    """Actionable cross-exchange hedge proposal."""

    pair_id: str
    symbol: str
    timestamp: datetime
    execution_style: ExecutionStyle
    leg_a: OrderLeg
    leg_b: OrderLeg
    gross_notional: float
    net_delta_notional: float
    delta_fraction: float
    hedge_ratio: float
    spread_z_score: float
    expected_convergence_return: float
    expected_funding_return: float
    round_trip_fee_hurdle: float
    expected_net_return: float
    estimated_holding_hours: float
    basis_divergence: Optional[float]
    funding_rate_divergence: float
    rationale: str


@dataclass(frozen=True)
class SignalDecision:
    """A complete signal evaluation, actionable or rejected."""

    pair_id: str
    timestamp: datetime
    actionable: bool
    reason: str
    analysis: CointegrationResult
    signal: Optional[ArbitrageSignal] = None


@dataclass(frozen=True)
class _PairObservation:
    """Internal synchronized pair observation."""

    timestamp_seconds: float
    snapshot_a: MarketSnapshot
    snapshot_b: MarketSnapshot
    kalman_residual: float


@dataclass
class _PairState:
    """Mutable internal state protected by the engine lock."""

    registration: PairRegistration
    observations: Deque[_PairObservation]
    kalman_residuals: Deque[float]
    latest_a: Optional[MarketSnapshot] = None
    latest_b: Optional[MarketSnapshot] = None
    last_processed_timestamp_pair: Optional[Tuple[float, float]] = None
    kalman_reference_price: Optional[float] = None
    kalman_theta: Optional[np.ndarray] = None
    kalman_covariance: Optional[np.ndarray] = None


class CrossExchangeStatisticalArbitrageEngine:
    """Thread-safe statistical arbitrage engine for perpetual futures venues.

    The engine does not send orders. It produces fully risk-checked hedge
    proposals for a separate execution and risk-control layer.

    Cointegration uses the Engle-Granger two-step procedure:

    1. OLS regression: ``price_b = alpha + beta * price_a + residual``.
    2. Augmented Dickey-Fuller test on the regression residual.

    The dynamic hedge ratio is tracked with a two-state Kalman filter on
    normalized price levels. The filter state is ``[intercept, hedge_ratio]``.
    All mutation, ingestion, registration, and signal generation is protected
    by a re-entrant lock.
    """

    _DEFAULT_FEES: Mapping[Venue, FeeSchedule] = {
        Venue.BINANCE_FUTURES: FeeSchedule(maker_rate=0.0002, taker_rate=0.0005),
        Venue.BYBIT_LINEAR: FeeSchedule(maker_rate=0.0002, taker_rate=0.0005),
        Venue.OKX_SWAP: FeeSchedule(maker_rate=0.0002, taker_rate=0.0005),
    }

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        fee_schedules: Optional[Mapping[Union[Venue, str], FeeSchedule]] = None,
    ) -> None:
        """Create an empty arbitrage engine.

        Args:
            config: Estimation, risk, and execution controls.
            fee_schedules: Optional venue fee overrides. Missing venues retain
                institutional default maker/taker assumptions of 2 bps / 5 bps.
        """
        self._config = config or EngineConfig()
        self._lock = RLock()
        self._pairs: Dict[str, _PairState] = {}
        self._fees: Dict[Venue, FeeSchedule] = dict(self._DEFAULT_FEES)

        if fee_schedules:
            for venue, schedule in fee_schedules.items():
                normalized_venue = self._coerce_venue(venue)
                if not isinstance(schedule, FeeSchedule):
                    raise ArbitrageConfigurationError(
                        "fee_schedules values must be FeeSchedule instances."
                    )
                self._fees[normalized_venue] = schedule

    @property
    def config(self) -> EngineConfig:
        """Return immutable engine configuration."""
        return self._config

    def register_pair(
        self,
        symbol: str,
        venue_a: Union[Venue, str],
        venue_b: Union[Venue, str],
        pair_id: Optional[str] = None,
    ) -> PairRegistration:
        """Register a same-symbol exchange pair for statistical arbitrage.

        The first venue is the regressor market ``A`` and the second venue is
        the dependent market ``B`` in the relationship:

        ``price_b = alpha + beta * price_a + residual``.
        """
        normalized_symbol = self._normalize_symbol(symbol)
        normalized_a = self._coerce_venue(venue_a)
        normalized_b = self._coerce_venue(venue_b)

        if normalized_a == normalized_b:
            raise ArbitrageConfigurationError(
                "A cross-exchange pair must use two distinct venues."
            )

        generated_id = (
            pair_id.strip()
            if pair_id is not None
            else f"{normalized_symbol}:{normalized_a.value}-{normalized_b.value}"
        )
        if not generated_id:
            raise ArbitrageConfigurationError("pair_id cannot be empty.")

        registration = PairRegistration(
            pair_id=generated_id,
            symbol=normalized_symbol,
            venue_a=normalized_a,
            venue_b=normalized_b,
        )

        with self._lock:
            if generated_id in self._pairs:
                raise ArbitrageConfigurationError(
                    f"Pair '{generated_id}' is already registered."
                )

            self._pairs[generated_id] = _PairState(
                registration=registration,
                observations=deque(maxlen=self._config.history_size),
                kalman_residuals=deque(maxlen=self._config.history_size),
            )

        return registration

    def unregister_pair(self, pair_id: str) -> None:
        """Remove a registered pair and all of its accumulated model state."""
        with self._lock:
            if pair_id not in self._pairs:
                raise ArbitrageConfigurationError(f"Unknown pair_id '{pair_id}'.")
            del self._pairs[pair_id]

    def registered_pairs(self) -> Tuple[PairRegistration, ...]:
        """Return a stable snapshot of registered pair definitions."""
        with self._lock:
            return tuple(state.registration for state in self._pairs.values())

    def ingest(self, snapshot: MarketSnapshot) -> Tuple[str, ...]:
        """Ingest one venue snapshot and update any synchronizable pairs.

        Stale or out-of-order snapshots cannot overwrite a newer snapshot from
        the same venue. A pair observation is added only when both venues have
        snapshots within the configured maximum timestamp skew.

        Returns:
            Pair identifiers for which a new synchronized observation was added.
        """
        self._validate_snapshot(snapshot)
        snapshot_venue = self._coerce_venue(snapshot.venue)
        normalized_symbol = self._normalize_symbol(snapshot.symbol)
        updated_pairs = []

        with self._lock:
            for pair_id, state in self._pairs.items():
                registration = state.registration
                if registration.symbol != normalized_symbol:
                    continue

                if snapshot_venue == registration.venue_a:
                    if self._is_newer_snapshot(snapshot, state.latest_a):
                        state.latest_a = snapshot
                elif snapshot_venue == registration.venue_b:
                    if self._is_newer_snapshot(snapshot, state.latest_b):
                        state.latest_b = snapshot
                else:
                    continue

                if self._append_synchronized_observation_locked(state):
                    updated_pairs.append(pair_id)

        return tuple(updated_pairs)

    def update_pair(
        self,
        pair_id: str,
        snapshot_a: MarketSnapshot,
        snapshot_b: MarketSnapshot,
    ) -> bool:
        """Atomically update both legs of a registered pair.

        This method is useful for feed handlers that already synchronize market
        data upstream. It returns ``True`` only if a new observation was added.
        """
        self._validate_snapshot(snapshot_a)
        self._validate_snapshot(snapshot_b)

        with self._lock:
            state = self._require_pair_locked(pair_id)
            registration = state.registration

            if self._coerce_venue(snapshot_a.venue) != registration.venue_a:
                raise MarketDataValidationError(
                    f"snapshot_a venue must be {registration.venue_a.value}."
                )
            if self._coerce_venue(snapshot_b.venue) != registration.venue_b:
                raise MarketDataValidationError(
                    f"snapshot_b venue must be {registration.venue_b.value}."
                )
            if self._normalize_symbol(snapshot_a.symbol) != registration.symbol:
                raise MarketDataValidationError(
                    "snapshot_a symbol does not match registered pair symbol."
                )
            if self._normalize_symbol(snapshot_b.symbol) != registration.symbol:
                raise MarketDataValidationError(
                    "snapshot_b symbol does not match registered pair symbol."
                )

            state.latest_a = snapshot_a
            state.latest_b = snapshot_b
            return self._append_synchronized_observation_locked(state)

    def cointegration(self, pair_id: str) -> CointegrationResult:
        """Run the current Engle-Granger and OU analyses for a pair."""
        with self._lock:
            state = self._require_pair_locked(pair_id)
            return self._cointegration_locked(state)

    def generate_signal(
        self,
        pair_id: str,
        desired_gross_notional: Optional[float] = None,
        execution_style: ExecutionStyle = ExecutionStyle.MAKER,
        as_of: Optional[Union[datetime, float, int]] = None,
    ) -> SignalDecision:
        """Generate a margin- and delta-constrained arbitrage decision.

        A trade is actionable only when all conditions hold:

        * The pair passes the Engle-Granger residual stationarity test.
        * The Kalman hedge ratio is positive and within configured limits.
        * The spread z-score meets the entry threshold, or funding carry alone
          passes its configured hurdle.
        * OU half-life is within risk limits.
        * Net delta satisfies the configured neutrality tolerance.
        * Gross notional respects both venue margin capacity constraints.
        * Expected convergence and funding PnL exceed round-trip fees and the
          configured minimum expected net-return hurdle.
        """
        if desired_gross_notional is not None:
            if not isfinite(desired_gross_notional) or desired_gross_notional <= 0.0:
                raise ArbitrageConfigurationError(
                    "desired_gross_notional must be finite and positive."
                )

        if not isinstance(execution_style, ExecutionStyle):
            execution_style = ExecutionStyle(str(execution_style).lower())

        as_of_seconds = self._timestamp_to_seconds(as_of) if as_of is not None else None

        with self._lock:
            state = self._require_pair_locked(pair_id)
            analysis = self._cointegration_locked(state)

            latest_timestamp = self._latest_timestamp_locked(state)
            decision_time = self._seconds_to_datetime(latest_timestamp or time_now())

            if state.latest_a is None or state.latest_b is None:
                return SignalDecision(
                    pair_id=pair_id,
                    timestamp=decision_time,
                    actionable=False,
                    reason="Both venue snapshots are required before signal generation.",
                    analysis=analysis,
                )

            reference_time = as_of_seconds if as_of_seconds is not None else time_now()
            if latest_timestamp is None or (
                reference_time - latest_timestamp > self._config.max_snapshot_age_seconds
            ):
                return SignalDecision(
                    pair_id=pair_id,
                    timestamp=decision_time,
                    actionable=False,
                    reason="Latest synchronized market observation is stale.",
                    analysis=analysis,
                )

            if not analysis.tested:
                return SignalDecision(
                    pair_id=pair_id,
                    timestamp=decision_time,
                    actionable=False,
                    reason=analysis.reason,
                    analysis=analysis,
                )

            if not analysis.cointegrated:
                return SignalDecision(
                    pair_id=pair_id,
                    timestamp=decision_time,
                    actionable=False,
                    reason="Pair fails the Engle-Granger cointegration criterion.",
                    analysis=analysis,
                )

            if (
                analysis.kalman_hedge_ratio is None
                or analysis.kalman_hedge_ratio <= 0.0
                or analysis.kalman_hedge_ratio > self._config.max_abs_hedge_ratio
            ):
                return SignalDecision(
                    pair_id=pair_id,
                    timestamp=decision_time,
                    actionable=False,
                    reason="Dynamic Kalman hedge ratio is outside permitted bounds.",
                    analysis=analysis,
                )

            if (
                analysis.half_life_seconds is None
                or analysis.half_life_seconds < self._config.min_half_life_seconds
                or analysis.half_life_seconds > self._config.max_half_life_seconds
            ):
                return SignalDecision(
                    pair_id=pair_id,
                    timestamp=decision_time,
                    actionable=False,
                    reason="Estimated OU half-life is outside permitted risk bounds.",
                    analysis=analysis,
                )

            if analysis.latest_z_score is None:
                return SignalDecision(
                    pair_id=pair_id,
                    timestamp=decision_time,
                    actionable=False,
                    reason="Insufficient dynamic residual history for z-score estimation.",
                    analysis=analysis,
                )

            snapshot_a = state.latest_a
            snapshot_b = state.latest_b
            hedge_ratio = analysis.kalman_hedge_ratio
            z_score = analysis.latest_z_score

            spread_triggered = abs(z_score) >= self._config.entry_z_score
            estimated_holding_hours = min(
                self._config.max_holding_hours,
                max(analysis.half_life_seconds / 3600.0, 1.0 / 60.0),
            )

            if spread_triggered:
                side_b_sign = -1.0 if z_score > 0.0 else 1.0
                trade_origin = "mean-reversion spread dislocation"
            elif self._config.allow_funding_only:
                funding_if_long_b = self._funding_return(
                    side_a_sign=-1.0,
                    side_b_sign=1.0,
                    hedge_ratio=hedge_ratio,
                    snapshot_a=snapshot_a,
                    snapshot_b=snapshot_b,
                    holding_hours=estimated_holding_hours,
                )
                funding_if_short_b = self._funding_return(
                    side_a_sign=1.0,
                    side_b_sign=-1.0,
                    hedge_ratio=hedge_ratio,
                    snapshot_a=snapshot_a,
                    snapshot_b=snapshot_b,
                    holding_hours=estimated_holding_hours,
                )
                side_b_sign = (
                    1.0 if funding_if_long_b >= funding_if_short_b else -1.0
                )
                trade_origin = "cross-exchange funding carry"
            else:
                return SignalDecision(
                    pair_id=pair_id,
                    timestamp=decision_time,
                    actionable=False,
                    reason="Spread z-score does not meet the entry threshold.",
                    analysis=analysis,
                )

            side_a_sign = -side_b_sign
            fee_rate_a = self._fees[self._coerce_venue(snapshot_a.venue)].rate_for(
                execution_style
            )
            fee_rate_b = self._fees[self._coerce_venue(snapshot_b.venue)].rate_for(
                execution_style
            )

            gross_per_unit_b = (
                snapshot_b.mark_price + hedge_ratio * snapshot_a.mark_price
            )
            if gross_per_unit_b <= 0.0:
                return SignalDecision(
                    pair_id=pair_id,
                    timestamp=decision_time,
                    actionable=False,
                    reason="Invalid hedge notional calculation.",
                    analysis=analysis,
                )

            delta_per_unit_b = (
                side_a_sign * hedge_ratio * snapshot_a.mark_price
                + side_b_sign * snapshot_b.mark_price
            )
            delta_fraction = abs(delta_per_unit_b) / gross_per_unit_b
            if delta_fraction > self._config.max_delta_fraction:
                return SignalDecision(
                    pair_id=pair_id,
                    timestamp=decision_time,
                    actionable=False,
                    reason=(
                        "Dynamic hedge ratio violates the net-delta neutrality "
                        "constraint."
                    ),
                    analysis=analysis,
                )

            max_gross = self._maximum_gross_notional(
                hedge_ratio=hedge_ratio,
                snapshot_a=snapshot_a,
                snapshot_b=snapshot_b,
            )
            requested_gross = (
                desired_gross_notional
                if desired_gross_notional is not None
                else self._config.max_gross_notional
            )
            gross_notional = min(requested_gross, self._config.max_gross_notional, max_gross)

            if gross_notional <= 0.0:
                return SignalDecision(
                    pair_id=pair_id,
                    timestamp=decision_time,
                    actionable=False,
                    reason="Insufficient available margin on one or both venues.",
                    analysis=analysis,
                )

            quantity_b = gross_notional / gross_per_unit_b
            quantity_a = hedge_ratio * quantity_b
            notional_a = quantity_a * snapshot_a.mark_price
            notional_b = quantity_b * snapshot_b.mark_price
            actual_gross = notional_a + notional_b

            convergence_return = 0.0
            if spread_triggered and analysis.latest_residual is not None:
                expected_convergence_pnl = abs(analysis.latest_residual) * quantity_b * 0.5
                convergence_return = expected_convergence_pnl / actual_gross

            funding_return = self._funding_return(
                side_a_sign=side_a_sign,
                side_b_sign=side_b_sign,
                hedge_ratio=hedge_ratio,
                snapshot_a=snapshot_a,
                snapshot_b=snapshot_b,
                holding_hours=estimated_holding_hours,
            )

            round_trip_fee_hurdle = 2.0 * (
                (notional_a / actual_gross) * fee_rate_a
                + (notional_b / actual_gross) * fee_rate_b
            )
            expected_net_return = (
                convergence_return + funding_return - round_trip_fee_hurdle
            )

            required_hurdle = (
                self._config.min_expected_net_return
                if spread_triggered
                else max(
                    self._config.min_expected_net_return,
                    self._config.funding_only_min_return,
                )
            )
            if expected_net_return < required_hurdle:
                return SignalDecision(
                    pair_id=pair_id,
                    timestamp=decision_time,
                    actionable=False,
                    reason=(
                        "Expected convergence and funding return does not clear "
                        "the round-trip fee and return hurdle."
                    ),
                    analysis=analysis,
                )

            margin_a = notional_a / snapshot_a.max_leverage
            margin_b = notional_b / snapshot_b.max_leverage
            side_a = OrderSide.BUY if side_a_sign > 0.0 else OrderSide.SELL
            side_b = OrderSide.BUY if side_b_sign > 0.0 else OrderSide.SELL

            signal = ArbitrageSignal(
                pair_id=pair_id,
                symbol=state.registration.symbol,
                timestamp=decision_time,
                execution_style=execution_style,
                leg_a=OrderLeg(
                    venue=state.registration.venue_a,
                    symbol=state.registration.symbol,
                    side=side_a,
                    base_quantity=quantity_a,
                    contract_quantity=quantity_a / snapshot_a.contract_multiplier,
                    mark_price=snapshot_a.mark_price,
                    notional=notional_a,
                    initial_margin=margin_a,
                    fee_rate=fee_rate_a,
                ),
                leg_b=OrderLeg(
                    venue=state.registration.venue_b,
                    symbol=state.registration.symbol,
                    side=side_b,
                    base_quantity=quantity_b,
                    contract_quantity=quantity_b / snapshot_b.contract_multiplier,
                    mark_price=snapshot_b.mark_price,
                    notional=notional_b,
                    initial_margin=margin_b,
                    fee_rate=fee_rate_b,
                ),
                gross_notional=actual_gross,
                net_delta_notional=delta_per_unit_b * quantity_b,
                delta_fraction=delta_fraction,
                hedge_ratio=hedge_ratio,
                spread_z_score=z_score,
                expected_convergence_return=convergence_return,
                expected_funding_return=funding_return,
                round_trip_fee_hurdle=round_trip_fee_hurdle,
                expected_net_return=expected_net_return,
                estimated_holding_hours=estimated_holding_hours,
                basis_divergence=analysis.basis_divergence,
                funding_rate_divergence=analysis.funding_rate_divergence or 0.0,
                rationale=(
                    f"{trade_origin}; Kalman hedge ratio={hedge_ratio:.6f}, "
                    f"z-score={z_score:.3f}, expected net return="
                    f"{expected_net_return:.5%}."
                ),
            )
            return SignalDecision(
                pair_id=pair_id,
                timestamp=decision_time,
                actionable=True,
                reason="Signal passes cointegration, risk, margin, and fee hurdles.",
                analysis=analysis,
                signal=signal,
            )

    def _append_synchronized_observation_locked(self, state: _PairState) -> bool:
        """Append synchronized data and update Kalman state under the lock."""
        snapshot_a = state.latest_a
        snapshot_b = state.latest_b
        if snapshot_a is None or snapshot_b is None:
            return False

        timestamp_a = self._timestamp_to_seconds(snapshot_a.timestamp)
        timestamp_b = self._timestamp_to_seconds(snapshot_b.timestamp)
        if abs(timestamp_a - timestamp_b) > self._config.max_snapshot_skew_seconds:
            return False

        timestamp_pair = (timestamp_a, timestamp_b)
        if state.last_processed_timestamp_pair == timestamp_pair:
            return False

        observation_timestamp = max(timestamp_a, timestamp_b)
        if state.observations and (
            observation_timestamp <= state.observations[-1].timestamp_seconds
        ):
            return False

        residual = self._kalman_update_locked(
            state=state,
            price_a=snapshot_a.mark_price,
            price_b=snapshot_b.mark_price,
        )
        state.observations.append(
            _PairObservation(
                timestamp_seconds=observation_timestamp,
                snapshot_a=snapshot_a,
                snapshot_b=snapshot_b,
                kalman_residual=residual,
            )
        )
        state.kalman_residuals.append(residual)
        state.last_processed_timestamp_pair = timestamp_pair
        return True

    def _kalman_update_locked(
        self,
        state: _PairState,
        price_a: float,
        price_b: float,
    ) -> float:
        """Perform a numerically stable normalized Kalman filter update."""
        if state.kalman_reference_price is None:
            state.kalman_reference_price = price_a
            normalized_b = price_b / price_a
            state.kalman_theta = np.array([0.0, normalized_b], dtype=float)
            state.kalman_covariance = np.diag([1.0, 1.0]).astype(float)
            return 0.0

        reference = state.kalman_reference_price
        theta = state.kalman_theta
        covariance = state.kalman_covariance
        if theta is None or covariance is None:
            raise RuntimeError("Kalman state invariant violated.")

        normalized_a = price_a / reference
        normalized_b = price_b / reference
        design = np.array([1.0, normalized_a], dtype=float)
        process_covariance = np.diag(
            [
                self._config.kalman_process_noise,
                self._config.kalman_process_noise,
            ]
        )
        predicted_covariance = covariance + process_covariance
        predicted_measurement = float(design @ theta)
        innovation = normalized_b - predicted_measurement
        innovation_variance = float(
            design @ predicted_covariance @ design
            + self._config.kalman_measurement_noise
        )

        if innovation_variance <= 0.0 or not isfinite(innovation_variance):
            raise RuntimeError("Kalman innovation variance is invalid.")

        gain = (predicted_covariance @ design) / innovation_variance
        updated_theta = theta + gain * innovation

        identity = np.eye(2)
        design_matrix = design.reshape(1, 2)
        gain_matrix = gain.reshape(2, 1)
        measurement_covariance = np.array(
            [[self._config.kalman_measurement_noise]]
        )
        updated_covariance = (
            (identity - gain_matrix @ design_matrix)
            @ predicted_covariance
            @ (identity - gain_matrix @ design_matrix).T
            + gain_matrix @ measurement_covariance @ gain_matrix.T
        )

        state.kalman_theta = updated_theta
        state.kalman_covariance = updated_covariance
        return innovation * reference

    def _cointegration_locked(self, state: _PairState) -> CointegrationResult:
        """Compute Engle-Granger, OU, dynamic residual, and carry diagnostics."""
        observations = list(state.observations)[-self._config.analysis_window :]
        sample_size = len(observations)
        registration = state.registration

        if sample_size < self._config.min_observations:
            return CointegrationResult(
                pair_id=registration.pair_id,
                tested=False,
                cointegrated=False,
                reason=(
                    f"Insufficient synchronized observations: {sample_size}/"
                    f"{self._config.min_observations}."
                ),
                sample_size=sample_size,
            )

        prices_a = np.asarray(
            [obs.snapshot_a.mark_price for obs in observations],
            dtype=float,
        )
        prices_b = np.asarray(
            [obs.snapshot_b.mark_price for obs in observations],
            dtype=float,
        )
        timestamps = np.asarray(
            [obs.timestamp_seconds for obs in observations],
            dtype=float,
        )

        try:
            alpha, beta, residuals = self._ols_with_intercept(prices_a, prices_b)
            adf_statistic = self._adf_statistic(
                residuals,
                lags=self._config.adf_lags,
            )
            residual_std = float(np.std(residuals, ddof=1))
            residual_mean = float(np.mean(residuals))
        except (np.linalg.LinAlgError, ValueError, FloatingPointError) as exc:
            return CointegrationResult(
                pair_id=registration.pair_id,
                tested=False,
                cointegrated=False,
                reason=f"Cointegration estimation failed: {exc}",
                sample_size=sample_size,
            )

        latest_residual = (
            float(state.kalman_residuals[-1])
            if state.kalman_residuals
            else float(residuals[-1])
        )
        dynamic_residuals = np.asarray(
            list(state.kalman_residuals)[-self._config.analysis_window :],
            dtype=float,
        )
        if len(dynamic_residuals) >= 20:
            dynamic_std = float(np.std(dynamic_residuals, ddof=1))
            dynamic_mean = float(np.mean(dynamic_residuals))
            dynamic_z_score = (
                (latest_residual - dynamic_mean) / dynamic_std
                if dynamic_std > 1.0e-12
                else 0.0
            )
        else:
            dynamic_z_score = None

        half_life = self._estimate_half_life_seconds(residuals, timestamps)
        latest_a = observations[-1].snapshot_a
        latest_b = observations[-1].snapshot_b

        if latest_a.basis is not None and latest_b.basis is not None:
            basis_divergence: Optional[float] = latest_b.basis - latest_a.basis
        else:
            basis_divergence = None

        kalman_hedge_ratio = None
        if state.kalman_theta is not None:
            kalman_hedge_ratio = float(state.kalman_theta[1])

        cointegrated = bool(
            isfinite(adf_statistic)
            and adf_statistic <= self._config.engle_granger_critical_value_5pct
            and beta > 0.0
        )

        reason = (
            "Residual ADF statistic passes the Engle-Granger 5% critical value."
            if cointegrated
            else "Residual ADF statistic does not pass the Engle-Granger 5% critical value."
        )

        return CointegrationResult(
            pair_id=registration.pair_id,
            tested=True,
            cointegrated=cointegrated,
            reason=reason,
            sample_size=sample_size,
            alpha=alpha,
            beta=beta,
            adf_statistic=adf_statistic,
            adf_critical_value=self._config.engle_granger_critical_value_5pct,
            residual_mean=residual_mean,
            residual_std=residual_std,
            latest_residual=latest_residual,
            latest_z_score=dynamic_z_score,
            half_life_seconds=half_life,
            kalman_hedge_ratio=kalman_hedge_ratio,
            basis_divergence=basis_divergence,
            funding_rate_divergence=latest_b.funding_rate - latest_a.funding_rate,
        )

    def _maximum_gross_notional(
        self,
        hedge_ratio: float,
        snapshot_a: MarketSnapshot,
        snapshot_b: MarketSnapshot,
    ) -> float:
        """Return the maximum gross notional permitted by both margin accounts."""
        unit_notional_a = hedge_ratio * snapshot_a.mark_price
        unit_notional_b = snapshot_b.mark_price
        gross_per_unit_b = unit_notional_a + unit_notional_b

        margin_per_unit_a = unit_notional_a / snapshot_a.max_leverage
        margin_per_unit_b = unit_notional_b / snapshot_b.max_leverage

        available_a = snapshot_a.available_margin * self._config.margin_utilization
        available_b = snapshot_b.available_margin * self._config.margin_utilization

        if margin_per_unit_a <= 0.0 or margin_per_unit_b <= 0.0:
            return 0.0

        max_quantity_b_a = available_a / margin_per_unit_a
        max_quantity_b_b = available_b / margin_per_unit_b
        return max(0.0, min(max_quantity_b_a, max_quantity_b_b) * gross_per_unit_b)

    @staticmethod
    def _funding_return(
        side_a_sign: float,
        side_b_sign: float,
        hedge_ratio: float,
        snapshot_a: MarketSnapshot,
        snapshot_b: MarketSnapshot,
        holding_hours: float,
    ) -> float:
        """Estimate funding PnL as a fraction of gross hedged notional."""
        notional_a = hedge_ratio * snapshot_a.mark_price
        notional_b = snapshot_b.mark_price
        gross = notional_a + notional_b
        if gross <= 0.0:
            return 0.0

        events_a = holding_hours / snapshot_a.funding_interval_hours
        events_b = holding_hours / snapshot_b.funding_interval_hours

        funding_pnl = (
            -side_a_sign * snapshot_a.funding_rate * events_a * notional_a
            - side_b_sign * snapshot_b.funding_rate * events_b * notional_b
        )
        return funding_pnl / gross

    @staticmethod
    def _ols_with_intercept(
        x: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[float, float, np.ndarray]:
        """Fit y = alpha + beta*x using least squares."""
        if len(x) != len(y) or len(x) < 3:
            raise ValueError("OLS requires at least three aligned observations.")
        if np.std(x) <= 1.0e-12:
            raise ValueError("Regressor price variance is too small.")

        design = np.column_stack((np.ones(len(x)), x))
        coefficients, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
        if rank < 2:
            raise ValueError("OLS design matrix is rank deficient.")

        alpha = float(coefficients[0])
        beta = float(coefficients[1])
        residuals = y - design @ coefficients
        return alpha, beta, residuals

    @staticmethod
    def _adf_statistic(residuals: np.ndarray, lags: int) -> float:
        """Return the ADF t-statistic for the residual unit-root coefficient."""
        residuals = np.asarray(residuals, dtype=float)
        differences = np.diff(residuals)
        n = len(residuals)

        if n < 12 + lags:
            raise ValueError("Insufficient observations for augmented Dickey-Fuller test.")

        start = lags
        target = differences[start:]
        lagged_level = residuals[start : n - 1]
        columns = [np.ones(len(target)), lagged_level]

        for lag in range(1, lags + 1):
            columns.append(differences[start - lag : len(differences) - lag])

        design = np.column_stack(columns)
        coefficients, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank < design.shape[1]:
            raise ValueError("ADF design matrix is rank deficient.")

        fitted = design @ coefficients
        regression_residuals = target - fitted
        degrees_of_freedom = len(target) - design.shape[1]
        if degrees_of_freedom <= 0:
            raise ValueError("ADF regression has insufficient degrees of freedom.")

        residual_variance = float(regression_residuals @ regression_residuals)
        residual_variance /= degrees_of_freedom
        covariance = residual_variance * np.linalg.inv(design.T @ design)
        standard_error = float(np.sqrt(covariance[1, 1]))

        if standard_error <= 0.0 or not isfinite(standard_error):
            raise ValueError("ADF unit-root coefficient standard error is invalid.")
        return float(coefficients[1] / standard_error)

    @staticmethod
    def _estimate_half_life_seconds(
        residuals: np.ndarray,
        timestamps: np.ndarray,
    ) -> Optional[float]:
        """Estimate OU half-life from residual first differences.

        For ``Δx_t = c + gamma*x_(t-1) + epsilon_t``, the continuous-time OU
        half-life approximation is ``-ln(2) * dt / gamma`` when gamma < 0.
        """
        if len(residuals) < 10 or len(timestamps) != len(residuals):
            return None

        lagged = residuals[:-1]
        delta = np.diff(residuals)
        if np.std(lagged) <= 1.0e-12:
            return None

        design = np.column_stack((np.ones(len(lagged)), lagged))
        coefficients, _, rank, _ = np.linalg.lstsq(design, delta, rcond=None)
        if rank < 2:
            return None

        gamma = float(coefficients[1])
        if not isfinite(gamma) or gamma >= 0.0:
            return None

        intervals = np.diff(timestamps)
        positive_intervals = intervals[intervals > 0.0]
        if len(positive_intervals) == 0:
            return None

        median_interval = float(np.median(positive_intervals))
        half_life = -log(2.0) * median_interval / gamma
        if not isfinite(half_life) or half_life <= 0.0:
            return None
        return half_life

    def _require_pair_locked(self, pair_id: str) -> _PairState:
        """Return a pair state or raise a deterministic configuration error."""
        try:
            return self._pairs[pair_id]
        except KeyError as exc:
            raise ArbitrageConfigurationError(f"Unknown pair_id '{pair_id}'.") from exc

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize and validate a canonical perpetual instrument symbol."""
        if not isinstance(symbol, str):
            raise MarketDataValidationError("symbol must be a string.")
        normalized = symbol.strip().upper()
        if len(normalized) < 3:
            raise MarketDataValidationError("symbol is too short.")
        return normalized

    @staticmethod
    def _coerce_venue(venue: Union[Venue, str]) -> Venue:
        """Convert supported venue aliases into a Venue enum."""
        if isinstance(venue, Venue):
            return venue
        if not isinstance(venue, str):
            raise MarketDataValidationError("venue must be a Venue or string.")

        normalized = venue.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "binance": Venue.BINANCE_FUTURES,
            "binance_futures": Venue.BINANCE_FUTURES,
            "binance_usdt_m": Venue.BINANCE_FUTURES,
            "bybit": Venue.BYBIT_LINEAR,
            "bybit_linear": Venue.BYBIT_LINEAR,
            "okx": Venue.OKX_SWAP,
            "okx_swap": Venue.OKX_SWAP,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            supported = ", ".join(v.value for v in Venue)
            raise MarketDataValidationError(
                f"Unsupported venue '{venue}'. Supported venues: {supported}."
            ) from exc

    def _validate_snapshot(self, snapshot: MarketSnapshot) -> None:
        """Validate all numerical market-data invariants before state mutation."""
        if not isinstance(snapshot, MarketSnapshot):
            raise MarketDataValidationError(
                "snapshot must be a MarketSnapshot instance."
            )

        self._coerce_venue(snapshot.venue)
        self._normalize_symbol(snapshot.symbol)
        self._timestamp_to_seconds(snapshot.timestamp)

        positive_fields = (
            ("mark_price", snapshot.mark_price),
            ("funding_interval_hours", snapshot.funding_interval_hours),
            ("max_leverage", snapshot.max_leverage),
            ("contract_multiplier", snapshot.contract_multiplier),
        )
        for name, value in positive_fields:
            if not isfinite(value) or value <= 0.0:
                raise MarketDataValidationError(
                    f"{name} must be finite and strictly positive."
                )

        if not isfinite(snapshot.funding_rate) or abs(snapshot.funding_rate) >= 1.0:
            raise MarketDataValidationError(
                "funding_rate must be finite and have absolute value below 1."
            )

        if not isfinite(snapshot.available_margin) or snapshot.available_margin < 0.0:
            raise MarketDataValidationError(
                "available_margin must be finite and non-negative."
            )

        if snapshot.index_price is not None and (
            not isfinite(snapshot.index_price) or snapshot.index_price <= 0.0
        ):
            raise MarketDataValidationError(
                "index_price must be None or finite and strictly positive."
            )

    @staticmethod
    def _timestamp_to_seconds(value: Union[datetime, float, int, None]) -> float:
        """Convert supported timestamps into UTC epoch seconds."""
        if value is None:
            raise MarketDataValidationError("timestamp cannot be None.")

        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            result = value.timestamp()
        elif isinstance(value, (float, int)):
            result = float(value)
        else:
            raise MarketDataValidationError(
                "timestamp must be datetime, float, or integer epoch seconds."
            )

        if not isfinite(result):
            raise MarketDataValidationError("timestamp must be finite.")
        return result

    @staticmethod
    def _seconds_to_datetime(timestamp_seconds: float) -> datetime:
        """Convert epoch seconds to an aware UTC datetime."""
        return datetime.fromtimestamp(timestamp_seconds, tz=timezone.utc)

    def _latest_timestamp_locked(self, state: _PairState) -> Optional[float]:
        """Return the latest synchronized timestamp for a pair."""
        if state.observations:
            return state.observations[-1].timestamp_seconds
        return None

    def _is_newer_snapshot(
        self,
        candidate: MarketSnapshot,
        existing: Optional[MarketSnapshot],
    ) -> bool:
        """Return whether candidate should replace an existing venue snapshot."""
        if existing is None:
            return True
        return self._timestamp_to_seconds(candidate.timestamp) > self._timestamp_to_seconds(
            existing.timestamp
        )


def time_now() -> float:
    """Return wall-clock UTC epoch seconds through a testable helper."""
    return datetime.now(tz=timezone.utc).timestamp()


if __name__ == "__main__":
    rng = np.random.default_rng(7)

    engine = CrossExchangeStatisticalArbitrageEngine(
        config=EngineConfig(
            min_observations=80,
            analysis_window=240,
            history_size=500,
            max_snapshot_age_seconds=60.0,
            entry_z_score=1.75,
            max_gross_notional=50_000.0,
        )
    )

    pair = engine.register_pair(
        symbol="BTCUSDT",
        venue_a=Venue.BINANCE_FUTURES,
        venue_b=Venue.BYBIT_LINEAR,
    )

    base_time = datetime.now(tz=timezone.utc) - timedelta(minutes=299)
    price_a = 60_000.0
    residual = 0.0

    for index in range(300):
        price_a *= float(np.exp(rng.normal(0.0, 0.0009)))
        residual = 0.91 * residual + float(rng.normal(0.0, 4.0))

        if index >= 294:
            residual += 24.0

        price_b = 1.0004 * price_a + 2.0 + residual
        timestamp = base_time + timedelta(minutes=index)

        engine.update_pair(
            pair.pair_id,
            MarketSnapshot(
                venue=Venue.BINANCE_FUTURES,
                symbol="BTCUSDT",
                timestamp=timestamp,
                mark_price=price_a,
                index_price=price_a * 0.9999,
                funding_rate=0.00010,
                funding_interval_hours=8.0,
                available_margin=10_000.0,
                max_leverage=10.0,
            ),
            MarketSnapshot(
                venue=Venue.BYBIT_LINEAR,
                symbol="BTCUSDT",
                timestamp=timestamp,
                mark_price=price_b,
                index_price=price_b * 1.0001,
                funding_rate=0.00035,
                funding_interval_hours=8.0,
                available_margin=10_000.0,
                max_leverage=10.0,
            ),
        )

    analysis = engine.cointegration(pair.pair_id)
    decision = engine.generate_signal(
        pair.pair_id,
        desired_gross_notional=25_000.0,
        execution_style=ExecutionStyle.MAKER,
    )

    print("Cointegration:", analysis)
    print("Decision:", decision)