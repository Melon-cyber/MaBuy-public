"""Continuous volatility-normalized trailing stop management."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple

__all__ = ["TrailingState", "ContinuousTrailingStopEngine"]


def _finite_float(value: float, field_name: str) -> float:
    """Return a finite float or raise a descriptive ValueError."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite numeric value, not bool.")

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite numeric value.") from exc

    if not math.isfinite(numeric_value):
        raise ValueError(f"{field_name} must be finite.")

    return numeric_value


def _positive_float(value: float, field_name: str) -> float:
    """Return a strictly positive finite float."""
    numeric_value = _finite_float(value, field_name)
    if numeric_value <= 0.0:
        raise ValueError(f"{field_name} must be strictly positive.")
    return numeric_value


def _non_negative_float(value: float, field_name: str) -> float:
    """Return a non-negative finite float."""
    numeric_value = _finite_float(value, field_name)
    if numeric_value < 0.0:
        raise ValueError(f"{field_name} must be non-negative.")
    return numeric_value


def _validate_side(side: int) -> int:
    """Validate and normalize a directional trading side."""
    if isinstance(side, bool) or side not in (-1, 1):
        raise ValueError("side must be +1 for LONG or -1 for SHORT.")
    return int(side)


@dataclass
class TrailingState:
    """
    Mutable state required to maintain a continuous trailing stop.

    ``current_sl_px`` represents the last stop level accepted for exchange
    amendment. The engine only advances this value when an amendment meets the
    configured minimum tightening threshold.
    """

    symbol: str
    side: int
    entry_px: float
    current_sl_px: float
    peak_px: float
    d_min: float = 0.012
    k_sigma: float = 1.8
    updated_at_ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Validate and normalize state values at construction time."""
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string.")

        self.symbol = self.symbol.strip()
        self.side = _validate_side(self.side)
        self.entry_px = _positive_float(self.entry_px, "entry_px")
        self.current_sl_px = _positive_float(
            self.current_sl_px,
            "current_sl_px",
        )
        self.peak_px = _positive_float(self.peak_px, "peak_px")
        self.d_min = _non_negative_float(self.d_min, "d_min")
        self.k_sigma = _non_negative_float(self.k_sigma, "k_sigma")
        self.updated_at_ts = _finite_float(self.updated_at_ts, "updated_at_ts")


class ContinuousTrailingStopEngine:
    """
    Institutional trailing-stop engine with a strict one-directional ratchet.

    The engine tracks the most favorable observed mark and derives a stop
    distance from the maximum of the configured minimum distance, normalized
    ATR distance, and optional structural-level distance.

    ``TrailingState.current_sl_px`` is updated only when the resulting stop
    tightening is sufficiently material to warrant an exchange amendment. This
    allows sub-threshold improvements to accumulate against the existing
    exchange stop rather than being lost to rate-limit suppression.
    """

    DEFAULT_EXCHANGE_UPDATE_THRESHOLD: ClassVar[float] = 0.0015

    def __init__(
        self,
        exchange_update_threshold: float = DEFAULT_EXCHANGE_UPDATE_THRESHOLD,
    ) -> None:
        """
        Initialize the engine.

        Args:
            exchange_update_threshold: Minimum fractional tightening relative
                to mark price required before an exchange stop amendment is
                requested. The default is 0.15%.

        Raises:
            ValueError: If the threshold is negative or non-finite.
        """
        self.exchange_update_threshold = _non_negative_float(
            exchange_update_threshold,
            "exchange_update_threshold",
        )

    def update_stop(
        self,
        state: TrailingState,
        mark_price: float,
        atr_norm: float,
        structural_invalidation_px: Optional[float] = None,
    ) -> Tuple[float, bool, str]:
        """
        Update the favorable price extreme and calculate a ratcheted stop.

        For a long position, the favorable extreme is the running maximum mark
        and the stop can only increase. For a short position, the favorable
        extreme is the running minimum mark and the stop can only decrease.

        Structural distance is calculated as the absolute normalized distance
        between the favorable extreme and the supplied structural invalidation
        level. This accommodates support levels for longs and resistance levels
        for shorts while preserving the ratchet invariant.

        Args:
            state: Mutable trailing-stop state.
            mark_price: Current positive mark price.
            atr_norm: Non-negative ATR expressed as a normalized fraction.
            structural_invalidation_px: Optional positive structural level.

        Returns:
            A tuple of:
            - proposed ratcheted stop level,
            - whether an exchange amendment should be submitted,
            - a machine-readable explanation string.

        Raises:
            TypeError: If ``state`` is not a TrailingState instance.
            ValueError: If prices, volatility inputs, or state values are
                invalid.
        """
        if not isinstance(state, TrailingState):
            raise TypeError("state must be an instance of TrailingState.")

        self._validate_state(state)

        mark = _positive_float(mark_price, "mark_price")
        normalized_atr = _non_negative_float(atr_norm, "atr_norm")

        structural_distance = 0.0
        if structural_invalidation_px is not None:
            structural_level = _positive_float(
                structural_invalidation_px,
                "structural_invalidation_px",
            )
        else:
            structural_level = None

        if state.side == 1:
            state.peak_px = max(state.peak_px, mark)
        else:
            state.peak_px = min(state.peak_px, mark)

        reference_price = state.peak_px
        if structural_level is not None:
            structural_distance = abs(reference_price - structural_level) / reference_price
            if not math.isfinite(structural_distance):
                raise ValueError(
                    "Structural distance is non-finite; check price magnitudes."
                )

        atr_distance = state.k_sigma * normalized_atr
        if not math.isfinite(atr_distance):
            raise ValueError("k_sigma * atr_norm must be finite.")

        trailing_distance = max(
            state.d_min,
            atr_distance,
            structural_distance,
        )

        if state.side == 1:
            candidate_sl = reference_price * (1.0 - trailing_distance)
            proposed_sl = max(state.current_sl_px, candidate_sl)
            tightening_amount = proposed_sl - state.current_sl_px
        else:
            candidate_sl = reference_price * (1.0 + trailing_distance)
            proposed_sl = min(state.current_sl_px, candidate_sl)
            tightening_amount = state.current_sl_px - proposed_sl

        if math.isnan(proposed_sl):
            raise ValueError("Calculated stop is NaN.")

        minimum_amendment = mark * self.exchange_update_threshold
        has_tightened = tightening_amount > 0.0
        should_update_exchange = (
            has_tightened and tightening_amount >= minimum_amendment
        )

        state.updated_at_ts = time.time()

        if should_update_exchange:
            state.current_sl_px = proposed_sl
            return (
                proposed_sl,
                True,
                "ratchet_tightened_exchange_update_required",
            )

        if has_tightened:
            return (
                proposed_sl,
                False,
                "ratchet_tightened_below_exchange_amendment_threshold",
            )

        return proposed_sl, False, "ratchet_invariant_prevented_stop_loosening"

    def check_breakeven_lock(
        self,
        state: TrailingState,
        mark_price: float,
        fee_buffer: float = 0.0010,
    ) -> Optional[float]:
        """
        Enforce a directional break-even-plus-fees stop after a 1% excursion.

        A long position qualifies when mark price is at least 1% above entry.
        A short position qualifies when mark price is at least 1% below entry.
        The resulting protected stop is:

        - LONG: ``entry_px * (1.0 + fee_buffer)``
        - SHORT: ``entry_px * (1.0 - fee_buffer)``

        The stop is always subject to the same non-loosening ratchet invariant.

        Args:
            state: Mutable trailing-stop state.
            mark_price: Current positive mark price.
            fee_buffer: Non-negative fractional fee/slippage buffer. For
                shorts it must be smaller than 100% to preserve a positive
                stop level.

        Returns:
            The resulting stop level when the break-even condition is active,
            otherwise ``None``.

        Raises:
            TypeError: If ``state`` is not a TrailingState instance.
            ValueError: If supplied values are invalid.
        """
        if not isinstance(state, TrailingState):
            raise TypeError("state must be an instance of TrailingState.")

        self._validate_state(state)

        mark = _positive_float(mark_price, "mark_price")
        fee = _non_negative_float(fee_buffer, "fee_buffer")

        if fee >= 1.0:
            raise ValueError("fee_buffer must be less than 1.0.")

        favorable_excursion = (
            mark / state.entry_px >= 1.010
            if state.side == 1
            else mark / state.entry_px <= 0.990
        )

        if not favorable_excursion:
            return None

        if state.side == 1:
            break_even_stop = state.entry_px * (1.0 + fee)
            if not math.isfinite(break_even_stop):
                raise ValueError("Calculated long break-even stop must be finite.")
            state.current_sl_px = max(state.current_sl_px, break_even_stop)
        else:
            break_even_stop = state.entry_px * (1.0 - fee)
            if not math.isfinite(break_even_stop) or break_even_stop <= 0.0:
                raise ValueError("Calculated short break-even stop must be positive.")
            state.current_sl_px = min(state.current_sl_px, break_even_stop)

        state.updated_at_ts = time.time()
        return state.current_sl_px

    @staticmethod
    def _validate_state(state: TrailingState) -> None:
        """Revalidate mutable state before use."""
        if not isinstance(state.symbol, str) or not state.symbol.strip():
            raise ValueError("state.symbol must be a non-empty string.")

        state.side = _validate_side(state.side)
        state.entry_px = _positive_float(state.entry_px, "state.entry_px")
        state.current_sl_px = _positive_float(
            state.current_sl_px,
            "state.current_sl_px",
        )
        state.peak_px = _positive_float(state.peak_px, "state.peak_px")
        state.d_min = _non_negative_float(state.d_min, "state.d_min")
        state.k_sigma = _non_negative_float(state.k_sigma, "state.k_sigma")
        state.updated_at_ts = _finite_float(
            state.updated_at_ts,
            "state.updated_at_ts",
        )