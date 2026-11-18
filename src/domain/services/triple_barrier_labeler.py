from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Hashable, Optional, Tuple

import numpy as np
import pandas as pd


__all__ = [
    "BarrierSpec",
    "BarrierResult",
    "TripleBarrierEngine",
    "compute_sample_uniqueness",
    "label_single_event",
    "make_barriers",
]


_REQUIRED_BAR_COLUMNS: frozenset[str] = frozenset(
    {"timestamp", "open", "high", "low", "close", "volume"}
)
_REQUIRED_EVENT_COLUMNS: frozenset[str] = frozenset(
    {
        "event_id",
        "symbol",
        "entry_ts",
        "side",
        "entry_px",
        "stop_frac",
        "target_frac",
        "vertical_ts",
    }
)
_VALID_TIE_POLICIES: frozenset[str] = frozenset({"conservative_stop", "censor"})


@dataclass(frozen=True, slots=True)
class BarrierSpec:
    """
    Immutable specification for one triple-barrier labeling event.

    Attributes:
        event_id: Unique, hashable event identifier.
        symbol: Instrument identifier.
        entry_ts: Event entry timestamp.
        side: Trading side, where +1 represents long and -1 represents short.
        entry_px: Positive entry price.
        stop_frac: Positive logarithmic stop distance.
        target_frac: Positive logarithmic target distance.
        vertical_ts: Vertical barrier timestamp, strictly after ``entry_ts``.
    """

    event_id: Hashable
    symbol: str
    entry_ts: pd.Timestamp
    side: int
    entry_px: float
    stop_frac: float
    target_frac: float
    vertical_ts: pd.Timestamp

    def __post_init__(self) -> None:
        try:
            hash(self.event_id)
        except TypeError as exc:
            raise ValueError("event_id must be hashable.") from exc

        if self.event_id is None or pd.isna(self.event_id):
            raise ValueError("event_id must not be null.")

        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string.")

        if isinstance(self.side, bool) or self.side not in (-1, 1):
            raise ValueError("side must be exactly +1 (long) or -1 (short).")

        entry_ts = _coerce_timestamp(self.entry_ts, "entry_ts")
        vertical_ts = _coerce_timestamp(self.vertical_ts, "vertical_ts")

        try:
            if vertical_ts <= entry_ts:
                raise ValueError("vertical_ts must be strictly later than entry_ts.")
        except TypeError as exc:
            raise ValueError(
                "entry_ts and vertical_ts must use compatible timezone conventions."
            ) from exc

        entry_px = _validate_positive_float(self.entry_px, "entry_px")
        stop_frac = _validate_nonnegative_float(self.stop_frac, "stop_frac")
        target_frac = _validate_nonnegative_float(self.target_frac, "target_frac")

        object.__setattr__(self, "entry_ts", entry_ts)
        object.__setattr__(self, "vertical_ts", vertical_ts)
        object.__setattr__(self, "entry_px", entry_px)
        object.__setattr__(self, "stop_frac", stop_frac)
        object.__setattr__(self, "target_frac", target_frac)


@dataclass(frozen=True, slots=True)
class BarrierResult:
    """
    Result of triple-barrier labeling for one event.

    Attributes:
        event_id: Source event identifier.
        label: +1 for target hit, -1 for stop hit, 0 for vertical timeout,
            or None when the event is ambiguous or cannot be labeled safely.
        barrier_ts: Timestamp at which the barrier outcome was observed.
        exit_px: Barrier execution price, final timeout close, or None for
            ambiguous outcomes.
        reason: Machine-readable explanation of the outcome.
        ambiguous: Whether the result is intentionally unlabeled.
    """

    event_id: Hashable
    label: Optional[int]
    barrier_ts: Optional[pd.Timestamp]
    exit_px: Optional[float]
    reason: str
    ambiguous: bool

    def __post_init__(self) -> None:
        try:
            hash(self.event_id)
        except TypeError as exc:
            raise ValueError("event_id must be hashable.") from exc

        if self.label not in (-1, 0, 1, None):
            raise ValueError("label must be one of -1, 0, +1, or None.")

        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string.")

        if self.label is None and not self.ambiguous:
            raise ValueError("label=None requires ambiguous=True.")

        if self.label is not None and self.ambiguous:
            raise ValueError("A resolved label cannot be marked ambiguous.")

        if self.barrier_ts is not None:
            object.__setattr__(
                self,
                "barrier_ts",
                _coerce_timestamp(self.barrier_ts, "barrier_ts"),
            )

        if self.exit_px is not None:
            object.__setattr__(
                self,
                "exit_px",
                _validate_positive_float(self.exit_px, "exit_px"),
            )


def make_barriers(
    entry_px: float,
    side: int,
    atr_norm: float,
    structural_invalidation_px: Optional[float] = None,
    c_win: float = 0.0004,
    c_loss: float = 0.0006,
) -> Tuple[float, float]:
    """
    Construct logarithmic stop and target barrier widths.

    The stop width is the greater of the structural invalidation distance and
    1.8 times normalized ATR, subject to an absolute 1.8% minimum floor.
    The target width explicitly includes estimated trading costs and guarantees
    a gross payoff ratio of at least 2.0 after the supplied cost adjustment.

    Args:
        entry_px: Positive event entry price.
        side: +1 for long or -1 for short.
        atr_norm: Non-negative normalized ATR expressed as a log/relative move.
        structural_invalidation_px: Optional adverse structural invalidation
            price. For a long it must not exceed entry price; for a short it
            must not be below entry price.
        c_win: Estimated round-trip target-side implementation cost.
        c_loss: Estimated round-trip stop-side implementation cost.

    Returns:
        Tuple of ``(stop_frac, target_frac)``.

    Raises:
        ValueError: If an input is invalid or structural invalidation is on the
            favorable side of the entry price.
    """
    validated_entry_px = _validate_positive_float(entry_px, "entry_px")

    if isinstance(side, bool) or side not in (-1, 1):
        raise ValueError("side must be exactly +1 (long) or -1 (short).")

    validated_atr_norm = _validate_nonnegative_float(atr_norm, "atr_norm")
    validated_c_win = _validate_nonnegative_float(c_win, "c_win")
    validated_c_loss = _validate_nonnegative_float(c_loss, "c_loss")

    structural_stop = 0.0
    if structural_invalidation_px is not None:
        invalidation_px = _validate_positive_float(
            structural_invalidation_px,
            "structural_invalidation_px",
        )

        if side == 1:
            if invalidation_px > validated_entry_px:
                raise ValueError(
                    "For long positions, structural_invalidation_px must be "
                    "less than or equal to entry_px."
                )
            structural_stop = log(validated_entry_px / invalidation_px)
        else:
            if invalidation_px < validated_entry_px:
                raise ValueError(
                    "For short positions, structural_invalidation_px must be "
                    "greater than or equal to entry_px."
                )
            structural_stop = log(invalidation_px / validated_entry_px)

    stop_frac = max(structural_stop, 1.8 * validated_atr_norm, 0.018)
    target_frac = 2.0 * (stop_frac + validated_c_loss) + validated_c_win

    if not isfinite(stop_frac) or not isfinite(target_frac):
        raise ValueError("Computed barrier fractions must be finite.")

    return float(stop_frac), float(target_frac)


def label_single_event(
    bars: pd.DataFrame,
    event: BarrierSpec,
    tie_policy: str = "conservative_stop",
) -> BarrierResult:
    """
    Label a single event using high-low triple-barrier logic.

    Bars are evaluated strictly after ``event.entry_ts`` and through
    ``event.vertical_ts`` inclusively. Target and stop contacts are inferred
    from each bar's high and low. When both barriers are touched within the
    same OHLC bar, no intrabar ordering is assumed.

    Args:
        bars: OHLCV DataFrame containing timestamp, open, high, low, close,
            and volume columns.
        event: Triple-barrier event specification.
        tie_policy: ``"conservative_stop"`` assigns simultaneous target/stop
            touches to the stop. ``"censor"`` emits an ambiguous result.

    Returns:
        A ``BarrierResult`` describing the first observed outcome.

    Raises:
        TypeError: If ``bars`` is not a DataFrame or ``event`` is invalid.
        ValueError: If data is malformed, timestamps are incompatible, or the
            tie policy is unsupported.
    """
    if not isinstance(event, BarrierSpec):
        raise TypeError("event must be an instance of BarrierSpec.")

    _validate_tie_policy(tie_policy)
    prepared_bars = _prepare_bars(bars)
    return _label_prepared_event(prepared_bars, event, tie_policy)


def compute_sample_uniqueness(
    events: pd.DataFrame,
    time_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Compute average sample uniqueness over overlapping event horizons.

    For every timestamp in ``time_index``, concurrency is defined as the number
    of event intervals containing that timestamp. Each event's uniqueness is:

        u_i = mean_t(1 / c_t), for t in [entry_ts, barrier_end_ts].

    Both interval endpoints are inclusive. The supplied time index represents
    the discrete observation grid over which the average is calculated.

    Args:
        events: DataFrame containing ``event_id``, ``entry_ts``, and
            ``barrier_end_ts`` columns.
        time_index: Sorted, unique DatetimeIndex defining the concurrency grid.

    Returns:
        Series named ``sample_uniqueness`` indexed by event_id. Events with no
        timestamps represented on ``time_index`` receive NaN.

    Raises:
        TypeError: If arguments have invalid types.
        ValueError: If required fields are absent, timestamps are invalid,
            event identifiers are duplicated, or horizons are malformed.
    """
    if not isinstance(events, pd.DataFrame):
        raise TypeError("events must be a pandas DataFrame.")
    if not isinstance(time_index, pd.DatetimeIndex):
        raise TypeError("time_index must be a pandas DatetimeIndex.")

    required_columns = {"event_id", "entry_ts", "barrier_end_ts"}
    missing_columns = required_columns.difference(events.columns)
    if missing_columns:
        raise ValueError(
            f"events is missing required columns: {sorted(missing_columns)}."
        )

    validated_index = _validate_time_index(time_index)

    if events.empty:
        return pd.Series(
            dtype="float64",
            index=pd.Index([], name="event_id"),
            name="sample_uniqueness",
        )

    event_ids = events["event_id"].tolist()
    _validate_unique_event_ids(event_ids)

    starts = _coerce_datetime_index(events["entry_ts"], "entry_ts")
    ends = _coerce_datetime_index(events["barrier_end_ts"], "barrier_end_ts")

    _assert_index_timezone_compatible(validated_index, starts, "entry_ts")
    _assert_index_timezone_compatible(validated_index, ends, "barrier_end_ts")

    try:
        invalid_horizon = ends < starts
    except TypeError as exc:
        raise ValueError(
            "entry_ts and barrier_end_ts must use compatible timezone conventions."
        ) from exc

    if bool(np.any(invalid_horizon)):
        raise ValueError("barrier_end_ts must not be earlier than entry_ts.")

    n_times = len(validated_index)
    starts_pos = validated_index.searchsorted(starts, side="left")
    ends_pos = validated_index.searchsorted(ends, side="right")

    concurrency_delta = np.zeros(n_times + 1, dtype=np.int64)
    for start_pos, end_pos in zip(starts_pos, ends_pos):
        if start_pos < end_pos:
            concurrency_delta[start_pos] += 1
            concurrency_delta[end_pos] -= 1

    concurrency = np.cumsum(concurrency_delta[:-1])
    inverse_concurrency = np.divide(
        1.0,
        concurrency,
        out=np.zeros(n_times, dtype=float),
        where=concurrency > 0,
    )
    inverse_prefix_sum = np.concatenate(
        (np.array([0.0], dtype=float), np.cumsum(inverse_concurrency))
    )

    uniqueness = np.full(len(events), np.nan, dtype=float)
    for position, (start_pos, end_pos) in enumerate(zip(starts_pos, ends_pos)):
        active_count = int(end_pos - start_pos)
        if active_count > 0:
            uniqueness[position] = (
                inverse_prefix_sum[end_pos] - inverse_prefix_sum[start_pos]
            ) / active_count

    return pd.Series(
        uniqueness,
        index=pd.Index(event_ids, name="event_id"),
        name="sample_uniqueness",
        dtype="float64",
    )


class TripleBarrierEngine:
    """
    High-level dataset labeling engine for institutional triple-barrier workflows.

    The engine validates OHLCV inputs once, labels each event, computes
    signed logarithmic realized returns, and attaches average sample uniqueness.
    """

    def __init__(self, tie_policy: str = "conservative_stop") -> None:
        """
        Initialize the engine.

        Args:
            tie_policy: Simultaneous target/stop policy passed to the event
                labeler. Supported values are ``"conservative_stop"`` and
                ``"censor"``.
        """
        _validate_tie_policy(tie_policy)
        self._tie_policy = tie_policy

    @property
    def tie_policy(self) -> str:
        """Return the configured same-bar target/stop resolution policy."""
        return self._tie_policy

    def label_events_dataset(
        self,
        bars_df: pd.DataFrame,
        events_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Label an events dataset and append barrier-derived outcome columns.

        Added columns are:
            - ``label``: +1 target, -1 stop, 0 timeout, or nullable missing.
            - ``barrier_ts``: Timestamp of first observed barrier outcome.
            - ``exit_px``: Barrier execution price or timeout close.
            - ``return``: Signed logarithmic realized return.
            - ``sample_uniqueness``: Average inverse concurrency per event.

        Events with censored intrabar ties or insufficient post-entry data have
        missing label, exit price, and return values.

        Args:
            bars_df: Valid OHLCV bar DataFrame.
            events_df: Event DataFrame containing all BarrierSpec fields.

        Returns:
            Copy of ``events_df`` with the derived output columns appended.

        Raises:
            TypeError: If inputs are not DataFrames.
            ValueError: If required fields, prices, timestamps, or event IDs
                are invalid.
        """
        if not isinstance(events_df, pd.DataFrame):
            raise TypeError("events_df must be a pandas DataFrame.")

        missing_columns = _REQUIRED_EVENT_COLUMNS.difference(events_df.columns)
        if missing_columns:
            raise ValueError(
                "events_df is missing required columns: "
                f"{sorted(missing_columns)}."
            )

        prepared_bars = _prepare_bars(bars_df)
        events = _build_event_specs(events_df)

        results = [
            _label_prepared_event(prepared_bars, event, self._tie_policy)
            for event in events
        ]

        output = events_df.copy()

        labels = [result.label for result in results]
        barrier_timestamps = [
            result.barrier_ts if result.barrier_ts is not None else pd.NaT
            for result in results
        ]
        exit_prices = [
            result.exit_px if result.exit_px is not None else np.nan
            for result in results
        ]

        realized_returns: list[float] = []
        for event, result in zip(events, results):
            if result.exit_px is None:
                realized_returns.append(np.nan)
            else:
                realized_returns.append(
                    float(event.side * log(result.exit_px / event.entry_px))
                )

        output["label"] = pd.array(labels, dtype="Int64")
        output["barrier_ts"] = barrier_timestamps
        output["exit_px"] = np.asarray(exit_prices, dtype=float)
        output["return"] = np.asarray(realized_returns, dtype=float)

        barrier_end_timestamps = [
            result.barrier_ts
            if result.barrier_ts is not None
            else event.vertical_ts
            for event, result in zip(events, results)
        ]

        uniqueness_events = pd.DataFrame(
            {
                "event_id": [event.event_id for event in events],
                "entry_ts": [event.entry_ts for event in events],
                "barrier_end_ts": barrier_end_timestamps,
            }
        )
        unique_time_index = pd.DatetimeIndex(
            prepared_bars["timestamp"].drop_duplicates().to_numpy()
        )
        uniqueness = compute_sample_uniqueness(
            uniqueness_events,
            unique_time_index,
        )
        uniqueness_by_event_id = uniqueness.to_dict()

        output["sample_uniqueness"] = np.asarray(
            [
                uniqueness_by_event_id[event.event_id]
                for event in events
            ],
            dtype=float,
        )

        return output


def _label_prepared_event(
    bars: pd.DataFrame,
    event: BarrierSpec,
    tie_policy: str,
) -> BarrierResult:
    """Label an event against an already validated, chronological bar frame."""
    bar_timestamps = pd.DatetimeIndex(bars["timestamp"])
    _assert_index_timestamp_compatible(bar_timestamps, event.entry_ts, "entry_ts")
    _assert_index_timestamp_compatible(
        bar_timestamps,
        event.vertical_ts,
        "vertical_ts",
    )

    try:
        target_px, stop_px = _compute_barrier_prices(event)
    except (OverflowError, ValueError) as exc:
        raise ValueError(
            f"Unable to compute finite barrier prices for event_id={event.event_id!r}."
        ) from exc

    eligible_mask = (
        (bars["timestamp"] > event.entry_ts)
        & (bars["timestamp"] <= event.vertical_ts)
    )
    eligible_bars = bars.loc[eligible_mask]

    for row in eligible_bars.itertuples(index=False):
        timestamp = pd.Timestamp(row.timestamp)
        high = float(row.high)
        low = float(row.low)

        if event.side == 1:
            target_touched = high >= target_px
            stop_touched = low <= stop_px
        else:
            target_touched = low <= target_px
            stop_touched = high >= stop_px

        if target_touched and stop_touched:
            if tie_policy == "conservative_stop":
                return BarrierResult(
                    event_id=event.event_id,
                    label=-1,
                    barrier_ts=timestamp,
                    exit_px=stop_px,
                    reason="simultaneous_target_stop_conservative_stop",
                    ambiguous=False,
                )
            return BarrierResult(
                event_id=event.event_id,
                label=None,
                barrier_ts=timestamp,
                exit_px=None,
                reason="simultaneous_target_stop_censored",
                ambiguous=True,
            )

        if target_touched:
            return BarrierResult(
                event_id=event.event_id,
                label=1,
                barrier_ts=timestamp,
                exit_px=target_px,
                reason="target_hit",
                ambiguous=False,
            )

        if stop_touched:
            return BarrierResult(
                event_id=event.event_id,
                label=-1,
                barrier_ts=timestamp,
                exit_px=stop_px,
                reason="stop_hit",
                ambiguous=False,
            )

    latest_bar_timestamp = pd.Timestamp(bars["timestamp"].iloc[-1])
    horizon_covered = latest_bar_timestamp >= event.vertical_ts

    if horizon_covered and not eligible_bars.empty:
        final_bar = eligible_bars.iloc[-1]
        return BarrierResult(
            event_id=event.event_id,
            label=0,
            barrier_ts=pd.Timestamp(final_bar["timestamp"]),
            exit_px=float(final_bar["close"]),
            reason="vertical_timeout",
            ambiguous=False,
        )

    if horizon_covered:
        return BarrierResult(
            event_id=event.event_id,
            label=None,
            barrier_ts=None,
            exit_px=None,
            reason="missing_post_entry_bar_before_vertical",
            ambiguous=True,
        )

    return BarrierResult(
        event_id=event.event_id,
        label=None,
        barrier_ts=None,
        exit_px=None,
        reason="insufficient_data_before_vertical",
        ambiguous=True,
    )


def _compute_barrier_prices(event: BarrierSpec) -> Tuple[float, float]:
    """Compute target and stop prices from event direction and log distances."""
    if event.side == 1:
        target_px = event.entry_px * exp(event.target_frac)
        stop_px = event.entry_px * exp(-event.stop_frac)
    else:
        target_px = event.entry_px * exp(-event.target_frac)
        stop_px = event.entry_px * exp(event.stop_frac)

    if (
        not isfinite(target_px)
        or not isfinite(stop_px)
        or target_px <= 0.0
        or stop_px <= 0.0
    ):
        raise ValueError("Computed target and stop prices must be finite and positive.")

    return float(target_px), float(stop_px)


def _prepare_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Validate, normalize, and chronologically sort an OHLCV bar frame."""
    if not isinstance(bars, pd.DataFrame):
        raise TypeError("bars must be a pandas DataFrame.")

    missing_columns = _REQUIRED_BAR_COLUMNS.difference(bars.columns)
    if missing_columns:
        raise ValueError(
            f"bars is missing required columns: {sorted(missing_columns)}."
        )

    if bars.empty:
        raise ValueError("bars must contain at least one row.")

    prepared = bars.loc[
        :,
        ["timestamp", "open", "high", "low", "close", "volume"],
    ].copy()

    prepared["timestamp"] = _coerce_datetime_series(
        prepared["timestamp"],
        "bars.timestamp",
    )

    for column in ("open", "high", "low", "close", "volume"):
        try:
            numeric_values = pd.to_numeric(prepared[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"bars.{column} must be numeric.") from exc

        array_values = numeric_values.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(array_values).all():
            raise ValueError(f"bars.{column} must contain only finite values.")

        prepared[column] = array_values

    if bool((prepared["high"] < prepared["low"]).any()):
        raise ValueError("Each bar must satisfy high >= low.")

    if bool((prepared[["open", "high", "low", "close"]] <= 0.0).any().any()):
        raise ValueError("OHLC prices must be strictly positive.")

    if bool((prepared["volume"] < 0.0).any()):
        raise ValueError("volume must be non-negative.")

    prepared = prepared.sort_values(
        "timestamp",
        kind="mergesort",
        ignore_index=True,
    )

    return prepared


def _build_event_specs(events_df: pd.DataFrame) -> list[BarrierSpec]:
    """Construct validated BarrierSpec instances from an event DataFrame."""
    specs: list[BarrierSpec] = []
    seen_event_ids: set[Hashable] = set()

    selected = events_df.loc[
        :,
        [
            "event_id",
            "symbol",
            "entry_ts",
            "side",
            "entry_px",
            "stop_frac",
            "target_frac",
            "vertical_ts",
        ],
    ]

    for values in selected.itertuples(index=False, name=None):
        spec = BarrierSpec(
            event_id=values[0],
            symbol=values[1],
            entry_ts=values[2],
            side=values[3],
            entry_px=values[4],
            stop_frac=values[5],
            target_frac=values[6],
            vertical_ts=values[7],
        )

        if spec.event_id in seen_event_ids:
            raise ValueError(
                f"events_df contains duplicate event_id: {spec.event_id!r}."
            )

        seen_event_ids.add(spec.event_id)
        specs.append(spec)

    return specs


def _validate_tie_policy(tie_policy: str) -> None:
    """Validate a simultaneous barrier contact resolution policy."""
    if tie_policy not in _VALID_TIE_POLICIES:
        raise ValueError(
            "tie_policy must be one of "
            f"{sorted(_VALID_TIE_POLICIES)}, got {tie_policy!r}."
        )


def _validate_positive_float(value: object, field_name: str) -> float:
    """Return a finite strictly positive float or raise ValueError."""
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite positive number.") from exc

    if not isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{field_name} must be a finite positive number.")

    return converted


def _validate_nonnegative_float(value: object, field_name: str) -> float:
    """Return a finite non-negative float or raise ValueError."""
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a finite non-negative number."
        ) from exc

    if not isfinite(converted) or converted < 0.0:
        raise ValueError(f"{field_name} must be a finite non-negative number.")

    return converted


def _coerce_timestamp(value: object, field_name: str) -> pd.Timestamp:
    """Coerce a scalar into a valid pandas timestamp."""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid timestamp.") from exc

    if pd.isna(timestamp):
        raise ValueError(f"{field_name} must not be null or NaT.")

    return timestamp


def _coerce_datetime_series(values: pd.Series, field_name: str) -> pd.Series:
    """Coerce a Series to a datetime Series while rejecting invalid values."""
    try:
        converted = pd.to_datetime(values, errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} contains invalid timestamps.") from exc

    if not isinstance(converted, pd.Series):
        converted = pd.Series(converted, index=values.index)

    if converted.isna().any():
        raise ValueError(f"{field_name} must not contain null or NaT values.")

    try:
        pd.DatetimeIndex(converted)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must use a consistent datetime timezone representation."
        ) from exc

    return converted


def _coerce_datetime_index(values: pd.Series, field_name: str) -> pd.DatetimeIndex:
    """Coerce a Series to a non-null DatetimeIndex."""
    converted = _coerce_datetime_series(values, field_name)
    try:
        return pd.DatetimeIndex(converted)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} could not be converted to DatetimeIndex.") from exc


def _validate_time_index(time_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Validate a sorted, unique, non-null datetime concurrency grid."""
    if time_index.hasnans:
        raise ValueError("time_index must not contain NaT values.")

    if not time_index.is_monotonic_increasing:
        raise ValueError("time_index must be sorted in ascending order.")

    if not time_index.is_unique:
        raise ValueError("time_index must contain unique timestamps.")

    return time_index


def _validate_unique_event_ids(event_ids: list[object]) -> None:
    """Validate event identifier hashability, nullability, and uniqueness."""
    seen: set[Hashable] = set()

    for event_id in event_ids:
        try:
            hash(event_id)
        except TypeError as exc:
            raise ValueError("All event_id values must be hashable.") from exc

        if event_id is None or pd.isna(event_id):
            raise ValueError("All event_id values must be non-null.")

        if event_id in seen:
            raise ValueError(f"events contains duplicate event_id: {event_id!r}.")

        seen.add(event_id)


def _assert_index_timestamp_compatible(
    index: pd.DatetimeIndex,
    timestamp: pd.Timestamp,
    field_name: str,
) -> None:
    """Ensure a scalar timestamp and DatetimeIndex are both naive or aware."""
    index_is_aware = index.tz is not None
    timestamp_is_aware = timestamp.tz is not None

    if index_is_aware != timestamp_is_aware:
        raise ValueError(
            f"{field_name} timezone convention is incompatible with bars.timestamp."
        )


def _assert_index_timezone_compatible(
    index: pd.DatetimeIndex,
    other_index: pd.DatetimeIndex,
    field_name: str,
) -> None:
    """Ensure two DatetimeIndexes are both naive or both timezone aware."""
    index_is_aware = index.tz is not None
    other_is_aware = other_index.tz is not None

    if index_is_aware != other_is_aware:
        raise ValueError(
            f"{field_name} timezone convention is incompatible with time_index."
        )