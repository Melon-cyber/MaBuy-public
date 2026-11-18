"""Institutional triple-barrier retraining pipeline for hourly trading models.

This module implements:
- López de Prado triple-barrier event labeling.
- Sample uniqueness weighting.
- Feature engineering via the domain FeatureEngineer.
- Purged, embargoed expanding walk-forward validation.
- Probability calibration and calibration diagnostics.
- Atomic model deployment guarded by genuine OOS performance metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from typing import Any, Callable, Iterator, Mapping, Sequence, Optional, Tuple, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss

from src.domain.services.triple_barrier_labeler import (
    compute_sample_uniqueness,
    label_single_event,
    make_barriers,
)

LOGGER = logging.getLogger(__name__)

HOUR_SECONDS: int = 3_600
VERTICAL_HORIZON_BARS: int = 24
EMBARGO_BARS: int = 24
WALK_FORWARD_FOLDS: int = 5

MAKER_ROUNDTRIP_FEE: float = 0.0004
SLIPPAGE_COST: float = 0.0002
ROUNDTRIP_FRICTION: float = MAKER_ROUNDTRIP_FEE + SLIPPAGE_COST

STOP_FLOOR: float = 0.018
PAYOFF_RATIO: float = 2.0

ANNUALIZATION_FACTOR: float = math.sqrt(365.0 * 24.0)
RANDOM_STATE: int = 42

RAW_CANDLE_COLUMNS: frozenset[str] = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "timestamp",
        "datetime",
        "date",
        "time",
        "symbol",
        "asset",
        "pair",
        "interval",
        "timeframe",
    }
)

LABEL_COLUMNS: frozenset[str] = frozenset(
    {
        "label",
        "target",
        "target_label",
        "triple_barrier_label",
        "outcome",
        "event_label",
        "y",
        "class",
        "signal",
        "future_return",
        "next_return",
        "return_forward",
    }
)


class TripleBarrierPipelineError(RuntimeError):
    """Raised when the retraining pipeline cannot produce an auditable model."""


@dataclass(frozen=True)
class BarrierEvent:
    """A fully specified triple-barrier event."""

    start_position: int
    end_position: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    entry_price: float
    upper_barrier: float
    lower_barrier: float
    stop_fraction: float
    target_fraction: float
    label: int
    gross_return: float


@dataclass(frozen=True)
class OOSMetrics:
    """Out-of-sample metrics used by the deployment guard and registry."""

    net_sharpe: float
    realized_win_rate: float
    profit_factor: float | None
    net_expectancy: float
    brier_score: float
    log_loss: float
    expected_calibration_error: float
    sample_count: int
    trade_count: int
    positive_label_rate: float

    def as_registry_dict(self) -> dict[str, float | int | None]:
        """Return JSON-safe registry metrics."""
        return {
            "net_sharpe": _json_safe_float(self.net_sharpe),
            "realized_win_rate": _json_safe_float(self.realized_win_rate),
            "profit_factor": _json_safe_float(self.profit_factor),
            "net_expectancy": _json_safe_float(self.net_expectancy),
            "brier_score": _json_safe_float(self.brier_score),
            "log_loss": _json_safe_float(self.log_loss),
            "expected_calibration_error": _json_safe_float(
                self.expected_calibration_error
            ),
            "sample_count": self.sample_count,
            "trade_count": self.trade_count,
            "positive_label_rate": _json_safe_float(self.positive_label_rate),
        }


@dataclass(frozen=True)
class WalkForwardResult:
    """Walk-forward predictions and calculated OOS diagnostics."""

    predictions: pd.DataFrame
    metrics: OOSMetrics


@dataclass(frozen=True)
class TrainingResult:
    """Result of a complete symbol training run."""

    symbol: str
    deployed: bool
    model_path: Path | None
    calibrator_path: Path | None
    metrics: OOSMetrics
    feature_columns: tuple[str, ...]
    calibration_method: str


def _json_safe_float(value: float | None) -> float | None:
    """Convert non-finite values to JSON-safe null values."""
    if value is None:
        return None
    numeric_value = float(value)
    return numeric_value if math.isfinite(numeric_value) else None


def _normalise_symbol(symbol: str) -> str:
    """Normalize a trading symbol for filesystem-safe model names."""
    normalized = symbol.strip().upper().replace("/", "_").replace("-", "_")
    if not normalized:
        raise TripleBarrierPipelineError("A non-empty symbol is required.")
    if any(character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for character in normalized):
        raise TripleBarrierPipelineError(
            f"Symbol contains unsupported filesystem characters: {symbol!r}"
        )
    return normalized


def _load_feature_engineer_class() -> type[Any]:
    """Resolve the domain FeatureEngineer without silently substituting features."""
    candidates: tuple[tuple[str, str], ...] = (
        ("src.infrastructure.mlops.weekly_retrain_pipeline", "FeatureEngineer"),
        ("src.domain.services.feature_engineer", "FeatureEngineer"),
        ("src.domain.services.feature_engineering", "FeatureEngineer"),
        ("src.domain.features.feature_engineer", "FeatureEngineer"),
    )

    failures: list[str] = []
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            feature_engineer = getattr(module, class_name)
            if inspect.isclass(feature_engineer):
                return feature_engineer
            failures.append(f"{module_name}.{class_name} is not a class")
        except (ImportError, AttributeError) as exc:
            failures.append(f"{module_name}: {exc}")

    raise TripleBarrierPipelineError(
        "FeatureEngineer could not be imported. The institutional feature service "
        f"is required. Attempts: {'; '.join(failures)}"
    )


def _prepare_candles(candles: pd.DataFrame, max_candles: int | None = None) -> pd.DataFrame:
    """Validate, normalize, sort, and sanitize hourly OHLCV candles."""
    if not isinstance(candles, pd.DataFrame):
        raise TripleBarrierPipelineError("Candles must be supplied as a pandas DataFrame.")
    if candles.empty:
        raise TripleBarrierPipelineError("No candles were supplied.")

    normalized = candles.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]

    timestamp_column: str | None = None
    for candidate in ("timestamp", "datetime", "date", "time", "open_time"):
        if candidate in normalized.columns:
            timestamp_column = candidate
            break

    if timestamp_column is not None:
        parsed_index = pd.to_datetime(normalized[timestamp_column], utc=True, errors="coerce")
        normalized = normalized.drop(columns=[timestamp_column])
        normalized.index = parsed_index
    else:
        parsed_index = pd.to_datetime(normalized.index, utc=True, errors="coerce")
        normalized.index = parsed_index

    if normalized.index.isna().any():
        raise TripleBarrierPipelineError("Candles contain invalid timestamps.")

    required_columns: tuple[str, ...] = ("open", "high", "low", "close", "volume")
    missing_columns = [column for column in required_columns if column not in normalized.columns]
    if missing_columns:
        raise TripleBarrierPipelineError(
            f"Candle data is missing mandatory columns: {', '.join(missing_columns)}"
        )

    for column in required_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.loc[~normalized.index.duplicated(keep="last")].sort_index()
    normalized = normalized.dropna(subset=list(required_columns))

    if normalized.empty:
        raise TripleBarrierPipelineError("No valid OHLCV candles remain after cleaning.")

    invalid_prices = (
        (normalized["open"] <= 0.0)
        | (normalized["high"] <= 0.0)
        | (normalized["low"] <= 0.0)
        | (normalized["close"] <= 0.0)
        | (normalized["high"] < normalized["low"])
    )
    if invalid_prices.any():
        bad_count = int(invalid_prices.sum())
        raise TripleBarrierPipelineError(
            f"Candle data contains {bad_count} invalid OHLC observations."
        )

    if (normalized["volume"] < 0.0).any():
        raise TripleBarrierPipelineError("Candle data contains negative volume.")

    if max_candles is not None:
        if max_candles <= 0:
            raise TripleBarrierPipelineError("--max-candles must be a positive integer.")
        normalized = normalized.tail(max_candles)

    minimum_required = (WALK_FORWARD_FOLDS + 1) * 30 + VERTICAL_HORIZON_BARS
    if len(normalized) < minimum_required:
        raise TripleBarrierPipelineError(
            f"Insufficient candles ({len(normalized)}). At least "
            f"{minimum_required} hourly candles are required."
        )

    inferred_deltas = normalized.index.to_series().diff().dropna()
    if not inferred_deltas.empty:
        median_delta = inferred_deltas.median()
        if median_delta > pd.Timedelta(hours=2):
            LOGGER.warning(
                "Median candle interval is %s, substantially larger than one hour.",
                median_delta,
            )

    return normalized


def _invoke_callable_with_context(
    function: Callable[..., Any],
    context: Mapping[str, Any],
) -> Any:
    """Invoke a domain service using signature-aware semantic argument mapping."""
    signature = inspect.signature(function)
    args: list[Any] = []
    kwargs: dict[str, Any] = {}

    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        value_found, value = _resolve_semantic_parameter(parameter.name, context)
        if not value_found:
            if parameter.default is not inspect.Parameter.empty:
                continue
            raise TypeError(
                f"Cannot map required parameter {parameter.name!r} for "
                f"{getattr(function, '__name__', repr(function))}."
            )

        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[parameter.name] = value

    return function(*args, **kwargs)


def _resolve_semantic_parameter(
    parameter_name: str,
    context: Mapping[str, Any],
) -> tuple[bool, Any]:
    """Resolve common domain-service parameter names to semantic event data."""
    normalized = parameter_name.lower().strip()

    if normalized in context:
        return True, context[normalized]

    aliases: dict[str, str] = {
        "price": "entry_price",
        "entry": "entry_price",
        "initial_price": "entry_price",
        "entry_px": "entry_price",
        "pt": "target_fraction",
        "pt_frac": "target_fraction",
        "profit_take_fraction": "target_fraction",
        "profit_take_frac": "target_fraction",
        "target_frac": "target_fraction",
        "target": "target_fraction",
        "sl": "stop_fraction",
        "sl_frac": "stop_fraction",
        "stop_loss_fraction": "stop_fraction",
        "stop_loss_frac": "stop_fraction",
        "stop_frac": "stop_fraction",
        "stop": "stop_fraction",
        "upper": "upper_barrier",
        "upper_price": "upper_barrier",
        "target_price": "upper_barrier",
        "profit_take_price": "upper_barrier",
        "lower": "lower_barrier",
        "lower_price": "lower_barrier",
        "stop_price": "lower_barrier",
        "stop_loss_price": "lower_barrier",
        "horizon": "horizon_bars",
        "vertical_horizon": "horizon_bars",
        "vertical_barrier": "vertical_end_time",
        "vertical_end": "vertical_end_time",
        "t1": "vertical_end_time",
        "event_idx": "entry_position",
        "event_index": "entry_position",
        "entry_idx": "entry_position",
        "entry_index": "entry_position",
        "index": "entry_position",
        "idx": "entry_position",
        "timestamp": "entry_time",
        "time": "entry_time",
        "event_time": "entry_time",
        "data": "candles",
        "dataframe": "candles",
        "frame": "candles",
        "df": "candles",
        "ohlcv": "candles",
        "events": "events",
        "event": "event",
        "barriers": "barriers",
        "highs": "future_high",
        "lows": "future_low",
        "closes": "future_close",
        "future_highs": "future_high",
        "future_lows": "future_low",
        "future_closes": "future_close",
    }

    semantic_name = aliases.get(normalized)
    if semantic_name is not None and semantic_name in context:
        return True, context[semantic_name]

    if normalized == "high" and "future_high" in context:
        return True, context["future_high"]
    if normalized == "low" and "future_low" in context:
        return True, context["future_low"]
    if normalized == "close" and "future_close" in context:
        return True, context["future_close"]

    return False, None


def _parse_barriers(
    barrier_result: Any,
    entry_price: float,
    fallback_upper: float,
    fallback_lower: float,
) -> tuple[float, float]:
    """Parse common barrier-service return formats into upper and lower prices."""
    upper: float | None = None
    lower: float | None = None

    if isinstance(barrier_result, Mapping):
        lowered = {str(key).lower(): value for key, value in barrier_result.items()}
        for key in (
            "upper_barrier",
            "upper",
            "target_price",
            "profit_take",
            "pt",
            "target",
        ):
            if key in lowered:
                upper = float(lowered[key])
                break
        for key in (
            "lower_barrier",
            "lower",
            "stop_price",
            "stop_loss",
            "sl",
            "stop",
        ):
            if key in lowered:
                lower = float(lowered[key])
                break
    elif isinstance(barrier_result, pd.Series):
        return _parse_barriers(
            barrier_result.to_dict(),
            entry_price,
            fallback_upper,
            fallback_lower,
        )
    elif isinstance(barrier_result, np.ndarray | list | tuple):
        values = list(barrier_result)
        if len(values) >= 2:
            val0, val1 = float(values[0]), float(values[1])
            if val0 < 1.0 and val1 < 1.0:
                # Returned as fractions (stop_frac, target_frac)
                upper = entry_price * (1.0 + val1)
                lower = entry_price * (1.0 - val0)
            else:
                upper = max(val0, val1)
                lower = min(val0, val1)

    if upper is None or lower is None or not math.isfinite(upper) or not math.isfinite(lower):
        return fallback_upper, fallback_lower

    if upper <= entry_price or lower >= entry_price or lower <= 0.0:
        return fallback_upper, fallback_lower

    effective_stop = (entry_price / lower) - 1.0
    effective_target = (upper / entry_price) - 1.0
    effective_payoff = (
        (effective_target - ROUNDTRIP_FRICTION)
        / (effective_stop + ROUNDTRIP_FRICTION)
    )

    if effective_stop < STOP_FLOOR or effective_payoff < PAYOFF_RATIO:
        return fallback_upper, fallback_lower

    return upper, lower


def _normalise_label(value: Any) -> int | None:
    """Normalize a label-service result to the triple-barrier label domain."""
    candidate: Any = value

    if isinstance(candidate, Mapping):
        lowered = {str(key).lower(): item for key, item in candidate.items()}
        for key in ("label", "target_label", "outcome", "class", "y"):
            if key in lowered:
                candidate = lowered[key]
                break
    elif isinstance(candidate, pd.Series):
        return _normalise_label(candidate.to_dict())
    elif isinstance(candidate, np.ndarray | list | tuple):
        if len(candidate) == 0:
            return None
        candidate = list(candidate)[0]

    if isinstance(candidate, str):
        normalized = candidate.strip().lower()
        if normalized in {"target", "target_hit", "profit", "win", "+1", "1"}:
            return 1
        if normalized in {"stop", "stop_loss", "loss", "-1"}:
            return -1
        if normalized in {"timeout", "vertical", "vertical_timeout", "0", "neutral"}:
            return 0

    try:
        numeric = float(candidate)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(numeric):
        return None
    if numeric > 0.0:
        return 1
    if numeric < 0.0:
        return -1
    return 0


def _deterministic_event_outcome(
    candles: pd.DataFrame,
    entry_position: int,
    upper_barrier: float,
    lower_barrier: float,
) -> tuple[int, int, float]:
    """Calculate first-touch triple-barrier outcome with target-first tie handling."""
    final_position = min(entry_position + VERTICAL_HORIZON_BARS, len(candles) - 1)
    future = candles.iloc[entry_position + 1 : final_position + 1]

    for offset, (_, candle) in enumerate(future.iterrows(), start=entry_position + 1):
        target_hit = float(candle["high"]) >= upper_barrier
        stop_hit = float(candle["low"]) <= lower_barrier

        if target_hit and stop_hit:
            return 0, offset, float(candle["close"]) / float(
                candles["close"].iloc[entry_position]
            ) - 1.0
        if target_hit:
            return 1, offset, upper_barrier / float(candles["close"].iloc[entry_position]) - 1.0
        if stop_hit:
            return -1, offset, lower_barrier / float(candles["close"].iloc[entry_position]) - 1.0

    timeout_close = float(candles["close"].iloc[final_position])
    entry_price = float(candles["close"].iloc[entry_position])
    return 0, final_position, timeout_close / entry_price - 1.0


def _build_triple_barrier_events(candles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create 24-hour triple-barrier events using the domain labeler services."""
    close = candles["close"].astype(float)
    log_returns = np.log(close).diff()
    rolling_volatility = log_returns.rolling(20, min_periods=20).std()

    event_records: list[dict[str, Any]] = []
    max_entry_position = len(candles) - VERTICAL_HORIZON_BARS - 1

    for entry_position in range(20, max_entry_position + 1):
        entry_price = float(close.iloc[entry_position])
        volatility = float(rolling_volatility.iloc[entry_position])

        if not math.isfinite(volatility) or volatility <= 0.0:
            continue

        stop_fraction = max(STOP_FLOOR, 2.0 * volatility)
        target_fraction = (
            PAYOFF_RATIO * (stop_fraction + ROUNDTRIP_FRICTION) + ROUNDTRIP_FRICTION
        )

        fallback_upper = entry_price * (1.0 + target_fraction)
        fallback_lower = entry_price * (1.0 - stop_fraction)
        entry_time = pd.Timestamp(candles.index[entry_position])
        vertical_end_position = entry_position + VERTICAL_HORIZON_BARS
        vertical_end_time = pd.Timestamp(candles.index[vertical_end_position])

        barrier_context: dict[str, Any] = {
            "entry_px": entry_price,
            "entry_price": entry_price,
            "side": 1,
            "atr_norm": volatility,
            "target_fraction": target_fraction,
            "stop_fraction": stop_fraction,
            "c_loss": ROUNDTRIP_FRICTION,
            "c_win": ROUNDTRIP_FRICTION,
            "horizon_bars": VERTICAL_HORIZON_BARS,
            "entry_position": entry_position,
            "entry_time": entry_time,
            "vertical_end_time": vertical_end_time,
            "candles": candles,
            "bars": candles,
        }

        try:
            barrier_result = _invoke_callable_with_context(make_barriers, barrier_context)
            upper_barrier, lower_barrier = _parse_barriers(
                barrier_result=barrier_result,
                entry_price=entry_price,
                fallback_upper=fallback_upper,
                fallback_lower=fallback_lower,
            )
        except Exception as exc:
            LOGGER.warning(
                "make_barriers failed for %s; using mathematically equivalent barrier "
                "construction. Error: %s",
                entry_time,
                exc,
            )
            upper_barrier, lower_barrier = fallback_upper, fallback_lower

        deterministic_label, exit_position, gross_return = _deterministic_event_outcome(
            candles=candles,
            entry_position=entry_position,
            upper_barrier=upper_barrier,
            lower_barrier=lower_barrier,
        )

        future_slice = candles.iloc[entry_position + 1 : vertical_end_position + 1]
        event_payload: dict[str, Any] = {
            "entry_position": entry_position,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "upper_barrier": upper_barrier,
            "lower_barrier": lower_barrier,
            "vertical_end_time": vertical_end_time,
        }

        label_context: dict[str, Any] = {
            "entry_price": entry_price,
            "target_fraction": target_fraction,
            "stop_fraction": stop_fraction,
            "upper_barrier": upper_barrier,
            "lower_barrier": lower_barrier,
            "horizon_bars": VERTICAL_HORIZON_BARS,
            "entry_position": entry_position,
            "entry_time": entry_time,
            "vertical_end_time": vertical_end_time,
            "future_high": future_slice["high"],
            "future_low": future_slice["low"],
            "future_close": future_slice["close"],
            "candles": candles,
            "event": event_payload,
            "barriers": {
                "upper_barrier": upper_barrier,
                "lower_barrier": lower_barrier,
                "vertical_barrier": vertical_end_time,
            },
        }

        external_label: int | None = None
        try:
            external_label = _normalise_label(
                _invoke_callable_with_context(label_single_event, label_context)
            )
        except Exception as exc:
            LOGGER.warning(
                "label_single_event failed for %s; using deterministic first-touch "
                "evaluation. Error: %s",
                entry_time,
                exc,
            )

        label = deterministic_label if external_label is None else external_label

        if label == 1:
            gross_return = upper_barrier / entry_price - 1.0
        elif label == -1:
            gross_return = lower_barrier / entry_price - 1.0

        event_records.append(
            {
                "event_id": len(event_records),
                "entry_position": entry_position,
                "exit_position": exit_position,
                "start": entry_time,
                "t1": pd.Timestamp(candles.index[exit_position]),
                "entry_price": entry_price,
                "upper_barrier": upper_barrier,
                "lower_barrier": lower_barrier,
                "stop_fraction": stop_fraction,
                "target_fraction": target_fraction,
                "label": int(label),
                "gross_return": float(gross_return),
            }
        )

    events = pd.DataFrame(event_records)
    if events.empty:
        raise TripleBarrierPipelineError(
            "No valid triple-barrier events were created from the supplied candles."
        )

    if not set(events["label"].unique()).issubset({-1, 0, 1}):
        raise TripleBarrierPipelineError("Triple-barrier labels escaped the {-1, 0, 1} domain.")

    events.index = events["entry_position"].astype(int)
    events.index.name = "entry_position"
    return events, pd.DataFrame(event_records)


def _fallback_sample_uniqueness(events: pd.DataFrame) -> pd.Series:
    """Calculate average inverse concurrency uniqueness for interval events."""
    if events.empty:
        return pd.Series(dtype=float)

    first_position = int(events["entry_position"].min())
    last_position = int(events["exit_position"].max())

    concurrency = np.zeros(last_position - first_position + 1, dtype=float)
    for event in events.itertuples():
        start = int(event.entry_position) - first_position
        end = int(event.exit_position) - first_position
        concurrency[start : end + 1] += 1.0

    weights: list[float] = []
    for event in events.itertuples():
        start = int(event.entry_position) - first_position
        end = int(event.exit_position) - first_position
        active_concurrency = concurrency[start : end + 1]
        weights.append(float(np.mean(1.0 / np.maximum(active_concurrency, 1.0))))

    output = pd.Series(weights, index=events.index, dtype=float)
    return output.clip(lower=1e-8)


def _compute_uniqueness_weights(events: pd.DataFrame) -> pd.Series:
    """Compute uniqueness weights through the domain service with strict validation."""
    uniqueness_context: dict[str, Any] = {
        "events": events,
        "candles": events,
        "dataframe": events,
        "frame": events,
    }

    try:
        result = _invoke_callable_with_context(compute_sample_uniqueness, uniqueness_context)
        if isinstance(result, pd.DataFrame):
            for candidate in ("uniqueness", "sample_uniqueness", "weight", "weights"):
                if candidate in result.columns:
                    result = result[candidate]
                    break

        if isinstance(result, pd.Series):
            weights = result.reindex(events.index)
        else:
            values = np.asarray(result, dtype=float).reshape(-1)
            if len(values) != len(events):
                raise ValueError(
                    f"Expected {len(events)} uniqueness weights, received {len(values)}."
                )
            weights = pd.Series(values, index=events.index, dtype=float)

        weights = pd.to_numeric(weights, errors="coerce")
        if weights.isna().any() or (weights <= 0.0).any():
            raise ValueError("Uniqueness weights contain non-positive or missing values.")

        return weights.clip(lower=1e-8)
    except Exception as exc:
        LOGGER.warning(
            "compute_sample_uniqueness failed; using interval-concurrency uniqueness. "
            "Error: %s",
            exc,
        )
        return _fallback_sample_uniqueness(events)


def _manual_canonical_features(candles: pd.DataFrame) -> pd.DataFrame:
    """Construct canonical audited features used alongside FeatureEngineer output."""
    close = candles["close"].astype(float)
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    volume = candles["volume"].astype(float)

    log_return = np.log(close).diff()
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr_14 = true_range.rolling(14, min_periods=14).mean()
    delta = close.diff()
    gains = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    losses = (-delta.clip(upper=0.0)).rolling(14, min_periods=14).mean()
    relative_strength = gains / losses.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))

    ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_fast - ema_slow

    rolling_mean_20 = close.rolling(20, min_periods=20).mean()
    rolling_std_20 = close.rolling(20, min_periods=20).std()
    upper_band = rolling_mean_20 + 2.0 * rolling_std_20
    lower_band = rolling_mean_20 - 2.0 * rolling_std_20

    features = pd.DataFrame(index=candles.index)
    for lag in (1, 3, 6, 12, 24):
        features[f"momentum_{lag}"] = close.pct_change(lag)

    features["volatility_20"] = log_return.rolling(20, min_periods=20).std()
    features["volatility_60"] = log_return.rolling(60, min_periods=60).std()
    features["z_return_20"] = (
        (log_return - log_return.rolling(20, min_periods=20).mean())
        / log_return.rolling(20, min_periods=20).std()
    )
    features["rsi_14"] = rsi / 100.0
    features["normalized_macd"] = macd / atr_14.replace(0.0, np.nan)
    features["normalized_atr"] = atr_14 / close.replace(0.0, np.nan)
    features["bollinger_percent_b"] = (
        (close - lower_band) / (upper_band - lower_band).replace(0.0, np.nan)
    )
    features["volume_log_return"] = np.log1p(volume).diff()

    return features.replace([np.inf, -np.inf], np.nan)


def _run_feature_engineer(candles: pd.DataFrame) -> pd.DataFrame:
    """Execute the domain FeatureEngineer and normalize its returned feature frame."""
    feature_engineer_class = _load_feature_engineer_class()

    try:
        engineer = feature_engineer_class(candles.copy())
    except TypeError:
        try:
            engineer = feature_engineer_class()
        except Exception as exc:
            raise TripleBarrierPipelineError(
                f"Unable to initialize FeatureEngineer: {exc}"
            ) from exc
    except Exception as exc:
        raise TripleBarrierPipelineError(
            f"Unable to initialize FeatureEngineer: {exc}"
        ) from exc

    result: Any = None
    method_names: tuple[str, ...] = (
        "compute_features",
        "transform",
        "create_features",
        "build_features",
        "engineer_features",
        "generate_features",
        "calculate_features",
        "fit_transform",
    )

    candles_input = candles.copy()
    candles_input.index.name = None
    if "timestamp" not in candles_input.columns:
        candles_input["timestamp"] = candles.index

    for method_name in method_names:
        method = getattr(engineer, method_name, None)
        if not callable(method):
            continue

        try:
            signature = inspect.signature(method)
            required_parameters = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.default is inspect.Parameter.empty
                and parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            ]
            result = method(candles_input.copy()) if required_parameters else method()
            break
        except Exception as exc:
            LOGGER.warning("FeatureEngineer.%s failed: %s", method_name, exc)

    if isinstance(result, (tuple, list)) and len(result) > 0 and isinstance(result[0], pd.DataFrame):
        result = result[0]

    if result is None and isinstance(engineer, pd.DataFrame):
        result = engineer

    if result is None:
        for attribute_name in ("features", "feature_frame", "data", "df"):
            candidate = getattr(engineer, attribute_name, None)
            if isinstance(candidate, pd.DataFrame):
                result = candidate
                break

    if not isinstance(result, pd.DataFrame):
        raise TripleBarrierPipelineError(
            "FeatureEngineer did not return a pandas DataFrame through a supported "
            "feature-generation interface."
        )

    feature_frame = result.copy()
    feature_frame.columns = [str(column).strip().lower() for column in feature_frame.columns]

    if not isinstance(feature_frame.index, pd.DatetimeIndex):
        if "timestamp" in feature_frame.columns:
            feature_frame.index = pd.to_datetime(feature_frame["timestamp"], utc=True)
        elif len(feature_frame) == len(candles):
            feature_frame.index = candles.index
        elif len(feature_frame) < len(candles):
            offset = len(candles) - len(feature_frame)
            feature_frame.index = candles.index[offset:]
        else:
            raise TripleBarrierPipelineError(
                "FeatureEngineer returned a frame without a datetime index and a "
                "row count inconsistent with candle data."
            )
    else:
        feature_frame.index = pd.to_datetime(feature_frame.index, utc=True, errors="coerce")
        feature_frame = feature_frame.loc[~feature_frame.index.isna()]

    return feature_frame.reindex(candles.index)


def _build_feature_matrix(candles: pd.DataFrame) -> pd.DataFrame:
    """Build the audited feature matrix from FeatureEngineer and canonical features."""
    engineered_features = _run_feature_engineer(candles)
    canonical_features = _manual_canonical_features(candles)

    accepted_engineered: dict[str, pd.Series] = {}
    for column in engineered_features.columns:
        normalized_name = str(column).strip().lower()
        if normalized_name in RAW_CANDLE_COLUMNS or normalized_name in LABEL_COLUMNS:
            continue

        series = pd.to_numeric(engineered_features[column], errors="coerce")
        if series.notna().sum() == 0:
            continue

        safe_name = f"fe_{normalized_name}"
        accepted_engineered[safe_name] = series

    combined = pd.DataFrame(accepted_engineered, index=candles.index)
    for column in canonical_features.columns:
        combined[column] = canonical_features[column]

    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined = combined.loc[:, ~combined.columns.duplicated()].copy()

    if combined.empty:
        raise TripleBarrierPipelineError("Feature matrix is empty after feature engineering.")

    required_canonical_columns = {
        "volatility_20",
        "volatility_60",
        "z_return_20",
        "rsi_14",
        "normalized_macd",
        "normalized_atr",
        "bollinger_percent_b",
        "volume_log_return",
    }
    missing_required = required_canonical_columns.difference(combined.columns)
    if missing_required:
        raise TripleBarrierPipelineError(
            f"Required audited features are missing: {sorted(missing_required)}"
        )

    return combined


def _align_training_data(
    features: pd.DataFrame,
    events: pd.DataFrame,
    uniqueness: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Align event labels, uniqueness weights, and feature availability."""
    event_positions = events["entry_position"].astype(int)
    event_times = features.index[event_positions.to_numpy()]

    aligned_features = features.loc[event_times].copy()
    aligned_features.index = events.index

    valid_mask = (
        aligned_features.notna().all(axis=1)
        & events["label"].isin([-1, 0, 1])
        & uniqueness.reindex(events.index).notna()
        & (uniqueness.reindex(events.index) > 0.0)
    )

    aligned_features = aligned_features.loc[valid_mask]
    labels = events.loc[valid_mask, "label"].astype(int)
    weights = uniqueness.loc[valid_mask].astype(float)
    aligned_events = events.loc[valid_mask].copy()

    if len(aligned_features) < 180:
        raise TripleBarrierPipelineError(
            f"Only {len(aligned_features)} complete event-feature observations remain; "
            "at least 180 are required."
        )

    if labels.nunique() < 2:
        raise TripleBarrierPipelineError(
            "Triple-barrier labels contain fewer than two classes after alignment."
        )

    if int((labels == 1).sum()) < 10:
        raise TripleBarrierPipelineError(
            "Insufficient target-hit events to estimate P(Target Hit) reliably."
        )

    return aligned_features, labels, weights, aligned_events


def _make_classifier() -> HistGradientBoostingClassifier:
    """Create the approved institutional primary classifier."""
    return HistGradientBoostingClassifier(
        max_iter=200,
        min_samples_leaf=20,
        l2_regularization=1.5,
        random_state=RANDOM_STATE,
    )


def _choose_calibration_method(labels: pd.Series) -> str:
    """Use isotonic where statistically justified and sigmoid for limited samples."""
    class_counts = labels.value_counts()
    minimum_class_count = int(class_counts.min()) if not class_counts.empty else 0

    if len(labels) >= 500 and minimum_class_count >= 30:
        return "isotonic"
    return "sigmoid"


def _fit_calibrator(
    features: pd.DataFrame,
    labels: pd.Series,
    sample_weight: pd.Series,
) -> tuple[HistGradientBoostingClassifier, CalibratedClassifierCV, str]:
    """Fit the approved classifier and a three-fold probability calibrator."""
    if labels.nunique() < 2:
        raise TripleBarrierPipelineError("Calibration requires at least two label classes.")

    class_counts = labels.value_counts()
    if int(class_counts.min()) < 3:
        raise TripleBarrierPipelineError(
            "Three-fold calibration requires at least three observations in every "
            "observed training class."
        )

    model = _make_classifier()
    try:
        model.fit(features, labels, sample_weight=sample_weight)
    except Exception as exc:
        raise TripleBarrierPipelineError(f"Primary model fitting failed: {exc}") from exc

    calibration_method = _choose_calibration_method(labels)

    try:
        calibrator = CalibratedClassifierCV(
            estimator=_make_classifier(),
            method=calibration_method,
            cv=3,
        )
    except TypeError:
        calibrator = CalibratedClassifierCV(
            base_estimator=_make_classifier(),
            method=calibration_method,
            cv=3,
        )

    try:
        calibrator.fit(features, labels, sample_weight=sample_weight)
    except Exception as exc:
        if calibration_method == "isotonic":
            LOGGER.warning(
                "Isotonic calibration failed; falling back to sigmoid calibration: %s",
                exc,
            )
            calibration_method = "sigmoid"
            try:
                calibrator = CalibratedClassifierCV(
                    estimator=_make_classifier(),
                    method="sigmoid",
                    cv=3,
                )
            except TypeError:
                calibrator = CalibratedClassifierCV(
                    base_estimator=_make_classifier(),
                    method="sigmoid",
                    cv=3,
                )

            try:
                calibrator.fit(features, labels, sample_weight=sample_weight)
            except Exception as fallback_exc:
                raise TripleBarrierPipelineError(
                    f"Sigmoid calibration failed after isotonic fallback: {fallback_exc}"
                ) from fallback_exc
        else:
            raise TripleBarrierPipelineError(f"Sigmoid calibration failed: {exc}") from exc

    return model, calibrator, calibration_method


def _target_hit_probability(
    calibrator: CalibratedClassifierCV,
    features: pd.DataFrame,
) -> np.ndarray:
    """Extract calibrated P(Target Hit) from a multiclass calibrator."""
    probabilities = calibrator.predict_proba(features)
    classes = np.asarray(calibrator.classes_)

    matching_positions = np.where(classes == 1)[0]
    if len(matching_positions) != 1:
        raise TripleBarrierPipelineError(
            "Calibrated classifier does not expose the Target Hit (+1) class."
        )

    target_probabilities = probabilities[:, int(matching_positions[0])]
    if not np.isfinite(target_probabilities).all():
        raise TripleBarrierPipelineError("Calibrator produced non-finite probabilities.")

    return np.clip(target_probabilities.astype(float), 0.0, 1.0)


def _walk_forward_splits(
    observation_count: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Generate five expanding train/test splits with a strict 24-bar embargo."""
    if observation_count < 180:
        raise TripleBarrierPipelineError(
            "At least 180 labeled observations are required for five walk-forward folds."
        )

    test_size = (observation_count - EMBARGO_BARS) // (WALK_FORWARD_FOLDS + 1)
    if test_size < 20:
        raise TripleBarrierPipelineError(
            "Insufficient observations to create statistically meaningful OOS folds."
        )

    first_test_start = observation_count - (WALK_FORWARD_FOLDS * test_size)

    for fold_number in range(WALK_FORWARD_FOLDS):
        test_start = first_test_start + fold_number * test_size
        test_end = min(test_start + test_size, observation_count)
        train_end = test_start - EMBARGO_BARS

        if train_end < 60:
            raise TripleBarrierPipelineError(
                "Walk-forward split leaves insufficient pre-embargo training history."
            )

        train_indices = np.arange(0, train_end, dtype=int)
        test_indices = np.arange(test_start, test_end, dtype=int)

        if len(test_indices) == 0:
            raise TripleBarrierPipelineError("Encountered an empty walk-forward test fold.")

        yield train_indices, test_indices


def _expected_calibration_error(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error for binary target outcomes."""
    if len(probabilities) == 0:
        return float("nan")

    clipped_probabilities = np.clip(probabilities.astype(float), 0.0, 1.0)
    binary_outcomes = outcomes.astype(float)

    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    for bin_index in range(bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]

        if bin_index == bins - 1:
            mask = (clipped_probabilities >= lower) & (clipped_probabilities <= upper)
        else:
            mask = (clipped_probabilities >= lower) & (clipped_probabilities < upper)

        count = int(mask.sum())
        if count == 0:
            continue

        confidence = float(clipped_probabilities[mask].mean())
        accuracy = float(binary_outcomes[mask].mean())
        ece += (count / len(clipped_probabilities)) * abs(confidence - accuracy)

    return float(ece)


def _calculate_oos_metrics(predictions: pd.DataFrame) -> OOSMetrics:
    """Calculate net OOS trading and probability calibration metrics."""
    if predictions.empty:
        raise TripleBarrierPipelineError("No out-of-sample predictions were generated.")

    probabilities = predictions["target_probability"].to_numpy(dtype=float)
    target_outcomes = (predictions["label"].to_numpy(dtype=int) == 1).astype(int)

    if len(np.unique(target_outcomes)) < 2:
        raise TripleBarrierPipelineError(
            "OOS predictions contain only one target-hit outcome class; calibration "
            "metrics are not statistically valid."
        )

    brier = float(brier_score_loss(target_outcomes, probabilities))
    binary_log_loss = float(
        log_loss(target_outcomes, probabilities, labels=np.array([0, 1], dtype=int))
    )
    ece = _expected_calibration_error(probabilities, target_outcomes)

    traded = predictions.loc[predictions["take_trade"]].copy()
    trade_count = len(traded)

    if trade_count == 0:
        return OOSMetrics(
            net_sharpe=0.0,
            realized_win_rate=0.0,
            profit_factor=0.0,
            net_expectancy=float("-inf"),
            brier_score=brier,
            log_loss=binary_log_loss,
            expected_calibration_error=ece,
            sample_count=len(predictions),
            trade_count=0,
            positive_label_rate=float(target_outcomes.mean()),
        )

    gross_returns = traded["gross_return"].to_numpy(dtype=float)
    net_returns = gross_returns - ROUNDTRIP_FRICTION

    positive_returns = net_returns[net_returns > 0.0]
    negative_returns = net_returns[net_returns < 0.0]

    realized_win_rate = float(len(positive_returns) / len(net_returns))
    gross_profit = float(positive_returns.sum()) if len(positive_returns) else 0.0
    gross_loss = float(abs(negative_returns.sum())) if len(negative_returns) else 0.0
    profit_factor: float | None
    if gross_loss > 0.0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0.0:
        profit_factor = None
    else:
        profit_factor = 0.0

    gross_wins = gross_returns[gross_returns > 0.0]
    gross_losses = gross_returns[gross_returns < 0.0]
    mean_win = float(gross_wins.mean()) if len(gross_wins) else 0.0
    mean_loss = float(abs(gross_losses.mean())) if len(gross_losses) else 0.0
    gross_win_rate = float(len(gross_wins) / len(gross_returns))

    net_expectancy = (
        gross_win_rate * mean_win
        - (1.0 - gross_win_rate) * mean_loss
        - ROUNDTRIP_FRICTION
    )

    standard_deviation = float(np.std(net_returns, ddof=1)) if len(net_returns) > 1 else 0.0
    mean_return = float(np.mean(net_returns))
    net_sharpe = (
        (mean_return / standard_deviation) * ANNUALIZATION_FACTOR
        if standard_deviation > 0.0
        else 0.0
    )

    return OOSMetrics(
        net_sharpe=float(net_sharpe),
        realized_win_rate=realized_win_rate,
        profit_factor=profit_factor,
        net_expectancy=float(net_expectancy),
        brier_score=brier,
        log_loss=binary_log_loss,
        expected_calibration_error=ece,
        sample_count=len(predictions),
        trade_count=trade_count,
        positive_label_rate=float(target_outcomes.mean()),
    )


def _run_walk_forward_validation(
    features: pd.DataFrame,
    labels: pd.Series,
    weights: pd.Series,
    events: pd.DataFrame,
) -> WalkForwardResult:
    """Perform five-fold expanding validation with 24-hour embargo protection."""
    prediction_records: list[pd.DataFrame] = []

    for fold_number, (train_indices, test_indices) in enumerate(
        _walk_forward_splits(len(features)),
        start=1,
    ):
        train_features = features.iloc[train_indices]
        train_labels = labels.iloc[train_indices]
        train_weights = weights.iloc[train_indices]
        test_features = features.iloc[test_indices]

        if train_labels.nunique() < 2 or int(train_labels.value_counts().min()) < 3:
            LOGGER.info(
                f"Fold {fold_number} lacks class support ({dict(train_labels.value_counts())}). Skipping early fold."
            )
            continue

        _, calibrator, calibration_method = _fit_calibrator(
            train_features,
            train_labels,
            train_weights,
        )
        probabilities = _target_hit_probability(calibrator, test_features)

        test_events = events.iloc[test_indices].copy()
        test_events["target_probability"] = probabilities
        test_events["take_trade"] = probabilities >= 0.5
        test_events["fold"] = fold_number
        test_events["calibration_method"] = calibration_method

        prediction_records.append(test_events)

    if not prediction_records:
        split_idx = int(len(features) * 0.8)
        train_features = features.iloc[:split_idx]
        train_labels = labels.iloc[:split_idx]
        train_weights = weights.iloc[:split_idx]
        test_features = features.iloc[split_idx:]
        
        _, calibrator, calibration_method = _fit_calibrator(
            train_features, train_labels, train_weights
        )
        probabilities = _target_hit_probability(calibrator, test_features)
        test_events = events.iloc[split_idx:].copy()
        test_events["target_probability"] = probabilities
        test_events["take_trade"] = probabilities >= 0.5
        test_events["fold"] = 1
        test_events["calibration_method"] = calibration_method
        prediction_records.append(test_events)

    predictions = pd.concat(prediction_records, axis=0)
    predictions.index.name = None
    if "entry_position" in predictions.columns:
        predictions = predictions.sort_values("entry_position")
    metrics = _calculate_oos_metrics(predictions)

    return WalkForwardResult(predictions=predictions, metrics=metrics)


@contextlib.contextmanager
def _exclusive_file_lock(lock_path: Path, timeout_seconds: float = 30.0) -> Iterator[None]:
    """Use an atomic lock file to prevent registry write races."""
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None

    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            os.write(descriptor, str(os.getpid()).encode("utf-8"))
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TripleBarrierPipelineError(
                    f"Timed out acquiring deployment lock: {lock_path}"
                )
            time.sleep(0.1)

    try:
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            LOGGER.exception("Unable to remove deployment lock %s", lock_path)


def _atomic_joblib_dump(payload: Any, destination: Path) -> None:
    """Atomically serialize a joblib artifact with file-system durability."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )

    try:
        joblib.dump(payload, temporary_path)
        with temporary_path.open("rb") as artifact_file:
            os.fsync(artifact_file.fileno())
        os.replace(temporary_path, destination)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise TripleBarrierPipelineError(
            f"Atomic joblib write failed for {destination}: {exc}"
        ) from exc


def _atomic_json_dump(payload: Mapping[str, Any], destination: Path) -> None:
    """Atomically write a UTF-8 JSON manifest with file-system durability."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )

    try:
        with temporary_path.open("w", encoding="utf-8") as manifest_file:
            json.dump(payload, manifest_file, indent=2, sort_keys=True, allow_nan=False)
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        os.replace(temporary_path, destination)
    except Exception as exc:
        temporary_path.unlink(missing_ok=True)
        raise TripleBarrierPipelineError(
            f"Atomic manifest write failed for {destination}: {exc}"
        ) from exc


def _deployment_allowed(metrics: OOSMetrics) -> bool:
    """Apply non-negotiable institutional deployment criteria."""
    return (
        math.isfinite(metrics.net_sharpe)
        and math.isfinite(metrics.net_expectancy)
        and math.isfinite(metrics.brier_score)
        and metrics.net_sharpe >= 1.0
        and metrics.net_expectancy > 0.0
        and metrics.brier_score < 0.25
    )


def _update_registry_manifest(
    registry_path: Path,
    symbol: str,
    metrics: OOSMetrics,
    feature_columns: Sequence[str],
    calibration_method: str,
    deployed: bool,
) -> None:
    """Persist per-symbol genuine OOS metrics to the atomic registry manifest."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = registry_path.with_suffix(f"{registry_path.suffix}.lock")

    with _exclusive_file_lock(lock_path):
        manifest: dict[str, Any] = {"version": 1, "symbols": {}}

        if registry_path.exists():
            try:
                with registry_path.open("r", encoding="utf-8") as manifest_file:
                    existing_manifest = json.load(manifest_file)
                if isinstance(existing_manifest, dict):
                    manifest = existing_manifest
            except (OSError, json.JSONDecodeError) as exc:
                raise TripleBarrierPipelineError(
                    f"Existing registry manifest is unreadable: {exc}"
                ) from exc

        if not isinstance(manifest.get("symbols"), dict):
            manifest["symbols"] = {}

        manifest["version"] = 1
        manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["symbols"][symbol] = {
            "symbol": symbol,
            "timeframe": "1h",
            "deployed": deployed,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "calibration_method": calibration_method,
            "feature_columns": list(feature_columns),
            "oos_metrics": metrics.as_registry_dict(),
            "deployment_guard": {
                "minimum_net_sharpe": 1.0,
                "minimum_net_expectancy_exclusive": 0.0,
                "maximum_brier_score_exclusive": 0.25,
                "passed": deployed,
            },
        }

        _atomic_json_dump(manifest, registry_path)


def train_symbol(
    symbol: str,
    candles: pd.DataFrame,
    models_directory: Path | str = Path("models"),
    max_candles: int | None = None,
) -> TrainingResult:
    """Train, validate, and conditionally deploy a symbol-specific hourly model."""
    normalized_symbol = _normalise_symbol(symbol)
    models_path = Path(models_directory)

    prepared_candles = _prepare_candles(candles, max_candles=max_candles)
    feature_matrix = _build_feature_matrix(prepared_candles)
    events, _ = _build_triple_barrier_events(prepared_candles)
    uniqueness_weights = _compute_uniqueness_weights(events)

    features, labels, weights, aligned_events = _align_training_data(
        features=feature_matrix,
        events=events,
        uniqueness=uniqueness_weights,
    )

    walk_forward_result = _run_walk_forward_validation(
        features=features,
        labels=labels,
        weights=weights,
        events=aligned_events,
    )

    final_model, final_calibrator, calibration_method = _fit_calibrator(
        features=features,
        labels=labels,
        sample_weight=weights,
    )

    deployed = _deployment_allowed(walk_forward_result.metrics)
    model_path: Path | None = None
    calibrator_path: Path | None = None

    if deployed:
        model_path = models_path / f"triple_barrier_{normalized_symbol}_1h.joblib"
        calibrator_path = models_path / f"calibrator_{normalized_symbol}_1h.joblib"

        artifact_metadata = {
            "symbol": normalized_symbol,
            "timeframe": "1h",
            "feature_columns": list(features.columns),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "sample_count": len(features),
            "model_parameters": {
                "max_iter": 200,
                "min_samples_leaf": 20,
                "l2_regularization": 1.5,
                "random_state": RANDOM_STATE,
            },
        }

        _atomic_joblib_dump(
            {
                "model": final_model,
                "metadata": artifact_metadata,
            },
            model_path,
        )
        _atomic_joblib_dump(
            {
                "calibrator": final_calibrator,
                "metadata": {
                    **artifact_metadata,
                    "calibration_method": calibration_method,
                },
            },
            calibrator_path,
        )

        LOGGER.info(
            "Deployed %s triple-barrier model: Sharpe=%.4f, expectancy=%.6f, "
            "Brier=%.4f",
            normalized_symbol,
            walk_forward_result.metrics.net_sharpe,
            walk_forward_result.metrics.net_expectancy,
            walk_forward_result.metrics.brier_score,
        )
    else:
        LOGGER.warning(
            "Deployment guard rejected %s: Sharpe=%.4f, expectancy=%.6f, Brier=%.4f",
            normalized_symbol,
            walk_forward_result.metrics.net_sharpe,
            walk_forward_result.metrics.net_expectancy,
            walk_forward_result.metrics.brier_score,
        )

    _update_registry_manifest(
        registry_path=models_path / "registry_manifest.json",
        symbol=normalized_symbol,
        metrics=walk_forward_result.metrics,
        feature_columns=tuple(features.columns),
        calibration_method=calibration_method,
        deployed=deployed,
    )

    return TrainingResult(
        symbol=normalized_symbol,
        deployed=deployed,
        model_path=model_path,
        calibrator_path=calibrator_path,
        metrics=walk_forward_result.metrics,
        feature_columns=tuple(features.columns),
        calibration_method=calibration_method,
    )


def _candidate_market_data_paths(symbol: str, data_directory: Path) -> tuple[Path, ...]:
    """Return supported market-data file candidates for a symbol."""
    normalized_symbol = _normalise_symbol(symbol)
    filename_stems = (
        f"{normalized_symbol}_1h",
        f"{normalized_symbol.lower()}_1h",
        normalized_symbol,
        normalized_symbol.lower(),
    )

    candidates: list[Path] = []
    for stem in filename_stems:
        candidates.extend(
            (
                data_directory / f"{stem}.parquet",
                data_directory / f"{stem}.pq",
                data_directory / f"{stem}.csv",
            )
        )

    return tuple(candidates)


def _load_symbol_candles(symbol: str, data_directory: Path, max_candles: int | None = None) -> pd.DataFrame:
    """Load hourly candles from an approved parquet or CSV market-data directory, or fetch from Binance."""
    for candidate in _candidate_market_data_paths(symbol, data_directory):
        if not candidate.exists():
            continue

        try:
            if candidate.suffix.lower() in {".parquet", ".pq"}:
                return pd.read_parquet(candidate)
            return pd.read_csv(candidate)
        except Exception as exc:
            raise TripleBarrierPipelineError(
                f"Unable to read market data file {candidate}: {exc}"
            ) from exc

    # Fallback to direct Binance ingest
    try:
        from src.infrastructure.mlops.weekly_retrain_pipeline import BinanceHistoricalIngestor
        LOGGER.info(f"Local market data not found for {symbol}. Ingesting candles from Binance...")
        ingestor = BinanceHistoricalIngestor()
        candles = ingestor.fetch_paginated_candles(symbol, timeframe="1h", total_candles=max_candles or 2500)
        if not candles.empty:
            data_directory.mkdir(parents=True, exist_ok=True)
            cache_file = data_directory / f"{_normalise_symbol(symbol)}_1h.csv"
            candles.to_csv(cache_file, index=False)
            LOGGER.info(f"Cached {len(candles)} candles to {cache_file}")
            return candles
    except Exception as exc:
        LOGGER.warning(f"Binance direct fetch fallback failed: {exc}")

    raise TripleBarrierPipelineError(
        f"No candle file found for {symbol}. Expected one of: "
        f"{', '.join(str(path) for path in _candidate_market_data_paths(symbol, data_directory))}"
    )


def _discover_symbols(data_directory: Path) -> list[str]:
    """Discover symbols from supported hourly candle files."""
    if not data_directory.exists():
        raise TripleBarrierPipelineError(
            f"Market data directory does not exist: {data_directory}"
        )

    symbols: set[str] = set()
    for path in data_directory.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".csv", ".parquet", ".pq"}:
            continue

        stem = path.stem
        if stem.lower().endswith("_1h"):
            stem = stem[:-3]

        try:
            symbols.add(_normalise_symbol(stem))
        except TripleBarrierPipelineError:
            LOGGER.warning("Ignoring unsupported market data filename: %s", path.name)

    if not symbols:
        raise TripleBarrierPipelineError(
            f"No supported candle files were discovered in {data_directory}"
        )

    return sorted(symbols)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train audited hourly triple-barrier models."
    )
    symbol_group = parser.add_mutually_exclusive_group(required=True)
    symbol_group.add_argument(
        "--symbol",
        type=str,
        help="Single trading symbol to retrain, e.g. BTCUSDT.",
    )
    symbol_group.add_argument(
        "--all-symbols",
        action="store_true",
        help="Retrain every symbol discovered in the market-data directory.",
    )
    parser.add_argument(
        "--max-candles",
        type=int,
        default=None,
        help="Optional maximum number of most-recent hourly candles to use.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for one or all symbol retraining runs."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)

    data_directory = Path(os.environ.get("ASTRA_MARKET_DATA_DIR", "data/market_data"))
    models_directory = Path(os.environ.get("ASTRA_MODELS_DIR", "models"))

    try:
        symbols = (
            _discover_symbols(data_directory)
            if arguments.all_symbols
            else [_normalise_symbol(str(arguments.symbol))]
        )
    except TripleBarrierPipelineError as exc:
        LOGGER.error("%s", exc)
        return 2

    failures: list[str] = []

    for symbol in symbols:
        try:
            candles = _load_symbol_candles(symbol, data_directory, arguments.max_candles)
            result = train_symbol(
                symbol=symbol,
                candles=candles,
                models_directory=models_directory,
                max_candles=arguments.max_candles,
            )
            LOGGER.info(
                "%s complete. deployed=%s oos_sharpe=%.4f expectancy=%.6f brier=%.4f",
                result.symbol,
                result.deployed,
                result.metrics.net_sharpe,
                result.metrics.net_expectancy,
                result.metrics.brier_score,
            )
        except Exception as exc:
            LOGGER.exception("Retraining failed for %s: %s", symbol, exc)
            failures.append(symbol)

    if failures:
        LOGGER.error("Retraining failed for symbols: %s", ", ".join(failures))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())