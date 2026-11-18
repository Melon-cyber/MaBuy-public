#!/usr/bin/env python3
"""
Institutional quantitative diagnostic and research advisory agent.

This module reads recent signal-log data from SQLite, computes trade-performance
and market-regime diagnostics, requests forensic recommendations from Astra,
and safely stages immutable advisories without modifying production parameters.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
import os
import signal
import sqlite3
import statistics
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable, Mapping, Sequence

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()
except ImportError:
    pass


LOGGER: Final[logging.Logger] = logging.getLogger("astra_quant_researcher")
UTC: Final[dt.tzinfo] = dt.timezone.utc

HEALTH_STATUSES: Final[frozenset[str]] = frozenset(
    {
        "OPTIMAL",
        "DEGRADED",
        "REGIME_SHIFT",
        "CIRCUIT_BREAKER_RECOMMENDED",
    }
)

CLOSED_STATUS_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "closed",
        "complete",
        "completed",
        "exited",
        "exit",
        "settled",
    }
)

VETO_TOKENS: Final[tuple[str, ...]] = (
    "veto",
    "reject",
    "rejected",
    "blocked",
    "deny",
    "denied",
    "prevented",
    "failed_gate",
)

TIMESTAMP_COLUMNS: Final[tuple[str, ...]] = (
    "closed_at",
    "close_time",
    "exit_time",
    "timestamp",
    "time",
    "created_at",
    "updated_at",
    "ts",
)

PNL_COLUMNS: Final[tuple[str, ...]] = (
    "realized_pnl",
    "net_pnl",
    "pnl_usd",
    "profit_loss",
    "profit",
    "pnl",
)

ENTRY_PRICE_COLUMNS: Final[tuple[str, ...]] = (
    "entry_fill_price",
    "entry_price",
    "open_price",
    "fill_entry_price",
)

EXIT_PRICE_COLUMNS: Final[tuple[str, ...]] = (
    "exit_fill_price",
    "exit_price",
    "close_price",
    "fill_exit_price",
)

EXPECTED_ENTRY_PRICE_COLUMNS: Final[tuple[str, ...]] = (
    "expected_entry_price",
    "signal_entry_price",
    "target_entry_price",
)

EXPECTED_EXIT_PRICE_COLUMNS: Final[tuple[str, ...]] = (
    "expected_exit_price",
    "signal_exit_price",
    "target_exit_price",
)

MARKET_PRICE_COLUMNS: Final[tuple[str, ...]] = (
    "mark_price",
    "mid_price",
    "last_price",
    "market_price",
    "price",
    "close",
    "close_price",
    "exit_price",
)

FUNDING_COLUMNS: Final[tuple[str, ...]] = (
    "funding_rate",
    "funding",
    "predicted_funding_rate",
)

SIDE_COLUMNS: Final[tuple[str, ...]] = (
    "side",
    "direction",
    "position_side",
    "trade_side",
)

QUANTITY_COLUMNS: Final[tuple[str, ...]] = (
    "quantity",
    "qty",
    "size",
    "filled_qty",
    "filled_quantity",
    "position_size",
)

STATUS_COLUMNS: Final[tuple[str, ...]] = (
    "status",
    "trade_status",
    "signal_status",
    "execution_status",
)

GATE_COLUMNS: Final[tuple[str, ...]] = (
    "gate_veto",
    "gate_decision",
    "gate_status",
    "veto_reason",
    "vetoed",
    "decision",
    "ofi_veto",
)

_AUDIT_LOCK: Final[threading.Lock] = threading.Lock()


def _utc_now() -> dt.datetime:
    """Return the current UTC timestamp."""
    return dt.datetime.now(UTC)


def _quote_sqlite_identifier(identifier: str) -> str:
    """Safely quote a SQLite identifier (table or column name)."""
    return f'"{identifier.replace(chr(34), chr(34)+chr(34))}"'


def _truncate_text(text: Any, max_chars: int) -> str:
    """Truncate text representation to max_chars."""
    if text is None:
        return ""
    return str(text)[:max_chars]


def _json_safe(obj: Any) -> Any:
    """Convert objects to JSON-serializable primitives, replacing non-finite floats."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (dt.datetime, dt.date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in obj]
    return str(obj)


def _canonical_json(obj: Any) -> str:
    """Serialize object to deterministic, sorted canonical JSON."""
    return json.dumps(
        _json_safe(obj),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _lock_file_if_supported(file_descriptor: int) -> None:
    """Apply advisory lock on file descriptor if supported by OS."""
    try:
        import fcntl
        fcntl.flock(file_descriptor, fcntl.LOCK_EX)
    except (ImportError, OSError):
        pass


def _unlock_file_if_supported(file_descriptor: int) -> None:
    """Release advisory lock on file descriptor if supported by OS."""
    try:
        import fcntl
        fcntl.flock(file_descriptor, fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass


def _number(value: Any) -> float | None:
    """Extract finite float from value, returning None on invalid/non-finite."""
    if value is None or isinstance(value, bool):
        return None
    try:
        val = float(value)
        return val if math.isfinite(val) else None
    except (ValueError, TypeError):
        return None


def _first_value(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    """Return first non-None matching key from row (case-insensitive)."""
    normalized = {str(k).lower(): v for k, v in row.items()}
    for key in keys:
        k_lower = key.lower()
        if k_lower in normalized and normalized[k_lower] is not None:
            return normalized[k_lower]
    return None


def _extract_timestamp(row: Mapping[str, Any]) -> dt.datetime | None:
    """Extract timezone-aware UTC datetime from row."""
    val = _first_value(row, TIMESTAMP_COLUMNS)
    if val is None:
        return None
    if isinstance(val, dt.datetime):
        return val if val.tzinfo else val.replace(tzinfo=UTC)
    if isinstance(val, (int, float)):
        try:
            ts = float(val)
            if ts > 1e11:
                ts /= 1000.0
            return dt.datetime.fromtimestamp(ts, tz=UTC)
        except Exception:
            return None
    if isinstance(val, str):
        try:
            s = val.replace("Z", "+00:00")
            parsed = dt.datetime.fromisoformat(s)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except Exception:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
            try:
                return dt.datetime.strptime(val, fmt).replace(tzinfo=UTC)
            except Exception:
                pass
    return None


def _safe_divide(num: float | None, den: float | None, fallback: float | None = None) -> float | None:
    """Safely divide numerator by denominator, returning fallback if zero or invalid."""
    if num is None or den is None or den == 0.0:
        return fallback
    try:
        res = num / den
        return res if math.isfinite(res) else fallback
    except Exception:
        return fallback


def _mean_or_none(values: Sequence[float]) -> float | None:
    """Return arithmetic mean or None if empty."""
    finite_vals = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.mean(finite_vals) if finite_vals else None


def _median_or_none(values: Sequence[float]) -> float | None:
    """Return median or None if empty."""
    finite_vals = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(finite_vals) if finite_vals else None


def _percentile(values: Sequence[float], pct: float) -> float | None:
    """Compute percentile (pct in 0.0..1.0 or 0..100) or None."""
    finite_vals = sorted([v for v in values if v is not None and math.isfinite(v)])
    if not finite_vals:
        return None
    p = pct if pct <= 1.0 else pct / 100.0
    idx = int(round(p * (len(finite_vals) - 1)))
    return finite_vals[max(0, min(idx, len(finite_vals) - 1))]


def _estimate_hurst_rs(log_returns: Sequence[float]) -> float | None:
    """Estimate Hurst exponent H via Rescaled Range (R/S) method."""
    returns = [r for r in log_returns if r is not None and math.isfinite(r)]
    if len(returns) < 20:
        return None
    try:
        lags = range(4, min(len(returns) // 2, 64))
        rs_values = []
        for lag in lags:
            rs_sub = []
            for i in range(0, len(returns) - lag + 1, lag):
                chunk = returns[i:i + lag]
                if len(chunk) < lag:
                    continue
                mean = statistics.mean(chunk)
                cum_dev = []
                acc = 0.0
                for c in chunk:
                    acc += (c - mean)
                    cum_dev.append(acc)
                r = max(cum_dev) - min(cum_dev)
                s = statistics.stdev(chunk) if len(chunk) > 1 else 0.0
                if s > 1e-8:
                    rs_sub.append(r / s)
            if rs_sub:
                rs_values.append((math.log(lag), math.log(statistics.mean(rs_sub))))
        if len(rs_values) < 3:
            return None
        # OLS slope
        x = [v[0] for v in rs_values]
        y = [v[1] for v in rs_values]
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        den = sum((xi - x_mean) ** 2 for xi in x)
        if den == 0.0:
            return None
        slope = num / den
        return float(max(0.0, min(1.0, slope)))
    except Exception:
        return None


def _stringify_model_value(val: Any) -> str:
    """Convert model recommendation field to clean string."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val
    if isinstance(val, Mapping):
        return json.dumps(val, ensure_ascii=False, indent=2)
    return str(val)


class AstraResearchError(RuntimeError):
    """Base exception for Astra quantitative research failures."""


class AstraAPIError(AstraResearchError):
    """Raised when the Astra API cannot return a usable response."""


@dataclass(frozen=True)
class ResearchConfig:
    """Runtime configuration for the quantitative research agent."""

    database_path: Path
    signal_table: str
    advisory_path: Path
    endpoint: str
    model: str
    api_key_env: str
    lookback_days: int
    max_rows: int
    timeout_seconds: float
    daemon_interval_seconds: float

    def __post_init__(self) -> None:
        if not self.signal_table.strip():
            raise ValueError("signal_table must not be empty.")
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive.")
        if self.max_rows < 10:
            raise ValueError("max_rows must be at least 10.")
        if self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive.")
        if self.daemon_interval_seconds <= 0.0:
            raise ValueError("daemon_interval_seconds must be positive.")

        parsed_endpoint = urllib.parse.urlparse(self.endpoint)
        if parsed_endpoint.scheme.lower() != "https" or not parsed_endpoint.netloc:
            raise ValueError("Astra endpoint must be an absolute HTTPS URL.")


@dataclass(frozen=True)
class AstraResponse:
    """Structured representation of an Astra API response."""

    raw_content: str
    parsed_content: Mapping[str, Any]


@dataclass(frozen=True)
class MarketObservation:
    """Minimal time-series point used for regime analysis."""

    timestamp: dt.datetime | None
    price: float | None
    funding_rate: float | None


class AstraQuantResearcher:
    """
    Scheduled or on-demand institutional quantitative diagnostic agent.

    The agent is deliberately advisory-only. It writes cryptographically hashed
    research advisories to a JSONL ledger and never changes production settings.
    """

    def __init__(self, config: ResearchConfig | None = None) -> None:
        """Initialize the agent with validated operational configuration."""
        if config is None:
            config = ResearchConfig(
                database_path=Path("data/mabuy_prod.db"),
                signal_table="signal_log",
                advisory_path=Path("models/research_advisories.jsonl"),
                endpoint="https://ttmapi.site/v1/chat/completions",
                model="gpt-6-astra",
                api_key_env="ASTRA_API_KEY",
                lookback_days=30,
                max_rows=1000,
                timeout_seconds=300.0,
                daemon_interval_seconds=21600.0,
            )
        self._config = config
        self._stop_event = threading.Event()

    def run_once(self) -> dict[str, Any]:
        """
        Execute one complete diagnostic, advisory, and immutable audit-log cycle.

        Returns:
            The advisory record successfully persisted to the audit ledger.

        Raises:
            OSError: If the advisory ledger cannot be written.
        """
        generated_at = _utc_now()
        payload = self._build_diagnostic_payload(generated_at)

        model_section: dict[str, Any]
        recommendation: dict[str, Any]

        try:
            api_key = (
                os.environ.get(self._config.api_key_env, "").strip()
                or os.environ.get("OPENAI_API_KEY", "").strip()
            )
            if not api_key:
                raise AstraAPIError(
                    f"Required API key environment variable "
                    f"{self._config.api_key_env!r} (or OPENAI_API_KEY) is not set."
                )

            response = self._request_astra(payload, api_key)
            recommendation = self._normalize_recommendation(response.parsed_content)
            model_section = {
                "request_status": "SUCCESS",
                "endpoint": self._config.endpoint,
                "model": self._config.model,
                "raw_response": _truncate_text(response.raw_content, 100_000),
            }
        except Exception as exc:  # Intentional: failure itself must be audited.
            LOGGER.exception("Astra recommendation request failed.")
            recommendation = {
                "strategy_health": "UNAVAILABLE",
                "model_parse_status": "NOT_AVAILABLE",
                "parameter_adjustments": {},
                "validated_staged_adjustments": [],
                "mathematical_justification": (
                    "No remote recommendation was accepted because the advisory "
                    "request failed. Existing production parameters remain unchanged."
                ),
                "risk_controls": [
                    "Do not modify production parameters.",
                    "Retry the diagnostic after the API or credential issue is resolved.",
                ],
                "failure_reason": _truncate_text(str(exc), 2_000),
            }
            model_section = {
                "request_status": "FAILED",
                "endpoint": self._config.endpoint,
                "model": self._config.model,
                "error_type": type(exc).__name__,
                "error": _truncate_text(str(exc), 2_000),
            }

        advisory_without_hash: dict[str, Any] = {
            "schema_version": "1.0",
            "event_type": "ASTRA_QUANT_RESEARCH_ADVISORY",
            "generated_at": generated_at.isoformat(),
            "diagnostic_payload": payload,
            "model_execution": model_section,
            "recommendation": recommendation,
            "safety_controls": {
                "production_parameters_modified": False,
                "staging_only": True,
                "validation_required_before_application": True,
                "required_validation": [
                    "Historical walk-forward validation",
                    "Out-of-sample evaluation",
                    "Paper-trading or shadow deployment",
                    "Risk-owner approval",
                ],
            },
        }

        canonical_advisory = _canonical_json(advisory_without_hash)
        audit_hash = hashlib.sha256(canonical_advisory.encode("utf-8")).hexdigest()
        advisory = {
            **advisory_without_hash,
            "audit_sha256": audit_hash,
        }

        self._append_advisory(advisory)
        LOGGER.info(
            "Persisted Astra research advisory hash=%s health=%s",
            audit_hash,
            recommendation.get("strategy_health", "UNKNOWN"),
        )
        return advisory

    def run_daemon(self) -> None:
        """
        Run diagnostics repeatedly at the configured interval until interrupted.

        Each cycle is independently protected so a transient data/API failure does
        not terminate the long-running scheduled diagnostic service.
        """
        LOGGER.info(
            "Starting Astra research daemon; interval_seconds=%.2f",
            self._config.daemon_interval_seconds,
        )

        previous_handlers: dict[int, Any] = {}

        def _handle_shutdown(signum: int, _frame: Any) -> None:
            LOGGER.info("Received shutdown signal %s; stopping daemon.", signum)
            self._stop_event.set()

        for signal_number in (signal.SIGINT, signal.SIGTERM):
            try:
                previous_handlers[signal_number] = signal.signal(
                    signal_number,
                    _handle_shutdown,
                )
            except ValueError:
                LOGGER.warning(
                    "Signal handlers could not be installed outside the main thread."
                )

        try:
            while not self._stop_event.is_set():
                cycle_started = time.monotonic()
                try:
                    self.run_once()
                except Exception:
                    LOGGER.exception(
                        "Diagnostic cycle failed before advisory persistence; "
                        "the daemon will continue."
                    )

                elapsed = time.monotonic() - cycle_started
                remaining = max(0.0, self._config.daemon_interval_seconds - elapsed)
                if self._stop_event.wait(remaining):
                    break
        finally:
            for signal_number, old_handler in previous_handlers.items():
                try:
                    signal.signal(signal_number, old_handler)
                except (ValueError, OSError):
                    LOGGER.debug("Could not restore signal handler %s.", signal_number)

            LOGGER.info("Astra research daemon stopped.")

    def _build_diagnostic_payload(self, generated_at: dt.datetime) -> dict[str, Any]:
        """
        Read SQLite signal data and construct a compact institutional payload.

        Database failures are represented explicitly in payload data-quality
        metadata so the audit trail remains complete even during outages.
        """
        warnings: list[str] = []
        rows: list[dict[str, Any]] = []
        source_metadata: dict[str, Any] = {
            "database_path": str(self._config.database_path),
            "signal_table": self._config.signal_table,
            "max_rows_requested": self._config.max_rows,
        }

        try:
            rows, source_metadata = self._read_signal_log()
        except Exception as exc:
            LOGGER.exception("Failed to read SQLite signal log.")
            warnings.append(
                f"SQLite ingestion failed: {type(exc).__name__}: "
                f"{_truncate_text(str(exc), 500)}"
            )

        cutoff = generated_at - dt.timedelta(days=self._config.lookback_days)
        recent_rows = self._filter_recent_rows(rows, cutoff)
        closed_rows = [row for row in recent_rows if self._record_is_closed(row)]

        performance = self._compute_performance_diagnostics(closed_rows, recent_rows)
        regime = self._compute_market_regime(recent_rows)
        data_quality = {
            "source_read_status": "SUCCESS" if not warnings else "DEGRADED",
            "warnings": warnings,
            "rows_read": len(rows),
            "rows_within_lookback": len(recent_rows),
            "closed_trades_within_lookback": len(closed_rows),
            "lookback_start": cutoff.isoformat(),
            "lookback_end": generated_at.isoformat(),
        }

        return _json_safe(
            {
                "payload_version": "1.0",
                "generated_at": generated_at.isoformat(),
                "agent": "AstraQuantResearcher",
                "source_metadata": source_metadata,
                "data_quality": data_quality,
                "performance_diagnostics": performance,
                "market_regime": regime,
                "methodology": {
                    "win_rate": (
                        "count(realized_pnl > 0) / count(trades with realized PnL)"
                    ),
                    "payoff_ratio": (
                        "mean(realized gains) / abs(mean(realized losses)); "
                        "undefined when no losses are observed"
                    ),
                    "slippage": (
                        "mean adverse execution basis points relative to the "
                        "available expected entry or expected exit price"
                    ),
                    "gate_veto_rate": (
                        "count(records containing veto/reject/block decision tokens) "
                        "/ count(observed signal records)"
                    ),
                    "volatility": (
                        "sample standard deviation of log returns annualized using "
                        "observed median timestamp spacing, or daily fallback"
                    ),
                    "trend_persistence_h": (
                        "Hurst exponent estimated by log-log ordinary least squares "
                        "fit of average rescaled-range R/S across multiple block sizes"
                    ),
                },
            }
        )

    def _read_signal_log(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Read recent rows from the configured SQLite signal table in read-only mode.

        Raises:
            AstraResearchError: If the database or requested table is unavailable.
        """
        database_path = self._config.database_path.expanduser()
        if not database_path.exists():
            raise AstraResearchError(f"SQLite database does not exist: {database_path}")
        if not database_path.is_file():
            raise AstraResearchError(
                f"SQLite database path is not a regular file: {database_path}"
            )

        resolved_path = database_path.resolve()
        sqlite_uri = f"{resolved_path.as_uri()}?mode=ro"
        table_identifier = _quote_sqlite_identifier(self._config.signal_table)

        with sqlite3.connect(sqlite_uri, uri=True, timeout=10.0) as connection:
            connection.row_factory = sqlite3.Row
            schema_rows = connection.execute(
                f"PRAGMA table_info({table_identifier})"
            ).fetchall()

            if not schema_rows:
                raise AstraResearchError(
                    f"SQLite table {self._config.signal_table!r} does not exist "
                    "or has no discoverable schema."
                )

            column_names = [str(row["name"]) for row in schema_rows]
            normalized_columns = {column.lower() for column in column_names}
            order_column = next(
                (
                    column
                    for column in TIMESTAMP_COLUMNS
                    if column in normalized_columns
                ),
                None,
            )

            query = f"SELECT * FROM {table_identifier}"
            if order_column is not None:
                query += f" ORDER BY {_quote_sqlite_identifier(order_column)} DESC"
            query += " LIMIT ?"

            result_rows = connection.execute(query, (self._config.max_rows,)).fetchall()
            records = [
                {str(key).lower(): row[key] for key in row.keys()}
                for row in result_rows
            ]

        metadata = {
            "database_path": str(resolved_path),
            "signal_table": self._config.signal_table,
            "table_columns": sorted(column_names),
            "rows_selected": len(records),
            "ordered_by": order_column,
            "read_only": True,
        }
        return records, metadata

    def _filter_recent_rows(
        self,
        rows: Sequence[Mapping[str, Any]],
        cutoff: dt.datetime,
    ) -> list[dict[str, Any]]:
        """Retain rows within the time window while preserving untimestamped rows."""
        filtered: list[dict[str, Any]] = []

        for row in rows:
            normalized_row = {str(key).lower(): value for key, value in row.items()}
            timestamp = _extract_timestamp(normalized_row)
            if timestamp is None or timestamp >= cutoff:
                filtered.append(normalized_row)

        return filtered

    def _record_is_closed(self, row: Mapping[str, Any]) -> bool:
        """Classify a signal-log row as a closed trade using conservative evidence."""
        status_value = _first_value(row, STATUS_COLUMNS)
        status = str(status_value).strip().lower() if status_value is not None else ""

        if any(token in status for token in CLOSED_STATUS_TOKENS):
            return True

        if _first_value(row, PNL_COLUMNS) is not None:
            return not any(token in status for token in VETO_TOKENS)

        has_entry = _number(_first_value(row, ENTRY_PRICE_COLUMNS)) is not None
        has_exit = _number(_first_value(row, EXIT_PRICE_COLUMNS)) is not None
        has_close_time = _first_value(
            row,
            ("closed_at", "close_time", "exit_time"),
        ) is not None

        return has_entry and has_exit and has_close_time

    def _compute_performance_diagnostics(
        self,
        closed_rows: Sequence[Mapping[str, Any]],
        all_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Calculate realized performance, execution quality, and gate diagnostics."""
        pnls: list[float] = []
        slippages_bps: list[float] = []

        for row in closed_rows:
            pnl = self._extract_realized_pnl(row)
            if pnl is not None and math.isfinite(pnl):
                pnls.append(pnl)

            slippage = self._extract_adverse_slippage_bps(row)
            if slippage is not None and math.isfinite(slippage):
                slippages_bps.append(slippage)

        wins = [pnl for pnl in pnls if pnl > 0.0]
        losses = [pnl for pnl in pnls if pnl < 0.0]
        breakeven_count = sum(1 for pnl in pnls if pnl == 0.0)

        win_rate = _safe_divide(float(len(wins)), float(len(pnls)))
        average_win = _mean_or_none(wins)
        average_loss = _mean_or_none(losses)
        payoff_ratio = (
            _safe_divide(average_win, abs(average_loss))
            if average_win is not None and average_loss is not None
            else None
        )
        profit_factor = (
            _safe_divide(sum(wins), abs(sum(losses)))
            if wins and losses
            else None
        )

        veto_count = 0
        observed_gate_count = 0
        for row in all_rows:
            values = [
                str(value).lower()
                for key, value in row.items()
                if key in GATE_COLUMNS or key in STATUS_COLUMNS
            ]

            if values:
                observed_gate_count += 1
                if any(
                    token in " ".join(values)
                    for token in VETO_TOKENS
                ):
                    veto_count += 1

        return {
            "trade_count": len(closed_rows),
            "pnl_observation_count": len(pnls),
            "win_count": len(wins),
            "loss_count": len(losses),
            "breakeven_count": breakeven_count,
            "win_rate": win_rate,
            "average_win": average_win,
            "average_loss": average_loss,
            "payoff_ratio": payoff_ratio,
            "profit_factor": profit_factor,
            "total_realized_pnl": sum(pnls) if pnls else None,
            "mean_realized_pnl": _mean_or_none(pnls),
            "median_realized_pnl": _median_or_none(pnls),
            "slippage_observation_count": len(slippages_bps),
            "mean_adverse_slippage_bps": _mean_or_none(slippages_bps),
            "median_adverse_slippage_bps": _median_or_none(slippages_bps),
            "p95_adverse_slippage_bps": _percentile(slippages_bps, 0.95),
            "gate_observation_count": observed_gate_count,
            "gate_veto_count": veto_count,
            "gate_veto_rate": _safe_divide(
                float(veto_count),
                float(observed_gate_count),
            ),
        }

    def _extract_realized_pnl(self, row: Mapping[str, Any]) -> float | None:
        """Extract recorded PnL, falling back to a directionally signed estimate."""
        direct_pnl = _number(_first_value(row, PNL_COLUMNS))
        if direct_pnl is not None:
            return direct_pnl

        entry = _number(_first_value(row, ENTRY_PRICE_COLUMNS))
        exit_price = _number(_first_value(row, EXIT_PRICE_COLUMNS))
        quantity = _number(_first_value(row, QUANTITY_COLUMNS))
        side_value = _first_value(row, SIDE_COLUMNS)

        if (
            entry is None
            or exit_price is None
            or quantity is None
            or quantity <= 0.0
            or side_value is None
        ):
            return None

        side = str(side_value).strip().lower()
        if side in {"buy", "long", "1", "+1"}:
            return (exit_price - entry) * quantity
        if side in {"sell", "short", "-1"}:
            return (entry - exit_price) * quantity
        return None

    def _extract_adverse_slippage_bps(
        self,
        row: Mapping[str, Any],
    ) -> float | None:
        """
        Estimate directionally adverse execution slippage in basis points.

        Exit-side expected prices are preferred. If absent, entry-side expected
        prices are used. Positive values denote adverse execution.
        """
        side_value = _first_value(row, SIDE_COLUMNS)
        if side_value is None:
            return None

        side = str(side_value).strip().lower()
        is_long = side in {"buy", "long", "1", "+1"}
        is_short = side in {"sell", "short", "-1"}
        if not is_long and not is_short:
            return None

        expected_exit = _number(_first_value(row, EXPECTED_EXIT_PRICE_COLUMNS))
        actual_exit = _number(_first_value(row, EXIT_PRICE_COLUMNS))
        if (
            expected_exit is not None
            and actual_exit is not None
            and expected_exit > 0.0
        ):
            if is_long:
                return ((expected_exit - actual_exit) / expected_exit) * 10_000.0
            return ((actual_exit - expected_exit) / expected_exit) * 10_000.0

        expected_entry = _number(_first_value(row, EXPECTED_ENTRY_PRICE_COLUMNS))
        actual_entry = _number(_first_value(row, ENTRY_PRICE_COLUMNS))
        if (
            expected_entry is None
            or actual_entry is None
            or expected_entry <= 0.0
        ):
            return None

        if is_long:
            return ((actual_entry - expected_entry) / expected_entry) * 10_000.0
        return ((expected_entry - actual_entry) / expected_entry) * 10_000.0

    def _compute_market_regime(
        self,
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Estimate volatility, trend direction, persistence H, and funding regime."""
        observations: list[MarketObservation] = []
        for row in rows:
            observations.append(
                MarketObservation(
                    timestamp=_extract_timestamp(row),
                    price=_number(_first_value(row, MARKET_PRICE_COLUMNS)),
                    funding_rate=_number(_first_value(row, FUNDING_COLUMNS)),
                )
            )

        observations.sort(
            key=lambda observation: (
                observation.timestamp is None,
                observation.timestamp or dt.datetime.min.replace(tzinfo=UTC),
            )
        )

        priced = [
            observation
            for observation in observations
            if observation.price is not None and observation.price > 0.0
        ]
        prices = [float(observation.price) for observation in priced]
        log_returns = [
            math.log(prices[index] / prices[index - 1])
            for index in range(1, len(prices))
            if prices[index - 1] > 0.0 and prices[index] > 0.0
        ]

        annualization_factor = self._infer_annualization_factor(priced)
        volatility = (
            statistics.stdev(log_returns) * math.sqrt(annualization_factor)
            if len(log_returns) >= 2
            else None
        )
        cumulative_return = (
            (prices[-1] / prices[0]) - 1.0 if len(prices) >= 2 and prices[0] > 0.0 else None
        )
        hurst = _estimate_hurst_rs(log_returns)

        if hurst is None:
            persistence_state = "INSUFFICIENT_DATA"
        elif hurst >= 0.55:
            persistence_state = "PERSISTENT"
        elif hurst <= 0.45:
            persistence_state = "MEAN_REVERTING"
        else:
            persistence_state = "NEUTRAL"

        if cumulative_return is None:
            trend_direction = "INSUFFICIENT_DATA"
        elif cumulative_return > 0.002:
            trend_direction = "UPTREND"
        elif cumulative_return < -0.002:
            trend_direction = "DOWNTREND"
        else:
            trend_direction = "RANGE_BOUND"

        funding_rates = [
            float(observation.funding_rate)
            for observation in observations
            if observation.funding_rate is not None
            and math.isfinite(observation.funding_rate)
        ]

        return {
            "price_observation_count": len(prices),
            "return_observation_count": len(log_returns),
            "annualization_factor": annualization_factor,
            "annualized_log_return_volatility": volatility,
            "cumulative_return": cumulative_return,
            "trend_direction": trend_direction,
            "trend_persistence_h": hurst,
            "persistence_state": persistence_state,
            "funding_observation_count": len(funding_rates),
            "mean_funding_rate": _mean_or_none(funding_rates),
            "median_funding_rate": _median_or_none(funding_rates),
            "funding_rate_stddev": (
                statistics.stdev(funding_rates) if len(funding_rates) >= 2 else None
            ),
            "latest_funding_rate": funding_rates[-1] if funding_rates else None,
        }

    def _infer_annualization_factor(
        self,
        observations: Sequence[MarketObservation],
    ) -> float:
        """Infer periods per year from timestamps, using a daily fallback."""
        timestamps = [
            observation.timestamp
            for observation in observations
            if observation.timestamp is not None
        ]
        if len(timestamps) < 2:
            return 365.0

        intervals = [
            (timestamps[index] - timestamps[index - 1]).total_seconds()
            for index in range(1, len(timestamps))
            if (timestamps[index] - timestamps[index - 1]).total_seconds() > 0.0
        ]
        if not intervals:
            return 365.0

        median_interval = statistics.median(intervals)
        if median_interval <= 0.0:
            return 365.0

        inferred = (365.25 * 24.0 * 60.0 * 60.0) / median_interval
        return min(max(inferred, 1.0), 10_000_000.0)

    def _request_astra(
        self,
        payload: Mapping[str, Any],
        api_key: str,
    ) -> AstraResponse:
        """
        Request forensic strategy analysis with multi-model cascading fallback and streaming.

        Raises:
            AstraAPIError: If all candidate models fail or return invalid content.
        """
        import openai

        base_endpoint = self._config.endpoint.rstrip("/")
        if "/chat/completions" in base_endpoint:
            base_url = base_endpoint.split("/chat/completions")[0]
        else:
            base_url = base_endpoint

        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self._config.timeout_seconds,
        )

        forensic_prompt = self._build_forensic_prompt(payload)
        candidate_models = [self._config.model, "gpt-5.6-sol", "gpt-5.6-luna"]
        seen_models = set()
        deduped_models = []
        for m in candidate_models:
            if m not in seen_models:
                seen_models.add(m)
                deduped_models.append(m)

        last_error: Exception | None = None
        for model in deduped_models:
            try:
                LOGGER.info("Consulting %s for quantitative audit...", model)
                role_name = "developer" if any(x in model for x in ["astra", "o1", "o3", "gpt-6"]) else "system"
                stream = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": role_name,
                            "content": (
                                "You are Astra, an institutional quantitative forensic "
                                "research analyst. Return only valid JSON adhering strictly to RFC 8259. "
                                "Never instruct direct production deployment; all recommendations must be "
                                "staged for validation."
                            ),
                        },
                        {
                            "role": "user",
                            "content": forensic_prompt,
                        },
                    ],
                    max_completion_tokens=8192,
                    response_format={"type": "json_object"},
                    stream=True,
                )
                chunks = []
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta_content = chunk.choices[0].delta.content
                        if delta_content:
                            chunks.append(delta_content)

                content = "".join(chunks).strip()
                if not content:
                    raise AstraAPIError(f"Model {model} returned empty content.")

                parsed_content = _parse_json_model_content(content)
                LOGGER.info("Successfully received and validated advisory from %s (%d chars).", model, len(content))
                return AstraResponse(
                    raw_content=content,
                    parsed_content=parsed_content,
                )
            except Exception as exc:
                LOGGER.warning("Candidate model %s failed: %s. Trying fallback...", model, exc)
                last_error = exc

        raise AstraAPIError(
            f"All candidate models failed: {last_error or 'unknown error'}"
        )

    def _build_forensic_prompt(self, payload: Mapping[str, Any]) -> str:
        """Construct a bounded, explicit, JSON-only forensic advisory request."""
        payload_json = _canonical_json(_json_safe(dict(payload)))
        return (
            "Perform a forensic quantitative review of the following diagnostic "
            "payload. Evaluate interaction among realized performance, execution "
            "slippage, gate veto behavior, volatility, trend direction, Hurst "
            "persistence, and funding conditions.\n\n"
            "Return exactly one JSON object with this schema:\n"
            "{\n"
            '  "strategy_health": "OPTIMAL|DEGRADED|REGIME_SHIFT|'
            'CIRCUIT_BREAKER_RECOMMENDED",\n'
            '  "parameter_adjustments": {\n'
            '    "atr_stop_floor": {"recommended": number, "rationale": string},\n'
            '    "tp_targets": {"recommended": [number], "rationale": string},\n'
            '    "ofi_veto_sensitivity": {"recommended": number, "rationale": string},\n'
            '    "gate_thresholds": {"recommended": number_or_object, '
            '"rationale": string}\n'
            "  },\n"
            '  "mathematical_justification": string,\n'
            '  "risk_controls": [string],\n'
            '  "confidence": number\n'
            "}\n\n"
            "Requirements:\n"
            "1. Explain mathematical evidence quantitatively, including relevant "
            "trade-off logic and uncertainty due to sample size.\n"
            "2. Give only parameter changes justified by the supplied metrics.\n"
            "3. If data quality is weak, explicitly recommend no change or "
            "additional validation.\n"
            "4. Recommendations are staging-only and must not be described as "
            "safe for direct production deployment.\n\n"
            f"Diagnostic payload:\n{payload_json}"
        )

    def _normalize_recommendation(
        self,
        model_response: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate response shape and extract safely staged parameter candidates."""
        raw_health = str(
            model_response.get(
                "strategy_health",
                model_response.get("health_status", "UNPARSED"),
            )
        ).strip().upper()

        health = raw_health if raw_health in HEALTH_STATUSES else "UNPARSED"
        raw_adjustments = model_response.get("parameter_adjustments", {})
        if not isinstance(raw_adjustments, Mapping):
            raw_adjustments = {}

        staged_adjustments = _validate_staged_adjustments(raw_adjustments)
        justification = model_response.get(
            "mathematical_justification",
            model_response.get("detailed_mathematical_justification", ""),
        )
        risk_controls = model_response.get("risk_controls", [])
        confidence = _number(model_response.get("confidence"))

        if not isinstance(risk_controls, list):
            risk_controls = [str(risk_controls)]

        return _json_safe(
            {
                "strategy_health": health,
                "model_parse_status": (
                    "VALIDATED"
                    if health != "UNPARSED"
                    else "PARTIALLY_VALIDATED"
                ),
                "parameter_adjustments": dict(raw_adjustments),
                "validated_staged_adjustments": staged_adjustments,
                "mathematical_justification": _truncate_text(
                    _stringify_model_value(justification),
                    20_000,
                ),
                "risk_controls": [
                    _truncate_text(str(control), 2_000)
                    for control in risk_controls[:50]
                ],
                "confidence": confidence,
                "validation_notice": (
                    "All extracted values are advisory candidates only. No production "
                    "parameter was changed by this process."
                ),
            }
        )

    def _append_advisory(self, advisory: Mapping[str, Any]) -> None:
        """
        Append one immutable JSONL audit event with restrictive file permissions.

        The SHA-256 is computed before this method and stored in the advisory body.
        """
        target = self._config.advisory_path.expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)

        line = (
            json.dumps(
                _json_safe(dict(advisory)),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")

        with _AUDIT_LOCK:
            flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
            file_descriptor = os.open(target, flags, 0o600)
            try:
                try:
                    os.chmod(target, 0o600)
                except OSError:
                    LOGGER.debug("Could not enforce permissions on %s.", target)

                _lock_file_if_supported(file_descriptor)
                try:
                    bytes_written = 0
                    while bytes_written < len(line):
                        written = os.write(file_descriptor, line[bytes_written:])
                        if written <= 0:
                            raise OSError("Failed to write complete advisory record.")
                        bytes_written += written
                    os.fsync(file_descriptor)
                finally:
                    _unlock_file_if_supported(file_descriptor)
            finally:
                os.close(file_descriptor)


def _validate_staged_adjustments(
    adjustments: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """
    Extract bounded numeric candidates from untrusted model output.

    Validation is intentionally restrictive because these records are for staging.
    Bounds prevent malformed or extreme model output from being mistaken for a
    viable candidate by downstream validation tooling.
    """
    specifications: dict[str, tuple[float, float]] = {
        "atr_stop_floor": (0.05, 50.0),
        "atr_stop_multiplier_floor": (0.05, 50.0),
        "tp_targets": (0.01, 1_000.0),
        "take_profit_targets": (0.01, 1_000.0),
        "ofi_veto_sensitivity": (0.0, 1.0),
        "gate_thresholds": (0.0, 1.0),
        "gate_threshold": (0.0, 1.0),
    }

    staged: list[dict[str, Any]] = []
    for parameter_name, candidate in adjustments.items():
        normalized_name = str(parameter_name).strip().lower()
        if normalized_name not in specifications:
            continue

        lower_bound, upper_bound = specifications[normalized_name]
        value = _extract_recommended_value(candidate)
        numeric_values = _flatten_numeric_values(value)

        if not numeric_values:
            continue

        if any(
            not math.isfinite(number)
            or number < lower_bound
            or number > upper_bound
            for number in numeric_values
        ):
            staged.append(
                {
                    "parameter": normalized_name,
                    "accepted_for_validation": False,
                    "reason": (
                        f"Candidate outside permitted staging range "
                        f"[{lower_bound}, {upper_bound}]."
                    ),
                }
            )
            continue

        staged.append(
            {
                "parameter": normalized_name,
                "candidate_value": value,
                "accepted_for_validation": True,
                "permitted_range": [lower_bound, upper_bound],
                "application_status": "STAGED_ONLY",
            }
        )

    return staged


def _extract_recommended_value(candidate: Any) -> Any:
    """Extract a semantic recommendation value from common model response forms."""
    if isinstance(candidate, Mapping):
        for key in ("recommended", "value", "proposed", "candidate"):
            if key in candidate:
                return candidate[key]
        return None
    return candidate


def _flatten_numeric_values(value: Any) -> list[float]:
    """Recursively collect finite numeric leaves from a model recommendation."""
    if isinstance(value, bool):
        return []

    if isinstance(value, (int, float)):
        return [float(value)]

    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric_values(item))
        return flattened

    if isinstance(value, Mapping):
        flattened = []
        for item in value.values():
            flattened.extend(_flatten_numeric_values(item))
        return flattened

    parsed = _number(value)
    return [parsed] if parsed is not None else []


def _extract_chat_content(response_json: Any) -> str:
    """Extract text content from common OpenAI-compatible response formats."""
    if not isinstance(response_json, Mapping):
        raise AstraAPIError("Astra response root is not a JSON object.")

    choices = response_json.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, Mapping):
            message = first_choice.get("message")
            if isinstance(message, Mapping):
                content = message.get("content")
                extracted = _content_to_text(content)
                if extracted:
                    return extracted

            text = first_choice.get("text")
            if isinstance(text, str) and text.strip():
                return text

    output = response_json.get("output")
    extracted_output = _content_to_text(output)
    if extracted_output:
        return extracted_output

    raise AstraAPIError("Astra response did not contain model text content.")


def _content_to_text(content: Any) -> str:
    """Convert string or structured model content into textual JSON candidate data."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                for key in ("text", "content", "value"):
                    value = item.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                        break
        return "\n".join(parts).strip()

    if isinstance(content, Mapping):
        return json.dumps(content, ensure_ascii=False)

    return ""


def _parse_json_model_content(content: str) -> Mapping[str, Any]:
    """Parse strict JSON or extract a single JSON object from fenced model output."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
            cleaned = "\n".join(lines[1:-1]).strip()

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace:last_brace + 1]
    else:
        candidate = cleaned

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, Mapping):
            return parsed
        raise AstraAPIError(f"Parsed JSON is not an object: {type(parsed).__name__}")
    except json.JSONDecodeError as exc:
        raise AstraAPIError(f"Failed to parse model JSON: {exc} | Content: {_truncate_text(cleaned, 300)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Astra Autonomous Strategy Research Daemon")
    parser.add_argument("--run-once", action="store_true", help="Execute one research cycle and exit")
    parser.add_argument("--daemon", action="store_true", help="Run continuously in daemon mode")
    parser.add_argument("--interval", type=int, default=21600, help="Interval in seconds between runs (default: 6 hours)")
    args = parser.parse_args()

    researcher = AstraQuantResearcher()
    print("[ASTRA QUANT RESEARCHER] Initialized.")
    if args.run_once or not args.daemon:
        print("[ASTRA QUANT RESEARCHER] Running single research cycle...")
        result = researcher.run_once()
        print(f"[ASTRA QUANT RESEARCHER] Cycle complete. Status: {result.get('status', 'OK')}")
        if "strategy_health" in result:
            print(f"Health: {result['strategy_health']}")
    else:
        print(f"[ASTRA QUANT RESEARCHER] Entering daemon loop (interval: {args.interval}s)...")
        while True:
            try:
                researcher.run_once()
            except Exception as e:
                LOGGER.error(f"Error in research cycle: {e}")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()