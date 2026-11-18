"""SQLite-backed tamper-evident institutional audit ledger."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import sqlite3
import threading
import time
from collections import deque
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Optional, Tuple


class AuditLedgerError(RuntimeError):
    """Raised when an audit-ledger persistence or integrity operation fails."""


class AuditLedger:
    """Thread-safe SQLite WAL audit ledger with tamper-evident hash chaining."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS trade_events (
        event_id TEXT PRIMARY KEY,
        symbol TEXT,
        side INTEGER,
        entry_ts_ns INTEGER,
        barrier_end_ts_ns INTEGER,
        entry_px REAL,
        stop_px REAL,
        target_px REAL,
        vertical_ts_ns INTEGER,
        label INTEGER,
        ambiguity_flag INTEGER,
        dataset_id TEXT,
        model_version TEXT,
        created_at_ns INTEGER
    );

    CREATE TABLE IF NOT EXISTS order_intents (
        intent_id TEXT PRIMARY KEY,
        client_order_id TEXT UNIQUE,
        symbol TEXT,
        purpose TEXT,
        side TEXT,
        quantity REAL,
        order_type TEXT,
        payload_json TEXT,
        state TEXT,
        exchange_order_id TEXT,
        created_at_ns INTEGER,
        updated_at_ns INTEGER
    );

    CREATE TABLE IF NOT EXISTS execution_fills (
        fill_id TEXT PRIMARY KEY,
        exchange_order_id TEXT,
        exchange_trade_id TEXT,
        symbol TEXT,
        side TEXT,
        price REAL,
        quantity REAL,
        commission REAL,
        commission_asset TEXT,
        liquidity TEXT,
        exchange_ts_ns INTEGER,
        UNIQUE(exchange_order_id, exchange_trade_id)
    );

    CREATE TABLE IF NOT EXISTS audit_ledger (
        sequence_no INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT,
        payload_json TEXT,
        previous_hash TEXT,
        event_hash TEXT UNIQUE,
        created_at_ns INTEGER
    );
    """

    def __init__(self, db_path: str) -> None:
        """
        Initialize the SQLite database, WAL settings, and required tables.

        Args:
            db_path: SQLite database file path. Use ``:memory:`` for an
                ephemeral in-memory database.

        Raises:
            AuditLedgerError: If the database cannot be initialized.
            ValueError: If ``db_path`` is empty.
        """
        if not isinstance(db_path, str) or not db_path.strip():
            raise ValueError("db_path must be a non-empty string")

        self._lock = threading.Lock()
        self._closed = False

        if db_path != ":memory:" and not db_path.startswith("file:"):
            try:
                Path(db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise AuditLedgerError(
                    f"Unable to create database directory for {db_path!r}"
                ) from exc

        try:
            self._connection = sqlite3.connect(
                db_path,
                timeout=5.0,
                check_same_thread=False,
                isolation_level=None,
            )
            self._connection.execute("PRAGMA journal_mode=WAL;")
            self._connection.execute("PRAGMA synchronous=FULL;")
            self._connection.execute("PRAGMA foreign_keys=ON;")
            self._connection.execute("PRAGMA busy_timeout=5000;")
            self._connection.executescript(self._SCHEMA)
        except sqlite3.Error as exc:
            try:
                self._connection.close()
            except (AttributeError, sqlite3.Error):
                pass
            raise AuditLedgerError(
                f"Unable to initialize audit ledger database at {db_path!r}"
            ) from exc

    def __enter__(self) -> "AuditLedger":
        """Return this ledger instance for context-manager use."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Any,
    ) -> None:
        """Close the database connection when leaving a context-manager block."""
        self.close()

    def close(self) -> None:
        """
        Close the underlying SQLite connection.

        Calling this method more than once is safe.
        """
        with self._lock:
            if self._closed:
                return
            try:
                self._connection.close()
            except sqlite3.Error as exc:
                raise AuditLedgerError("Unable to close audit ledger database") from exc
            finally:
                self._closed = True

    def record_event(self, event_type: str, payload: dict[str, Any]) -> str:
        """
        Append a generic event to the tamper-evident audit hash chain.

        The event hash is SHA-256 over:
        ``f"{previous_hash}|{event_type}|{payload_json}|{created_at_ns}"``.

        Args:
            event_type: Stable event classification string.
            payload: JSON-serializable event payload.

        Returns:
            The newly generated SHA-256 event hash.

        Raises:
            AuditLedgerError: If the event cannot be persisted.
            ValueError: If the event data is invalid or not JSON serializable.
        """
        self._require_text(event_type, "event_type")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dictionary")

        payload_json = self._canonical_json(payload)
        created_at_ns = self._utc_now_ns()

        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                event_hash = self._append_event_cursor(
                    event_type=event_type,
                    payload_json=payload_json,
                    created_at_ns=created_at_ns,
                )
                self._connection.commit()
                return event_hash
            except Exception as exc:
                self._rollback_safely()
                if isinstance(exc, sqlite3.Error):
                    raise AuditLedgerError("Unable to record audit event") from exc
                raise

    def verify_chain_integrity(self) -> Tuple[bool, Optional[int]]:
        """
        Verify sequence continuity and all hashes in the audit chain.

        Returns:
            A tuple of ``(is_valid, failed_sequence_no)``. For a valid chain,
            ``failed_sequence_no`` is ``None``. An empty ledger is valid.
        """
        with self._lock:
            self._ensure_open()
            try:
                rows = self._connection.execute(
                    """
                    SELECT
                        sequence_no,
                        event_type,
                        payload_json,
                        previous_hash,
                        event_hash,
                        created_at_ns
                    FROM audit_ledger
                    ORDER BY sequence_no ASC
                    """
                ).fetchall()
            except sqlite3.Error as exc:
                raise AuditLedgerError("Unable to read audit ledger chain") from exc

        expected_sequence = 1
        expected_previous_hash = ""

        for row in rows:
            sequence_no = row[0]
            event_type = row[1]
            payload_json = row[2]
            previous_hash = row[3]
            event_hash = row[4]
            created_at_ns = row[5]

            if sequence_no != expected_sequence:
                return False, sequence_no

            if not all(
                isinstance(value, str)
                for value in (event_type, payload_json, previous_hash, event_hash)
            ):
                return False, sequence_no

            if not isinstance(created_at_ns, int):
                return False, sequence_no

            if previous_hash != expected_previous_hash:
                return False, sequence_no

            material = (
                f"{previous_hash}|{event_type}|{payload_json}|{created_at_ns}"
            ).encode("utf-8")
            computed_hash = hashlib.sha256(material).hexdigest()

            if not hmac.compare_digest(computed_hash, event_hash):
                return False, sequence_no

            expected_previous_hash = event_hash
            expected_sequence += 1

        return True, None

    def record_trade_event(
        self,
        event_id: str,
        symbol: str,
        side: int,
        entry_ts_ns: int,
        barrier_end_ts_ns: int,
        entry_px: float,
        stop_px: float,
        target_px: float,
        vertical_ts_ns: Optional[int],
        label: Optional[int],
        ambiguity_flag: bool | int,
        dataset_id: str,
        model_version: str,
        created_at_ns: Optional[int] = None,
    ) -> str:
        """
        Persist a trade-labeling event and atomically audit its insertion.

        Returns:
            The supplied ``event_id``.
        """
        self._require_text(event_id, "event_id")
        self._require_text(symbol, "symbol")
        self._require_integer(side, "side")
        self._require_timestamp(entry_ts_ns, "entry_ts_ns")
        self._require_timestamp(barrier_end_ts_ns, "barrier_end_ts_ns")
        self._require_finite_number(entry_px, "entry_px")
        self._require_finite_number(stop_px, "stop_px")
        self._require_finite_number(target_px, "target_px")
        self._require_optional_timestamp(vertical_ts_ns, "vertical_ts_ns")
        self._require_optional_integer(label, "label")
        self._require_text(dataset_id, "dataset_id")
        self._require_text(model_version, "model_version")

        if not isinstance(ambiguity_flag, (bool, int)):
            raise ValueError("ambiguity_flag must be a boolean or integer")

        created = self._utc_now_ns() if created_at_ns is None else created_at_ns
        self._require_timestamp(created, "created_at_ns")

        normalized_ambiguity_flag = int(ambiguity_flag)
        audit_payload: dict[str, Any] = {
            "event_id": event_id,
            "symbol": symbol,
            "side": side,
            "entry_ts_ns": entry_ts_ns,
            "barrier_end_ts_ns": barrier_end_ts_ns,
            "entry_px": float(entry_px),
            "stop_px": float(stop_px),
            "target_px": float(target_px),
            "vertical_ts_ns": vertical_ts_ns,
            "label": label,
            "ambiguity_flag": normalized_ambiguity_flag,
            "dataset_id": dataset_id,
            "model_version": model_version,
            "created_at_ns": created,
        }

        self._write_domain_record(
            sql="""
                INSERT INTO trade_events (
                    event_id,
                    symbol,
                    side,
                    entry_ts_ns,
                    barrier_end_ts_ns,
                    entry_px,
                    stop_px,
                    target_px,
                    vertical_ts_ns,
                    label,
                    ambiguity_flag,
                    dataset_id,
                    model_version,
                    created_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            parameters=(
                event_id,
                symbol,
                side,
                entry_ts_ns,
                barrier_end_ts_ns,
                float(entry_px),
                float(stop_px),
                float(target_px),
                vertical_ts_ns,
                label,
                normalized_ambiguity_flag,
                dataset_id,
                model_version,
                created,
            ),
            event_type="TRADE_EVENT",
            audit_payload=audit_payload,
        )
        return event_id

    def record_order_intent(
        self,
        intent_id: str,
        client_order_id: str,
        symbol: str,
        purpose: str,
        side: str,
        quantity: float,
        order_type: str,
        payload: dict[str, Any],
        state: str,
        exchange_order_id: Optional[str] = None,
        created_at_ns: Optional[int] = None,
        updated_at_ns: Optional[int] = None,
    ) -> str:
        """
        Persist an order intent and atomically audit its insertion.

        Returns:
            The supplied ``intent_id``.
        """
        self._require_text(intent_id, "intent_id")
        self._require_text(client_order_id, "client_order_id")
        self._require_text(symbol, "symbol")
        self._require_text(purpose, "purpose")
        self._require_text(side, "side")
        self._require_positive_number(quantity, "quantity")
        self._require_text(order_type, "order_type")
        self._require_text(state, "state")

        if exchange_order_id is not None:
            self._require_text(exchange_order_id, "exchange_order_id")
        if not isinstance(payload, dict):
            raise ValueError("payload must be a dictionary")

        payload_json = self._canonical_json(payload)
        created = self._utc_now_ns() if created_at_ns is None else created_at_ns
        updated = created if updated_at_ns is None else updated_at_ns
        self._require_timestamp(created, "created_at_ns")
        self._require_timestamp(updated, "updated_at_ns")

        audit_payload: dict[str, Any] = {
            "intent_id": intent_id,
            "client_order_id": client_order_id,
            "symbol": symbol,
            "purpose": purpose,
            "side": side,
            "quantity": float(quantity),
            "order_type": order_type,
            "payload": payload,
            "state": state,
            "exchange_order_id": exchange_order_id,
            "created_at_ns": created,
            "updated_at_ns": updated,
        }

        self._write_domain_record(
            sql="""
                INSERT INTO order_intents (
                    intent_id,
                    client_order_id,
                    symbol,
                    purpose,
                    side,
                    quantity,
                    order_type,
                    payload_json,
                    state,
                    exchange_order_id,
                    created_at_ns,
                    updated_at_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            parameters=(
                intent_id,
                client_order_id,
                symbol,
                purpose,
                side,
                float(quantity),
                order_type,
                payload_json,
                state,
                exchange_order_id,
                created,
                updated,
            ),
            event_type="ORDER_INTENT",
            audit_payload=audit_payload,
        )
        return intent_id

    def record_fill(
        self,
        fill_id: str,
        exchange_order_id: str,
        exchange_trade_id: str,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        commission: float,
        commission_asset: str,
        liquidity: str,
        exchange_ts_ns: Optional[int] = None,
    ) -> str:
        """
        Persist an execution fill and atomically audit its insertion.

        ``exchange_ts_ns`` defaults to the current UTC epoch nanosecond value.

        Returns:
            The supplied ``fill_id``.
        """
        self._require_text(fill_id, "fill_id")
        self._require_text(exchange_order_id, "exchange_order_id")
        self._require_text(exchange_trade_id, "exchange_trade_id")
        self._require_text(symbol, "symbol")
        self._require_text(side, "side")
        self._require_positive_number(price, "price")
        self._require_positive_number(quantity, "quantity")
        self._require_finite_number(commission, "commission")
        self._require_text(commission_asset, "commission_asset")
        self._require_text(liquidity, "liquidity")

        timestamp = self._utc_now_ns() if exchange_ts_ns is None else exchange_ts_ns
        self._require_timestamp(timestamp, "exchange_ts_ns")

        audit_payload: dict[str, Any] = {
            "fill_id": fill_id,
            "exchange_order_id": exchange_order_id,
            "exchange_trade_id": exchange_trade_id,
            "symbol": symbol,
            "side": side,
            "price": float(price),
            "quantity": float(quantity),
            "commission": float(commission),
            "commission_asset": commission_asset,
            "liquidity": liquidity,
            "exchange_ts_ns": timestamp,
        }

        self._write_domain_record(
            sql="""
                INSERT INTO execution_fills (
                    fill_id,
                    exchange_order_id,
                    exchange_trade_id,
                    symbol,
                    side,
                    price,
                    quantity,
                    commission,
                    commission_asset,
                    liquidity,
                    exchange_ts_ns
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            parameters=(
                fill_id,
                exchange_order_id,
                exchange_trade_id,
                symbol,
                side,
                float(price),
                float(quantity),
                float(commission),
                commission_asset,
                liquidity,
                timestamp,
            ),
            event_type="EXECUTION_FILL",
            audit_payload=audit_payload,
        )
        return fill_id

    def get_reconciliation_summary(self) -> dict[str, Any]:
        """
        Produce an execution reconciliation summary.

        Realized PnL is calculated using chronological FIFO lot matching per
        symbol. Commissions are reported separately and are not deducted from
        realized PnL because commission assets may differ from quote currency.

        Returns:
            A dictionary containing total fills, aggregate commission amount,
            aggregate filled quantity, and gross FIFO realized PnL.

        Raises:
            AuditLedgerError: If stored fill data is malformed or unreadable.
        """
        with self._lock:
            self._ensure_open()
            try:
                rows = self._connection.execute(
                    """
                    SELECT
                        fill_id,
                        symbol,
                        side,
                        price,
                        quantity,
                        commission,
                        exchange_ts_ns
                    FROM execution_fills
                    ORDER BY symbol ASC, exchange_ts_ns ASC, fill_id ASC
                    """
                ).fetchall()
            except sqlite3.Error as exc:
                raise AuditLedgerError(
                    "Unable to query execution fills for reconciliation"
                ) from exc

        total_volume = Decimal("0")
        total_fees = Decimal("0")
        realized_pnl = Decimal("0")
        open_lots: dict[str, deque[tuple[int, Decimal, Decimal]]] = {}

        try:
            for fill_id, symbol, side, price, quantity, commission, _timestamp in rows:
                if not isinstance(fill_id, str) or not isinstance(symbol, str):
                    raise AuditLedgerError("Execution fill contains invalid identifiers")
                if not isinstance(side, str):
                    raise AuditLedgerError(
                        f"Execution fill {fill_id!r} contains an invalid side"
                    )

                normalized_side = side.strip().upper()
                if normalized_side in {"BUY", "B"}:
                    direction = 1
                elif normalized_side in {"SELL", "S"}:
                    direction = -1
                else:
                    raise AuditLedgerError(
                        f"Execution fill {fill_id!r} has unsupported side {side!r}"
                    )

                decimal_price = self._decimal_from_db(price, "price", fill_id)
                decimal_quantity = self._decimal_from_db(
                    quantity, "quantity", fill_id
                )
                decimal_commission = self._decimal_from_db(
                    commission, "commission", fill_id
                )

                if decimal_price <= 0 or decimal_quantity <= 0:
                    raise AuditLedgerError(
                        f"Execution fill {fill_id!r} has non-positive price or quantity"
                    )

                total_volume += decimal_quantity
                total_fees += decimal_commission

                lots = open_lots.setdefault(symbol, deque())
                remaining_quantity = decimal_quantity

                while (
                    remaining_quantity > 0
                    and lots
                    and lots[0][0] != direction
                ):
                    lot_direction, lot_price, lot_quantity = lots.popleft()
                    matched_quantity = min(remaining_quantity, lot_quantity)

                    realized_pnl += (
                        (decimal_price - lot_price)
                        * Decimal(lot_direction)
                        * matched_quantity
                    )

                    remaining_quantity -= matched_quantity
                    lot_quantity -= matched_quantity

                    if lot_quantity > 0:
                        lots.appendleft((lot_direction, lot_price, lot_quantity))

                if remaining_quantity > 0:
                    lots.append((direction, decimal_price, remaining_quantity))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise AuditLedgerError(
                "Malformed numeric execution fill data prevents reconciliation"
            ) from exc

        return {
            "total_fills": len(rows),
            "total_fees_paid": self._decimal_to_finite_float(total_fees, "fees"),
            "total_volume": self._decimal_to_finite_float(total_volume, "volume"),
            "realized_pnl": self._decimal_to_finite_float(
                realized_pnl, "realized_pnl"
            ),
        }

    def _write_domain_record(
        self,
        sql: str,
        parameters: tuple[Any, ...],
        event_type: str,
        audit_payload: dict[str, Any],
    ) -> None:
        """Atomically insert a domain row and append its audit-chain event."""
        payload_json = self._canonical_json(audit_payload)
        created_at_ns = self._utc_now_ns()

        with self._lock:
            self._ensure_open()
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                self._connection.execute(sql, parameters)
                self._append_event_cursor(
                    event_type=event_type,
                    payload_json=payload_json,
                    created_at_ns=created_at_ns,
                )
                self._connection.commit()
            except Exception as exc:
                self._rollback_safely()
                if isinstance(exc, sqlite3.Error):
                    raise AuditLedgerError(
                        f"Unable to persist {event_type.lower()} record"
                    ) from exc
                raise

    def _append_event_cursor(
        self,
        event_type: str,
        payload_json: str,
        created_at_ns: int,
    ) -> str:
        """Append a hash-chained ledger event within an active transaction."""
        row = self._connection.execute(
            """
            SELECT event_hash
            FROM audit_ledger
            ORDER BY sequence_no DESC
            LIMIT 1
            """
        ).fetchone()

        previous_hash = "" if row is None else row[0]
        if not isinstance(previous_hash, str):
            raise AuditLedgerError("Audit ledger contains an invalid previous hash")

        material = (
            f"{previous_hash}|{event_type}|{payload_json}|{created_at_ns}"
        ).encode("utf-8")
        event_hash = hashlib.sha256(material).hexdigest()

        self._connection.execute(
            """
            INSERT INTO audit_ledger (
                event_type,
                payload_json,
                previous_hash,
                event_hash,
                created_at_ns
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                event_type,
                payload_json,
                previous_hash,
                event_hash,
                created_at_ns,
            ),
        )
        return event_hash

    def _rollback_safely(self) -> None:
        """Rollback an active transaction without masking the original error."""
        try:
            if self._connection.in_transaction:
                self._connection.rollback()
        except sqlite3.Error:
            pass

    def _ensure_open(self) -> None:
        """Raise if the ledger has already been closed."""
        if self._closed:
            raise AuditLedgerError("Audit ledger is closed")

    @staticmethod
    def _utc_now_ns() -> int:
        """Return the current UTC Unix epoch timestamp in nanoseconds."""
        return time.time_ns()

    @staticmethod
    def _canonical_json(payload: dict[str, Any]) -> str:
        """Serialize a payload deterministically for durable hashing."""
        try:
            return json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "payload must be JSON serializable and cannot contain NaN or Infinity"
            ) from exc

    @staticmethod
    def _require_text(value: Any, name: str) -> None:
        """Validate a required non-empty text field."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")

    @staticmethod
    def _require_integer(value: Any, name: str) -> None:
        """Validate a signed SQLite-compatible integer field."""
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if value < -(2**63) or value > (2**63 - 1):
            raise ValueError(f"{name} is outside SQLite INTEGER range")

    @classmethod
    def _require_optional_integer(cls, value: Any, name: str) -> None:
        """Validate an optional integer field."""
        if value is not None:
            cls._require_integer(value, name)

    @classmethod
    def _require_timestamp(cls, value: Any, name: str) -> None:
        """Validate an integer UTC epoch timestamp expressed in nanoseconds."""
        cls._require_integer(value, name)

    @classmethod
    def _require_optional_timestamp(cls, value: Any, name: str) -> None:
        """Validate an optional nanosecond timestamp."""
        if value is not None:
            cls._require_timestamp(value, name)

    @staticmethod
    def _require_finite_number(value: Any, name: str) -> None:
        """Validate a finite real number while excluding booleans."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")

    @classmethod
    def _require_positive_number(cls, value: Any, name: str) -> None:
        """Validate a finite strictly positive real number."""
        cls._require_finite_number(value, name)
        if float(value) <= 0:
            raise ValueError(f"{name} must be greater than zero")

    @staticmethod
    def _decimal_from_db(value: Any, field_name: str, fill_id: str) -> Decimal:
        """Convert a stored SQLite numeric value into a validated Decimal."""
        if isinstance(value, bool) or value is None:
            raise AuditLedgerError(
                f"Execution fill {fill_id!r} contains invalid {field_name}"
            )

        try:
            decimal_value = Decimal(str(value))
        except (InvalidOperation, ValueError) as exc:
            raise AuditLedgerError(
                f"Execution fill {fill_id!r} contains invalid {field_name}"
            ) from exc

        if not decimal_value.is_finite():
            raise AuditLedgerError(
                f"Execution fill {fill_id!r} contains non-finite {field_name}"
            )
        return decimal_value

    @staticmethod
    def _decimal_to_finite_float(value: Decimal, field_name: str) -> float:
        """Convert a Decimal aggregate to a finite float for API consumers."""
        result = float(value)
        if not math.isfinite(result):
            raise AuditLedgerError(
                f"Reconciliation {field_name} cannot be represented as a finite float"
            )
        return result