import pytest
import sqlite3
import os
import time
from decimal import Decimal

from src.infrastructure.persistence.audit_ledger import (
    AuditLedger,
    AuditLedgerError
)

@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_audit.db")
    ledger = AuditLedger(db_file)
    yield ledger, db_file
    ledger.close()

def test_audit_ledger_initialization(temp_db):
    ledger, db_file = temp_db
    conn = sqlite3.connect(db_file)
    
    # Verify WAL mode
    journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    assert journal_mode.lower() == "wal"

    # Verify tables
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()}
    assert "audit_ledger" in tables
    assert "trade_events" in tables
    assert "order_intents" in tables
    assert "execution_fills" in tables
    conn.close()

def test_hash_chain_integrity_and_tamper_detection(temp_db):
    ledger, db_file = temp_db

    # Record 3 events
    h1 = ledger.record_event("STATE_CHANGE", {"from": "FLAT", "to": "ENTRY_PENDING"})
    h2 = ledger.record_event("ORDER_PLACED", {"symbol": "SOLUSDT", "side": "BUY", "qty": 1.0})
    h3 = ledger.record_event("FILL_RECEIVED", {"fill_id": "f1", "price": 105.0})

    # Verify chain integrity
    is_valid, broken_seq = ledger.verify_chain_integrity()
    assert is_valid is True
    assert broken_seq is None

    # Tamper with an event in SQLite directly
    conn = sqlite3.connect(db_file)
    conn.execute("UPDATE audit_ledger SET payload_json='{\"hacked\": true}' WHERE sequence_no=2;")
    conn.commit()
    conn.close()

    # Verify tamper is immediately detected
    is_valid_tampered, broken_seq_tampered = ledger.verify_chain_integrity()
    assert is_valid_tampered is False
    assert broken_seq_tampered == 2

def test_record_trade_event_and_intents(temp_db):
    ledger, _ = temp_db
    now_ns = time.time_ns()

    ledger.record_trade_event(
        event_id="evt_101",
        symbol="SOLUSDT",
        side=1,
        entry_ts_ns=now_ns,
        barrier_end_ts_ns=now_ns + 3600_000_000_000,
        entry_px=100.0,
        stop_px=98.0,
        target_px=104.5,
        vertical_ts_ns=now_ns + 86400_000_000_000,
        label=1,
        ambiguity_flag=0,
        dataset_id="ds_live_01",
        model_version="v5_titan"
    )

    ledger.record_order_intent(
        intent_id="intent_001",
        client_order_id="mb-entry-998877",
        symbol="SOLUSDT",
        purpose="ENTRY",
        side="BUY",
        quantity=0.5,
        order_type="MARKET",
        payload={"timeInForce": "IOC"},
        state="FILLED"
    )

    is_valid, _ = ledger.verify_chain_integrity()
    assert is_valid is True

def test_reconciliation_summary_decimal_accounting(temp_db):
    ledger, _ = temp_db
    now_ns = time.time_ns()

    # Buy 1.0 SOL at 100.0 (fee 0.05)
    ledger.record_fill(
        fill_id="f1",
        exchange_order_id="ord_1",
        exchange_trade_id="trd_1",
        symbol="SOLUSDT",
        side="BUY",
        price=100.0,
        quantity=1.0,
        commission=0.05,
        commission_asset="USDT",
        liquidity="MAKER",
        exchange_ts_ns=now_ns
    )

    # Sell 1.0 SOL at 105.0 (fee 0.05) -> Realized PnL = +5.00
    ledger.record_fill(
        fill_id="f2",
        exchange_order_id="ord_2",
        exchange_trade_id="trd_2",
        symbol="SOLUSDT",
        side="SELL",
        price=105.0,
        quantity=1.0,
        commission=0.05,
        commission_asset="USDT",
        liquidity="MAKER",
        exchange_ts_ns=now_ns + 1000
    )

    summary = ledger.get_reconciliation_summary()
    assert summary["total_fills"] == 2
    assert summary["total_fees_paid"] == 0.10
    assert summary["realized_pnl"] == 5.0