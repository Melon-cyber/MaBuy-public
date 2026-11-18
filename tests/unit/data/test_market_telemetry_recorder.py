"""Unit tests for MarketTelemetryRecorder."""

import gzip
import json
import os
import shutil
import tempfile
import time
from unittest.mock import patch
import pytest

from src.infrastructure.data.market_telemetry_recorder import (
    MarketTelemetryRecorder,
    MarketTelemetrySnapshot,
)


@pytest.fixture
def temp_storage():
    temp_dir = tempfile.mkdtemp(prefix="test_mkt_telemetry_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_snapshot_creation_and_normalization():
    snapshot = MarketTelemetrySnapshot(
        timestamp_utc="2026-09-13T10:00:00Z",
        timestamp_ms=1789380000000,
        symbol="BTC",
        best_bid=60000.0,
        best_ask=60002.0,
        mid_price=60001.0,
        microprice=60001.2,
        spread_bps=0.33,
        bids_l2=[(60000.0, 1.5)],
        asks_l2=[(60002.0, 2.0)],
    )
    d = snapshot.to_dict()
    assert d["symbol"] == "BTC"
    assert json.loads(d["bids_l2"]) == [[60000.0, 1.5]]
    assert json.loads(d["asks_l2"]) == [[60002.0, 2.0]]


def test_recorder_buffering_and_flush_jsonl(temp_storage):
    recorder = MarketTelemetryRecorder(
        storage_dir=temp_storage,
        max_buffer_size=10,
        flush_interval_sec=3600.0,
        storage_format="jsonl.gz",
        auto_start_worker=False,
    )

    # Ingest 5 records
    for i in range(5):
        recorder.record(
            symbol="BTC/USDT",
            best_bid=60000.0 + i,
            best_ask=60001.0 + i,
            bids_l2=[(60000.0 + i, 1.0)],
            asks_l2=[(60001.0 + i, 1.0)],
            ofi_norm=0.25,
            cvd_delta=10.5,
        )

    assert len(recorder._buffer) == 5
    out_file = recorder.flush()
    assert out_file is not None
    assert os.path.exists(out_file)
    assert out_file.endswith(".jsonl.gz")
    assert len(recorder._buffer) == 0

    # Read back gzip jsonl
    with gzip.open(out_file, "rt", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 5
    rec0 = json.loads(lines[0])
    assert rec0["symbol"] == "BTC"
    assert rec0["best_bid"] == 60000.0
    assert rec0["ofi_norm"] == 0.25


def test_recorder_auto_flush_on_max_buffer(temp_storage):
    recorder = MarketTelemetryRecorder(
        storage_dir=temp_storage,
        max_buffer_size=3,
        flush_interval_sec=3600.0,
        storage_format="jsonl.gz",
        auto_start_worker=False,
    )

    # Ingest 3 records -> should trigger auto-flush
    for i in range(3):
        recorder.record(
            symbol="SOL/USDT:USDT",
            best_bid=100.0 + i,
            best_ask=100.05 + i,
        )

    # Buffer should be emptied because auto-flush was invoked
    assert len(recorder._buffer) == 0
    stats = recorder.get_stats()
    assert stats["total_recorded"] == 3
    assert stats["storage_size_mb"] >= 0.0


def test_recorder_l2_depth_clipping(temp_storage):
    recorder = MarketTelemetryRecorder(
        storage_dir=temp_storage,
        max_buffer_size=10,
        auto_start_worker=False,
    )

    deep_bids = [(100.0 - i, 1.0) for i in range(50)]
    deep_asks = [(100.0 + i, 1.0) for i in range(50)]

    snap = recorder.record(
        symbol="ETHUSDT",
        best_bid=100.0,
        best_ask=100.1,
        bids_l2=deep_bids,
        asks_l2=deep_asks,
    )

    # Should be clipped to 20 levels
    assert len(snap.bids_l2) == 20
    assert len(snap.asks_l2) == 20
    assert snap.symbol == "ETH"


def test_disk_guard_quota_pruning(temp_storage):
    # Set quota very small: 0.001 MB (~1 KB)
    recorder = MarketTelemetryRecorder(
        storage_dir=temp_storage,
        max_buffer_size=5,
        max_storage_mb=0.0001,
        storage_format="jsonl.gz",
        auto_start_worker=False,
    )

    # Create dummy old partition directory
    old_partition = os.path.join(temp_storage, "2026-09-01")
    os.makedirs(old_partition, exist_ok=True)
    with open(os.path.join(old_partition, "dummy.dat"), "wb") as f:
        f.write(b"0" * 1024 * 100)  # 100 KB

    assert os.path.exists(old_partition)

    # Flush some data
    recorder.record("BTC", 50000.0, 50001.0)
    recorder.flush()

    # The old partition should have been pruned because storage exceeded 0.0001 MB
    assert not os.path.exists(old_partition)
