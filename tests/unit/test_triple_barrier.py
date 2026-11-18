import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.domain.services.triple_barrier_labeler import (
    BarrierSpec,
    BarrierResult,
    label_single_event,
    make_barriers,
    compute_sample_uniqueness,
    TripleBarrierEngine
)

def test_make_barriers_geometry_and_floor():
    entry_px = 100.0
    side = 1  # LONG
    atr_norm = 0.005  # Low ATR (0.5%), should trigger floor 0.018
    c_win = 0.0004
    c_loss = 0.0006

    stop_frac, target_frac = make_barriers(
        entry_px=entry_px,
        side=side,
        atr_norm=atr_norm,
        c_win=c_win,
        c_loss=c_loss
    )

    # Verify stop floor: max(1.8 * 0.005, 0.018) = 0.018
    assert stop_frac >= 0.018
    # Verify nominal Payoff Ratio R >= 2.0: target_frac >= 2 * (stop_frac + c_loss) + c_win
    assert target_frac >= 2.0 * (stop_frac + c_loss) + c_win
    assert target_frac > stop_frac * 2.0

def test_label_single_event_long_target():
    entry_ts = pd.Timestamp("2026-09-11 12:00:00", tz="UTC")
    vertical_ts = entry_ts + pd.Timedelta(hours=24)
    entry_px = 100.0
    stop_frac = 0.02
    target_frac = 0.04

    event = BarrierSpec(
        event_id="evt_001",
        symbol="BTCUSDT",
        entry_ts=entry_ts,
        side=1,
        entry_px=entry_px,
        stop_frac=stop_frac,
        target_frac=target_frac,
        vertical_ts=vertical_ts
    )

    target_px = entry_px * np.exp(target_frac)
    stop_px = entry_px * np.exp(-stop_frac)

    timestamps = [entry_ts + pd.Timedelta(hours=i) for i in range(1, 5)]
    bars = pd.DataFrame([
        {"timestamp": timestamps[0], "open": 100.5, "high": 101.0, "low": 99.5, "close": 100.8, "volume": 10.0},
        {"timestamp": timestamps[1], "open": 101.0, "high": 102.5, "low": 100.5, "close": 102.0, "volume": 15.0},
        {"timestamp": timestamps[2], "open": 102.0, "high": target_px + 0.5, "low": 101.5, "close": target_px, "volume": 20.0},
        {"timestamp": timestamps[3], "open": 104.0, "high": 105.0, "low": 103.5, "close": 104.5, "volume": 12.0},
    ])

    res = label_single_event(bars, event, tie_policy="conservative_stop")
    assert res.label == 1
    assert res.reason == "target_hit"
    assert res.barrier_ts == timestamps[2]

def test_label_single_event_long_stop():
    entry_ts = pd.Timestamp("2026-09-11 12:00:00", tz="UTC")
    vertical_ts = entry_ts + pd.Timedelta(hours=24)
    entry_px = 100.0
    stop_frac = 0.02
    target_frac = 0.04

    event = BarrierSpec(
        event_id="evt_002",
        symbol="BTCUSDT",
        entry_ts=entry_ts,
        side=1,
        entry_px=entry_px,
        stop_frac=stop_frac,
        target_frac=target_frac,
        vertical_ts=vertical_ts
    )

    stop_px = entry_px * np.exp(-stop_frac)

    timestamps = [entry_ts + pd.Timedelta(hours=i) for i in range(1, 4)]
    bars = pd.DataFrame([
        {"timestamp": timestamps[0], "open": 99.5, "high": 100.2, "low": 99.0, "close": 99.2, "volume": 10.0},
        {"timestamp": timestamps[1], "open": 99.0, "high": 99.3, "low": stop_px - 0.5, "close": stop_px - 0.2, "volume": 25.0},
        {"timestamp": timestamps[2], "open": 97.0, "high": 98.0, "low": 96.5, "close": 97.5, "volume": 15.0},
    ])

    res = label_single_event(bars, event, tie_policy="conservative_stop")
    assert res.label == -1
    assert res.reason == "stop_hit"
    assert res.barrier_ts == timestamps[1]

def test_label_single_event_short_target_and_stop():
    entry_ts = pd.Timestamp("2026-09-11 12:00:00", tz="UTC")
    vertical_ts = entry_ts + pd.Timedelta(hours=24)
    entry_px = 100.0
    stop_frac = 0.02
    target_frac = 0.04

    event_short = BarrierSpec(
        event_id="evt_short",
        symbol="ETHUSDT",
        entry_ts=entry_ts,
        side=-1,
        entry_px=entry_px,
        stop_frac=stop_frac,
        target_frac=target_frac,
        vertical_ts=vertical_ts
    )

    target_px = entry_px * np.exp(-target_frac)

    timestamps = [entry_ts + pd.Timedelta(hours=i) for i in range(1, 3)]
    bars = pd.DataFrame([
        {"timestamp": timestamps[0], "open": 99.0, "high": 99.5, "low": 98.0, "close": 98.5, "volume": 5.0},
        {"timestamp": timestamps[1], "open": 98.0, "high": 98.2, "low": target_px - 0.5, "close": target_px, "volume": 8.0},
    ])

    res = label_single_event(bars, event_short)
    assert res.label == 1
    assert res.reason == "target_hit"

def test_label_vertical_timeout():
    entry_ts = pd.Timestamp("2026-09-11 12:00:00", tz="UTC")
    vertical_ts = entry_ts + pd.Timedelta(hours=2)
    entry_px = 100.0

    event = BarrierSpec(
        event_id="evt_timeout",
        symbol="SOLUSDT",
        entry_ts=entry_ts,
        side=1,
        entry_px=entry_px,
        stop_frac=0.05,
        target_frac=0.10,
        vertical_ts=vertical_ts
    )

    timestamps = [entry_ts + pd.Timedelta(hours=1), vertical_ts]
    bars = pd.DataFrame([
        {"timestamp": timestamps[0], "open": 100.1, "high": 100.5, "low": 99.8, "close": 100.2, "volume": 5.0},
        {"timestamp": timestamps[1], "open": 100.2, "high": 100.4, "low": 99.9, "close": 100.1, "volume": 6.0},
    ])

    res = label_single_event(bars, event)
    assert res.label == 0
    assert "vertical" in res.reason.lower()

def test_ambiguous_bar_resolution():
    entry_ts = pd.Timestamp("2026-09-11 12:00:00", tz="UTC")
    vertical_ts = entry_ts + pd.Timedelta(hours=10)
    entry_px = 100.0
    stop_frac = 0.02
    target_frac = 0.04

    event = BarrierSpec(
        event_id="evt_ambig",
        symbol="BTCUSDT",
        entry_ts=entry_ts,
        side=1,
        entry_px=entry_px,
        stop_frac=stop_frac,
        target_frac=target_frac,
        vertical_ts=vertical_ts
    )

    target_px = entry_px * np.exp(target_frac)
    stop_px = entry_px * np.exp(-stop_frac)

    # Flash-wick bar: hits both target (high >= target_px) and stop (low <= stop_px)
    bars = pd.DataFrame([
        {"timestamp": entry_ts + pd.Timedelta(hours=1), "open": 100.0, "high": target_px + 2.0, "low": stop_px - 2.0, "close": 100.0, "volume": 100.0},
    ])

    # Conservative policy treats as stop
    res_cons = label_single_event(bars, event, tie_policy="conservative_stop")
    assert res_cons.label == -1
    assert "conservative_stop" in res_cons.reason

    # Censor policy excludes it
    res_censor = label_single_event(bars, event, tie_policy="censor")
    assert res_censor.label is None
    assert res_censor.ambiguous is True

def test_sample_uniqueness():
    t0 = pd.Timestamp("2026-09-11 00:00:00", tz="UTC")
    time_index = pd.date_range(t0, periods=10, freq="h")

    # 3 concurrent events spanning t0+1h to t0+5h
    events = pd.DataFrame([
        {"event_id": "e1", "entry_ts": t0 + pd.Timedelta(hours=1), "barrier_end_ts": t0 + pd.Timedelta(hours=4)},
        {"event_id": "e2", "entry_ts": t0 + pd.Timedelta(hours=2), "barrier_end_ts": t0 + pd.Timedelta(hours=5)},
        {"event_id": "e3", "entry_ts": t0 + pd.Timedelta(hours=2), "barrier_end_ts": t0 + pd.Timedelta(hours=4)},
    ])

    u = compute_sample_uniqueness(events, time_index)
    assert len(u) == 3
    for val in u.values:
        assert 0.0 < val < 1.0

    # Completely disjoint events have uniqueness == 1.0
    disjoint_events = pd.DataFrame([
        {"event_id": "d1", "entry_ts": t0, "barrier_end_ts": t0 + pd.Timedelta(hours=1)},
        {"event_id": "d2", "entry_ts": t0 + pd.Timedelta(hours=3), "barrier_end_ts": t0 + pd.Timedelta(hours=4)},
    ])
    u_disjoint = compute_sample_uniqueness(disjoint_events, time_index)
    for val in u_disjoint.values:
        assert val == 1.0