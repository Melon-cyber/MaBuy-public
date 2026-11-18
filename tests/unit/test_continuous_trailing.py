import pytest
import time
from src.domain.services.continuous_trailing_stop_engine import (
    ContinuousTrailingStopEngine,
    TrailingState
)

def test_long_ratchet_invariant():
    engine = ContinuousTrailingStopEngine(exchange_update_threshold=0.0015)
    
    # LONG trade entry at 100.0, initial stop at 98.2 (1.8% SL)
    state = TrailingState(
        symbol="BTCUSDT",
        side=1,
        entry_px=100.0,
        current_sl_px=98.2,
        peak_px=100.0,
        d_min=0.012,
        k_sigma=1.8,
        updated_at_ts=time.time()
    )

    # Price rallies to 105.0 with low ATR (0.5% => 1.8 * 0.005 = 0.009, so d_min=0.012 dominates)
    # Stop should move to 105.0 * (1 - 0.012) = 103.74
    new_sl, should_update, reason = engine.update_stop(state, mark_price=105.0, atr_norm=0.005)
    assert new_sl > 98.2
    assert should_update is True
    assert state.peak_px == 105.0

    # Price pulls back from 105.0 to 102.0
    # Stop must NEVER loosen! It must remain at 103.74
    sl_pullback, should_update_pb, reason_pb = engine.update_stop(state, mark_price=102.0, atr_norm=0.005)
    assert sl_pullback == new_sl
    assert should_update_pb is False
    assert "prevented_stop_loosening" in reason_pb

def test_short_ratchet_invariant():
    engine = ContinuousTrailingStopEngine(exchange_update_threshold=0.0015)
    
    # SHORT trade entry at 100.0, initial stop at 101.8 (1.8% SL)
    state = TrailingState(
        symbol="ETHUSDT",
        side=-1,
        entry_px=100.0,
        current_sl_px=101.8,
        peak_px=100.0,
        d_min=0.012,
        k_sigma=1.8,
        updated_at_ts=time.time()
    )

    # Price drops to 92.0 (favorable for short)
    # Candidate SL = 92.0 * (1 + 0.012) = 93.104
    new_sl, should_update, reason = engine.update_stop(state, mark_price=92.0, atr_norm=0.005)
    assert new_sl < 101.8
    assert should_update is True
    assert state.peak_px == 92.0

    # Price bounces back to 95.0
    # Stop must NEVER loosen (never increase above 93.104)
    sl_bounce, should_update_b, reason_b = engine.update_stop(state, mark_price=95.0, atr_norm=0.005)
    assert sl_bounce == new_sl
    assert should_update_b is False
    assert "prevented_stop_loosening" in reason_b

def test_breakeven_lock():
    engine = ContinuousTrailingStopEngine()
    
    state = TrailingState(
        symbol="SOLUSDT",
        side=1,
        entry_px=100.0,
        current_sl_px=98.2,
        peak_px=100.0,
        d_min=0.015,
        k_sigma=2.0,
        updated_at_ts=time.time()
    )

    # Small move +0.5% (below 1.0% breakeven trigger)
    be_sl = engine.check_breakeven_lock(state, mark_price=100.5, fee_buffer=0.0010)
    assert be_sl is None
    assert state.current_sl_px == 98.2

    # Move +1.5% (triggers breakeven lock)
    be_sl_triggered = engine.check_breakeven_lock(state, mark_price=101.5, fee_buffer=0.0010)
    assert be_sl_triggered is not None
    # Protected stop is entry * (1 + 0.0010) = 100.10
    assert be_sl_triggered == 100.10
    assert state.current_sl_px == 100.10