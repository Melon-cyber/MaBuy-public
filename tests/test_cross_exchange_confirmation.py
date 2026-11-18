import time
import pytest
from src.domain.services.cross_exchange_state import CrossExchangeState
from src.domain.services.cross_exchange_confirmation import CrossExchangeConfirmation


def test_cross_exchange_state_ring_buffer():
    state = CrossExchangeState(symbols=["BTC"], venues=["binance", "bybit", "okx"], buffer_len=5)
    now = time.time()
    for i in range(10):
        state.update_quote("BTC", "binance", 60000.0 + i, timestamp=now + i)

    # Ring buffer should hold only the latest 5 observations
    obs = state.get_latest_quote("BTC", "binance")
    assert obs is not None
    assert obs.price == 60009.0
    assert obs.timestamp == now + 9


def test_cross_exchange_stale_detection():
    state = CrossExchangeState(symbols=["BTC"])
    now = time.time()
    # Quote from 10 seconds ago
    state.update_quote("BTC", "bybit", 65000.0, timestamp=now - 10.0)

    fresh = state.get_fresh_venues("BTC", max_age_sec=5.0, now_ts=now)
    assert "bybit" not in fresh


def test_cross_exchange_confirmation_pass():
    state = CrossExchangeState(symbols=["BTC"])
    now = time.time()

    # Initial prices
    state.update_quote("BTC", "binance", 65000.0, timestamp=now - 3.0)
    state.update_quote("BTC", "bybit", 65000.0, timestamp=now - 3.0)
    state.update_quote("BTC", "okx", 65000.0, timestamp=now - 3.0)

    # Bybit and OKX pumped +12 bps
    state.update_quote("BTC", "bybit", 65078.0, timestamp=now)
    state.update_quote("BTC", "okx", 65078.0, timestamp=now)
    state.update_quote("BTC", "binance", 65010.0, timestamp=now)

    cecl = CrossExchangeConfirmation(state=state)
    decision = cecl.evaluate(symbol="BTC", side="LONG", now_ts=now)

    assert decision.status == "PASS"
    assert decision.score >= 0.20
    assert "Подтверждено" in decision.reason


def test_cross_exchange_confirmation_veto_momentum():
    state = CrossExchangeState(symbols=["BTC"])
    now = time.time()

    # Initial prices
    state.update_quote("BTC", "binance", 65000.0, timestamp=now - 3.0)
    state.update_quote("BTC", "bybit", 65000.0, timestamp=now - 3.0)
    state.update_quote("BTC", "okx", 65000.0, timestamp=now - 3.0)

    # External venues dumping while proposed side is LONG
    state.update_quote("BTC", "bybit", 64920.0, timestamp=now)
    state.update_quote("BTC", "okx", 64920.0, timestamp=now)
    state.update_quote("BTC", "binance", 65000.0, timestamp=now)

    cecl = CrossExchangeConfirmation(state=state)
    decision = cecl.evaluate(symbol="BTC", side="LONG", now_ts=now)

    assert decision.status == "VETO"
    assert decision.score <= -0.35


def test_cross_exchange_confirmation_veto_dislocation():
    state = CrossExchangeState(symbols=["BTC"])
    now = time.time()

    # External market at 65000, Binance at huge premium (+20 bps > 8 bps threshold)
    state.update_quote("BTC", "bybit", 65000.0, timestamp=now)
    state.update_quote("BTC", "okx", 65000.0, timestamp=now)
    state.update_quote("BTC", "binance", 65130.0, timestamp=now)

    cecl = CrossExchangeConfirmation(state=state)
    decision = cecl.evaluate(symbol="BTC", side="LONG", now_ts=now)

    assert decision.status == "VETO"
    assert "Премия Binance" in decision.reason


def test_cross_exchange_degraded_when_no_externals():
    state = CrossExchangeState(symbols=["BTC"])
    now = time.time()
    state.update_quote("BTC", "binance", 65000.0, timestamp=now)

    cecl = CrossExchangeConfirmation(state=state)
    decision = cecl.evaluate(symbol="BTC", side="LONG", now_ts=now)

    assert decision.status == "DEGRADED"
