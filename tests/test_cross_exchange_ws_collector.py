import json
import time
import pytest

from src.domain.services.cross_exchange_state import CrossExchangeState, QuoteUpdate
from src.domain.services.cross_exchange_ws_collector import (
    CrossExchangeWSCollector,
    BINANCE,
    BYBIT,
    OKX,
)


def test_binance_bookticker_parsing():
    state = CrossExchangeState()
    collector = CrossExchangeWSCollector(state=state)

    payload = {
        "e": "bookTicker",
        "u": 123456,
        "s": "BTCUSDT",
        "b": "65000.10",
        "B": "1.5",
        "a": "65000.20",
        "A": "2.0",
        "T": 1710000000000,
    }

    quotes = collector._parse_message(BINANCE, json.dumps(payload), {})
    assert len(quotes) == 1
    q = quotes[0]
    assert q.exchange == "binance"
    assert q.symbol == "BTCUSDT"
    assert q.bid == 65000.10
    assert q.ask == 65000.20
    assert q.mid_price == pytest.approx(65000.15, abs=1e-5)
    assert q.source == "ws"


def test_bybit_snapshot_and_delta_merge():
    state = CrossExchangeState()
    collector = CrossExchangeWSCollector(state=state)
    cache = {}

    # Snapshot message
    snap_payload = {
        "topic": "tickers.BTCUSDT",
        "type": "snapshot",
        "ts": 1710000000000,
        "data": {
            "symbol": "BTCUSDT",
            "bid1Price": "65000.10",
            "ask1Price": "65000.30",
            "lastPrice": "65000.20",
        },
    }

    quotes1 = collector._parse_message(BYBIT, json.dumps(snap_payload), cache)
    assert len(quotes1) == 1
    assert quotes1[0].bid == 65000.10
    assert quotes1[0].ask == 65000.30

    # Delta message: only ask changed
    delta_payload = {
        "topic": "tickers.BTCUSDT",
        "type": "delta",
        "ts": 1710000001000,
        "data": {
            "symbol": "BTCUSDT",
            "ask1Price": "65000.25",
        },
    }

    quotes2 = collector._parse_message(BYBIT, json.dumps(delta_payload), cache)
    assert len(quotes2) == 1
    # Bid should be preserved from snapshot cache!
    assert quotes2[0].bid == 65000.10
    assert quotes2[0].ask == 65000.25


def test_okx_ticker_parsing():
    state = CrossExchangeState()
    collector = CrossExchangeWSCollector(state=state)

    payload = {
        "arg": {"channel": "tickers", "instId": "BTC-USDT-SWAP"},
        "data": [
            {
                "instId": "BTC-USDT-SWAP",
                "bidPx": "65000.05",
                "askPx": "65000.15",
                "last": "65000.10",
                "ts": "1710000000000",
            }
        ],
    }

    quotes = collector._parse_message(OKX, json.dumps(payload), {})
    assert len(quotes) == 1
    q = quotes[0]
    assert q.exchange == "okx"
    assert q.symbol == "BTCUSDT"
    assert q.bid == 65000.05
    assert q.ask == 65000.15


def test_invalid_crossed_quote_rejected():
    state = CrossExchangeState()
    collector = CrossExchangeWSCollector(state=state)

    # Inverted spread: ask < bid
    payload = {
        "e": "bookTicker",
        "s": "BTCUSDT",
        "b": "65010.00",
        "a": "65000.00",
    }
    quotes = collector._parse_message(BINANCE, json.dumps(payload), {})
    assert len(quotes) == 0


def test_quote_update_monotonic_ordering():
    state = CrossExchangeState()

    now_ns = time.monotonic_ns()
    q1 = QuoteUpdate(
        exchange="binance",
        symbol="BTC",
        bid=65000.0,
        ask=65001.0,
        exchange_ts_ms=1000,
        received_monotonic_ns=now_ns,
        source="ws",
    )
    # Stale/out-of-order quote arrived later with earlier monotonic ts
    q_stale = QuoteUpdate(
        exchange="binance",
        symbol="BTC",
        bid=64990.0,
        ask=64991.0,
        exchange_ts_ms=900,
        received_monotonic_ns=now_ns - 1_000_000,
        source="ws",
    )

    assert state.update_quote_update(q1) is True
    assert state.update_quote_update(q_stale) is False

    # State still contains q1
    latest = state.get_quote("binance", "BTC")
    assert latest.bid == 65000.0


def test_backoff_bounded():
    collector = CrossExchangeWSCollector(state=CrossExchangeState())
    for attempt in range(1, 15):
        delay = collector._backoff_delay_s(attempt)
        assert 0.0 <= delay <= 30.0
