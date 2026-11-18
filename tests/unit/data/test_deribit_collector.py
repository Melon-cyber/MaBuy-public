"""Unit tests for DeribitCollector."""

import json
from unittest.mock import MagicMock, patch
import pytest

from src.domain.services.deribit_collector import (
    DeribitCollector,
    DeribitQuote,
    DeribitDVOL,
)


def test_deribit_quote_properties():
    q = DeribitQuote(
        symbol="BTC",
        instrument_name="BTC-PERPETUAL",
        timestamp_ms=1789280000000,
        best_bid=77000.0,
        best_ask=77001.0,
        mid_price=77000.5,
        mark_price=77000.4,
        index_price=77000.0,
        spread_bps=0.13,
        funding_8h=0.0001,
        open_interest=50000000.0,
    )
    assert q.is_valid is True
    assert q.symbol == "BTC"
    assert q.mid_price == 77000.5


def test_deribit_dvol_properties():
    d = DeribitDVOL(
        currency="BTC",
        timestamp_ms=1789280000000,
        dvol_open=36.5,
        dvol_high=38.0,
        dvol_low=36.0,
        dvol_close=37.2,
    )
    assert d.currency == "BTC"
    assert d.dvol_close == 37.2


@patch.object(DeribitCollector, "_http_get_json")
def test_deribit_collector_poll(mock_http):
    # Mock responses for ticker and DVOL
    def side_effect(url: str):
        if "ticker" in url:
            return {
                "result": {
                    "best_bid_price": 77100.0,
                    "best_ask_price": 77101.0,
                    "mark_price": 77100.5,
                    "index_price": 77100.0,
                    "funding_8h": 0.00005,
                    "open_interest": 80000000,
                    "timestamp": 1789280000000,
                }
            }
        elif "get_volatility_index_data" in url:
            return {
                "result": {
                    "data": [
                        [1789280000000, 36.0, 37.5, 35.8, 37.0]
                    ]
                }
            }
        return {}

    mock_http.side_effect = side_effect

    collector = DeribitCollector(auto_start=False)
    collector.poll_all_now()

    q_btc = collector.get_quote("BTC")
    assert q_btc is not None
    assert q_btc.best_bid == 77100.0
    assert q_btc.best_ask == 77101.0
    assert q_btc.mid_price == 77100.5

    # Test symbol normalization
    q_btc_slash = collector.get_quote("BTC/USDT")
    assert q_btc_slash is not None
    assert q_btc_slash.symbol == "BTC"

    # Test DVOL
    dvol_btc = collector.get_dvol("BTC")
    assert dvol_btc is not None
    assert dvol_btc.dvol_close == 37.0
