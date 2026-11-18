import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from src.infrastructure.data.live_microstructure_collector import (
    LiveMicrostructureCollector,
    MicrostructureSnapshot,
    MicrostructureCollectionError
)

def test_microstructure_snapshot_fields():
    snap = MicrostructureSnapshot(
        symbol="SOLUSDT",
        timestamp="2026-09-11T21:00:00Z",
        mark_price=101.5,
        ofi_norm=0.35,
        microprice_delta=0.012,
        cvd_delta=1250.0,
        aggressor_buyer_ratio=0.62,
        funding_rate=0.0001,
        funding_velocity=0.000005,
        oi=500000.0,
        oi_velocity=0.015,
        composite_flow_bias=0.45,
        flow_veto=False,
        veto_reason=None
    )
    assert snap.symbol == "SOLUSDT"
    assert snap.mark_price == 101.5
    assert snap.flow_veto is False
    assert snap.composite_flow_bias == 0.45

def test_fallback_top_of_book():
    collector = LiveMicrostructureCollector()
    orderbook = pd.DataFrame([
        {"bid_px": 100.0, "bid_qty": 10.0, "ask_px": 100.2, "ask_qty": 5.0},
        {"bid_px": 99.8, "bid_qty": 20.0, "ask_px": 100.4, "ask_qty": 15.0}
    ])
    ofi, delta_mu = collector._fallback_top_of_book_features(orderbook)
    assert -1.0 <= ofi <= 1.0
    assert np.isfinite(delta_mu)

def test_fallback_cvd_features():
    collector = LiveMicrostructureCollector()
    trades = pd.DataFrame([
        {"timestamp": 1000, "price": 100.0, "quantity": 2.0, "is_buyer_maker": False}, # buyer aggression
        {"timestamp": 2000, "price": 100.1, "quantity": 1.0, "is_buyer_maker": True},  # seller aggression
        {"timestamp": 3000, "price": 100.2, "quantity": 3.0, "is_buyer_maker": False}, # buyer aggression
    ])
    cvd, buyer_ratio = collector._fallback_cvd_features(trades)
    assert cvd == pytest.approx(4.0, rel=1e-3)
    assert buyer_ratio == pytest.approx(5.0 / 6.0, rel=1e-3)

def test_flow_veto_logic():
    collector = LiveMicrostructureCollector()
    orderbook = pd.DataFrame([
        {"bid_px": 100.0, "bid_qty": 1.0, "ask_px": 100.2, "ask_qty": 50.0}
    ])
    trades = pd.DataFrame([
        {"timestamp": 1000, "price": 100.0, "quantity": 100.0, "is_buyer_maker": True}
    ])
    with patch.object(collector, 'fetch_orderbook_snapshot', return_value=orderbook), \
         patch.object(collector, 'fetch_recent_agg_trades', return_value=trades), \
         patch.object(collector, 'fetch_funding_and_oi', return_value=(0.0005, 100.0, 10000.0)):
        
        snap = collector.get_live_microstructure("SOLUSDT", proposed_direction="BUY")
        assert snap.composite_flow_bias < 0.0
