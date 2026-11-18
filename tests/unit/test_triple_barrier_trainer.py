import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.infrastructure.mlops.triple_barrier_retrain_pipeline import (
    BarrierEvent,
    OOSMetrics,
    _prepare_candles,
    _normalise_symbol,
    train_symbol,
    TripleBarrierPipelineError
)

def test_barrier_event_dataclass():
    now = pd.Timestamp.now(tz="UTC")
    event = BarrierEvent(
        start_position=10,
        end_position=25,
        start_time=now,
        end_time=now + pd.Timedelta(hours=15),
        entry_price=100.0,
        upper_barrier=104.0,
        lower_barrier=98.0,
        stop_fraction=0.02,
        target_fraction=0.04,
        label=1,
        gross_return=0.04
    )
    assert event.label == 1
    assert event.entry_price == 100.0
    assert event.upper_barrier == 104.0

def test_oos_metrics_registry_dict():
    metrics = OOSMetrics(
        net_sharpe=1.85,
        realized_win_rate=0.58,
        profit_factor=2.15,
        net_expectancy=0.0035,
        brier_score=0.18,
        log_loss=0.55,
        expected_calibration_error=0.04,
        sample_count=500,
        trade_count=120,
        positive_label_rate=0.45
    )
    d = metrics.as_registry_dict()
    assert d["net_sharpe"] == 1.85
    assert d["profit_factor"] == 2.15
    assert d["sample_count"] == 500

def test_normalise_symbol():
    assert _normalise_symbol("btc/usdt") == "BTC_USDT"
    assert _normalise_symbol("sol-usdt") == "SOL_USDT"
    assert _normalise_symbol("ETHUSDT") == "ETHUSDT"

def test_prepare_candles_validation():
    # Valid dataframe
    dates = pd.date_range("2026-01-01", periods=250, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.linspace(100, 110, 250),
        "high": np.linspace(101, 111, 250),
        "low": np.linspace(99, 109, 250),
        "close": np.linspace(100.5, 110.5, 250),
        "volume": np.ones(250) * 1000.0
    })
    prepared = _prepare_candles(df)
    assert len(prepared) == 250
    assert isinstance(prepared.index, pd.DatetimeIndex)

    # Missing columns should raise error
    with pytest.raises(TripleBarrierPipelineError):
        _prepare_candles(pd.DataFrame({"open": [1, 2], "close": [1, 2]}))
