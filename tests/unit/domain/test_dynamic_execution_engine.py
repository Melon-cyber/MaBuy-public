# -*- coding: utf-8 -*-
import pytest
import numpy as np
from src.domain.services.dynamic_execution_engine import (
    DynamicParametricExecutionEngine,
    ExecutionTargetParameters,
    TrailingUpdateResult
)

@pytest.fixture
def engine():
    return DynamicParametricExecutionEngine()

def test_volatility_estimators(engine):
    highs = np.array([102, 104, 103, 105, 106, 105, 107, 108, 107, 109, 110, 112, 111, 113], dtype=float)
    lows = np.array([98, 100, 99, 101, 102, 101, 103, 104, 103, 105, 106, 108, 107, 109], dtype=float)
    opens = np.array([99, 101, 100, 102, 103, 102, 104, 105, 104, 106, 107, 109, 108, 110], dtype=float)
    closes = np.array([101, 103, 102, 104, 105, 104, 106, 107, 106, 108, 109, 111, 110, 112], dtype=float)

    sig_p = engine.calculate_parkinson_volatility(highs, lows)
    sig_gk = engine.calculate_garman_klass_volatility(opens, highs, lows, closes)
    sig_natr = engine.calculate_natr(highs, lows, closes)

    assert sig_p > 0
    assert sig_gk > 0
    assert sig_natr > 0

def test_dynamic_risk_reward_expansion(engine):
    rr_low, _ = engine.calculate_dynamic_risk_reward(model_prob_win=0.55, ofi_aligned=0.0)
    rr_high, _ = engine.calculate_dynamic_risk_reward(model_prob_win=0.85, ofi_aligned=0.60)
    assert rr_low >= 1.80
    assert rr_high > rr_low
    assert rr_high <= 3.20

def test_entry_targets_calculation(engine):
    highs = np.array([102, 104, 103, 105, 106, 105, 107, 108, 107, 109, 110, 112, 111, 113], dtype=float)
    lows = np.array([98, 100, 99, 101, 102, 101, 103, 104, 103, 105, 106, 108, 107, 109], dtype=float)
    opens = np.array([99, 101, 100, 102, 103, 102, 104, 105, 104, 106, 107, 109, 108, 110], dtype=float)
    closes = np.array([101, 103, 102, 104, 105, 104, 106, 107, 106, 108, 109, 111, 110, 112], dtype=float)

    res = engine.calculate_entry_targets(
        symbol='BTCUSDT',
        direction='BULLISH',
        current_price=100000.0,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        model_prob_win=0.78,
        raw_ofi=0.35,
        market_regime='BULL_TREND'
    )
    assert res.entry_price == 100000.0
    assert res.stop_loss_price < 100000.0
    assert res.take_profit_price > 100000.0
    assert res.risk_reward_ratio >= 2.0

def test_momentum_trailing_stop(engine):
    prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110], dtype=float)
    res = engine.update_trailing_stop(
        direction='BULLISH',
        entry_price=100.0,
        current_price=106.0,
        peak_price=106.0,
        initial_sl_price=98.0,
        previous_sl_price=98.0,
        tp_price=110.0,
        sigma_eff=0.015,
        recent_price_series=prices
    )
    assert res.is_break_even_active is True
    assert res.updated_stop_loss >= 100.0  # BE active
