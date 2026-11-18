"""Unit tests for SyntheticSpreadEngine (Astra Step 3 StatArb)."""

import math
import pytest
from src.domain.services.synthetic_spread_engine import (
    SpreadAction,
    SpreadPosition,
    SyntheticSpreadConfig,
    SyntheticSpreadEngine,
)


def test_initialization_and_warmup():
    config = SyntheticSpreadConfig(initial_beta=1.0)
    engine = SyntheticSpreadEngine(config)

    assert engine.position == SpreadPosition.FLAT
    assert engine.beta == 1.0

    t0 = 1700000000.0
    sol_p = 150.0
    btc_p = 60000.0

    telemetry = engine.update(sol_mid=sol_p, btc_mid=btc_p, timestamp=t0)
    assert telemetry.status == "WARMUP"
    assert telemetry.action == SpreadAction.HOLD
    assert telemetry.tradable is False
    assert telemetry.alpha is not None
    # alpha should equal ln(150) - 1.0 * ln(60000)
    expected_alpha = math.log(150.0) - math.log(60000.0)
    assert math.isclose(telemetry.alpha, expected_alpha, rel_tol=1e-6)


def test_invalid_and_stale_prices():
    engine = SyntheticSpreadEngine()
    now = 1700000000.0

    # Negative / zero prices
    tel_zero = engine.update(sol_mid=0.0, btc_mid=60000.0, timestamp=now)
    assert tel_zero.status == "BLOCKED"
    assert tel_zero.reason == "invalid_non_positive_price"

    tel_nan = engine.update(sol_mid=float("nan"), btc_mid=60000.0, timestamp=now)
    assert tel_nan.status == "BLOCKED"
    assert tel_nan.reason == "invalid_non_positive_price"

    # Stale quotes
    tel_stale = engine.update(
        sol_mid=150.0,
        btc_mid=60000.0,
        timestamp=now,
        sol_quote_timestamp=now - 5.0,  # 5s old > 3s limit
        btc_quote_timestamp=now,
    )
    assert tel_stale.status == "BLOCKED"
    assert tel_stale.reason == "sol_quote_stale"

    # Timestamp skew
    tel_skew = engine.update(
        sol_mid=150.0,
        btc_mid=60000.0,
        timestamp=now,
        sol_quote_timestamp=now,
        btc_quote_timestamp=now - 4.0,
    )
    assert tel_skew.status == "BLOCKED"
    assert tel_skew.reason in ("btc_quote_stale", "sol_btc_quote_timestamp_skew")


def test_kalman_beta_tracking():
    # Test that Kalman filter adapts beta towards true data generating process
    config = SyntheticSpreadConfig(
        initial_beta=1.0,
        q_beta_per_second=1e-5,  # Accelerated for rapid unit-test convergence
        measurement_r=1e-7,
    )
    engine = SyntheticSpreadEngine(config)

    true_alpha = -6.0
    true_beta = 1.25
    now = 1700000000.0

    for i in range(100):
        btc = 50000.0 + i * 200.0
        ln_sol = true_alpha + true_beta * math.log(btc)
        sol = math.exp(ln_sol)

        engine.update(sol_mid=sol, btc_mid=btc, timestamp=now + i * 2.0)

    # Beta should move positively towards 1.25 from initial 1.0
    assert engine.beta > 1.005


def test_ou_convergence_and_telemetry():
    # Test OU estimator on synthetic mean-reverting AR(1) process
    config = SyntheticSpreadConfig(
        ou_sample_interval_seconds=5.0,
        ou_sample_tolerance_ratio=0.50,
        min_ou_effective_samples=15.0,
        ou_estimation_half_life_seconds=600.0,
        min_half_life_seconds=10.0,
        max_half_life_seconds=10000.0,
    )
    engine = SyntheticSpreadEngine(config)
    now = 1700000000.0

    btc_base = 60000.0
    alpha = math.log(150.0) - math.log(btc_base)
    s = 0.0
    phi = 0.85

    for i in range(50):
        t = now + i * 5.0
        s = phi * s + 0.005 * math.sin(i * 0.4)
        sol = math.exp(alpha + math.log(btc_base) + s)

        tel = engine.update(sol_mid=sol, btc_mid=btc_base, timestamp=t)

    # After 50 samples (> 15 effective), OU should be ready
    assert engine._ou_mu is not None
    assert engine._sigma_eq is not None
    assert engine._sigma_eq > 0.0
    assert engine._half_life_seconds is not None
    assert config.min_half_life_seconds <= engine._half_life_seconds <= config.max_half_life_seconds


def test_signal_triggers_and_positions():
    config = SyntheticSpreadConfig(
        ou_sample_interval_seconds=5.0,
        min_ou_effective_samples=10.0,
        min_half_life_seconds=10.0,
        max_half_life_seconds=50000.0,
        entry_z=2.0,
        exit_z=0.3,
        catastrophic_z=3.8,
        sol_one_way_fee_bps=1.0,
        btc_one_way_fee_bps=1.0,
        sol_one_way_slippage_bps=0.5,
        btc_one_way_slippage_bps=0.5,
        edge_safety_buffer_bps=1.0,
    )
    engine = SyntheticSpreadEngine(config)
    now = 1700000000.0
    btc = 60000.0
    sol_base = 150.0
    alpha = math.log(sol_base) - math.log(btc)

    # Warmup with small oscillations around 0
    for i in range(30):
        t = now + i * 5.0
        s = 0.002 * math.sin(i * 0.5)
        sol = math.exp(alpha + math.log(btc) + s)
        engine.update(sol_mid=sol, btc_mid=btc, timestamp=t)

    assert engine._sigma_eq is not None
    sigma_eq = engine._sigma_eq
    mu = engine._ou_mu or 0.0

    # 1. Trigger ENTER_LONG_SPREAD: Target Z = -2.5
    t = now + 31 * 5.0
    pred_y = engine._alpha + engine._beta * math.log(btc)
    target_sol = math.exp(pred_y + mu - 2.5 * sigma_eq)

    tel_long = engine.update(sol_mid=target_sol, btc_mid=btc, timestamp=t)
    assert tel_long.action == SpreadAction.ENTER_LONG_SPREAD
    assert engine.position == SpreadPosition.LONG_SPREAD
    assert tel_long.zscore is not None and tel_long.zscore <= -2.0

    # 2. While in position, moderate move keeps it OPEN (Z = -1.2)
    t += 5.0
    pred_y = engine._alpha + engine._beta * math.log(btc)
    target_sol = math.exp(pred_y + mu - 1.2 * sigma_eq)
    tel_hold = engine.update(sol_mid=target_sol, btc_mid=btc, timestamp=t)
    assert tel_hold.action == SpreadAction.HOLD
    assert tel_hold.status == "IN_POSITION"
    assert engine.position == SpreadPosition.LONG_SPREAD

    # 3. Mean reversion to 0 (|Z| <= 0.3) -> EXIT_MEAN_REVERSION
    t += 5.0
    pred_y = engine._alpha + engine._beta * math.log(btc)
    target_sol = math.exp(pred_y + (engine._ou_mu or 0.0))
    tel_exit = engine.update(sol_mid=target_sol, btc_mid=btc, timestamp=t)
    assert tel_exit.action in (SpreadAction.EXIT_MEAN_REVERSION, SpreadAction.EXIT_ZERO_CROSS)
    assert engine.position == SpreadPosition.FLAT

    # 4. Trigger ENTER_SHORT_SPREAD: Target Z >= +2.0
    t += 5.0
    pred_y = engine._alpha + engine._beta * math.log(btc)
    target_sol = math.exp(pred_y + (engine._ou_mu or 0.0) + 3.5 * engine._sigma_eq)
    tel_short = engine.update(sol_mid=target_sol, btc_mid=btc, timestamp=t)
    assert tel_short.action == SpreadAction.ENTER_SHORT_SPREAD
    assert engine.position == SpreadPosition.SHORT_SPREAD
    assert tel_short.zscore is not None and tel_short.zscore >= 2.0

    # 5. Catastrophic explosion (structural breakdown |Z| >= 3.8 while in short position)
    t += 5.0
    pred_y = engine._alpha + engine._beta * math.log(btc)
    target_sol = math.exp(pred_y + (engine._ou_mu or 0.0) + 6.0 * engine._sigma_eq)
    tel_stop = engine.update(sol_mid=target_sol, btc_mid=btc, timestamp=t)
    assert tel_stop.action == SpreadAction.EXIT_STRUCTURAL_STOP
    assert engine.position == SpreadPosition.FLAT


def test_russian_summary_and_logging_format():
    engine = SyntheticSpreadEngine()
    tel = engine.update(sol_mid=150.0, btc_mid=60000.0, timestamp=1700000000.0)

    summary = tel.format_russian_summary()
    assert isinstance(summary, str)
    assert "Z=" in summary
    assert "Beta=" in summary
    assert "Статус:" in summary

    d = tel.to_log_dict()
    assert d["event"] == "synthetic_spread_telemetry"
    assert d["sol_mid"] == 150.0
    assert d["btc_mid"] == 60000.0
    assert d["action"] == "HOLD"
