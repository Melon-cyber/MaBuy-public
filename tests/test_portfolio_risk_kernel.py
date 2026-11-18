"""
Unit tests for Unified Portfolio Risk Kernel & Dynamic Sizing (Astra Step 4 Specification).
"""

import pytest
from src.domain.services.portfolio_risk_kernel import (
    CandidateOrder,
    PlattCalibrator,
    PortfolioRiskKernel,
    PortfolioSnapshot,
    PositionSnapshot,
    RiskKernelConfig,
    canonical_pair,
)


def test_canonical_pair_normalization():
    assert canonical_pair("BTC/USDT:USDT", "ETH/USDT") == ("BTC", "ETH")
    assert canonical_pair("SOLUSDT", "BTCUSDT") == ("BTC", "SOL")
    assert canonical_pair("eth", "btc") == ("BTC", "ETH")


def test_platt_calibrator():
    calibrator = PlattCalibrator(a=1.20, b=-0.10)
    p_neutral = calibrator.calibrate(0.50)
    assert 0.45 <= p_neutral <= 0.55

    p_high = calibrator.calibrate(0.80)
    assert p_high > p_neutral
    assert 0.70 <= p_high <= 0.85

    p_low = calibrator.calibrate(0.20)
    assert p_low < p_neutral

    # Edge cases
    assert calibrator.calibrate(0.0) >= 0.001
    assert calibrator.calibrate(1.0) <= 0.999


def test_veto_invalid_stop_and_tp_directions():
    kernel = PortfolioRiskKernel()
    portfolio = PortfolioSnapshot(equity_usd=100.0, day_start_equity_usd=100.0)

    # Long stop above entry
    bad_long = CandidateOrder(
        symbol="SOLUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=105.0,  # Invalid: above entry
        take_profit_price=110.0,
        raw_confidence=0.75,
    )
    res = kernel.evaluate_candidate_order(portfolio, bad_long)
    assert not res.approved
    assert "LONG stop must be below entry" in res.veto_reason

    # Short stop below entry
    bad_short = CandidateOrder(
        symbol="SOLUSDT",
        side="SHORT",
        entry_price=100.0,
        stop_price=95.0,  # Invalid: below entry
        take_profit_price=90.0,
        raw_confidence=0.75,
    )
    res = kernel.evaluate_candidate_order(portfolio, bad_short)
    assert not res.approved
    assert "SHORT stop must be above entry" in res.veto_reason


def test_veto_tp_does_not_cover_costs():
    config = RiskKernelConfig(
        entry_fee_fraction=0.001,
        exit_fee_fraction=0.001,
        entry_slippage_fraction=0.001,
        exit_slippage_fraction=0.001,
    )  # Round trip cost = 0.004 (0.40%)
    kernel = PortfolioRiskKernel(config=config)
    portfolio = PortfolioSnapshot(equity_usd=100.0, day_start_equity_usd=100.0)

    # TP is only 0.20% above entry -> does not cover 0.40% round trip cost
    cand = CandidateOrder(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=98.0,
        take_profit_price=100.20,
        raw_confidence=0.80,
    )
    res = kernel.evaluate_candidate_order(portfolio, cand)
    assert not res.approved
    assert res.veto_reason == "TAKE_PROFIT_DOES_NOT_COVER_COSTS"


def test_veto_non_positive_net_kelly():
    kernel = PortfolioRiskKernel()
    portfolio = PortfolioSnapshot(equity_usd=100.0, day_start_equity_usd=100.0)

    # Very low confidence (0.15) with low payoff ratio (1:1) -> negative Kelly expectation
    cand = CandidateOrder(
        symbol="ETHUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=98.0,
        take_profit_price=102.0,
        raw_confidence=0.15,
    )
    res = kernel.evaluate_candidate_order(portfolio, cand)
    assert not res.approved
    assert res.veto_reason == "NON_POSITIVE_NET_KELLY"


def test_veto_daily_drawdown_limit():
    kernel = PortfolioRiskKernel(
        config=RiskKernelConfig(daily_drawdown_limit_fraction=0.02)
    )
    # Day started at $100, current equity is $97.50 -> drawdown is 2.5% > 2.0%
    portfolio = PortfolioSnapshot(equity_usd=97.50, day_start_equity_usd=100.0)

    cand = CandidateOrder(
        symbol="SOLUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=98.0,
        take_profit_price=105.0,
        raw_confidence=0.80,
    )
    res = kernel.evaluate_candidate_order(portfolio, cand)
    assert not res.approved
    assert res.veto_reason == "DAILY_DRAWDOWN_LIMIT_REACHED"


def test_veto_margin_utilization_limit():
    kernel = PortfolioRiskKernel(
        config=RiskKernelConfig(max_margin_utilization_fraction=0.40)
    )
    # Equity = $100, existing position uses $45 margin -> 45% > 40%
    portfolio = PortfolioSnapshot(
        equity_usd=100.0,
        day_start_equity_usd=100.0,
        positions=(
            PositionSnapshot(symbol="BTCUSDT", signed_notional_usd=135.0, margin_usd=45.0, leverage=3.0),
        ),
    )

    cand = CandidateOrder(
        symbol="SOLUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=98.0,
        take_profit_price=105.0,
        raw_confidence=0.80,
    )
    res = kernel.evaluate_candidate_order(portfolio, cand)
    assert not res.approved
    assert res.veto_reason == "MARGIN_UTILIZATION_LIMIT_REACHED"


def test_directional_correlation_haircut():
    kernel = PortfolioRiskKernel(
        config=RiskKernelConfig(
            min_notional_usd=1.0,
            default_missing_correlation=0.80,
            alpha=0.20,
            f_max=0.0125,
        )
    )

    cand = CandidateOrder(
        symbol="SOLUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=98.0,
        take_profit_price=106.0,
        raw_confidence=0.82,
        requested_leverage=3.0,
    )

    # Scenario A: Portfolio is flat
    port_flat = PortfolioSnapshot(equity_usd=100.0, day_start_equity_usd=100.0)
    res_flat = kernel.evaluate_candidate_order(port_flat, cand)
    assert res_flat.approved
    assert res_flat.correlation_haircut == 1.0

    # Scenario B: Portfolio holds LONG ETH ($60 notional, same direction as new LONG SOL)
    port_long_eth = PortfolioSnapshot(
        equity_usd=100.0,
        day_start_equity_usd=100.0,
        positions=(
            PositionSnapshot(symbol="ETHUSDT", signed_notional_usd=60.0, margin_usd=20.0, leverage=3.0),
        ),
        correlations={("ETH", "SOL"): 0.85},
    )
    res_long = kernel.evaluate_candidate_order(port_long_eth, cand)
    assert res_long.approved
    # Haircut must reduce size because directional overlap is positive
    assert res_long.correlation_haircut < 1.0
    assert res_long.approved_notional_usd < res_flat.approved_notional_usd

    # Scenario C: Portfolio holds SHORT ETH (opposing direction, hedge)
    port_short_eth = PortfolioSnapshot(
        equity_usd=100.0,
        day_start_equity_usd=100.0,
        positions=(
            PositionSnapshot(symbol="ETHUSDT", signed_notional_usd=-60.0, margin_usd=20.0, leverage=3.0),
        ),
        correlations={("ETH", "SOL"): 0.85},
    )
    res_short = kernel.evaluate_candidate_order(port_short_eth, cand)
    assert res_short.approved
    # Opposing position does not increase directional risk -> no haircut penalty
    assert res_short.correlation_haircut == 1.0


def test_liquidation_distance_safety():
    kernel = PortfolioRiskKernel(
        config=RiskKernelConfig(
            min_liquidation_distance_fraction=0.20,
            require_exchange_liquidation_distance=False,
        )
    )
    portfolio = PortfolioSnapshot(equity_usd=100.0, day_start_equity_usd=100.0)

    cand = CandidateOrder(
        symbol="SOLUSDT",
        side="LONG",
        entry_price=100.0,
        stop_price=98.0,
        take_profit_price=106.0,
        raw_confidence=0.80,
        requested_leverage=3.0,
    )
    res = kernel.evaluate_candidate_order(portfolio, cand)
    assert res.approved
    # For 3x leverage, liquidation distance is approx 1/3 - 0.01 ~= 32.3% >= 20%
    assert res.liquidation_distance_fraction >= 0.20
    assert res.recommended_leverage == 3.0


def test_approved_risk_invariant():
    config = RiskKernelConfig(
        max_trade_risk_fraction=0.0125,  # 1.25% of equity = $1.25 max risk on $100
        min_notional_usd=5.0,
    )
    kernel = PortfolioRiskKernel(config=config)
    portfolio = PortfolioSnapshot(equity_usd=100.0, day_start_equity_usd=100.0)

    cand = CandidateOrder(
        symbol="BTCUSDT",
        side="LONG",
        entry_price=70000.0,
        stop_price=68740.0,  # 1.8% SL
        take_profit_price=72450.0,  # 3.5% TP
        raw_confidence=0.78,
        requested_leverage=3.0,
    )
    res = kernel.evaluate_candidate_order(portfolio, cand)
    assert res.approved
    assert res.estimated_trade_risk_usd <= 100.0 * 0.0125 + 1e-9

    # Verify Russian summary formatting
    summary = res.format_russian_summary("BTCUSDT", "LONG")
    assert "РИСК-ОДОБРЕНО" in summary
    assert "symbol=BTCUSDT" in summary
    assert "lev=3.00x" in summary
