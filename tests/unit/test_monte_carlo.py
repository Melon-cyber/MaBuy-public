import pytest
import numpy as np

from src.domain.services.monte_carlo_risk_simulator import (
    AssetContract,
    SimulationConfig,
    MonteCarloRiskSimulator
)

def test_asset_contract_initialization():
    btc = AssetContract(
        symbol="BTCUSDT",
        reference_price=85000.0,
        minimum_quantity=0.001,
        target_weight=0.35,
        annual_expected_return=0.12,
        annual_volatility=0.65
    )
    assert btc.symbol == "BTCUSDT"
    assert btc.minimum_notional == pytest.approx(85.0)

def test_monte_carlo_run_and_metrics():
    cfg = SimulationConfig(
        batch_size=512
    )
    simulator = MonteCarloRiskSimulator(
        equity=150.0,
        trades=20,
        simulations=500,
        seed=42,
        config=cfg
    )
    results = simulator.run_simulation()
    
    assert 30 in results
    assert 90 in results
    assert 365 in results
    
    res_30 = results[30]
    assert 0.0 <= res_30.probability_of_ruin <= 1.0
    assert 0.0 <= res_30.probability_capital_breach_5pct <= 1.0
    assert np.isfinite(res_30.var_99)
    assert np.isfinite(res_30.cvar_99)
    assert np.isfinite(res_30.maximum_drawdown_median)

def test_monte_carlo_advisory_report():
    simulator = MonteCarloRiskSimulator(
        equity=39.54,
        trades=10,
        simulations=200,
        seed=123
    )
    report = simulator.generate_advisory_report()
    assert "ASTRA INSTITUTIONAL AUDIT" in report
    assert "CAPITAL ALLOCATION ADVISORY" in report
    assert "$39.54" in report
