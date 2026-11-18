import pytest
import numpy as np
import pandas as pd
from src.domain.services.dynamic_covariance_engine import DynamicCovarianceEngine

def generate_synthetic_returns(n_periods: int = 150) -> pd.DataFrame:
    np.random.seed(42)
    cov_true = np.array([
        [0.0009, 0.0007, 0.0006],
        [0.0007, 0.0016, 0.0009],
        [0.0006, 0.0009, 0.0025]
    ])
    mean = np.array([0.0002, 0.0003, 0.0005])
    returns = np.random.multivariate_normal(mean, cov_true, size=n_periods)
    return pd.DataFrame(returns, columns=["BTC", "ETH", "SOL"])

def test_covariance_matrix_properties():
    engine = DynamicCovarianceEngine()
    df = generate_synthetic_returns(100)
    engine.update_returns(df)
    
    cov = engine.get_covariance_matrix()
    corr = engine.get_correlation_matrix()
    
    assert cov.shape == (3, 3), "Covariance matrix must be 3x3"
    assert np.allclose(cov, cov.T, atol=1e-7), "Covariance matrix must be symmetric"
    
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues >= -1e-9), f"Covariance matrix must be positive semi-definite: {eigenvalues}"
    
    assert np.allclose(np.diag(corr), 1.0, atol=1e-5), "Correlation diagonals must be 1.0"
    assert np.all(corr >= -1.0 - 1e-5) and np.all(corr <= 1.0 + 1e-5), "Correlations must be in [-1, 1]"

def test_risk_parity_weights():
    engine = DynamicCovarianceEngine()
    df = generate_synthetic_returns(120)
    engine.update_returns(df)
    
    weights = engine.compute_risk_parity_weights(["BTC", "ETH", "SOL"])
    
    assert isinstance(weights, dict), "Weights must be a dict"
    assert set(weights.keys()) == {"BTC", "ETH", "SOL"}
    total_w = sum(weights.values())
    assert abs(total_w - 1.0) < 1e-4, f"Weights must sum to 1.0, got {total_w}"
    
    # In synthetic covariance, SOL has highest vol (0.0025) and BTC lowest (0.0009).
    # Risk Parity must allocate higher weight to BTC than SOL.
    assert weights["BTC"] > weights["SOL"], f"Risk Parity must weight BTC ({weights['BTC']:.3f}) > SOL ({weights['SOL']:.3f})"

def test_position_sizing_micro_capital():
    engine = DynamicCovarianceEngine()
    df = generate_synthetic_returns(100)
    engine.update_returns(df)
    
    equity = 39.54
    active_signals = {"SOL": "BULLISH"}
    current_prices = {"SOL": 130.0}
    
    sizes = engine.calculate_position_sizes(
        equity=equity,
        active_signals=active_signals,
        current_prices=current_prices
    )
    
    assert isinstance(sizes, dict)
    if "SOL" in sizes and sizes["SOL"] > 0:
        notional = sizes["SOL"] * current_prices["SOL"]
        assert notional <= equity * 2.5, f"Position notional ${notional:.2f} breaches leverage guard"

def test_portfolio_risk_report():
    engine = DynamicCovarianceEngine()
    df = generate_synthetic_returns(80)
    engine.update_returns(df)
    
    report = engine.get_portfolio_risk_report()
    assert isinstance(report, dict)
