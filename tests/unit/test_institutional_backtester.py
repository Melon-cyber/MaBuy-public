import pytest
import numpy as np
import pandas as pd
from src.domain.services.institutional_backtester import InstitutionalBacktester

def generate_synthetic_ohlcv(n_bars: int = 300) -> pd.DataFrame:
    np.random.seed(123)
    timestamps = pd.date_range("2026-01-01", periods=n_bars, freq="1h")
    close = 100.0 + np.cumsum(np.random.normal(0.05, 1.0, size=n_bars))
    high = close + np.random.uniform(0.2, 1.5, size=n_bars)
    low = close - np.random.uniform(0.2, 1.5, size=n_bars)
    open_p = (high + low) / 2.0
    volume = np.random.uniform(1000, 5000, size=n_bars)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume
    })
    df.set_index("timestamp", inplace=True)
    return df

def test_backtester_execution_and_metrics():
    df = generate_synthetic_ohlcv(200)
    # Generate alternating signals
    signals = pd.Series(0, index=df.index)
    signals.iloc[10] = 1   # Buy
    signals.iloc[40] = -1  # Sell
    signals.iloc[80] = 1   # Buy
    signals.iloc[130] = -1 # Sell
    
    bt = InstitutionalBacktester(initial_capital=1000.0, maker_fee=0.0002, taker_fee=0.0005)
    result = bt.run_backtest(df=df, signals=signals, stop_atr_mult=2.0, tp_atr_mult=3.0, maker_execution=True)
    
    assert result is not None
    assert hasattr(result, "net_pnl")
    assert hasattr(result, "sharpe_ratio")
    assert hasattr(result, "profit_factor")
    assert hasattr(result, "trades")
    
    report = bt.generate_report()
    assert isinstance(report, str)
    assert len(report) > 50

def test_deflated_sharpe_ratio():
    bt = InstitutionalBacktester()
    # Test DSR calculation with arbitrary returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.01, size=250)
    dsr = bt.compute_deflated_sharpe_ratio(returns=returns, n_trials=50)
    
    assert 0.0 <= dsr <= 1.0, f"DSR must be a probability in [0, 1], got {dsr}"
