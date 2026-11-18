import pytest
import numpy as np
import pandas as pd

from src.domain.services.microstructure_feature_engine import (
    compute_top_of_book_features,
    compute_cvd_features,
    compute_funding_velocity_and_oi,
    compute_cross_asset_beta_residuals,
    MicrostructureFeatureEngine
)

def test_top_of_book_features():
    book = pd.DataFrame([
        {"ts": pd.Timestamp("2026-09-11 12:00:00", tz="UTC"), "bid_px": 100.0, "bid_qty": 5.0, "ask_px": 100.1, "ask_qty": 5.0},
        {"ts": pd.Timestamp("2026-09-11 12:00:01", tz="UTC"), "bid_px": 100.05, "bid_qty": 8.0, "ask_px": 100.15, "ask_qty": 4.0},
    ])

    features = compute_top_of_book_features(book)
    assert "microprice" in features.columns
    assert "microprice_delta" in features.columns
    assert "ofi_norm" in features.columns
    assert features["microprice"].iloc[0] == pytest.approx(100.05, 1e-4)

def test_cvd_features():
    trades = pd.DataFrame([
        {"ts": pd.Timestamp("2026-09-11 12:00:00", tz="UTC"), "price": 100.0, "quantity": 2.0, "buyer_is_maker": False}, # Aggressive buy (+2)
        {"ts": pd.Timestamp("2026-09-11 12:00:01", tz="UTC"), "price": 100.2, "quantity": 1.5, "buyer_is_maker": True},  # Aggressive sell (-1.5)
        {"ts": pd.Timestamp("2026-09-11 12:00:02", tz="UTC"), "price": 100.1, "quantity": 3.0, "buyer_is_maker": False}, # Aggressive buy (+3)
    ])

    cvd_df = compute_cvd_features(trades)
    assert "cvd" in cvd_df.columns
    assert "cvd_divergence" in cvd_df.columns
    # Cumulative volume delta should be: 2.0 -> 0.5 -> 3.5
    assert cvd_df["cvd"].iloc[0] == 2.0
    assert cvd_df["cvd"].iloc[1] == 0.5
    assert cvd_df["cvd"].iloc[2] == 3.5

def test_cross_asset_beta_residuals():
    np.random.seed(42)
    btc_rets = pd.Series(np.random.normal(0.0001, 0.01, 150))
    # Altcoin is 1.5x beta + idiosyncratic noise
    alt_rets = 1.5 * btc_rets + pd.Series(np.random.normal(0, 0.005, 150))

    residuals = compute_cross_asset_beta_residuals(alt_rets, btc_rets, window=30)
    assert len(residuals) == 150
    valid_res = residuals.dropna()
    assert len(valid_res) > 50
    assert np.isfinite(valid_res).all()

def test_microstructure_feature_engine_integration():
    engine = MicrostructureFeatureEngine()

    orderbook = {
        "ts": pd.Timestamp("2026-09-11 12:00:00", tz="UTC"),
        "bid_px": 100.0,
        "bid_qty": 10.0,
        "ask_px": 100.1,
        "ask_qty": 10.0
    }
    recent_trades = [
        {"ts": pd.Timestamp("2026-09-11 12:00:00", tz="UTC"), "price": 100.05, "quantity": 1.0, "buyer_is_maker": False}
    ]

    features = engine.extract_features(
        symbol="SOLUSDT",
        orderbook=orderbook,
        recent_trades=recent_trades,
        funding_rate=0.0001,
        btc_return=0.002
    )

    expected_keys = [
        "spread_bps", "microprice_delta", "ofi_norm", "cvd_divergence",
        "funding_zscore", "long_squeeze_score", "short_squeeze_score",
        "cross_asset_residual_z"
    ]
    for key in expected_keys:
        assert key in features
        assert np.isfinite(features[key])
