"""Parity, lifecycle, and fallback tests for `src.native.rust_bridge` and integrated services."""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.domain.services.hurst_regime_service import HurstRegimeService
from src.domain.services.order_flow_imbalance import OrderFlowImbalanceService
from src.infrastructure.ml.v4_titan_ensemble import HurstRegimeDetector
from src.native import rust_bridge
from src.native.parity.check_parity import py_single_scale_hurst_ddof0


@pytest.fixture(autouse=True)
def _restore_rust_backend(monkeypatch: pytest.MonkeyPatch):
    """Ensure Rust backend is restored after tests that toggle env vars."""
    yield
    monkeypatch.delenv("MABUY_DISABLE_RUST", raising=False)
    monkeypatch.delenv("MABUY_RUST_LIB", raising=False)
    rust_bridge.reload_backend()


def _make_ar1_prices(n: int, phi: float, seed: int = 42, drift: float = 0.0) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.float64)
    if n == 1:
        return np.array([100.0], dtype=np.float64)
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 0.004, size=n - 1)
    rets = np.empty(n - 1, dtype=np.float64)
    prev = 0.002 if phi > 0 else 0.0
    for i, e in enumerate(eps):
        prev = phi * prev + drift + float(e)
        rets[i] = prev
    return np.concatenate(([100.0], 100.0 * np.exp(np.cumsum(rets))))


def test_rust_backend_loaded_and_available():
    assert rust_bridge.is_rust_available() is True
    info = rust_bridge.get_backend_info()
    assert info["enabled"] is True
    assert info["available"] is True
    assert info["disabled_by_env"] is False
    assert info["lib_path"] is not None
    assert info["lib_path"].endswith(("mabuy_engine_core.dll", ".so", ".dylib"))


@pytest.mark.parametrize("n", [0, 5, 19, 20, 36, 100, 256, 1000])
@pytest.mark.parametrize(
    "kind,phi",
    [
        ("trending", 0.70),
        ("random_walk", 0.0),
        ("mean_reverting", -0.70),
    ],
)
def test_hurst_rs_multiscale_parity_lengths_and_regimes(
    monkeypatch: pytest.MonkeyPatch, n: int, kind: str, phi: float
):
    prices = _make_ar1_prices(n, phi, seed=101 + n)
    h_rust = HurstRegimeService.calculate_hurst_rs(prices)

    monkeypatch.setenv("MABUY_DISABLE_RUST", "1")
    h_py = HurstRegimeService.calculate_hurst_rs(prices)

    assert math.isfinite(h_rust)
    assert abs(h_rust - h_py) <= 1e-12, f"{kind} n={n}: rust={h_rust} vs py={h_py}"


def test_hurst_rs_multiscale_edge_cases(monkeypatch: pytest.MonkeyPatch):
    rng = np.random.default_rng(17)
    base = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=120)))

    with_bad = base.copy()
    with_bad[5] = 0.0
    with_bad[15] = -12.5
    with_bad[45] = np.nan

    cases = [
        np.full(100, 100.0),          # constant
        np.zeros(50),                 # all zeros
        np.full(50, -10.0),           # all negative
        np.full(50, np.nan),          # all NaN
        with_bad,                     # mixed valid + 0 + negative + NaN
        base.tolist(),                # Python list input
    ]

    for arr in cases:
        monkeypatch.delenv("MABUY_DISABLE_RUST", raising=False)
        h_rust = HurstRegimeService.calculate_hurst_rs(arr)
        monkeypatch.setenv("MABUY_DISABLE_RUST", "1")
        h_py = HurstRegimeService.calculate_hurst_rs(arr)
        assert abs(h_rust - h_py) <= 1e-12

    # Inf handling: Rust explicitly filters non-finite prices (`v.is_finite() && v > 0.0`)
    monkeypatch.delenv("MABUY_DISABLE_RUST", raising=False)
    all_inf = np.full(50, np.inf)
    assert HurstRegimeService.calculate_hurst_rs(all_inf) == 0.50
    with_inf = base.copy()
    with_inf[20] = np.inf
    with_inf[60] = -np.inf
    h_inf = HurstRegimeService.calculate_hurst_rs(with_inf)
    assert 0.05 <= h_inf <= 0.95


@pytest.mark.parametrize("n", [0, 19, 20, 50, 200, 1000])
@pytest.mark.parametrize("mode", ["trending", "oscillating", "random_walk"])
def test_hurst_2scale_parity_lengths_and_regimes(
    monkeypatch: pytest.MonkeyPatch, n: int, mode: str
):
    if mode == "trending":
        prices = _make_ar1_prices(n, 0.70, seed=303 + n)
    elif mode == "oscillating":
        prices = np.array([100.0 if i % 2 == 0 else 101.5 for i in range(n)], dtype=np.float64)
    else:
        prices = _make_ar1_prices(n, 0.0, seed=404 + n)

    h_rust = HurstRegimeDetector.calculate_hurst_from_prices(prices)
    monkeypatch.setenv("MABUY_DISABLE_RUST", "1")
    h_py = HurstRegimeDetector.calculate_hurst_from_prices(prices)

    assert abs(h_rust - h_py) <= 1e-12, f"{mode} n={n}: rust={h_rust} vs py={h_py}"


def test_hurst_2scale_edge_cases(monkeypatch: pytest.MonkeyPatch):
    base = _make_ar1_prices(80, 0.5, seed=99)
    with_bad = base.copy()
    with_bad[3] = 0.0
    with_bad[11] = -3.0
    with_bad[29] = np.nan

    for arr in [
        np.full(80, 42.0),
        np.zeros(40),
        np.full(40, -1.0),
        np.full(40, np.nan),
        with_bad,
        base.tolist(),
    ]:
        monkeypatch.delenv("MABUY_DISABLE_RUST", raising=False)
        h_rust = HurstRegimeDetector.calculate_hurst_from_prices(arr)
        monkeypatch.setenv("MABUY_DISABLE_RUST", "1")
        h_py = HurstRegimeDetector.calculate_hurst_from_prices(arr)
        assert abs(h_rust - h_py) <= 1e-12

    monkeypatch.delenv("MABUY_DISABLE_RUST", raising=False)
    assert HurstRegimeDetector.calculate_hurst_from_prices(np.full(40, np.inf)) == 0.50


def test_obi_snapshot_parity_across_scenarios():
    svc = OrderFlowImbalanceService()
    scenarios = [
        # balanced
        (
            [[60000.0, 10.0], [59950.0, 10.0], [59900.0, 10.0], [59850.0, 10.0]],
            [[60050.0, 10.0], [60100.0, 10.0], [60150.0, 10.0], [60200.0, 10.0]],
        ),
        # bid dominance
        (
            [[60000.0, 50.0], [59950.0, 40.0]],
            [[60050.0, 5.0], [60100.0, 5.0]],
        ),
        # ask dominance
        (
            [[60000.0, 5.0], [59950.0, 5.0]],
            [[60050.0, 50.0], [60100.0, 40.0]],
        ),
        # bid wall
        (
            [[59000.0, 50.0], [58900.0, 15.0], [58800.0, 15.0], [58700.0, 20.0]],
            [[60100.0, 25.0], [60200.0, 25.0], [60300.0, 25.0], [60400.0, 25.0]],
        ),
        # ask wall
        (
            [[59000.0, 25.0], [58900.0, 25.0], [58800.0, 25.0], [58700.0, 25.0]],
            [[60100.0, 60.0], [60200.0, 15.0], [60300.0, 15.0], [60400.0, 10.0]],
        ),
        # SOL small order (< $150k)
        (
            [[140.0, 6.0], [139.9, 2.0], [139.8, 1.0], [139.7, 1.0]],
            [[140.1, 2.5], [140.2, 2.5], [140.3, 2.5], [140.4, 2.5]],
        ),
        # SOL whale wall (>= $150k)
        (
            [[140.0, 1500.0], [139.9, 500.0], [139.8, 500.0], [139.7, 500.0]],
            [[140.1, 2.5], [140.2, 2.5], [140.3, 2.5], [140.4, 2.5]],
        ),
        # 1-level book
        (
            [[60000.0, 1.5]],
            [[60010.0, 2.5]],
        ),
        # >20 levels (only top 20 must be included)
        (
            [[60000.0 - i * 10.0, 1.0 + (i % 3)] for i in range(30)],
            [[60010.0 + i * 10.0, 1.5 + (i % 4)] for i in range(30)],
        ),
    ]

    for bids, asks in scenarios:
        py_res = svc.calculate_obi(bids, asks)
        rs_res = rust_bridge.obi_snapshot(bids, asks)
        assert rs_res is not None
        for key in (
            "obi_score",
            "microprice",
            "depth_microprice",
            "total_bid_vol",
            "total_ask_vol",
            "bid_ask_ratio",
        ):
            assert abs(rs_res[key] - py_res[key]) <= 1e-12, f"Mismatch on {key}: {rs_res} vs {py_res}"
        assert rs_res["has_bid_wall"] == py_res["has_bid_wall"]
        assert rs_res["has_ask_wall"] == py_res["has_ask_wall"]
        assert rs_res["bid_wall_price"] == py_res["bid_wall_price"]
        assert rs_res["ask_wall_price"] == py_res["ask_wall_price"]

        # Also check 2D ndarray input and 1D array input
        b_np = np.asarray(bids, dtype=np.float64)
        a_np = np.asarray(asks, dtype=np.float64)
        rs_np = rust_bridge.obi_snapshot(b_np, a_np)
        assert rs_np == rs_res
        rs_arr = rust_bridge.obi_snapshot_arrays(b_np[:, 0], b_np[:, 1], a_np[:, 0], a_np[:, 1])
        assert rs_arr == rs_res
        svc_np = svc.calculate_obi(b_np, a_np)
        assert svc_np == py_res

    # Empty book
    py_empty = svc.calculate_obi([], [])
    rs_empty = rust_bridge.obi_snapshot([], [])
    assert rs_empty is not None
    assert rs_empty["obi_score"] == py_empty["obi_score"] == 0.0
    assert rs_empty["bid_ask_ratio"] == py_empty["bid_ask_ratio"] == 1.0
    assert rs_empty["has_bid_wall"] is False
    assert rs_empty["has_ask_wall"] is False


def test_obi_summary_threshold_boundary_unrounded_parity():
    """Verify summary threshold uses unrounded obi_score on both list[list] and 2D ndarray."""
    svc = OrderFlowImbalanceService(confirmation_threshold=0.20)
    # Exact raw obi_score = (5999.8 - 4000.2) / 10000.0 = 0.19996 (< 0.20),
    # while round(0.19996, 4) == 0.2000 (>= 0.20).
    for bids, asks in [
        ([[60000.0, 5999.8]], [[60050.0, 4000.2]]),
        ([[60000.0, 4000.2]], [[60050.0, 5999.8]]),
    ]:
        res_list = svc.calculate_obi(bids, asks)
        res_np = svc.calculate_obi(
            np.asarray(bids, dtype=np.float64),
            np.asarray(asks, dtype=np.float64),
        )
        assert res_np == res_list
        assert "Сбалансированный стакан" in res_np["summary"]


def test_hurst_single_scale_and_rolling_volatility_parity():
    rng = np.random.default_rng(77)
    returns = rng.normal(0.0002, 0.015, size=200)

    h_rust = rust_bridge.hurst_single_scale(returns)
    h_py = py_single_scale_hurst_ddof0(returns)
    assert h_rust is not None and h_py is not None
    assert abs(h_rust - h_py) <= 1e-11

    assert rust_bridge.hurst_single_scale(returns[:7]) is None
    assert rust_bridge.hurst_single_scale(np.zeros(20)) is None

    for window in (2, 10, 32, 64, 200):
        v_rust = rust_bridge.rolling_volatility(returns, window)
        v_py = float(np.std(returns[-window:], ddof=0))
        assert v_rust is not None
        assert abs(v_rust - v_py) <= 1e-12

    # Non-finite returns filtered when computing rolling volatility
    with_nan = np.array([0.01, np.nan, -0.01, np.inf, 0.02, -0.02])
    v_clean = rust_bridge.rolling_volatility(with_nan, 4)
    v_expected = float(np.std(np.array([0.01, -0.01, 0.02, -0.02]), ddof=0))
    assert v_clean is not None
    assert abs(v_clean - v_expected) <= 1e-12
    assert rust_bridge.rolling_volatility(returns, 1) is None


def test_native_microstructure_session_lifecycle_and_transitions():
    with pytest.raises(ValueError):
        rust_bridge.NativeMicrostructureSession(tick_size_usd=0.0)

    with rust_bridge.NativeMicrostructureSession(
        tick_size_usd=0.01,
        hurst_center=0.55,
        hurst_deadband=0.03,
        funding_bias_limit=0.0005,
        volatility_trip=0.05,
    ) as session:
        snap0 = session.snapshot()
        assert snap0["risk_state"] == "ReduceOnly"
        assert snap0["hurst_regime"] == "Neutral"
        assert snap0["mid_price_usd"] is None

        sides = np.array([0, 1], dtype=np.uint8)
        ticks = np.array([10_000, 10_002], dtype=np.int64)
        qtys = np.array([3.0, 1.5], dtype=np.float64)
        ts = np.array([1, 2], dtype=np.uint64)
        cum_ofi = session.apply_l2_batch(sides, ticks, qtys, ts)
        # +100.0*3.0 - 100.02*1.5 = +149.97
        assert abs(cum_ofi - 149.97) <= 1e-12

        # Persistent trending trade series -> Trending / Normal
        price = 100.0
        for i in range(40):
            session.observe_trade(10 + i, price)
            price *= math.exp(0.004 + 0.0003 * math.sin(i * 0.25))

        snap_trend = session.snapshot()
        assert snap_trend["mid_price_usd"] == pytest.approx(100.01, abs=1e-12)
        assert snap_trend["bid_count"] == 1
        assert snap_trend["ask_count"] == 1
        assert snap_trend["hurst"] is not None
        assert snap_trend["volatility"] is not None
        assert snap_trend["hurst_regime"] == "Trending"
        assert snap_trend["risk_state"] == "Normal"

        # Positive funding bias -> LongRestricted
        session.set_funding_rate(0.001)
        session.observe_trade(100, price * 1.004)
        assert session.snapshot()["risk_state"] == "LongRestricted"

        # Negative funding bias -> ShortRestricted
        session.set_funding_rate(-0.001)
        session.observe_trade(101, price * 1.008)
        assert session.snapshot()["risk_state"] == "ShortRestricted"

        # Volatility trip -> Halted, then clear_circuit_breaker -> ReduceOnly
        session.set_funding_rate(0.0)
        session.observe_trade(102, price * 1.50)
        assert session.snapshot()["risk_state"] == "Halted"
        assert session.clear_circuit_breaker() is True
        assert session.snapshot()["risk_state"] == "ReduceOnly"

        # Double close safety
        session.close()
        session.close()
        with pytest.raises(RuntimeError):
            session.snapshot()


def test_fallback_when_disabled_or_missing_dll(monkeypatch: pytest.MonkeyPatch):
    prices = _make_ar1_prices(100, 0.65, seed=55)
    expected_rs = HurstRegimeService.calculate_hurst_rs(prices)
    expected_2s = HurstRegimeDetector.calculate_hurst_from_prices(prices)

    # 1. Disable via env var
    for flag in ("1", "true", "yes"):
        monkeypatch.setenv("MABUY_DISABLE_RUST", flag)
        assert rust_bridge.is_rust_available() is False
        assert rust_bridge.hurst_rs_multiscale(prices) is None
        assert rust_bridge.hurst_2scale(prices) is None
        assert rust_bridge.hurst_single_scale(np.diff(np.log(prices))) is None
        assert rust_bridge.rolling_volatility(np.diff(np.log(prices)), 10) is None
        assert rust_bridge.obi_snapshot([[100.0, 1.0]], [[101.0, 1.0]]) is None
        with pytest.raises(RuntimeError):
            rust_bridge.NativeMicrostructureSession()
        assert abs(HurstRegimeService.calculate_hurst_rs(prices) - expected_rs) <= 1e-12
        assert abs(HurstRegimeDetector.calculate_hurst_from_prices(prices) - expected_2s) <= 1e-12

    # 2. Missing DLL path via MABUY_RUST_LIB
    monkeypatch.delenv("MABUY_DISABLE_RUST", raising=False)
    monkeypatch.setenv("MABUY_RUST_LIB", r"C:\nonexistent\path\mabuy_engine_core.dll")
    assert rust_bridge.reload_backend() is False
    assert rust_bridge.is_rust_available() is False
    assert abs(HurstRegimeService.calculate_hurst_rs(prices) - expected_rs) <= 1e-12
    assert abs(HurstRegimeDetector.calculate_hurst_from_prices(prices) - expected_2s) <= 1e-12
