"""Honest end-to-end Python benchmark suite comparing Rust ctypes FFI vs pure Python.

Measures full Python-caller latency (including NumPy conversion / ctypes overhead)
against `MABUY_DISABLE_RUST=1` pure Python fallback, using the release shared library.

Run with:
    python -B src/native/parity/bench_integration.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.domain.services.hurst_regime_service import HurstRegimeService  # noqa: E402
from src.domain.services.order_flow_imbalance import OrderFlowImbalanceService  # noqa: E402
from src.infrastructure.ml.v4_titan_ensemble import HurstRegimeDetector  # noqa: E402
from src.native import rust_bridge  # noqa: E402
from src.native.parity.check_parity import generate_l2_deltas, py_incremental_l2_ofi  # noqa: E402


def _time_fn(fn: Callable[[], object], reps: int, warmup: int = 5) -> Tuple[float, object]:
    for _ in range(warmup):
        _ = fn()
    t0 = time.perf_counter()
    last = None
    for _ in range(reps):
        last = fn()
    elapsed = time.perf_counter() - t0
    return elapsed, last


def _make_prices(n: int, seed: int = 2026) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0001, 0.008, size=n - 1)
    return np.concatenate(([100.0], 100.0 * np.exp(np.cumsum(rets))))


def bench_hurst_rs() -> None:
    print("=== 1. HurstRegimeService.calculate_hurst_rs (Rust ON vs MABUY_DISABLE_RUST=1) ===")
    cases = [(100, 500), (1000, 200), (10000, 50)]
    for n, reps in cases:
        prices = _make_prices(n, seed=100 + n)
        os.environ.pop("MABUY_DISABLE_RUST", None)
        assert rust_bridge.is_rust_available(), "Rust backend must be available"
        t_rust, h_rust = _time_fn(lambda: HurstRegimeService.calculate_hurst_rs(prices), reps)

        os.environ["MABUY_DISABLE_RUST"] = "1"
        t_py, h_py = _time_fn(lambda: HurstRegimeService.calculate_hurst_rs(prices), reps)
        os.environ.pop("MABUY_DISABLE_RUST", None)

        diff = abs(float(h_rust) - float(h_py))
        assert diff <= 1e-12, f"Parity mismatch at N={n}: {diff}"
        print(
            f"  N={n:5d} x {reps:3d} reps | "
            f"Rust={t_rust:.6f}s ({t_rust * 1e3 / reps:.4f} ms/call) | "
            f"Python={t_py:.6f}s ({t_py * 1e3 / reps:.4f} ms/call) | "
            f"Speedup={t_py / t_rust:.2f}x | |diff|={diff:.2e}"
        )


def bench_hurst_2scale() -> None:
    print("=== 2. HurstRegimeDetector.calculate_hurst_from_prices (Rust ON vs MABUY_DISABLE_RUST=1) ===")
    cases = [(200, 500), (1000, 200), (10000, 50)]
    for n, reps in cases:
        prices = _make_prices(n, seed=200 + n)
        os.environ.pop("MABUY_DISABLE_RUST", None)
        assert rust_bridge.is_rust_available(), "Rust backend must be available"
        t_rust, h_rust = _time_fn(lambda: HurstRegimeDetector.calculate_hurst_from_prices(prices), reps)

        os.environ["MABUY_DISABLE_RUST"] = "1"
        t_py, h_py = _time_fn(lambda: HurstRegimeDetector.calculate_hurst_from_prices(prices), reps)
        os.environ.pop("MABUY_DISABLE_RUST", None)

        diff = abs(float(h_rust) - float(h_py))
        assert diff <= 1e-12, f"Parity mismatch at N={n}: {diff}"
        print(
            f"  N={n:5d} x {reps:3d} reps | "
            f"Rust={t_rust:.6f}s ({t_rust * 1e3 / reps:.4f} ms/call) | "
            f"Python={t_py:.6f}s ({t_py * 1e3 / reps:.4f} ms/call) | "
            f"Speedup={t_py / t_rust:.2f}x | |diff|={diff:.2e}"
        )


def bench_obi() -> None:
    print("=== 3. OrderFlowImbalanceService.calculate_obi vs Rust FFI (20 levels x 10,000 reps) ===")
    svc = OrderFlowImbalanceService()
    bids_list = [[60000.0 - i * 0.5, 10.0 + (i % 5)] for i in range(20)]
    asks_list = [[60001.0 + i * 0.5, 9.0 + (i % 4)] for i in range(20)]
    bids_np = np.array(bids_list, dtype=np.float64)
    asks_np = np.array(asks_list, dtype=np.float64)
    bp = np.ascontiguousarray(bids_np[:, 0])
    bq = np.ascontiguousarray(bids_np[:, 1])
    ap = np.ascontiguousarray(asks_np[:, 0])
    aq = np.ascontiguousarray(asks_np[:, 1])

    reps = 10_000
    os.environ.pop("MABUY_DISABLE_RUST", None)
    t_py_list, res_py = _time_fn(lambda: svc.calculate_obi(bids_list, asks_list), reps, warmup=50)
    t_rs_svc_np, res_rs_svc_np = _time_fn(lambda: svc.calculate_obi(bids_np, asks_np), reps, warmup=50)
    t_rs_list, res_rs_list = _time_fn(lambda: rust_bridge.obi_snapshot(bids_list, asks_list), reps, warmup=50)
    t_rs_np, res_rs_np = _time_fn(lambda: rust_bridge.obi_snapshot(bids_np, asks_np), reps, warmup=50)
    t_rs_arr, res_rs_arr = _time_fn(
        lambda: rust_bridge.obi_snapshot_arrays(bp, bq, ap, aq), reps, warmup=50
    )

    os.environ["MABUY_DISABLE_RUST"] = "1"
    t_py_np, res_py_np = _time_fn(lambda: svc.calculate_obi(bids_np, asks_np), reps, warmup=50)
    os.environ.pop("MABUY_DISABLE_RUST", None)

    assert abs(res_py["obi_score"] - res_rs_list["obi_score"]) <= 1e-12
    assert abs(res_py_np["obi_score"] - res_rs_np["obi_score"]) <= 1e-12
    assert abs(res_py_np["obi_score"] - res_rs_svc_np["obi_score"]) <= 1e-12
    assert abs(res_py_np["obi_score"] - res_rs_arr["obi_score"]) <= 1e-12

    print(
        f"  Python calculate_obi (list[list]):          {t_py_list:.6f}s ({t_py_list * 1e6 / reps:.2f} us/call) [baseline]"
    )
    print(
        f"  Rust FFI obi_snapshot (list[list]):         {t_rs_list:.6f}s ({t_rs_list * 1e6 / reps:.2f} us/call) "
        f"[vs py list = {t_py_list / t_rs_list:.2f}x]"
    )
    print(
        f"  Python calculate_obi (ndarray 20x2, OFF):   {t_py_np:.6f}s ({t_py_np * 1e6 / reps:.2f} us/call)"
    )
    print(
        f"  Svc calculate_obi (ndarray 20x2, Rust ON):  {t_rs_svc_np:.6f}s ({t_rs_svc_np * 1e6 / reps:.2f} us/call) "
        f"[vs py np = {t_py_np / t_rs_svc_np:.2f}x]"
    )
    print(
        f"  Rust FFI obi_snapshot (ndarray 20x2):       {t_rs_np:.6f}s ({t_rs_np * 1e6 / reps:.2f} us/call) "
        f"[vs py list = {t_py_list / t_rs_np:.2f}x, vs py np = {t_py_np / t_rs_np:.2f}x]"
    )
    print(
        f"  Rust FFI obi_snapshot_arrays (1D):          {t_rs_arr:.6f}s ({t_rs_arr * 1e6 / reps:.2f} us/call) "
        f"[vs py list = {t_py_list / t_rs_arr:.2f}x, vs py np = {t_py_np / t_rs_arr:.2f}x]"
    )


def bench_l2_batch() -> None:
    print("=== 4. NativeMicrostructureSession.apply_l2_batch vs Python incremental L2+OFI ===")
    n_deltas = 2_000
    reps = 100
    deltas = generate_l2_deltas(seed=999, count=n_deltas)

    sides = np.array([0 if s == "B" else 1 for s, _, _, _ in deltas], dtype=np.uint8)
    ticks = np.array([p for _, p, _, _ in deltas], dtype=np.int64)
    qtys = np.array([q for _, _, q, _ in deltas], dtype=np.float64)
    ts = np.array([t for _, _, _, t in deltas], dtype=np.uint64)

    def run_rust() -> float:
        with rust_bridge.NativeMicrostructureSession(tick_size_usd=0.01) as session:
            return session.apply_l2_batch(sides, ticks, qtys, ts)

    t_rust, cum_rust = _time_fn(run_rust, reps, warmup=3)
    t_py, py_res = _time_fn(lambda: py_incremental_l2_ofi(0.01, deltas, depth=32), reps, warmup=3)
    cum_py = float(py_res[0])
    diff = abs(float(cum_rust) - cum_py)
    assert diff < 1e-8, f"OFI batch parity mismatch: {diff}"

    total_updates = n_deltas * reps
    print(
        f"  {n_deltas} deltas x {reps} reps = {total_updates:,} updates | "
        f"Rust={t_rust:.6f}s ({t_rust * 1e6 / total_updates:.4f} us/delta) | "
        f"Python={t_py:.6f}s ({t_py * 1e6 / total_updates:.4f} us/delta) | "
        f"Speedup={t_py / t_rust:.2f}x | |cum_ofi_diff|={diff:.2e}"
    )


def main() -> int:
    os.environ.pop("MABUY_DISABLE_RUST", None)
    rust_bridge.reload_backend()
    info = rust_bridge.get_backend_info()
    print(f"Backend info: {info}")
    if not info["enabled"]:
        print("ERROR: Rust backend is not enabled", file=sys.stderr)
        return 1
    if "release" not in (info["lib_path"] or "").lower():
        print(f"WARNING: expected release library, got {info['lib_path']}", file=sys.stderr)

    bench_hurst_rs()
    bench_hurst_2scale()
    bench_obi()
    bench_l2_batch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
