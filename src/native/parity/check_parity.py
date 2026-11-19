"""Parity and benchmark suite comparing Rust `mabuy_engine_core` (`engine_cli`)
against MaBuy Python references (`HurstRegimeService`, `OrderFlowImbalanceService`)
and exact mathematical counterparts.

Run with:
    python src/native/parity/check_parity.py [--bench]
"""

from __future__ import annotations

import argparse
import math
import os
import site
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# `-S` disables site-packages; restore it inside the script so numpy is importable.
site.main()

import numpy as np

NATIVE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.domain.services.hurst_regime_service import HurstRegimeService  # noqa: E402
from src.domain.services.order_flow_imbalance import OrderFlowImbalanceService  # noqa: E402


def find_engine_cli(explicit_path: Optional[str] = None, prefer_release: bool = True) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_file():
            raise FileNotFoundError(f"engine_cli not found at {path}")
        return path

    target_root = Path(
        os.environ.get("CARGO_TARGET_DIR", str(NATIVE_DIR / "target"))
    )
    order = ("release", "debug") if prefer_release else ("debug", "release")
    for profile in order:
        for name in ("engine_cli.exe", "engine_cli"):
            candidate = target_root / profile / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Could not locate engine_cli under {target_root}")


def run_cli(cli_path: Path, payload: str) -> str:
    proc = subprocess.run(
        [str(cli_path)],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"engine_cli failed (exit={proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def py_single_scale_hurst_ddof0(samples: Sequence[float]) -> Optional[float]:
    """Exact Python counterpart of Rust `HurstKernel::estimate`:
    single-window R/S with population std (ddof=0) and ln(R/S)/ln(n).
    """
    arr = np.asarray(samples, dtype=np.float64)
    n = len(arr)
    if n < 8:
        return None
    mean = float(np.mean(arr))
    var = max(0.0, float(np.mean(arr * arr) - mean * mean))
    if not math.isfinite(var) or var <= np.finfo(np.float64).eps:
        return None
    std = math.sqrt(var)
    cum = np.cumsum(arr - mean)
    cum_min = min(0.0, float(np.min(cum)))
    cum_max = max(0.0, float(np.max(cum)))
    rng = cum_max - cum_min
    if not math.isfinite(rng) or rng <= np.finfo(np.float64).eps:
        return None
    rs = rng / std
    if not math.isfinite(rs) or rs <= 0.0:
        return None
    h = math.log(rs) / math.log(float(n))
    return h if math.isfinite(h) else None


def py_incremental_l2_ofi(
    tick_size: float,
    deltas: Sequence[Tuple[str, int, float, int]],
    depth: int = 32,
) -> Tuple[float, float, List[List[float]], List[List[float]]]:
    """Exact Python counterpart of Rust `L2OrderBook<32, _>` + `OfiCalculator`."""
    bids: List[List[float]] = []  # [price_ticks, qty], descending price_ticks
    asks: List[List[float]] = []  # [price_ticks, qty], ascending price_ticks
    cumulative_usd = 0.0
    last_increment_usd = 0.0

    for side, price_ticks, qty, _ts in deltas:
        if price_ticks <= 0 or not math.isfinite(qty) or qty < 0.0:
            raise ValueError(f"Invalid delta {(side, price_ticks, qty)}")

        levels = bids if side == "B" else asks
        existing_idx = None
        for i, lvl in enumerate(levels):
            if lvl[0] == price_ticks:
                existing_idx = i
                break

        applied = False
        old_qty = 0.0
        new_qty = qty

        if existing_idx is not None:
            old_qty = levels[existing_idx][1]
            if qty == 0.0:
                levels.pop(existing_idx)
            else:
                levels[existing_idx][1] = qty
            applied = True
        elif qty > 0.0:
            insert_at = len(levels)
            for i, lvl in enumerate(levels):
                if (side == "B" and price_ticks > lvl[0]) or (
                    side == "A" and price_ticks < lvl[0]
                ):
                    insert_at = i
                    break
            if not (len(levels) == depth and insert_at >= depth):
                if len(levels) == depth:
                    levels.pop()
                levels.insert(insert_at, [price_ticks, qty])
                applied = True

        if not applied:
            last_increment_usd = 0.0
        else:
            price_usd = float(price_ticks) * tick_size
            signed_notional = (
                price_usd * (new_qty - old_qty)
                if side == "B"
                else -price_usd * (new_qty - old_qty)
            )
            last_increment_usd = signed_notional if math.isfinite(signed_notional) else 0.0
            cumulative_usd += last_increment_usd

    bid_usd = [[lvl[0] * tick_size, lvl[1]] for lvl in bids]
    ask_usd = [[lvl[0] * tick_size, lvl[1]] for lvl in asks]
    return cumulative_usd, last_increment_usd, bid_usd, ask_usd


def generate_hurst_series(seed: int = 42, n_prices: int = 256) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 0.004, size=n_prices - 1)

    def ar1_prices(phi: float) -> np.ndarray:
        rets = np.empty(n_prices - 1, dtype=np.float64)
        prev = 0.002 if phi > 0 else 0.0
        for i, e in enumerate(eps):
            prev = phi * prev + float(e)
            rets[i] = prev
        return np.concatenate(([100.0], 100.0 * np.exp(np.cumsum(rets))))

    return {
        "trending_ar1_pos": ar1_prices(0.70),
        "random_walk_iid": ar1_prices(0.0),
        "mean_reverting_ar1_neg": ar1_prices(-0.70),
    }


def check_hurst_parity(cli_path: Path) -> None:
    print("=== 1. HURST PARITY ===")
    series_map = generate_hurst_series(seed=42, n_prices=256)

    for label, prices in series_map.items():
        log_returns = np.diff(np.log(prices))
        price_str = " ".join(f"{p:.17g}" for p in prices)
        ret_str = " ".join(f"{r:.17g}" for r in log_returns)

        out = run_cli(
            cli_path,
            f"hurst_prices {len(prices)} {price_str}\nhurst {len(log_returns)} {ret_str}\n",
        ).splitlines()
        h_rust_prices = float(out[0].split()[1])
        h_rust_returns = float(out[1].split()[1])
        h_py_single = py_single_scale_hurst_ddof0(log_returns)
        assert h_py_single is not None
        h_py_service = HurstRegimeService.calculate_hurst_rs(prices.tolist())

        diff_prices_vs_returns = abs(h_rust_prices - h_rust_returns)
        diff_single_scale = abs(h_rust_returns - h_py_single)
        diff_vs_service = h_rust_returns - h_py_service

        print(
            f"  [{label}] rust_prices={h_rust_prices:.12f}, rust_rets={h_rust_returns:.12f}, "
            f"py_single_rs={h_py_single:.12f}, py_service_rs={h_py_service:.6f}"
        )
        print(
            f"    |rust_prices - rust_rets| = {diff_prices_vs_returns:.3e} | "
            f"|rust - py_single_rs| = {diff_single_scale:.3e} | "
            f"(rust - py_service) = {diff_vs_service:+.6f} [different estimator]"
        )
        assert diff_prices_vs_returns < 1e-12
        assert diff_single_scale < 1e-11


def generate_l2_deltas(seed: int = 123, count: int = 200) -> List[Tuple[str, int, float, int]]:
    rng = np.random.default_rng(seed)
    deltas: List[Tuple[str, int, float, int]] = []
    # Seed initial book levels across >32 ticks to exercise updates, deletes, and depth overflow
    for i in range(count):
        side = "B" if rng.random() < 0.5 else "A"
        if side == "B":
            price_ticks = int(rng.integers(9960, 10005))
        else:
            price_ticks = int(rng.integers(10006, 10050))
        # Mix inserts, updates, and zero-quantity deletes
        if rng.random() < 0.25:
            qty = 0.0
        else:
            qty = float(round(float(rng.uniform(0.5, 25.0)), 4))
        deltas.append((side, price_ticks, qty, i + 1))
    return deltas


def check_ofi_parity(cli_path: Path) -> None:
    print("=== 2. OFI / L2 BOOK / OBI PARITY ===")
    tick_size = 0.01
    deltas = generate_l2_deltas(seed=123, count=200)
    flat = " ".join(f"{s} {p} {q:.17g} {ts}" for s, p, q, ts in deltas)
    out = run_cli(cli_path, f"ofi {tick_size:.17g} {len(deltas)} {flat}\n")

    tokens = out.split()
    assert tokens[0] == "ofi"
    rust_cum_ofi = float(tokens[1])
    rust_last_inc = float(tokens[2])
    assert tokens[3] == "bids"
    n_bids = int(tokens[4])
    idx = 5
    rust_bids = []
    for _ in range(n_bids):
        p_s, q_s = tokens[idx].split(":")
        rust_bids.append([float(p_s), float(q_s)])
        idx += 1
    assert tokens[idx] == "asks"
    n_asks = int(tokens[idx + 1])
    idx += 2
    rust_asks = []
    for _ in range(n_asks):
        p_s, q_s = tokens[idx].split(":")
        rust_asks.append([float(p_s), float(q_s)])
        idx += 1

    py_cum_ofi, py_last_inc, py_bids, py_asks = py_incremental_l2_ofi(tick_size, deltas, depth=32)

    cum_err = abs(rust_cum_ofi - py_cum_ofi)
    inc_err = abs(rust_last_inc - py_last_inc)
    assert len(rust_bids) == len(py_bids) and len(rust_asks) == len(py_asks)
    max_book_err = 0.0
    for (rp, rq), (pp, pq) in zip(rust_bids + rust_asks, py_bids + py_asks):
        max_book_err = max(max_book_err, abs(rp - pp), abs(rq - pq))

    obi_svc = OrderFlowImbalanceService()
    obi_res = obi_svc.calculate_obi(rust_bids, rust_asks)

    # Direct numpy verification of OBI & microprice on top-20 / top-5 book levels
    top20_b_vol = sum(q for _, q in rust_bids[:20])
    top20_a_vol = sum(q for _, q in rust_asks[:20])
    expected_obi = round((top20_b_vol - top20_a_vol) / (top20_b_vol + top20_a_vol), 4)
    expected_micro = round(
        (rust_asks[0][0] * rust_bids[0][1] + rust_bids[0][0] * rust_asks[0][1])
        / (rust_bids[0][1] + rust_asks[0][1]),
        4,
    )
    assert obi_res["obi_score"] == expected_obi
    assert obi_res["microprice"] == expected_micro

    print(
        f"  [ofi_incremental] rust_cum={rust_cum_ofi:.6f} USD, py_cum={py_cum_ofi:.6f} USD, "
        f"|diff|={cum_err:.3e}, |last_inc_diff|={inc_err:.3e}"
    )
    print(
        f"  [l2_book_state]   bids={len(rust_bids)}, asks={len(rust_asks)}, "
        f"max_level_err={max_book_err:.3e}"
    )
    print(
        f"  [py_obi_snapshot] obi_score={obi_res['obi_score']:+.4f}, "
        f"microprice={obi_res['microprice']:.4f}, depth_microprice={obi_res['depth_microprice']:.4f} "
        f"(snapshot metric; differs by definition from incremental USD OFI)"
    )
    assert cum_err < 1e-8
    assert inc_err < 1e-10
    assert max_book_err < 1e-12


def check_volatility_parity(cli_path: Path) -> None:
    print("=== 3. ROLLING VOLATILITY PARITY ===")
    rng = np.random.default_rng(77)
    returns = rng.normal(0.0002, 0.015, size=200)
    ret_str = " ".join(f"{r:.17g}" for r in returns)

    for window in (20, 32, 64, 100):
        # Test both partial window (n < window) and evicted window (n = 200 > window)
        sub_len = window // 2
        sub_str = " ".join(f"{r:.17g}" for r in returns[:sub_len])
        out = run_cli(
            cli_path,
            f"vol {window} {sub_len} {sub_str}\nvol {window} {len(returns)} {ret_str}\n",
        ).splitlines()
        rust_partial = float(out[0].split()[1])
        rust_full = float(out[1].split()[1])

        py_partial = float(np.std(returns[:sub_len], ddof=0))
        py_full = float(np.std(returns[-window:], ddof=0))

        err_partial = abs(rust_partial - py_partial)
        err_full = abs(rust_full - py_full)
        print(
            f"  [window={window:3d}] partial(n={sub_len}): rust={rust_partial:.12f} "
            f"py={py_partial:.12f} |diff|={err_partial:.3e} | "
            f"evicted(n=200): rust={rust_full:.12f} py={py_full:.12f} |diff|={err_full:.3e}"
        )
        assert err_partial < 1e-12
        assert err_full < 1e-12


def run_benchmarks(cli_path: Path) -> None:
    print("=== 4. BENCHMARKS (Rust Release vs Python) ===")
    # A. Hurst benchmark: 10,000 points, 100 repetitions
    rng = np.random.default_rng(2026)
    n_pts = 10_000
    reps = 100
    log_returns = rng.normal(0.0, 0.01, size=n_pts)
    prices = np.concatenate(([100.0], 100.0 * np.exp(np.cumsum(log_returns))))
    ret_list = log_returns.tolist()
    price_list = prices.tolist()

    ret_str = " ".join(f"{r:.17g}" for r in ret_list)
    rust_out = run_cli(cli_path, f"bench_hurst {reps} {n_pts} {ret_str}\n")
    parts = rust_out.split()
    rust_full_sec = float(parts[3])
    rust_eval_sec = float(parts[4])

    # Python single-scale R/S (numpy, matching Rust math)
    t0 = time.perf_counter()
    for _ in range(reps):
        _ = py_single_scale_hurst_ddof0(ret_list)
    py_single_sec = time.perf_counter() - t0

    # Python HurstRegimeService.calculate_hurst_rs (reference implementation)
    t1 = time.perf_counter()
    for _ in range(reps):
        _ = HurstRegimeService.calculate_hurst_rs(price_list)
    py_service_sec = time.perf_counter() - t1

    print(
        f"  [Hurst 10,000 pts x {reps} reps]\n"
        f"    Rust (push+estimate): {rust_full_sec:.6f} s ({rust_full_sec * 1e3 / reps:.4f} ms/rep)\n"
        f"    Rust (estimate only): {rust_eval_sec:.6f} s ({rust_eval_sec * 1e3 / reps:.4f} ms/rep)\n"
        f"    Python (numpy single-scale R/S): {py_single_sec:.6f} s ({py_single_sec * 1e3 / reps:.4f} ms/rep) "
        f"[speedup vs push+est={py_single_sec / rust_full_sec:.1f}x, vs est={py_single_sec / rust_eval_sec:.1f}x]\n"
        f"    Python (HurstRegimeService 3-chunk): {py_service_sec:.6f} s ({py_service_sec * 1e3 / reps:.4f} ms/rep) "
        f"[speedup vs push+est={py_service_sec / rust_full_sec:.1f}x, vs est={py_service_sec / rust_eval_sec:.1f}x]"
    )

    # B. OFI loop benchmark: 2,000 deltas x 100 repetitions (200,000 updates)
    n_deltas = 2_000
    ofi_reps = 100
    deltas = generate_l2_deltas(seed=999, count=n_deltas)
    flat = " ".join(f"{s} {p} {q:.17g} {ts}" for s, p, q, ts in deltas)
    ofi_out = run_cli(cli_path, f"bench_ofi {ofi_reps} 0.01 {n_deltas} {flat}\n")
    rust_ofi_sec = float(ofi_out.split()[3])

    t2 = time.perf_counter()
    for _ in range(ofi_reps):
        _ = py_incremental_l2_ofi(0.01, deltas, depth=32)
    py_ofi_sec = time.perf_counter() - t2

    # Also measure OrderFlowImbalanceService.calculate_obi on 20-level book x (ofi_reps * n_deltas) / 20
    # Use 10,000 snapshot calls to report per-call latency accurately
    snap_reps = 10_000
    _, _, sample_bids, sample_asks = py_incremental_l2_ofi(0.01, deltas, depth=32)
    sample_bids_20 = sample_bids[:20]
    sample_asks_20 = sample_asks[:20]
    svc = OrderFlowImbalanceService()
    t3 = time.perf_counter()
    for _ in range(snap_reps):
        _ = svc.calculate_obi(sample_bids_20, sample_asks_20)
    py_obi_sec = time.perf_counter() - t3

    print(
        f"  [OFI loop {n_deltas} deltas x {ofi_reps} reps = {n_deltas * ofi_reps:,} updates]\n"
        f"    Rust (L2OrderBook + OfiCalculator): {rust_ofi_sec:.6f} s "
        f"({rust_ofi_sec * 1e6 / (n_deltas * ofi_reps):.4f} us/delta)\n"
        f"    Python (incremental L2 + OFI loop): {py_ofi_sec:.6f} s "
        f"({py_ofi_sec * 1e6 / (n_deltas * ofi_reps):.4f} us/delta) "
        f"[speedup={py_ofi_sec / rust_ofi_sec:.1f}x]\n"
        f"    Python (OrderFlowImbalanceService.calculate_obi x {snap_reps:,} snapshots): "
        f"{py_obi_sec:.6f} s ({py_obi_sec * 1e6 / snap_reps:.2f} us/snapshot)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="MaBuy Rust vs Python parity & benchmark")
    parser.add_argument("--cli", type=str, default=None, help="Path to engine_cli executable")
    parser.add_argument("--bench", action="store_true", help="Also run performance benchmarks")
    args = parser.parse_args()

    cli_path = find_engine_cli(args.cli, prefer_release=True)
    print(f"Using engine_cli: {cli_path}")

    check_hurst_parity(cli_path)
    check_ofi_parity(cli_path)
    check_volatility_parity(cli_path)
    if args.bench:
        run_benchmarks(cli_path)
    print("ALL PARITY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
