"""Zero-dependency ctypes bridge to the MaBuy Rust microstructure core (`mabuy_engine_core`)."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

_MODULE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_DIR.parents[1]
_WORKSPACE_ROOT = _REPO_ROOT.parent

_c_double_p = ctypes.POINTER(ctypes.c_double)
_c_uint8_p = ctypes.POINTER(ctypes.c_uint8)
_c_int64_p = ctypes.POINTER(ctypes.c_int64)
_c_uint64_p = ctypes.POINTER(ctypes.c_uint64)

# Non-null sentinel buffer for zero-length slice calls across C ABI.
_EMPTY_F64 = (ctypes.c_double * 1)(0.0)
_EMPTY_F64_PTR = ctypes.cast(_EMPTY_F64, _c_double_p)

_RISK_STATE_NAMES = {
    0: "Normal",
    1: "LongRestricted",
    2: "ShortRestricted",
    3: "ReduceOnly",
    4: "Halted",
}

_HURST_REGIME_NAMES = {
    0: "Neutral",
    1: "Trending",
    2: "MeanReverting",
}


class ObiSnapshotC(ctypes.Structure):
    _fields_ = [
        ("obi_score", ctypes.c_double),
        ("microprice", ctypes.c_double),
        ("depth_microprice", ctypes.c_double),
        ("total_bid_vol", ctypes.c_double),
        ("total_ask_vol", ctypes.c_double),
        ("bid_ask_ratio", ctypes.c_double),
        ("bid_wall_price", ctypes.c_double),
        ("ask_wall_price", ctypes.c_double),
        ("has_bid_wall", ctypes.c_uint8),
        ("has_ask_wall", ctypes.c_uint8),
        ("has_bid_wall_price", ctypes.c_uint8),
        ("has_ask_wall_price", ctypes.c_uint8),
    ]


class EngineSnapshotC(ctypes.Structure):
    _fields_ = [
        ("cumulative_ofi_usd", ctypes.c_double),
        ("last_ofi_increment_usd", ctypes.c_double),
        ("mid_price_usd", ctypes.c_double),
        ("has_mid_price", ctypes.c_uint8),
        ("hurst", ctypes.c_double),
        ("has_hurst", ctypes.c_uint8),
        ("volatility", ctypes.c_double),
        ("has_volatility", ctypes.c_uint8),
        ("risk_state", ctypes.c_uint8),
        ("hurst_regime", ctypes.c_uint8),
        ("bid_count", ctypes.c_size_t),
        ("ask_count", ctypes.c_size_t),
        ("generation", ctypes.c_uint64),
    ]


def _shared_lib_filename() -> str:
    if sys.platform.startswith("win") or os.name == "nt":
        return "mabuy_engine_core.dll"
    if sys.platform == "darwin":
        return "libmabuy_engine_core.dylib"
    return "libmabuy_engine_core.so"


def _is_disabled_by_env() -> bool:
    val = os.environ.get("MABUY_DISABLE_RUST", "").strip().lower()
    return val in {"1", "true", "yes"}


def _discover_shared_lib() -> Optional[Path]:
    lib_name = _shared_lib_filename()

    env_lib = os.environ.get("MABUY_RUST_LIB", "").strip()
    if env_lib:
        p = Path(env_lib)
        return p if p.is_file() else None

    candidates: list[Path] = []

    cargo_target = os.environ.get("CARGO_TARGET_DIR", "").strip()
    if cargo_target:
        t_root = Path(cargo_target)
        candidates.extend([t_root / "release" / lib_name, t_root / "debug" / lib_name])

    repo_target = _REPO_ROOT / "src" / "native" / "target"
    candidates.extend([repo_target / "release" / lib_name, repo_target / "debug" / lib_name])

    candidates.extend(
        [
            _MODULE_DIR / lib_name,
            _MODULE_DIR / "release" / lib_name,
            _MODULE_DIR / "debug" / lib_name,
        ]
    )

    if not cargo_target:
        scratch_target = _WORKSPACE_ROOT / "_scratch" / "target"
        candidates.extend(
            [scratch_target / "release" / lib_name, scratch_target / "debug" / lib_name]
        )

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


_lib: Optional[ctypes.CDLL] = None
_lib_path: Optional[Path] = None


def _bind_library(lib: ctypes.CDLL) -> None:
    lib.mabuy_hurst_rs_multiscale.argtypes = [_c_double_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.mabuy_hurst_rs_multiscale.restype = ctypes.c_double

    lib.mabuy_hurst_2scale.argtypes = [_c_double_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.mabuy_hurst_2scale.restype = ctypes.c_double

    lib.mabuy_hurst_single_scale.argtypes = [_c_double_p, ctypes.c_size_t, _c_double_p]
    lib.mabuy_hurst_single_scale.restype = ctypes.c_int32

    lib.mabuy_rolling_volatility.argtypes = [
        _c_double_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _c_double_p,
    ]
    lib.mabuy_rolling_volatility.restype = ctypes.c_int32

    lib.mabuy_obi_snapshot.argtypes = [
        _c_double_p,
        _c_double_p,
        ctypes.c_size_t,
        _c_double_p,
        _c_double_p,
        ctypes.c_size_t,
        ctypes.c_double,
        ctypes.POINTER(ObiSnapshotC),
    ]
    lib.mabuy_obi_snapshot.restype = ctypes.c_int32

    lib.mabuy_engine_new.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.mabuy_engine_new.restype = ctypes.c_void_p

    lib.mabuy_engine_free.argtypes = [ctypes.c_void_p]
    lib.mabuy_engine_free.restype = None

    lib.mabuy_engine_apply_l2_batch.argtypes = [
        ctypes.c_void_p,
        _c_uint8_p,
        _c_int64_p,
        _c_double_p,
        _c_uint64_p,
        ctypes.c_size_t,
        _c_double_p,
    ]
    lib.mabuy_engine_apply_l2_batch.restype = ctypes.c_int32

    lib.mabuy_engine_observe_trade.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_double,
    ]
    lib.mabuy_engine_observe_trade.restype = ctypes.c_int32

    lib.mabuy_engine_set_funding_rate.argtypes = [ctypes.c_void_p, ctypes.c_double]
    lib.mabuy_engine_set_funding_rate.restype = ctypes.c_int32

    lib.mabuy_engine_clear_circuit_breaker.argtypes = [ctypes.c_void_p]
    lib.mabuy_engine_clear_circuit_breaker.restype = ctypes.c_int32

    lib.mabuy_engine_snapshot.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(EngineSnapshotC),
    ]
    lib.mabuy_engine_snapshot.restype = ctypes.c_int32


def reload_backend() -> bool:
    """Re-discover and load the Rust shared library from current environment."""
    global _lib, _lib_path
    _lib = None
    _lib_path = None

    path = _discover_shared_lib()
    if path is None:
        return False

    try:
        loaded = ctypes.CDLL(str(path))
        _bind_library(loaded)
        _lib = loaded
        _lib_path = path
        return True
    except (OSError, AttributeError):
        _lib = None
        _lib_path = None
        return False


reload_backend()


def is_rust_available() -> bool:
    """Return True when the Rust shared library is loaded and not disabled by env."""
    return (not _is_disabled_by_env()) and (_lib is not None)


def get_backend_info() -> Dict[str, Any]:
    """Return diagnostic metadata about the Rust FFI backend state."""
    disabled = _is_disabled_by_env()
    available = _lib is not None
    return {
        "enabled": (not disabled) and available,
        "available": available,
        "disabled_by_env": disabled,
        "lib_path": str(_lib_path) if _lib_path is not None else None,
    }


def _to_f64_1d(values: Any) -> np.ndarray:
    arr = np.ascontiguousarray(values, dtype=np.float64)
    if arr.ndim != 1:
        arr = np.ascontiguousarray(arr.ravel(), dtype=np.float64)
    return arr


def hurst_rs_multiscale(prices: Sequence[float], min_chunk: int = 8) -> Optional[float]:
    """Compute multiscale R/S Hurst exponent via Rust FFI, or None if unavailable."""
    if not is_rust_available() or prices is None:
        return None
    lib = _lib
    if lib is None:
        return None
    try:
        mc = int(min_chunk)
        if mc <= 0:
            return None
        arr = _to_f64_1d(prices)
        ptr = arr.ctypes.data_as(_c_double_p) if arr.size > 0 else _EMPTY_F64_PTR
        return float(lib.mabuy_hurst_rs_multiscale(ptr, arr.size, mc))
    except (ValueError, TypeError, OverflowError, ctypes.ArgumentError):
        return None


def hurst_2scale(prices: Sequence[float], min_len: int = 20) -> Optional[float]:
    """Compute 2-scale R/S Hurst exponent via Rust FFI, or None if unavailable."""
    if not is_rust_available() or prices is None:
        return None
    lib = _lib
    if lib is None:
        return None
    try:
        ml = int(min_len)
        if ml < 0:
            return None
        arr = _to_f64_1d(prices)
        ptr = arr.ctypes.data_as(_c_double_p) if arr.size > 0 else _EMPTY_F64_PTR
        return float(lib.mabuy_hurst_2scale(ptr, arr.size, ml))
    except (ValueError, TypeError, OverflowError, ctypes.ArgumentError):
        return None


def hurst_single_scale(returns: Sequence[float]) -> Optional[float]:
    """Compute single-scale ddof=0 R/S Hurst estimate via Rust FFI."""
    if not is_rust_available() or returns is None:
        return None
    lib = _lib
    if lib is None:
        return None
    try:
        arr = _to_f64_1d(returns)
        ptr = arr.ctypes.data_as(_c_double_p) if arr.size > 0 else _EMPTY_F64_PTR
        out = ctypes.c_double(0.0)
        rc = lib.mabuy_hurst_single_scale(ptr, arr.size, ctypes.byref(out))
        if rc == 1:
            return float(out.value)
        return None
    except (ValueError, TypeError, OverflowError, ctypes.ArgumentError):
        return None


def rolling_volatility(returns: Sequence[float], window: int) -> Optional[float]:
    """Compute population std (ddof=0) over the latest `window` finite returns via Rust FFI."""
    if not is_rust_available() or returns is None:
        return None
    lib = _lib
    if lib is None:
        return None
    try:
        win = int(window)
        if win < 0:
            return None
        arr = _to_f64_1d(returns)
        ptr = arr.ctypes.data_as(_c_double_p) if arr.size > 0 else _EMPTY_F64_PTR
        out = ctypes.c_double(0.0)
        rc = lib.mabuy_rolling_volatility(ptr, arr.size, win, ctypes.byref(out))
        if rc == 1:
            return float(out.value)
        return None
    except (ValueError, TypeError, OverflowError, ctypes.ArgumentError):
        return None


def _extract_side_arrays(levels: Any) -> tuple[Any, Any, int, tuple[Any, ...]]:
    if isinstance(levels, np.ndarray):
        arr = np.ascontiguousarray(levels[:20], dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("Orderbook ndarray must have shape (N, >=2)")
        n = int(arr.shape[0])
        if n == 0:
            return None, None, 0, ()
        prices = np.ascontiguousarray(arr[:, 0], dtype=np.float64)
        qtys = np.ascontiguousarray(arr[:, 1], dtype=np.float64)
        return (
            prices.ctypes.data_as(_c_double_p),
            qtys.ctypes.data_as(_c_double_p),
            n,
            (prices, qtys),
        )

    n = min(len(levels), 20)
    if n == 0:
        return None, None, 0, ()
    prices_c = (ctypes.c_double * n)()
    qtys_c = (ctypes.c_double * n)()
    for i in range(n):
        lvl = levels[i]
        prices_c[i] = float(lvl[0])
        qtys_c[i] = float(lvl[1])
    return prices_c, qtys_c, n, (prices_c, qtys_c)


def _obi_snapshot_raw(
    bids: Any,
    asks: Any,
    min_wall_usd: float = 150_000.0,
) -> Optional[tuple[Dict[str, Any], float]]:
    """Compute OBI snapshot via Rust FFI, returning (rounded_dict, raw_obi_score)."""
    if not is_rust_available() or bids is None or asks is None:
        return None
    lib = _lib
    if lib is None:
        return None

    try:
        bp, bq, n_bids, b_keepalive = _extract_side_arrays(bids)
        ap, aq, n_asks, a_keepalive = _extract_side_arrays(asks)
        snap = ObiSnapshotC()
        rc = lib.mabuy_obi_snapshot(
            bp,
            bq,
            n_bids,
            ap,
            aq,
            n_asks,
            float(min_wall_usd),
            ctypes.byref(snap),
        )
        # Keep temporary buffers explicitly referenced until C call completes.
        _ = (b_keepalive, a_keepalive)
        if rc != 0:
            return None
        raw_obi = float(snap.obi_score)
        rounded = {
            "obi_score": round(raw_obi, 4),
            "microprice": round(float(snap.microprice), 4),
            "depth_microprice": round(float(snap.depth_microprice), 4),
            "total_bid_vol": round(float(snap.total_bid_vol), 4),
            "total_ask_vol": round(float(snap.total_ask_vol), 4),
            "bid_ask_ratio": round(float(snap.bid_ask_ratio), 2),
            "has_bid_wall": bool(snap.has_bid_wall),
            "has_ask_wall": bool(snap.has_ask_wall),
            "bid_wall_price": float(snap.bid_wall_price) if snap.has_bid_wall_price else None,
            "ask_wall_price": float(snap.ask_wall_price) if snap.has_ask_wall_price else None,
        }
        return rounded, raw_obi
    except (IndexError, ValueError, TypeError, OverflowError, ctypes.ArgumentError):
        return None


def obi_snapshot(
    bids: Any,
    asks: Any,
    min_wall_usd: float = 150_000.0,
) -> Optional[Dict[str, Any]]:
    """Compute OBI snapshot via Rust FFI, returning rounded fields matching Python service."""
    res = _obi_snapshot_raw(bids, asks, min_wall_usd=min_wall_usd)
    return res[0] if res is not None else None


def obi_snapshot_arrays(
    bid_prices: np.ndarray,
    bid_qtys: np.ndarray,
    ask_prices: np.ndarray,
    ask_qtys: np.ndarray,
    min_wall_usd: float = 150_000.0,
) -> Optional[Dict[str, Any]]:
    """Zero-copy OBI snapshot directly from contiguous 1D float64 NumPy arrays."""
    if not is_rust_available():
        return None
    lib = _lib
    if lib is None:
        return None
    try:
        bp = _to_f64_1d(bid_prices)
        bq = _to_f64_1d(bid_qtys)
        ap = _to_f64_1d(ask_prices)
        aq = _to_f64_1d(ask_qtys)
        n_bids = min(bp.size, bq.size)
        n_asks = min(ap.size, aq.size)
        snap = ObiSnapshotC()
        rc = lib.mabuy_obi_snapshot(
            bp.ctypes.data_as(_c_double_p) if n_bids > 0 else None,
            bq.ctypes.data_as(_c_double_p) if n_bids > 0 else None,
            n_bids,
            ap.ctypes.data_as(_c_double_p) if n_asks > 0 else None,
            aq.ctypes.data_as(_c_double_p) if n_asks > 0 else None,
            n_asks,
            float(min_wall_usd),
            ctypes.byref(snap),
        )
        if rc != 0:
            return None
        return {
            "obi_score": round(float(snap.obi_score), 4),
            "microprice": round(float(snap.microprice), 4),
            "depth_microprice": round(float(snap.depth_microprice), 4),
            "total_bid_vol": round(float(snap.total_bid_vol), 4),
            "total_ask_vol": round(float(snap.total_ask_vol), 4),
            "bid_ask_ratio": round(float(snap.bid_ask_ratio), 2),
            "has_bid_wall": bool(snap.has_bid_wall),
            "has_ask_wall": bool(snap.has_ask_wall),
            "bid_wall_price": float(snap.bid_wall_price) if snap.has_bid_wall_price else None,
            "ask_wall_price": float(snap.ask_wall_price) if snap.has_ask_wall_price else None,
        }
    except (ValueError, TypeError, OverflowError, ctypes.ArgumentError):
        return None


class NativeMicrostructureSession:
    """Stateful wrapper around Rust `MicrostructureEngine<32, 1024, 256, 64, 4096>`."""

    def __init__(
        self,
        tick_size_usd: float = 0.01,
        hurst_center: float = 0.55,
        hurst_deadband: float = 0.03,
        funding_bias_limit: float = 0.0005,
        volatility_trip: float = 0.025,
    ) -> None:
        if not is_rust_available() or _lib is None:
            raise RuntimeError("Rust backend is not available or disabled via MABUY_DISABLE_RUST")

        self._lib = _lib
        self._free_fn = _lib.mabuy_engine_free
        ptr = self._lib.mabuy_engine_new(
            float(tick_size_usd),
            float(hurst_center),
            float(hurst_deadband),
            float(funding_bias_limit),
            float(volatility_trip),
        )
        if not ptr:
            raise ValueError("Invalid configuration parameters for NativeMicrostructureSession")
        self._ptr: Optional[int] = int(ptr)

    def _ensure_open(self) -> int:
        if not self._ptr:
            raise RuntimeError("NativeMicrostructureSession is closed")
        return self._ptr

    def apply_l2_batch(
        self,
        sides: Any,
        price_ticks: Any,
        quantities: Any,
        timestamps_ns: Any,
    ) -> float:
        ptr = self._ensure_open()

        if isinstance(sides, np.ndarray) and sides.dtype == np.uint8:
            sides_arr = np.ascontiguousarray(sides.ravel(), dtype=np.uint8)
        elif isinstance(sides, (list, tuple)) and sides and isinstance(sides[0], str):
            sides_arr = np.fromiter(
                (0 if s in ("B", "b", "bid", "BID") else 1 if s in ("A", "a", "ask", "ASK") else 255 for s in sides),
                dtype=np.uint8,
                count=len(sides),
            )
        else:
            sides_arr = np.ascontiguousarray(sides, dtype=np.uint8).ravel()

        ticks_arr = np.ascontiguousarray(price_ticks, dtype=np.int64).ravel()
        qtys_arr = np.ascontiguousarray(quantities, dtype=np.float64).ravel()
        ts_arr = np.ascontiguousarray(timestamps_ns, dtype=np.uint64).ravel()

        count = sides_arr.size
        if ticks_arr.size != count or qtys_arr.size != count or ts_arr.size != count:
            raise ValueError("Batch arrays must all have equal length")

        cum_ofi = ctypes.c_double(0.0)
        rc = self._lib.mabuy_engine_apply_l2_batch(
            ctypes.c_void_p(ptr),
            sides_arr.ctypes.data_as(_c_uint8_p) if count > 0 else None,
            ticks_arr.ctypes.data_as(_c_int64_p) if count > 0 else None,
            qtys_arr.ctypes.data_as(_c_double_p) if count > 0 else None,
            ts_arr.ctypes.data_as(_c_uint64_p) if count > 0 else None,
            count,
            ctypes.byref(cum_ofi),
        )
        if rc < 0:
            raise ValueError(f"mabuy_engine_apply_l2_batch failed with code {rc}")
        return float(cum_ofi.value)

    def observe_trade(self, ts_ns: int, price_usd: float) -> None:
        ptr = self._ensure_open()
        rc = self._lib.mabuy_engine_observe_trade(
            ctypes.c_void_p(ptr),
            int(ts_ns),
            float(price_usd),
        )
        if rc < 0:
            raise ValueError(f"mabuy_engine_observe_trade failed with code {rc}")

    def set_funding_rate(self, funding_rate: float) -> None:
        ptr = self._ensure_open()
        rc = self._lib.mabuy_engine_set_funding_rate(
            ctypes.c_void_p(ptr),
            float(funding_rate),
        )
        if rc < 0:
            raise ValueError(f"mabuy_engine_set_funding_rate failed with code {rc}")

    def clear_circuit_breaker(self) -> bool:
        ptr = self._ensure_open()
        rc = self._lib.mabuy_engine_clear_circuit_breaker(ctypes.c_void_p(ptr))
        if rc < 0:
            raise RuntimeError(f"mabuy_engine_clear_circuit_breaker failed with code {rc}")
        return rc == 1

    def snapshot(self) -> Dict[str, Any]:
        ptr = self._ensure_open()
        snap = EngineSnapshotC()
        rc = self._lib.mabuy_engine_snapshot(ctypes.c_void_p(ptr), ctypes.byref(snap))
        if rc != 0:
            raise RuntimeError(f"mabuy_engine_snapshot failed with code {rc}")
        return {
            "cumulative_ofi_usd": float(snap.cumulative_ofi_usd),
            "last_ofi_increment_usd": float(snap.last_ofi_increment_usd),
            "mid_price_usd": float(snap.mid_price_usd) if snap.has_mid_price else None,
            "hurst": float(snap.hurst) if snap.has_hurst else None,
            "volatility": float(snap.volatility) if snap.has_volatility else None,
            "risk_state": _RISK_STATE_NAMES.get(int(snap.risk_state), "Unknown"),
            "risk_state_code": int(snap.risk_state),
            "hurst_regime": _HURST_REGIME_NAMES.get(int(snap.hurst_regime), "Unknown"),
            "hurst_regime_code": int(snap.hurst_regime),
            "bid_count": int(snap.bid_count),
            "ask_count": int(snap.ask_count),
            "generation": int(snap.generation),
        }

    def close(self) -> None:
        ptr = getattr(self, "_ptr", None)
        if ptr:
            self._ptr = None
            free_fn = getattr(self, "_free_fn", None)
            if free_fn is not None:
                free_fn(ctypes.c_void_p(ptr))

    def __enter__(self) -> "NativeMicrostructureSession":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
