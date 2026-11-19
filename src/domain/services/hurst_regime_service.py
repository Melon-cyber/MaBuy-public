"""
MaBuy Hurst Exponent Dynamic Regime Detector.
"""

import numpy as np
import logging
from typing import Sequence, Optional
from dataclasses import dataclass
from enum import Enum

from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher
from src.native import rust_bridge

logger = logging.getLogger("HurstRegimeService")


class HurstRegimeType(str, Enum):
    PERSISTENT_TREND = "PERSISTENT_TREND"
    RANDOM_WALK_NOISE = "RANDOM_WALK_NOISE"
    ANTI_PERSISTENT_MEAN_REVERT = "ANTI_PERSISTENT_MEAN_REVERT"
    MOMENTUM_EXHAUSTION = "MOMENTUM_EXHAUSTION"


@dataclass(frozen=True)
class HurstAnalysisResult:
    symbol: str
    hurst_exponent: float
    regime: HurstRegimeType
    recommended_tp_pct: float
    recommended_sl_pct: float
    min_confidence_required: float
    size_multiplier: float
    description: str
    hurst_fast: float = 0.50
    hurst_med: float = 0.50


class HurstRegimeService:
    DEFAULT_LOOKBACK: int = 100  # Medium window (~4.1 days)
    FAST_LOOKBACK: int = 36      # Fast adaptive window (~1.5 days)
    PERSISTENT_THRESHOLD: float = 0.55
    ANTI_PERSISTENT_THRESHOLD: float = 0.45

    TP_PERSISTENT_PCT: float = 0.042   # +4.2%
    TP_EXHAUSTION_PCT: float = 0.025   # +2.5% (Breakout dying down)
    TP_MEAN_REVERT_PCT: float = 0.020  # +2.0%
    TP_DEFAULT_PCT: float = 0.035      # +3.5%
    SL_DEFAULT_PCT: float = 0.018      # -1.8%

    RANDOM_WALK_MIN_CONF: float = 0.78 # 78%

    def __init__(self, fetcher: Optional[CryptoDataFetcher] = None):
        self.fetcher = fetcher or CryptoDataFetcher(max_cache_age_sec=300)

    @staticmethod
    def calculate_hurst_rs(prices: Sequence[float], min_chunk: int = 8) -> float:
        native_res = rust_bridge.hurst_rs_multiscale(prices, min_chunk=min_chunk)
        if native_res is not None:
            return native_res

        arr = np.asarray(prices, dtype=np.float64)
        valid = arr[arr > 0]
        if len(valid) < 20:
            return 0.50

        log_returns = np.diff(np.log(valid))
        n_total = len(log_returns)
        if n_total < 16:
            return 0.50

        max_chunk = n_total // 2
        chunk_sizes = [min_chunk, int(min_chunk * 1.5), max_chunk]
        chunk_sizes = sorted(list(set([c for c in chunk_sizes if c <= max_chunk and c >= min_chunk])))
        if len(chunk_sizes) < 2:
            return 0.50

        log_chunks = []
        log_rs_means = []

        for n in chunk_sizes:
            num_subs = n_total // n
            rs_vals = []
            for i in range(num_subs):
                sub = log_returns[i * n : (i + 1) * n]
                s = np.std(sub, ddof=1) if n > 1 else np.std(sub)
                if s > 1e-12:
                    cum = np.cumsum(sub - np.mean(sub))
                    r = np.max(cum) - np.min(cum)
                    if r > 1e-12:
                        rs_vals.append(r / s)
            if rs_vals:
                log_chunks.append(np.log(float(n)))
                log_rs_means.append(np.log(float(np.mean(rs_vals))))

        if len(log_chunks) >= 2:
            try:
                poly = np.polyfit(log_chunks, log_rs_means, 1)
                return float(np.clip(poly[0], 0.05, 0.95))
            except Exception:
                pass
        return 0.50

    def analyze_symbol(self, symbol: str) -> HurstAnalysisResult:
        clean_sym = symbol.upper().replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").replace("USDT", "").strip()
        H_med = 0.50
        H_fast = 0.50
        try:
            df = self.fetcher.fetch_ohlcv(clean_sym, timeframe="1h", limit=self.DEFAULT_LOOKBACK)
            if df is not None and not df.empty and "close" in df.columns:
                closes = df["close"].values
                H_med = self.calculate_hurst_rs(closes)
                if len(closes) >= self.FAST_LOOKBACK:
                    H_fast = self.calculate_hurst_rs(closes[-self.FAST_LOOKBACK:])
                else:
                    H_fast = H_med
        except Exception:
            H_med = 0.50
            H_fast = 0.50

        # Primary composite metric (weighted towards recent adaptive window):
        # 60% weight on recent 36-bar dynamics, 40% on macro 100-bar background
        H_composite = 0.60 * H_fast + 0.40 * H_med

        # Detect Structural Regime Breaks (Astra Finding 2)
        if H_med > self.PERSISTENT_THRESHOLD and H_fast < self.ANTI_PERSISTENT_THRESHOLD:
            # Macro trend is old, but fast regime is already mean-reverting/breaking down!
            regime = HurstRegimeType.MOMENTUM_EXHAUSTION
            tp = self.TP_EXHAUSTION_PCT
            min_conf = 0.70
            size_mult = 0.85
            desc = f"Momentum Exhaustion (H_fast={H_fast:.2f} < 0.45 while H_med={H_med:.2f}). Capping TP to +2.5%."
        elif H_composite > self.PERSISTENT_THRESHOLD and H_fast >= 0.50:
            regime = HurstRegimeType.PERSISTENT_TREND
            tp = self.TP_PERSISTENT_PCT
            min_conf = 0.65
            size_mult = 1.0
            desc = f"Persistent Trend Confirmed (H_fast={H_fast:.2f}, H_med={H_med:.2f}). Loosened TP to +4.2%."
        elif H_composite < self.ANTI_PERSISTENT_THRESHOLD:
            regime = HurstRegimeType.ANTI_PERSISTENT_MEAN_REVERT
            tp = self.TP_MEAN_REVERT_PCT
            min_conf = 0.65
            size_mult = 1.0
            desc = f"Mean-Reverting (H_fast={H_fast:.2f}, H_med={H_med:.2f}). Quick Scalp TP at +2.0%."
        else:
            regime = HurstRegimeType.RANDOM_WALK_NOISE
            tp = self.TP_DEFAULT_PCT
            min_conf = self.RANDOM_WALK_MIN_CONF
            size_mult = 0.60
            desc = f"Brownian Random Walk (H_comp={H_composite:.2f}). Confidence barrier elevated to >= 78%."

        return HurstAnalysisResult(
            symbol=clean_sym,
            hurst_exponent=round(H_composite, 4),
            regime=regime,
            recommended_tp_pct=tp,
            recommended_sl_pct=self.SL_DEFAULT_PCT,
            min_confidence_required=min_conf,
            size_multiplier=size_mult,
            description=desc,
            hurst_fast=round(H_fast, 4),
            hurst_med=round(H_med, 4)
        )
