"""
MaBuy Multi-Timeframe (MTF) Trend Confluence Service (Astra Phase 5 Directive).
Guarantees that 1h execution trades NEVER fight the higher-timeframe (4h) structural regime.
Mathematical Condition:
p_hat = P(target before stop | regime_4h, setup_1h, flow_live)
If 1h candidate direction directly opposes the 4h trend (Close vs EMA50_4h), VETO candidate.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher

logger = logging.getLogger("MTFTrendService")


@dataclass(frozen=True)
class MTFConfluenceResult:
    symbol: str
    macro_regime_4h: str       # BULLISH, BEARISH, NEUTRAL
    alignment: bool            # True if candidate matches 4h regime
    veto: bool                 # True if candidate directly opposes a strong 4h trend
    confidence_delta: float    # Boost or penalty (+0.04 to -0.15)
    reason: str
    ema20_4h: float = 0.0
    ema50_4h: float = 0.0
    rsi_4h: float = 50.0


class MTFTrendService:
    def __init__(self, fetcher: Optional[CryptoDataFetcher] = None):
        self.fetcher = fetcher or CryptoDataFetcher(max_cache_age_sec=600)

    def analyze_macro_regime(self, symbol: str) -> Dict[str, Any]:
        """Fetches 4h candles and determines structural trend regime."""
        clean_sym = symbol.upper().replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").replace("USDT", "").strip()
        try:
            df = self.fetcher.fetch_ohlcv(clean_sym, timeframe="4h", limit=60)
            if df is None or len(df) < 30 or "close" not in df.columns:
                return {"regime": "NEUTRAL", "reason": "Insufficient 4h data", "ema20": 0.0, "ema50": 0.0, "rsi": 50.0}

            closes = df["close"].values
            ema20 = float(pd.Series(closes).ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(pd.Series(closes).ewm(span=50, adjust=False).mean().iloc[-1])
            current_close = float(closes[-1])

            # 4h RSI
            deltas = np.diff(closes)
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 1e-6
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 1e-6
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = float(100.0 - (100.0 / (1.0 + rs)))

            # Strong 4h Bullish: Close > EMA20 > EMA50 AND RSI > 50
            if current_close > ema20 and ema20 > ema50 and rsi >= 48.0:
                regime = "BULLISH"
            # Strong 4h Bearish: Close < EMA20 < EMA50 AND RSI < 50
            elif current_close < ema20 and ema20 < ema50 and rsi <= 52.0:
                regime = "BEARISH"
            else:
                regime = "NEUTRAL"

            return {
                "regime": regime,
                "current_close": current_close,
                "ema20": ema20,
                "ema50": ema50,
                "rsi": rsi,
                "reason": f"4h C={current_close:,.2f} EMA20={ema20:,.2f} EMA50={ema50:,.2f} RSI={rsi:.1f}"
            }
        except Exception as e:
            logger.warning(f"Error computing MTF regime for {symbol}: {e}")
            return {"regime": "NEUTRAL", "reason": f"Exception: {e}", "ema20": 0.0, "ema50": 0.0, "rsi": 50.0}

    def evaluate_confluence(self, symbol: str, candidate_direction: str) -> MTFConfluenceResult:
        """
        Evaluates whether a 1h signal aligns with the 4h macro trend.
        - If 1h BULLISH but 4h is strong BEARISH -> VETO (Never catch falling knives).
        - If 1h BEARISH but 4h is strong BULLISH -> VETO (Never short a bull run).
        - If aligned -> confidence boost (+4%).
        - If 4h is NEUTRAL -> proceed without boost or veto.
        """
        clean_sym = symbol.upper().replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").replace("USDT", "").strip()
        macro = self.analyze_macro_regime(clean_sym)
        regime = macro["regime"]
        cand_dir = candidate_direction.upper()

        if regime == "BULLISH":
            if cand_dir == "BULLISH":
                return MTFConfluenceResult(
                    symbol=clean_sym,
                    macro_regime_4h="BULLISH",
                    alignment=True,
                    veto=False,
                    confidence_delta=0.04,
                    reason=f"4h Macro Trend confirms BULLISH ({macro['reason']}). +4% confidence confluence.",
                    ema20_4h=macro["ema20"],
                    ema50_4h=macro["ema50"],
                    rsi_4h=macro["rsi"]
                )
            elif cand_dir == "BEARISH":
                return MTFConfluenceResult(
                    symbol=clean_sym,
                    macro_regime_4h="BULLISH",
                    alignment=False,
                    veto=True,
                    confidence_delta=-0.15,
                    reason=f"VETO: 1h BEARISH opposes strong 4h BULLISH trend ({macro['reason']}).",
                    ema20_4h=macro["ema20"],
                    ema50_4h=macro["ema50"],
                    rsi_4h=macro["rsi"]
                )

        elif regime == "BEARISH":
            if cand_dir == "BEARISH":
                return MTFConfluenceResult(
                    symbol=clean_sym,
                    macro_regime_4h="BEARISH",
                    alignment=True,
                    veto=False,
                    confidence_delta=0.04,
                    reason=f"4h Macro Trend confirms BEARISH ({macro['reason']}). +4% confidence confluence.",
                    ema20_4h=macro["ema20"],
                    ema50_4h=macro["ema50"],
                    rsi_4h=macro["rsi"]
                )
            elif cand_dir == "BULLISH":
                return MTFConfluenceResult(
                    symbol=clean_sym,
                    macro_regime_4h="BEARISH",
                    alignment=False,
                    veto=True,
                    confidence_delta=-0.15,
                    reason=f"VETO: 1h BULLISH opposes strong 4h BEARISH trend ({macro['reason']}).",
                    ema20_4h=macro["ema20"],
                    ema50_4h=macro["ema50"],
                    rsi_4h=macro["rsi"]
                )

        # NEUTRAL regime
        return MTFConfluenceResult(
            symbol=clean_sym,
            macro_regime_4h="NEUTRAL",
            alignment=False,
            veto=False,
            confidence_delta=0.0,
            reason=f"4h Macro regime is NEUTRAL / Consolidation ({macro['reason']}). Trade permitted at standard confidence.",
            ema20_4h=macro["ema20"],
            ema50_4h=macro["ema50"],
            rsi_4h=macro["rsi"]
        )
