"""
MaBuy Institutional Funding Rate & Liquidation Bias Engine.
"""

import time
import json
import logging
import urllib.request
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from threading import Lock

logger = logging.getLogger("FundingRateService")


class FundingBiasRegime(str, Enum):
    EXTREME_LONGS_OVERHEATED = "EXTREME_LONGS_OVERHEATED"
    CROWDED_LONGS = "CROWDED_LONGS"
    BALANCED_NEUTRAL = "BALANCED_NEUTRAL"
    CROWDED_SHORTS = "CROWDED_SHORTS"
    EXTREME_SHORTS_SQUEEZE = "EXTREME_SHORTS_SQUEEZE"


@dataclass(frozen=True)
class FundingRateData:
    symbol: str
    funding_rate: float
    funding_rate_pct: float
    mark_price: float
    timestamp: float


@dataclass(frozen=True)
class FundingModulationResult:
    symbol: str
    direction: str
    raw_confidence: float
    modulated_confidence: float
    confidence_delta: float
    funding_rate_pct: float
    regime: FundingBiasRegime
    veto: bool
    veto_reason: Optional[str]
    narrative: str


class FundingRateService:
    CACHE_TTL_SECONDS: int = 60
    ENDPOINT_URL: str = "https://fapi.binance.com/fapi/v1/premiumIndex"

    HIGH_POSITIVE_THRESHOLD_PCT: float = 0.03      # +0.03%
    EXTREME_POSITIVE_THRESHOLD_PCT: float = 0.06   # +0.06%
    NEGATIVE_THRESHOLD_PCT: float = -0.02          # -0.02%
    EXTREME_NEGATIVE_THRESHOLD_PCT: float = -0.05  # -0.05%

    BOOST_AMOUNT: float = 0.06
    PENALTY_AMOUNT: float = 0.08

    def __init__(self, cache_ttl_seconds: int = 60):
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, Tuple[float, FundingRateData]] = {}
        self._lock = Lock()

    def _normalize_pair(self, raw_symbol: str) -> str:
        clean = raw_symbol.upper().replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").replace("USDT", "").strip()
        return f"{clean}USDT"

    def fetch_funding_rate(self, symbol: str, exchange: Optional[Any] = None) -> FundingRateData:
        pair = self._normalize_pair(symbol)
        now = time.time()

        with self._lock:
            if pair in self._cache:
                cached_time, cached_data = self._cache[pair]
                if now - cached_time < self.cache_ttl:
                    return cached_data

        # 1. CCXT session fallback
        if exchange is not None and hasattr(exchange, "fetch_funding_rate"):
            try:
                ccxt_symbol = f"{symbol.upper()}/USDT:USDT"
                res = exchange.fetch_funding_rate(ccxt_symbol)
                fr = float(res.get("fundingRate", 0.0001))
                data = FundingRateData(
                    symbol=symbol.upper(),
                    funding_rate=fr,
                    funding_rate_pct=fr * 100.0,
                    mark_price=float(res.get("markPrice", 0.0)),
                    timestamp=now
                )
                with self._lock:
                    self._cache[pair] = (now, data)
                return data
            except Exception:
                pass

        # 2. Direct REST Fallback
        url = f"{self.ENDPOINT_URL}?symbol={pair}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MaBuy-Institutional/5.0"})
            with urllib.request.urlopen(req, timeout=3.5) as response:
                payload = json.loads(response.read().decode("utf-8"))

            fr = float(payload.get("lastFundingRate", 0.0001))
            data = FundingRateData(
                symbol=symbol.upper(),
                funding_rate=fr,
                funding_rate_pct=fr * 100.0,
                mark_price=float(payload.get("markPrice", 0.0)),
                timestamp=now
            )
            with self._lock:
                self._cache[pair] = (now, data)
            return data
        except Exception:
            with self._lock:
                if pair in self._cache:
                    return self._cache[pair][1]

        return FundingRateData(symbol=symbol.upper(), funding_rate=0.0001, funding_rate_pct=0.01, mark_price=0.0, timestamp=now)

    def classify_regime(self, funding_rate_pct: float) -> FundingBiasRegime:
        if funding_rate_pct > self.EXTREME_POSITIVE_THRESHOLD_PCT:
            return FundingBiasRegime.EXTREME_LONGS_OVERHEATED
        elif funding_rate_pct > self.HIGH_POSITIVE_THRESHOLD_PCT:
            return FundingBiasRegime.CROWDED_LONGS
        elif funding_rate_pct < self.EXTREME_NEGATIVE_THRESHOLD_PCT:
            return FundingBiasRegime.EXTREME_SHORTS_SQUEEZE
        elif funding_rate_pct < self.NEGATIVE_THRESHOLD_PCT:
            return FundingBiasRegime.CROWDED_SHORTS
        return FundingBiasRegime.BALANCED_NEUTRAL

    def modulate_signal(self, symbol: str, direction: str, confidence: float, exchange: Optional[Any] = None) -> FundingModulationResult:
        funding_data = self.fetch_funding_rate(symbol, exchange=exchange)
        norm_conf = confidence / 100.0 if confidence > 1.0 else confidence
        norm_dir = "BULLISH" if direction.upper() in ("BUY", "LONG", "BULLISH") else "BEARISH"

        fr_pct = funding_data.funding_rate_pct
        regime = self.classify_regime(fr_pct)

        delta = 0.0
        veto = False
        veto_reason = None
        narrative = ""

        if regime == FundingBiasRegime.EXTREME_LONGS_OVERHEATED:
            if norm_dir == "BULLISH":
                veto = True
                veto_reason = f"EXTREME_FUNDING_VETO: Funding +{fr_pct:.4f}% > +{self.EXTREME_POSITIVE_THRESHOLD_PCT}%. Overheated longs liquidation hazard."
            else:
                delta = self.BOOST_AMOUNT
                narrative = f"Overheated longs (+{fr_pct:.4f}%). Boosted Short confidence +{self.BOOST_AMOUNT*100:.0f}%."

        elif regime == FundingBiasRegime.CROWDED_LONGS:
            if norm_dir == "BULLISH":
                delta = -self.PENALTY_AMOUNT
                narrative = f"Crowded longs (+{fr_pct:.4f}%). Penalized Bullish confidence -{self.PENALTY_AMOUNT*100:.0f}%."
            else:
                delta = self.BOOST_AMOUNT
                narrative = f"Crowded longs (+{fr_pct:.4f}%). Boosted Bearish confidence +{self.BOOST_AMOUNT*100:.0f}%."

        elif regime == FundingBiasRegime.EXTREME_SHORTS_SQUEEZE:
            if norm_dir == "BEARISH":
                veto = True
                veto_reason = f"EXTREME_FUNDING_VETO: Funding {fr_pct:.4f}% < {self.EXTREME_NEGATIVE_THRESHOLD_PCT}%. Short squeeze hazard."
            else:
                delta = self.BOOST_AMOUNT
                narrative = f"Extreme short crowding ({fr_pct:.4f}%). Boosted Long confidence +{self.BOOST_AMOUNT*100:.0f}%."

        elif regime == FundingBiasRegime.CROWDED_SHORTS:
            if norm_dir == "BULLISH":
                delta = self.BOOST_AMOUNT
                narrative = f"Crowded shorts ({fr_pct:.4f}%). Boosted Bullish confidence +{self.BOOST_AMOUNT*100:.0f}%."
            else:
                delta = -self.PENALTY_AMOUNT
                narrative = f"Crowded shorts ({fr_pct:.4f}%). Penalized Bearish confidence -{self.PENALTY_AMOUNT*100:.0f}%."
        else:
            narrative = f"Funding balanced ({fr_pct:.4f}%)."

        modulated = max(0.0, min(1.0, norm_conf + delta)) if not veto else 0.0

        delta_val = round(modulated - norm_conf, 4)
        return FundingModulationResult(
            symbol=symbol.upper(),
            direction=norm_dir,
            raw_confidence=confidence,
            modulated_confidence=round(modulated, 4),
            confidence_delta=delta_val,
            funding_rate_pct=round(fr_pct, 4),
            regime=regime,
            veto=veto,
            veto_reason=veto_reason,
            narrative=narrative
        )

    def get_carry_sizing_multiplier(self, direction: str, funding_rate_pct: float, alpha: float = 5.0) -> Tuple[float, str]:
        """
        Calculates Carry Arbitrage position sizing multiplier based on real-time funding rate differentials.
        Mathematical Formulation:
          Position_Size_Multiplier = clamp(1.0 + alpha * carry_yield_pct, 0.80, 1.20)

        Where:
          - For SHORT when funding > 0: carry_yield_pct = +funding_rate_pct (longs pay shorts).
          - For LONG when funding > 0: carry_yield_pct = -funding_rate_pct (longs pay funding cost).
          - For LONG when funding < 0: carry_yield_pct = +abs(funding_rate_pct) (shorts pay longs).
          - For SHORT when funding < 0: carry_yield_pct = -abs(funding_rate_pct) (shorts pay funding cost).

        Returns: (multiplier: float, narrative: str)
        """
        norm_dir = "BULLISH" if str(direction).upper() in ("BUY", "LONG", "BULLISH") else "BEARISH"

        if norm_dir == "BEARISH":
            carry_yield = funding_rate_pct
        else:
            carry_yield = -funding_rate_pct

        raw_multiplier = 1.0 + (alpha * carry_yield)
        clamped_multiplier = max(0.80, min(1.20, raw_multiplier))

        if clamped_multiplier > 1.0:
            narrative = f"Positive Carry Yield (+{carry_yield:+.4f}%). Size boosted by +{(clamped_multiplier-1.0)*100:.1f}%."
        elif clamped_multiplier < 1.0:
            narrative = f"Adverse Carry Drag ({carry_yield:+.4f}%). Size scaled down by {(1.0-clamped_multiplier)*100:.1f}%."
        else:
            narrative = "Neutral Carry. No size adjustment."

        return round(clamped_multiplier, 4), narrative

