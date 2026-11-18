import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class DistanceDecayedOBIResult:
    d_obi_score: float # [-1.0, +1.0] Distance-Decayed Order Book Imbalance
    raw_obi_score: float # Standard linear OBI score
    total_bid_volume: float
    total_ask_volume: float
    weighted_bid_volume: float
    weighted_ask_volume: float
    is_bid_dominated: bool
    is_ask_dominated: bool
    liquidity_wall_detected: bool
    dominant_side: str # "BID", "ASK", or "NEUTRAL"
    microstructure_veto: bool
    veto_reason: Optional[str]


class DistanceDecayedOBIService:
    """
    Computes Distance-Decayed Order Book Imbalance (D-OBI).
    Levels closer to Mid-Price receive exponentially higher weight,
    neutralizing passive spoofing walls placed far from the spread.
    
    Formula:
        w_i = exp(-decay_factor * (|P_i - P_mid| / P_mid) * 1000)
    """

    def __init__(self, decay_factor: float = 0.5, veto_threshold: float = 0.40):
        self.decay_factor = decay_factor
        self.veto_threshold = veto_threshold

    def calculate_d_obi(
        self,
        bids: List[List[float]], # [[price, qty], ...]
        asks: List[List[float]], # [[price, qty], ...]
        signal_direction: Optional[str] = None # "BULLISH", "BEARISH", or None
    ) -> DistanceDecayedOBIResult:
        if not bids or not asks:
            return DistanceDecayedOBIResult(
                d_obi_score=0.0,
                raw_obi_score=0.0,
                total_bid_volume=0.0,
                total_ask_volume=0.0,
                weighted_bid_volume=0.0,
                weighted_ask_volume=0.0,
                is_bid_dominated=False,
                is_ask_dominated=False,
                liquidity_wall_detected=False,
                dominant_side="NEUTRAL",
                microstructure_veto=False,
                veto_reason=None
            )

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2.0

        if mid_price <= 0:
            mid_price = best_bid

        raw_bid_vol = sum(float(b[1]) for b in bids)
        raw_ask_vol = sum(float(a[1]) for a in asks)
        raw_total = raw_bid_vol + raw_ask_vol
        raw_obi = (raw_bid_vol - raw_ask_vol) / raw_total if raw_total > 0 else 0.0

        weighted_bid_vol = 0.0
        weighted_ask_vol = 0.0
        wall_detected = False

        # Weighted bids
        for price_val, qty_val in bids:
            p = float(price_val)
            q = float(qty_val)
            dist_rel = abs(p - mid_price) / mid_price
            weight = math.exp(-self.decay_factor * dist_rel * 1000.0)
            weighted_bid_vol += q * weight
            if raw_bid_vol > 0 and (q / raw_bid_vol) >= 0.35:
                wall_detected = True

        # Weighted asks
        for price_val, qty_val in asks:
            p = float(price_val)
            q = float(qty_val)
            dist_rel = abs(p - mid_price) / mid_price
            weight = math.exp(-self.decay_factor * dist_rel * 1000.0)
            weighted_ask_vol += q * weight
            if raw_ask_vol > 0 and (q / raw_ask_vol) >= 0.35:
                wall_detected = True

        weighted_total = weighted_bid_vol + weighted_ask_vol
        d_obi = (weighted_bid_vol - weighted_ask_vol) / weighted_total if weighted_total > 0 else 0.0
        d_obi = max(-1.0, min(1.0, d_obi))

        is_bid_dominated = bool(d_obi >= 0.20)
        is_ask_dominated = bool(d_obi <= -0.20)
        dominant_side = "BID" if is_bid_dominated else ("ASK" if is_ask_dominated else "NEUTRAL")

        # Microstructure Veto Logic
        microstructure_veto = False
        veto_reason = None

        if signal_direction:
            clean_dir = signal_direction.upper()
            if clean_dir == "BULLISH" and d_obi <= -self.veto_threshold:
                microstructure_veto = True
                veto_reason = f"VETO: Bullish signal rejected by heavy Ask order book pressure (D-OBI = {d_obi:.2f} <= -{self.veto_threshold})"
            elif clean_dir == "BEARISH" and d_obi >= self.veto_threshold:
                microstructure_veto = True
                veto_reason = f"VETO: Bearish signal rejected by heavy Bid order book support (D-OBI = {d_obi:.2f} >= +{self.veto_threshold})"

        return DistanceDecayedOBIResult(
            d_obi_score=round(d_obi, 4),
            raw_obi_score=round(raw_obi, 4),
            total_bid_volume=round(raw_bid_vol, 2),
            total_ask_volume=round(raw_ask_vol, 2),
            weighted_bid_volume=round(weighted_bid_vol, 2),
            weighted_ask_volume=round(weighted_ask_vol, 2),
            is_bid_dominated=is_bid_dominated,
            is_ask_dominated=is_ask_dominated,
            liquidity_wall_detected=wall_detected,
            dominant_side=dominant_side,
            microstructure_veto=microstructure_veto,
            veto_reason=veto_reason
        )
