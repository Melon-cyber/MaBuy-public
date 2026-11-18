import asyncio
import collections
import json
import logging
import math
import time
from typing import Dict, List, Optional, Tuple, Callable

logger = logging.getLogger("LiquidationStreamer")


class LiquidationEvent:
    """
    Structured liquidation order event synchronized by exchange kernel event_time.
    """
    __slots__ = ("symbol", "side", "order_type", "time_in_force", "orig_qty", "price", "avg_price", "status", "event_time", "volume_usd")

    def __init__(
        self,
        symbol: str,
        side: str, # "BUY" (short liquidated) or "SELL" (long liquidated)
        orig_qty: float,
        price: float,
        event_time: int, # Exchange kernel millisecond timestamp
        status: str = "FILLED"
    ):
        self.symbol = symbol.upper()
        self.side = side.upper()
        self.orig_qty = orig_qty
        self.price = price
        self.event_time = event_time
        self.status = status
        self.volume_usd = orig_qty * price


class LiquidationCascadeRadar:
    """
    In-Memory Bounded Liquidation & CVD Radar.
    
    Guarantees:
    - Bounded O(1) Memory: Uses fixed-capacity collections.deque(maxlen=X) preventing memory leaks.
    - Exchange Time Sync: All aggregations aligned strictly on event_time (exchange epoch ms).
    - Capitulation Detection: Computes rolling 1h Z-Score of 5m liquidation volume.
    """

    def __init__(self, max_history_events: int = 5000):
        # symbol -> deque of LiquidationEvent
        self._events_by_symbol: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=max_history_events)
        )
        # symbol -> deque of (event_time_ms, delta_usd) for CVD
        self._trade_deltas: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=max_history_events)
        )

    def record_liquidation(
        self,
        symbol: str,
        side: str,
        orig_qty: float,
        price: float,
        event_time_ms: int
    ) -> LiquidationEvent:
        """
        Records a liquidation event in bounded memory.
        """
        event = LiquidationEvent(
            symbol=symbol,
            side=side,
            orig_qty=orig_qty,
            price=price,
            event_time=event_time_ms
        )
        self._events_by_symbol[symbol.upper()].append(event)
        return event

    def record_trade_delta(self, symbol: str, is_buyer_maker: bool, volume_usd: float, event_time_ms: int) -> None:
        """
        Records market trade direction for Cumulative Volume Delta (CVD).
        If is_buyer_maker is True => Taker Sell (Market Sell) => delta is -volume_usd.
        If is_buyer_maker is False => Taker Buy (Market Buy) => delta is +volume_usd.
        """
        delta = -volume_usd if is_buyer_maker else volume_usd
        self._trade_deltas[symbol.upper()].append((event_time_ms, delta))

    def calculate_liquidation_zscore(
        self,
        symbol: str,
        reference_time_ms: Optional[int] = None
    ) -> Tuple[float, float, str, Dict[str, float]]:
        """
        Computes 5m liquidation volume Z-Score relative to rolling 1h baseline.
        Returns:
            z_score: Standard deviations above mean
            volume_5m_usd: 5-minute liquidation volume in USD
            dominant_side: "LONG_LIQ" (longs capitulating), "SHORT_LIQ" (shorts squeezed), or "NEUTRAL"
            metrics: breakdown dictionary
        """
        sym = symbol.upper()
        events = self._events_by_symbol.get(sym, None)
        if not events:
            return 0.0, 0.0, "NEUTRAL", {"z_score": 0.0, "volume_5m_usd": 0.0, "mean_1h": 0.0, "std_1h": 1.0}

        now_ms = reference_time_ms if reference_time_ms is not None else events[-1].event_time
        
        # 1. Bucket events in 5-minute windows over the past 1 hour (12 buckets)
        bucket_size_ms = 5 * 60 * 1000 # 300,000 ms
        window_1h_ms = 60 * 60 * 1000  # 3,600,000 ms
        cutoff_ms = now_ms - window_1h_ms

        buckets = [0.0] * 12 # 12 five-minute buckets in 1h
        long_vol_5m = 0.0
        short_vol_5m = 0.0

        for ev in events:
            if ev.event_time < cutoff_ms or ev.event_time > now_ms:
                continue
            
            # Bucket index [0..11]
            elapsed = ev.event_time - cutoff_ms
            idx = min(11, max(0, int(elapsed // bucket_size_ms)))
            buckets[idx] += ev.volume_usd

            # Check if event is in the most recent 5 minutes
            if ev.event_time >= (now_ms - bucket_size_ms):
                if ev.side in ("SELL", "LONG_LIQ"):
                    long_vol_5m += ev.volume_usd
                else:
                    short_vol_5m += ev.volume_usd

        current_5m_vol = buckets[-1]
        
        # 2. Compute mean and standard deviation over the 12 buckets
        mean_1h = sum(buckets) / 12.0
        variance = sum((b - mean_1h) ** 2 for b in buckets) / 12.0
        std_1h = math.sqrt(variance)

        # 3. Compute Z-Score with epsilon protection
        z_score = (current_5m_vol - mean_1h) / (std_1h + 1000.0)

        # 4. Determine dominant liquidation side
        if current_5m_vol > 0:
            if long_vol_5m > short_vol_5m * 1.5:
                dominant_side = "LONG_LIQ"
            elif short_vol_5m > long_vol_5m * 1.5:
                dominant_side = "SHORT_LIQ"
            else:
                dominant_side = "MIXED"
        else:
            dominant_side = "NEUTRAL"

        metrics = {
            "z_score": round(z_score, 3),
            "volume_5m_usd": round(current_5m_vol, 2),
            "long_vol_5m": round(long_vol_5m, 2),
            "short_vol_5m": round(short_vol_5m, 2),
            "mean_1h": round(mean_1h, 2),
            "std_1h": round(std_1h, 2),
            "is_capitulation_spike": bool(z_score >= 3.0)
        }
        return z_score, current_5m_vol, dominant_side, metrics

    def calculate_cvd_5m(self, symbol: str, reference_time_ms: Optional[int] = None) -> float:
        """
        Calculates 5-minute Cumulative Volume Delta (CVD) in USD aligned by event_time.
        Positive => Aggressive market buying pressure.
        Negative => Aggressive market selling pressure.
        """
        sym = symbol.upper()
        trades = self._trade_deltas.get(sym, None)
        if not trades:
            return 0.0

        now_ms = reference_time_ms if reference_time_ms is not None else trades[-1][0]
        cutoff_ms = now_ms - (5 * 60 * 1000)

        cvd_5m = 0.0
        for ts, delta in trades:
            if ts >= cutoff_ms and ts <= now_ms:
                cvd_5m += delta
                
        return cvd_5m
