"""Lightweight in-memory Cross-Exchange Market State.

Stores real-time perpetual quotes across Binance, Bybit, and OKX using
standard-library ring buffers (collections.deque). Zero external heavy dependencies
(no pandas, no numpy overhead in the hot path) for optimal execution on 1 vCPU / 2GB RAM.
Supports both synchronous tick ingestion and high-speed WebSocket QuoteUpdate ingestion.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_VENUES: Tuple[str, ...] = ("binance", "bybit", "okx", "deribit")
DEFAULT_SYMBOLS: Tuple[str, ...] = ("BTC", "ETH", "SOL")


def canonical_symbol(symbol: str) -> str:
    """Normalizes symbol string to pure asset ticker (e.g. 'BTCUSDT' -> 'BTC')."""
    s = symbol.upper().strip()
    for suffix in ("/USDT:USDT", ":USDT", "/USDT", "-USDT-SWAP", "-USDT", "USDT"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return s


@dataclass(frozen=True, slots=True)
class QuoteObservation:
    timestamp: float
    price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    source: str = "rest"


@dataclass(frozen=True, slots=True)
class QuoteUpdate:
    exchange: str
    symbol: str
    bid: float
    ask: float
    exchange_ts_ms: Optional[int]
    received_monotonic_ns: int
    source: str = "ws"

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2.0


class CrossExchangeState:
    """Thread-safe multi-venue quotation cache and rolling return estimator."""

    def __init__(
        self,
        symbols: Sequence[str] = DEFAULT_SYMBOLS,
        venues: Sequence[str] = DEFAULT_VENUES,
        buffer_len: int = 300,
    ) -> None:
        self.symbols = tuple(canonical_symbol(s) for s in symbols)
        self.venues = tuple(v.strip().lower() for v in venues)
        self.buffer_len = buffer_len

        self._lock = threading.RLock()
        # Symbol -> Venue -> deque of QuoteObservation
        self._buffers: Dict[str, Dict[str, deque[QuoteObservation]]] = {
            sym: {v: deque(maxlen=self.buffer_len) for v in self.venues}
            for sym in self.symbols
        }
        # Venue -> Symbol -> latest QuoteUpdate
        self._latest_quote_updates: Dict[str, Dict[str, QuoteUpdate]] = {
            v: {} for v in self.venues
        }

    def update_quote(
        self,
        symbol: str,
        venue: str,
        price: float,
        timestamp: Optional[float] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        source: str = "rest",
    ) -> None:
        """Records a new market price observation for a specific venue and symbol."""
        if price <= 0.0 or not math.isfinite(price):
            return

        sym = canonical_symbol(symbol)
        ven = venue.strip().lower()
        now = time.time() if timestamp is None else float(timestamp)

        with self._lock:
            if sym not in self._buffers:
                self._buffers[sym] = {v: deque(maxlen=self.buffer_len) for v in self.venues}
            if ven not in self._buffers[sym]:
                self._buffers[sym][ven] = deque(maxlen=self.buffer_len)

            obs = QuoteObservation(
                timestamp=now,
                price=price,
                bid=bid,
                ask=ask,
                source=source,
            )
            self._buffers[sym][ven].append(obs)

    def update_quote_update(self, quote: QuoteUpdate) -> bool:
        """
        Atomically records a normalized QuoteUpdate from WebSocket or REST.
        Rejects inverted or negative books. Updates both ring buffer and latest map.
        """
        if quote.bid <= 0.0 or quote.ask <= 0.0 or not math.isfinite(quote.bid) or not math.isfinite(quote.ask):
            return False
        if quote.ask < quote.bid:
            return False

        sym = canonical_symbol(quote.symbol)
        ven = quote.exchange.strip().lower()
        now = time.time()

        with self._lock:
            exchange_quotes = self._latest_quote_updates.setdefault(ven, {})
            previous = exchange_quotes.get(sym)

            if previous is not None and quote.received_monotonic_ns < previous.received_monotonic_ns:
                return False  # Out-of-order drop

            exchange_quotes[sym] = quote

            if sym not in self._buffers:
                self._buffers[sym] = {v: deque(maxlen=self.buffer_len) for v in self.venues}
            if ven not in self._buffers[sym]:
                self._buffers[sym][ven] = deque(maxlen=self.buffer_len)

            mid = quote.mid_price
            obs = QuoteObservation(
                timestamp=now,
                price=mid,
                bid=quote.bid,
                ask=quote.ask,
                source=quote.source,
            )
            self._buffers[sym][ven].append(obs)
            return True

    def get_latest_quote(
        self,
        symbol: str,
        venue: str,
    ) -> Optional[QuoteObservation]:
        """Returns the most recent price observation for the given venue."""
        sym = canonical_symbol(symbol)
        ven = venue.strip().lower()

        with self._lock:
            buf = self._buffers.get(sym, {}).get(ven)
            if buf and len(buf) > 0:
                return buf[-1]
            return None

    def get_quote(
        self,
        exchange: str,
        symbol: str,
        max_age_s: Optional[float] = None,
    ) -> Optional[QuoteUpdate]:
        """Returns latest QuoteUpdate if it is fresh within max_age_s."""
        ven = exchange.strip().lower()
        sym = canonical_symbol(symbol)

        with self._lock:
            quote = self._latest_quote_updates.get(ven, {}).get(sym)

        if quote is None:
            return None

        if max_age_s is not None:
            age_s = (time.monotonic_ns() - quote.received_monotonic_ns) / 1_000_000_000.0
            if age_s > max_age_s:
                return None

        return quote

    def get_fresh_venues(
        self,
        symbol: str,
        max_age_sec: float = 5.0,
        now_ts: Optional[float] = None,
    ) -> List[str]:
        """Returns the list of venues that reported quotes within max_age_sec."""
        sym = canonical_symbol(symbol)
        now = time.time() if now_ts is None else float(now_ts)
        fresh: List[str] = []

        with self._lock:
            venues_dict = self._buffers.get(sym, {})
            for v, buf in venues_dict.items():
                if buf and len(buf) > 0:
                    last_obs = buf[-1]
                    if (now - last_obs.timestamp) <= max_age_sec:
                        fresh.append(v)
        return fresh

    def get_rolling_return(
        self,
        symbol: str,
        venue: str,
        window_sec: float = 3.0,
        now_ts: Optional[float] = None,
    ) -> Optional[float]:
        """Computes log return r = ln(P_now / P_past) over window_sec for a venue."""
        sym = canonical_symbol(symbol)
        ven = venue.strip().lower()
        now = time.time() if now_ts is None else float(now_ts)

        with self._lock:
            buf = self._buffers.get(sym, {}).get(ven)
            if not buf or len(buf) < 2:
                return None

            latest = buf[-1]
            if (now - latest.timestamp) > (window_sec * 3.0):
                return None  # Stale data

            target_ts = latest.timestamp - window_sec
            past_obs: Optional[QuoteObservation] = None

            # Reverse scan for the observation closest to target_ts
            for obs in reversed(buf):
                if obs.timestamp <= target_ts:
                    past_obs = obs
                    break

            if past_obs is None:
                past_obs = buf[0]  # Oldest available in buffer

            if past_obs.price <= 0.0 or latest.price <= 0.0:
                return None

            return math.log(latest.price / past_obs.price)

    def get_median_external_return(
        self,
        symbol: str,
        window_sec: float = 3.0,
        max_age_sec: float = 5.0,
        now_ts: Optional[float] = None,
    ) -> Optional[float]:
        """Computes median log return across external venues (Bybit, OKX)."""
        sym = canonical_symbol(symbol)
        now = time.time() if now_ts is None else float(now_ts)
        returns: List[float] = []

        with self._lock:
            fresh_v = self.get_fresh_venues(sym, max_age_sec=max_age_sec, now_ts=now)
            for v in ("bybit", "okx"):
                if v in fresh_v:
                    ret = self.get_rolling_return(sym, v, window_sec=window_sec, now_ts=now)
                    if ret is not None and math.isfinite(ret):
                        returns.append(ret)

        if not returns:
            return None
        if len(returns) == 1:
            return returns[0]

        returns.sort()
        mid = len(returns) // 2
        if len(returns) % 2 == 1:
            return returns[mid]
        return (returns[mid - 1] + returns[mid]) / 2.0

    def get_dislocation_bps(
        self,
        symbol: str,
        max_age_sec: float = 5.0,
        now_ts: Optional[float] = None,
    ) -> Optional[float]:
        """
        Computes price dislocation in basis points:
        d_bps = ((P_binance - median(P_bybit, P_okx)) / median(P_bybit, P_okx)) * 10000.0.
        Positive = Binance is more expensive (premium).
        Negative = Binance is cheaper (discount).
        """
        sym = canonical_symbol(symbol)
        now = time.time() if now_ts is None else float(now_ts)

        with self._lock:
            fresh_v = self.get_fresh_venues(sym, max_age_sec=max_age_sec, now_ts=now)
            if "binance" not in fresh_v:
                return None

            binance_obs = self.get_latest_quote(sym, "binance")
            if not binance_obs:
                return None

            ext_prices: List[float] = []
            for v in ("bybit", "okx"):
                if v in fresh_v:
                    q = self.get_latest_quote(sym, v)
                    if q and q.price > 0.0:
                        ext_prices.append(q.price)

            if not ext_prices:
                return None

            ext_prices.sort()
            mid = len(ext_prices) // 2
            median_ext = ext_prices[mid] if len(ext_prices) % 2 == 1 else (ext_prices[mid - 1] + ext_prices[mid]) / 2.0

            if median_ext <= 0.0:
                return None

            return ((binance_obs.price - median_ext) / median_ext) * 10000.0
