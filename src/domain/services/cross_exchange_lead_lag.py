from __future__ import annotations

import json
import math
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType
from typing import Any, Callable, Deque, Dict, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_SYMBOLS: Tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
DEFAULT_VENUES: Tuple[str, ...] = ("binance", "bybit", "okx")
SUPPORTED_VENUES = frozenset(DEFAULT_VENUES)


class VenueUnavailableError(RuntimeError):
    """Raised when a venue's public REST API cannot provide valid market data."""


@dataclass(frozen=True)
class VenueSnapshot:
    """Latest normalized market state received from one exchange venue."""

    symbol: str
    venue: str
    timestamp: float
    price: float
    mark_price: Optional[float]
    funding_rate: Optional[float]
    best_bid: Optional[float]
    best_ask: Optional[float]
    bid_notional: float
    ask_notional: float
    order_flow_imbalance: float


@dataclass(frozen=True)
class VenueHealth:
    """Availability and recent error state for a venue-symbol data source."""

    symbol: str
    venue: str
    reachable: bool
    last_success: Optional[float]
    last_error: Optional[str]
    consecutive_failures: int


@dataclass(frozen=True)
class LeadLagCorrelation:
    """
    Lagged return correlation between two venues.

    A positive lag means ``base_venue`` leads ``other_venue``:
    rho(tau) = Corr(r_base[t], r_other[t + tau]).
    """

    symbol: str
    timeframe_seconds: int
    base_venue: str
    other_venue: str
    correlations: Mapping[int, float]
    best_lag_ticks: Optional[int]
    best_correlation: Optional[float]
    samples: int
    leader: Optional[str]
    confidence: float


@dataclass(frozen=True)
class FundingDivergence:
    """
    Cross-venue funding-rate dispersion and directional positioning pressure.

    Positive funding directional pressure denotes net bullish long positioning.
    A positive spread means the highest-funding venue is materially richer than
    the lowest-funding venue, creating a potential short-high / long-low basis
    convergence trade.
    """

    symbol: str
    timestamp: float
    rates: Mapping[str, float]
    average_rate: float
    spread: float
    highest_funding_venue: Optional[str]
    lowest_funding_venue: Optional[str]
    directional_pressure: float
    distortion: float
    arbitrage_direction: Optional[str]


@dataclass(frozen=True)
class DirectionalBias:
    """Composite cross-exchange momentum bias with an explicit confidence score."""

    symbol: str
    timestamp: float
    score: float
    confidence: float
    momentum_component: float
    funding_component: float
    order_flow_component: float
    leading_venue: Optional[str]
    rationale: Tuple[str, ...]


@dataclass(frozen=True)
class _PriceTick:
    timestamp: float
    price: float
    order_flow_imbalance: float
    funding_rate: Optional[float]


HttpGetter = Callable[[str, float], Any]


class CrossExchangeLeadLagService:
    """
    Thread-safe cross-exchange lead-lag and funding divergence service.

    The service polls public REST endpoints in a daemon background thread and
    maintains an in-memory cache. Read methods never perform network I/O and
    remain safe to call concurrently from strategy, API, and monitoring threads.

    Binance Futures remains independently useful when Bybit or OKX becomes
    unavailable. Secondary venue failures are isolated per venue and symbol;
    cached data is only used while it remains within ``stale_after_seconds``.
    """

    BINANCE_PREMIUM_INDEX_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
    BINANCE_DEPTH_URL = "https://fapi.binance.com/fapi/v1/depth"

    BYBIT_TICKERS_URL = "https://api.bybit.com/v5/market/tickers"
    BYBIT_ORDERBOOK_URL = "https://api.bybit.com/v5/market/orderbook"

    OKX_TICKER_URL = "https://www.okx.com/api/v5/market/ticker"
    OKX_BOOKS_URL = "https://www.okx.com/api/v5/market/books"
    OKX_FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate"

    def __init__(
        self,
        symbols: Sequence[str] = DEFAULT_SYMBOLS,
        venues: Sequence[str] = DEFAULT_VENUES,
        polling_interval_seconds: float = 5.0,
        request_timeout_seconds: float = 3.0,
        history_seconds: int = 7200,
        max_lag_ticks: int = 3,
        min_correlation_samples: int = 8,
        stale_after_seconds: Optional[float] = None,
        max_retries: int = 1,
        http_get: Optional[HttpGetter] = None,
    ) -> None:
        normalized_symbols = tuple(self._canonical_symbol(symbol) for symbol in symbols)
        normalized_venues = tuple(venue.strip().lower() for venue in venues)

        if not normalized_symbols:
            raise ValueError("At least one symbol must be configured.")
        if len(set(normalized_symbols)) != len(normalized_symbols):
            raise ValueError("Symbols must be unique after normalization.")
        if not normalized_venues:
            raise ValueError("At least one venue must be configured.")
        if len(set(normalized_venues)) != len(normalized_venues):
            raise ValueError("Venues must be unique.")
        if any(venue not in SUPPORTED_VENUES for venue in normalized_venues):
            unsupported = sorted(set(normalized_venues) - SUPPORTED_VENUES)
            raise ValueError(f"Unsupported venues: {unsupported}")
        if polling_interval_seconds <= 0:
            raise ValueError("polling_interval_seconds must be positive.")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive.")
        if history_seconds < 300:
            raise ValueError("history_seconds must be at least 300 seconds.")
        if max_lag_ticks < 0:
            raise ValueError("max_lag_ticks cannot be negative.")
        if min_correlation_samples < 3:
            raise ValueError("min_correlation_samples must be at least 3.")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        self.symbols = normalized_symbols
        self.venues = normalized_venues
        self.polling_interval_seconds = float(polling_interval_seconds)
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.history_seconds = int(history_seconds)
        self.max_lag_ticks = int(max_lag_ticks)
        self.min_correlation_samples = int(min_correlation_samples)
        self.stale_after_seconds = (
            float(stale_after_seconds)
            if stale_after_seconds is not None
            else max(self.polling_interval_seconds * 3.0, 15.0)
        )
        self.max_retries = int(max_retries)
        self._http_get = http_get

        max_history_length = max(
            128,
            int(math.ceil(self.history_seconds / self.polling_interval_seconds)) + 16,
        )

        self._state_lock = threading.RLock()
        self._cycle_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._executor: Optional[ThreadPoolExecutor] = None

        self._snapshots: Dict[str, Dict[str, VenueSnapshot]] = {
            symbol: {} for symbol in self.symbols
        }
        self._history: Dict[str, Dict[str, Deque[_PriceTick]]] = {
            symbol: {
                venue: deque(maxlen=max_history_length)
                for venue in self.venues
            }
            for symbol in self.symbols
        }
        self._health: Dict[str, Dict[str, VenueHealth]] = {
            symbol: {
                venue: VenueHealth(
                    symbol=symbol,
                    venue=venue,
                    reachable=False,
                    last_success=None,
                    last_error="not polled",
                    consecutive_failures=0,
                )
                for venue in self.venues
            }
            for symbol in self.symbols
        }

    def __enter__(self) -> "CrossExchangeLeadLagService":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.stop()

    def start(self) -> None:
        """Start the non-blocking daemon polling loop if it is not already running."""
        with self._lifecycle_lock:
            if self._poll_thread is not None and self._poll_thread.is_alive():
                return

            self._stop_event.clear()
            self._executor = ThreadPoolExecutor(
                max_workers=max(1, len(self.symbols) * len(self.venues)),
                thread_name_prefix="cross-exchange-feed",
            )
            self._poll_thread = threading.Thread(
                target=self._polling_loop,
                name="cross-exchange-lead-lag",
                daemon=True,
            )
            self._poll_thread.start()

    def stop(self, join_timeout_seconds: float = 5.0) -> None:
        """Request background polling shutdown and release worker resources."""
        with self._lifecycle_lock:
            self._stop_event.set()
            thread = self._poll_thread

        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, join_timeout_seconds))

        with self._lifecycle_lock:
            if self._poll_thread is thread and (
                thread is None or not thread.is_alive()
            ):
                self._poll_thread = None

            executor = self._executor
            if executor is not None and (
                self._poll_thread is None or not self._poll_thread.is_alive()
            ):
                self._executor = None
                executor.shutdown(wait=False, cancel_futures=True)

    def is_running(self) -> bool:
        """Return whether the daemon polling thread is currently active."""
        with self._lifecycle_lock:
            return self._poll_thread is not None and self._poll_thread.is_alive()

    def poll_once(self) -> int:
        """
        Perform one synchronous cache refresh.

        Network work is concurrent across configured venue-symbol pairs. If a
        background cycle is already in progress, this method returns zero rather
        than blocking behind it.
        """
        if not self._cycle_lock.acquire(blocking=False):
            return 0

        temporary_executor: Optional[ThreadPoolExecutor] = None
        try:
            with self._lifecycle_lock:
                executor = self._executor

            if executor is None:
                temporary_executor = ThreadPoolExecutor(
                    max_workers=max(1, len(self.symbols) * len(self.venues)),
                    thread_name_prefix="cross-exchange-manual",
                )
                executor = temporary_executor

            futures: Dict[Future[Tuple[Optional[VenueSnapshot], Optional[str]]], Tuple[str, str]] = {}
            for symbol in self.symbols:
                for venue in self.venues:
                    future = executor.submit(self._safe_fetch_venue_symbol, venue, symbol)
                    futures[future] = (symbol, venue)

            successes: list[VenueSnapshot] = []
            failures: list[Tuple[str, str, str]] = []

            for future in as_completed(futures):
                symbol, venue = futures[future]
                try:
                    snapshot, error = future.result()
                except Exception as exc:
                    snapshot, error = None, self._error_text(exc)

                if snapshot is not None:
                    successes.append(snapshot)
                else:
                    failures.append((symbol, venue, error or "unknown fetch failure"))

            now = time.time()
            with self._state_lock:
                for snapshot in successes:
                    self._snapshots[snapshot.symbol][snapshot.venue] = snapshot
                    self._history[snapshot.symbol][snapshot.venue].append(
                        _PriceTick(
                            timestamp=snapshot.timestamp,
                            price=snapshot.price,
                            order_flow_imbalance=snapshot.order_flow_imbalance,
                            funding_rate=snapshot.funding_rate,
                        )
                    )
                    self._health[snapshot.symbol][snapshot.venue] = VenueHealth(
                        symbol=snapshot.symbol,
                        venue=snapshot.venue,
                        reachable=True,
                        last_success=snapshot.timestamp,
                        last_error=None,
                        consecutive_failures=0,
                    )

                for symbol, venue, error in failures:
                    previous = self._health[symbol][venue]
                    self._health[symbol][venue] = VenueHealth(
                        symbol=symbol,
                        venue=venue,
                        reachable=False,
                        last_success=previous.last_success,
                        last_error=error,
                        consecutive_failures=previous.consecutive_failures + 1,
                    )

            return len(successes)
        finally:
            if temporary_executor is not None:
                temporary_executor.shutdown(wait=False, cancel_futures=True)
            self._cycle_lock.release()

    def get_snapshots(
        self,
        symbol: str,
        include_stale: bool = False,
    ) -> Mapping[str, VenueSnapshot]:
        """Return the latest cached normalized snapshots for a symbol."""
        normalized = self._validate_symbol(symbol)
        now = time.time()

        with self._state_lock:
            snapshots = dict(self._snapshots[normalized])

        if not include_stale:
            snapshots = {
                venue: snapshot
                for venue, snapshot in snapshots.items()
                if now - snapshot.timestamp <= self.stale_after_seconds
            }

        return MappingProxyType(snapshots)

    def get_venue_health(self, symbol: str) -> Mapping[str, VenueHealth]:
        """Return current per-venue fetch health for a configured symbol."""
        normalized = self._validate_symbol(symbol)
        with self._state_lock:
            health = dict(self._health[normalized])
        return MappingProxyType(health)

    def get_cross_correlations(
        self,
        symbol: str,
        timeframe: int | str = "1m",
    ) -> Tuple[LeadLagCorrelation, ...]:
        """
        Compute all available pairwise lagged return correlations.

        Supported timeframes are ``"1m"``, ``"5m"``, 60, and 300 seconds.
        """
        normalized = self._validate_symbol(symbol)
        timeframe_seconds = self._normalize_timeframe(timeframe)

        with self._state_lock:
            history = {
                venue: tuple(self._history[normalized][venue])
                for venue in self.venues
            }

        results = []
        for base_venue, other_venue in combinations(self.venues, 2):
            result = self._calculate_cross_correlation(
                symbol=normalized,
                base_venue=base_venue,
                other_venue=other_venue,
                base_history=history[base_venue],
                other_history=history[other_venue],
                timeframe_seconds=timeframe_seconds,
            )
            results.append(result)

        return tuple(results)

    def get_lead_lag(
        self,
        symbol: str,
        timeframe: int | str = "1m",
    ) -> Tuple[LeadLagCorrelation, ...]:
        """Alias for ``get_cross_correlations`` for strategy-facing usage."""
        return self.get_cross_correlations(symbol, timeframe)

    def get_funding_divergence(self, symbol: str) -> Optional[FundingDivergence]:
        """
        Return funding basis dispersion using all fresh reachable venues.

        Returns ``None`` only when no fresh venue has a valid funding rate.
        """
        normalized = self._validate_symbol(symbol)
        now = time.time()
        snapshots = self.get_snapshots(normalized, include_stale=False)

        rates = {
            venue: snapshot.funding_rate
            for venue, snapshot in snapshots.items()
            if snapshot.funding_rate is not None
            and math.isfinite(snapshot.funding_rate)
        }
        if not rates:
            return None

        typed_rates = {venue: float(rate) for venue, rate in rates.items()}
        highest_venue = max(typed_rates, key=typed_rates.get)
        lowest_venue = min(typed_rates, key=typed_rates.get)
        highest = typed_rates[highest_venue]
        lowest = typed_rates[lowest_venue]
        average = sum(typed_rates.values()) / len(typed_rates)
        spread = highest - lowest

        directional_pressure = self._clamp(average / 0.00030)
        distortion = self._clamp(spread / 0.00040)

        arbitrage_direction: Optional[str] = None
        if len(typed_rates) >= 2 and highest_venue != lowest_venue:
            arbitrage_direction = f"short_{highest_venue}_long_{lowest_venue}"

        return FundingDivergence(
            symbol=normalized,
            timestamp=now,
            rates=MappingProxyType(dict(typed_rates)),
            average_rate=average,
            spread=spread,
            highest_funding_venue=highest_venue,
            lowest_funding_venue=lowest_venue,
            directional_pressure=directional_pressure,
            distortion=distortion,
            arbitrage_direction=arbitrage_direction,
        )

    def get_directional_bias(self, symbol: str) -> DirectionalBias:
        """
        Produce a normalized momentum score in [-1.0, +1.0] and confidence.

        The score combines short-horizon price momentum, lag-adjusted leader
        momentum, funding positioning pressure, and visible order-book flow.
        Confidence decays automatically when secondary exchanges are missing,
        stale, or lack enough synchronized return history.
        """
        normalized = self._validate_symbol(symbol)
        now = time.time()
        snapshots = self.get_snapshots(normalized, include_stale=False)
        funding = self.get_funding_divergence(normalized)
        correlations = self.get_cross_correlations(normalized, "1m")

        with self._state_lock:
            histories = {
                venue: tuple(self._history[normalized][venue])
                for venue in self.venues
            }

        venue_momentum: Dict[str, float] = {}
        venue_momentum_quality: Dict[str, float] = {}
        for venue, ticks in histories.items():
            momentum, quality = self._recent_return(ticks, lookback_seconds=60)
            venue_momentum[venue] = momentum
            venue_momentum_quality[venue] = quality

        active_momenta = [
            venue_momentum[venue]
            for venue in snapshots
            if venue_momentum_quality.get(venue, 0.0) > 0.0
        ]

        leading_venue: Optional[str] = None
        lead_weight_total = 0.0
        lead_momentum_total = 0.0
        strongest_lead_confidence = -1.0

        for correlation in correlations:
            if (
                correlation.best_lag_ticks is None
                or correlation.best_lag_ticks == 0
                or correlation.best_correlation is None
                or correlation.leader is None
            ):
                continue

            source_venue = correlation.leader
            source_momentum = venue_momentum.get(source_venue, 0.0)
            source_quality = venue_momentum_quality.get(source_venue, 0.0)
            if source_quality <= 0.0:
                continue

            relationship_sign = 1.0 if correlation.best_correlation >= 0.0 else -1.0
            weight = correlation.confidence * source_quality
            lead_momentum_total += source_momentum * relationship_sign * weight
            lead_weight_total += weight

            if weight > strongest_lead_confidence:
                strongest_lead_confidence = weight
                leading_venue = source_venue

        if lead_weight_total > 0.0:
            raw_momentum = lead_momentum_total / lead_weight_total
            lead_quality = min(1.0, lead_weight_total)
        elif active_momenta:
            raw_momentum = sum(active_momenta) / len(active_momenta)
            lead_quality = 0.20 * (
                sum(venue_momentum_quality[v] for v in snapshots) / max(1, len(snapshots))
            )
        else:
            raw_momentum = 0.0
            lead_quality = 0.0

        momentum_component = self._clamp(raw_momentum / 0.0010)

        if funding is not None:
            funding_component = funding.directional_pressure
            funding_quality = len(funding.rates) / len(self.venues)
        else:
            funding_component = 0.0
            funding_quality = 0.0

        imbalances = [
            snapshot.order_flow_imbalance
            for snapshot in snapshots.values()
            if math.isfinite(snapshot.order_flow_imbalance)
        ]
        order_flow_component = (
            self._clamp(sum(imbalances) / len(imbalances)) if imbalances else 0.0
        )
        order_flow_quality = len(imbalances) / len(self.venues)

        score = self._clamp(
            0.55 * momentum_component
            + 0.30 * funding_component
            + 0.15 * order_flow_component
        )

        fresh_venue_ratio = len(snapshots) / len(self.venues)
        confidence = self._clamp(
            (
                0.55 * lead_quality
                + 0.30 * funding_quality
                + 0.15 * order_flow_quality
            )
            * fresh_venue_ratio
        )

        rationale = []
        if leading_venue is not None:
            rationale.append(f"{leading_venue} is the strongest detected price-discovery leader")
        elif active_momenta:
            rationale.append("momentum uses available venue returns without confirmed lead-lag")
        else:
            rationale.append("insufficient recent return history for price momentum")

        if funding is not None:
            rationale.append(
                f"funding pressure={funding.directional_pressure:+.3f}, "
                f"dispersion={funding.spread:.6f}"
            )
        else:
            rationale.append("funding unavailable from fresh venue cache")

        if len(snapshots) < len(self.venues):
            unavailable = sorted(set(self.venues) - set(snapshots))
            rationale.append(f"fallback mode active; unavailable or stale venues={','.join(unavailable)}")

        return DirectionalBias(
            symbol=normalized,
            timestamp=now,
            score=score,
            confidence=confidence,
            momentum_component=momentum_component,
            funding_component=funding_component,
            order_flow_component=order_flow_component,
            leading_venue=leading_venue,
            rationale=tuple(rationale),
        )

    def get_signals(self, symbol: str) -> Mapping[str, Any]:
        """Return a complete strategy-facing signal bundle from the local cache."""
        normalized = self._validate_symbol(symbol)
        return MappingProxyType(
            {
                "snapshots": self.get_snapshots(normalized),
                "health": self.get_venue_health(normalized),
                "lead_lag_1m": self.get_cross_correlations(normalized, "1m"),
                "lead_lag_5m": self.get_cross_correlations(normalized, "5m"),
                "funding_divergence": self.get_funding_divergence(normalized),
                "directional_bias": self.get_directional_bias(normalized),
            }
        )

    def _polling_loop(self) -> None:
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                self.poll_once()
            except Exception:
                # poll_once already isolates individual venue failures. This
                # final guard ensures an unexpected implementation error never
                # terminates the market-data service thread.
                pass

            elapsed = time.monotonic() - started
            self._stop_event.wait(max(0.0, self.polling_interval_seconds - elapsed))

    def _safe_fetch_venue_symbol(
        self,
        venue: str,
        symbol: str,
    ) -> Tuple[Optional[VenueSnapshot], Optional[str]]:
        try:
            if venue == "binance":
                return self._fetch_binance(symbol), None
            if venue == "bybit":
                return self._fetch_bybit(symbol), None
            if venue == "okx":
                return self._fetch_okx(symbol), None
            raise VenueUnavailableError(f"Unsupported venue: {venue}")
        except Exception as exc:
            return None, self._error_text(exc)

    def _fetch_binance(self, symbol: str) -> VenueSnapshot:
        ticker_url = f"{self.BINANCE_PREMIUM_INDEX_URL}?{urlencode({'symbol': symbol})}"
        ticker = self._fetch_json(ticker_url)

        if not isinstance(ticker, Mapping):
            raise VenueUnavailableError("Binance returned an invalid premium index payload")
        if "code" in ticker and int(self._safe_float(ticker.get("code"), 0.0)) != 0:
            raise VenueUnavailableError(str(ticker.get("msg", "Binance API error")))

        mark_price = self._required_positive_float(
            ticker.get("markPrice", ticker.get("price")),
            "Binance mark price",
        )
        funding_rate = self._optional_float(ticker.get("lastFundingRate"))

        best_bid, best_ask, bid_notional, ask_notional, imbalance = self._fetch_binance_book(
            symbol
        )
        return VenueSnapshot(
            symbol=symbol,
            venue="binance",
            timestamp=time.time(),
            price=mark_price,
            mark_price=mark_price,
            funding_rate=funding_rate,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_notional=bid_notional,
            ask_notional=ask_notional,
            order_flow_imbalance=imbalance,
        )

    def _fetch_bybit(self, symbol: str) -> VenueSnapshot:
        ticker_url = f"{self.BYBIT_TICKERS_URL}?{urlencode({'category': 'linear', 'symbol': symbol})}"
        payload = self._fetch_json(ticker_url)

        if not isinstance(payload, Mapping) or int(payload.get("retCode", -1)) != 0:
            message = payload.get("retMsg", "Bybit API error") if isinstance(payload, Mapping) else "invalid payload"
            raise VenueUnavailableError(str(message))

        result = payload.get("result")
        rows = result.get("list") if isinstance(result, Mapping) else None
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], Mapping):
            raise VenueUnavailableError("Bybit returned no linear ticker data")

        ticker = rows[0]
        price = self._required_positive_float(
            ticker.get("lastPrice", ticker.get("markPrice")),
            "Bybit last price",
        )
        mark_price = self._optional_float(ticker.get("markPrice"))
        funding_rate = self._optional_float(ticker.get("fundingRate"))

        best_bid, best_ask, bid_notional, ask_notional, imbalance = self._fetch_bybit_book(
            symbol
        )
        return VenueSnapshot(
            symbol=symbol,
            venue="bybit",
            timestamp=time.time(),
            price=price,
            mark_price=mark_price,
            funding_rate=funding_rate,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_notional=bid_notional,
            ask_notional=ask_notional,
            order_flow_imbalance=imbalance,
        )

    def _fetch_okx(self, symbol: str) -> VenueSnapshot:
        instrument = self._okx_instrument(symbol)
        ticker_url = f"{self.OKX_TICKER_URL}?{urlencode({'instId': instrument})}"
        payload = self._fetch_json(ticker_url)

        if not isinstance(payload, Mapping) or str(payload.get("code", "1")) != "0":
            message = payload.get("msg", "OKX API error") if isinstance(payload, Mapping) else "invalid payload"
            raise VenueUnavailableError(str(message))

        rows = payload.get("data")
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], Mapping):
            raise VenueUnavailableError("OKX returned no swap ticker data")

        ticker = rows[0]
        price = self._required_positive_float(ticker.get("last"), "OKX last price")

        funding_rate: Optional[float] = None
        try:
            funding_url = f"{self.OKX_FUNDING_URL}?{urlencode({'instId': instrument})}"
            funding_payload = self._fetch_json(funding_url)
            funding_rows = (
                funding_payload.get("data")
                if isinstance(funding_payload, Mapping)
                and str(funding_payload.get("code", "1")) == "0"
                else None
            )
            if isinstance(funding_rows, list) and funding_rows and isinstance(funding_rows[0], Mapping):
                funding_rate = self._optional_float(funding_rows[0].get("fundingRate"))
        except Exception:
            funding_rate = None

        best_bid, best_ask, bid_notional, ask_notional, imbalance = self._fetch_okx_book(
            instrument
        )
        return VenueSnapshot(
            symbol=symbol,
            venue="okx",
            timestamp=time.time(),
            price=price,
            mark_price=None,
            funding_rate=funding_rate,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_notional=bid_notional,
            ask_notional=ask_notional,
            order_flow_imbalance=imbalance,
        )

    def _fetch_binance_book(
        self,
        symbol: str,
    ) -> Tuple[Optional[float], Optional[float], float, float, float]:
        try:
            url = f"{self.BINANCE_DEPTH_URL}?{urlencode({'symbol': symbol, 'limit': 10})}"
            payload = self._fetch_json(url)
            if not isinstance(payload, Mapping):
                raise VenueUnavailableError("invalid Binance order book")
            return self._book_metrics(payload.get("bids"), payload.get("asks"))
        except Exception:
            return None, None, 0.0, 0.0, 0.0

    def _fetch_bybit_book(
        self,
        symbol: str,
    ) -> Tuple[Optional[float], Optional[float], float, float, float]:
        try:
            url = f"{self.BYBIT_ORDERBOOK_URL}?{urlencode({'category': 'linear', 'symbol': symbol, 'limit': 10})}"
            payload = self._fetch_json(url)
            result = payload.get("result") if isinstance(payload, Mapping) else None
            if (
                not isinstance(payload, Mapping)
                or int(payload.get("retCode", -1)) != 0
                or not isinstance(result, Mapping)
            ):
                raise VenueUnavailableError("invalid Bybit order book")
            return self._book_metrics(result.get("b"), result.get("a"))
        except Exception:
            return None, None, 0.0, 0.0, 0.0

    def _fetch_okx_book(
        self,
        instrument: str,
    ) -> Tuple[Optional[float], Optional[float], float, float, float]:
        try:
            url = f"{self.OKX_BOOKS_URL}?{urlencode({'instId': instrument, 'sz': 10})}"
            payload = self._fetch_json(url)
            rows = payload.get("data") if isinstance(payload, Mapping) else None
            if (
                not isinstance(payload, Mapping)
                or str(payload.get("code", "1")) != "0"
                or not isinstance(rows, list)
                or not rows
                or not isinstance(rows[0], Mapping)
            ):
                raise VenueUnavailableError("invalid OKX order book")
            return self._book_metrics(rows[0].get("bids"), rows[0].get("asks"))
        except Exception:
            return None, None, 0.0, 0.0, 0.0

    def _fetch_json(self, url: str) -> Any:
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                if self._http_get is not None:
                    response = self._http_get(url, self.request_timeout_seconds)
                    if isinstance(response, (bytes, bytearray)):
                        return json.loads(response.decode("utf-8"))
                    if isinstance(response, str):
                        return json.loads(response)
                    return response

                request = Request(
                    url,
                    headers={
                        "Accept": "application/json",
                        "User-Agent": "cross-exchange-lead-lag/1.0",
                    },
                )
                with urlopen(request, timeout=self.request_timeout_seconds) as response:
                    raw = response.read(2_000_000)
                return json.loads(raw.decode("utf-8"))
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(0.25 * (2**attempt), 1.0))

        raise VenueUnavailableError(self._error_text(last_error))

    def _calculate_cross_correlation(
        self,
        symbol: str,
        base_venue: str,
        other_venue: str,
        base_history: Sequence[_PriceTick],
        other_history: Sequence[_PriceTick],
        timeframe_seconds: int,
    ) -> LeadLagCorrelation:
        base_returns = self._bucket_returns(base_history, timeframe_seconds)
        other_returns = self._bucket_returns(other_history, timeframe_seconds)

        correlations: Dict[int, float] = {}
        sample_counts: Dict[int, int] = {}

        for lag in range(-self.max_lag_ticks, self.max_lag_ticks + 1):
            x_values = []
            y_values = []

            for bucket, base_return in base_returns.items():
                other_return = other_returns.get(bucket + lag)
                if other_return is not None:
                    x_values.append(base_return)
                    y_values.append(other_return)

            correlation = self._pearson_correlation(x_values, y_values)
            if correlation is not None:
                correlations[lag] = correlation
                sample_counts[lag] = len(x_values)

        if not correlations:
            return LeadLagCorrelation(
                symbol=symbol,
                timeframe_seconds=timeframe_seconds,
                base_venue=base_venue,
                other_venue=other_venue,
                correlations=MappingProxyType({}),
                best_lag_ticks=None,
                best_correlation=None,
                samples=0,
                leader=None,
                confidence=0.0,
            )

        best_lag = max(
            correlations,
            key=lambda lag: (abs(correlations[lag]), -abs(lag)),
        )
        best_correlation = correlations[best_lag]
        samples = sample_counts[best_lag]

        if best_lag > 0:
            leader: Optional[str] = base_venue
        elif best_lag < 0:
            leader = other_venue
        else:
            leader = None

        sample_factor = min(1.0, math.sqrt(max(0, samples - 2) / 30.0))
        confidence = self._clamp(abs(best_correlation) * sample_factor)

        return LeadLagCorrelation(
            symbol=symbol,
            timeframe_seconds=timeframe_seconds,
            base_venue=base_venue,
            other_venue=other_venue,
            correlations=MappingProxyType(dict(correlations)),
            best_lag_ticks=best_lag,
            best_correlation=best_correlation,
            samples=samples,
            leader=leader,
            confidence=confidence,
        )

    def _bucket_returns(
        self,
        ticks: Sequence[_PriceTick],
        timeframe_seconds: int,
    ) -> Dict[int, float]:
        closes: Dict[int, Tuple[float, float]] = {}

        for tick in ticks:
            if tick.price <= 0.0 or not math.isfinite(tick.price):
                continue

            bucket = int(tick.timestamp // timeframe_seconds)
            previous = closes.get(bucket)
            if previous is None or tick.timestamp >= previous[0]:
                closes[bucket] = (tick.timestamp, tick.price)

        returns: Dict[int, float] = {}
        for bucket in sorted(closes):
            prior = closes.get(bucket - 1)
            current = closes[bucket]
            if prior is None or prior[1] <= 0.0:
                continue

            value = math.log(current[1] / prior[1])
            if math.isfinite(value):
                returns[bucket] = value

        return returns

    def _recent_return(
        self,
        ticks: Sequence[_PriceTick],
        lookback_seconds: int,
    ) -> Tuple[float, float]:
        if len(ticks) < 2:
            return 0.0, 0.0

        latest = ticks[-1]
        if latest.price <= 0.0:
            return 0.0, 0.0

        target_timestamp = latest.timestamp - lookback_seconds
        baseline: Optional[_PriceTick] = None

        for tick in reversed(ticks[:-1]):
            if tick.timestamp <= target_timestamp:
                baseline = tick
                break

        if baseline is None or baseline.price <= 0.0:
            return 0.0, 0.0

        elapsed = latest.timestamp - baseline.timestamp
        if elapsed < lookback_seconds * 0.75:
            return 0.0, 0.0

        value = math.log(latest.price / baseline.price)
        if not math.isfinite(value):
            return 0.0, 0.0

        quality = self._clamp(elapsed / float(lookback_seconds))
        return value, quality

    @staticmethod
    def _book_metrics(
        bids: Any,
        asks: Any,
    ) -> Tuple[Optional[float], Optional[float], float, float, float]:
        if not isinstance(bids, list) or not isinstance(asks, list):
            raise VenueUnavailableError("order book side missing")

        def parse_side(rows: Iterable[Any]) -> Tuple[Optional[float], float]:
            best: Optional[float] = None
            notional = 0.0

            for row in rows:
                if not isinstance(row, (list, tuple)) or len(row) < 2:
                    continue

                try:
                    price = float(row[0])
                    quantity = float(row[1])
                except (TypeError, ValueError):
                    continue

                if (
                    not math.isfinite(price)
                    or not math.isfinite(quantity)
                    or price <= 0.0
                    or quantity <= 0.0
                ):
                    continue

                if best is None:
                    best = price
                notional += price * quantity

            return best, notional

        best_bid, bid_notional = parse_side(bids)
        best_ask, ask_notional = parse_side(asks)

        total = bid_notional + ask_notional
        imbalance = (bid_notional - ask_notional) / total if total > 0.0 else 0.0
        return best_bid, best_ask, bid_notional, ask_notional, imbalance

    @staticmethod
    def _pearson_correlation(
        x_values: Sequence[float],
        y_values: Sequence[float],
    ) -> Optional[float]:
        if len(x_values) != len(y_values) or len(x_values) < 3:
            return None

        n = len(x_values)
        mean_x = sum(x_values) / n
        mean_y = sum(y_values) / n

        covariance = 0.0
        variance_x = 0.0
        variance_y = 0.0

        for x_value, y_value in zip(x_values, y_values):
            dx = x_value - mean_x
            dy = y_value - mean_y
            covariance += dx * dy
            variance_x += dx * dx
            variance_y += dy * dy

        if variance_x <= 0.0 or variance_y <= 0.0:
            return None

        correlation = covariance / math.sqrt(variance_x * variance_y)
        if not math.isfinite(correlation):
            return None
        return max(-1.0, min(1.0, correlation))

    @staticmethod
    def _canonical_symbol(symbol: str) -> str:
        normalized = symbol.strip().upper().replace("-", "").replace("_", "")
        normalized = normalized.replace("SWAP", "")
        if not normalized:
            raise ValueError("Symbol cannot be empty.")
        return normalized

    def _validate_symbol(self, symbol: str) -> str:
        normalized = self._canonical_symbol(symbol)
        if normalized not in self.symbols:
            raise KeyError(f"Symbol is not configured: {normalized}")
        return normalized

    @staticmethod
    def _normalize_timeframe(timeframe: int | str) -> int:
        if isinstance(timeframe, str):
            normalized = timeframe.strip().lower()
            mapping = {"1m": 60, "60s": 60, "5m": 300, "300s": 300}
            if normalized not in mapping:
                raise ValueError("Supported timeframes are '1m', '5m', 60, and 300.")
            return mapping[normalized]

        if timeframe in (60, 300):
            return int(timeframe)
        raise ValueError("Supported timeframes are '1m', '5m', 60, and 300.")

    @staticmethod
    def _okx_instrument(symbol: str) -> str:
        for quote in ("USDT", "USDC", "USD"):
            if symbol.endswith(quote) and len(symbol) > len(quote):
                base = symbol[: -len(quote)]
                return f"{base}-{quote}-SWAP"
        raise ValueError(f"Cannot construct OKX swap instrument for symbol: {symbol}")

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return default
        return parsed if math.isfinite(parsed) else default

    @classmethod
    def _optional_float(cls, value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        parsed = cls._safe_float(value, float("nan"))
        return parsed if math.isfinite(parsed) else None

    @classmethod
    def _required_positive_float(cls, value: Any, field_name: str) -> float:
        parsed = cls._optional_float(value)
        if parsed is None or parsed <= 0.0:
            raise VenueUnavailableError(f"{field_name} is invalid")
        return parsed

    @staticmethod
    def _clamp(value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return max(-1.0, min(1.0, value))

    @staticmethod
    def _error_text(error: Optional[BaseException]) -> str:
        if error is None:
            return "unknown error"
        text = f"{type(error).__name__}: {error}".strip()
        return text[:500]


if __name__ == "__main__":
    service = CrossExchangeLeadLagService(
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        venues=("binance", "bybit", "okx"),
        polling_interval_seconds=5.0,
        request_timeout_seconds=3.0,
    )

    try:
        successful_updates = service.poll_once()
        print(f"successful venue updates: {successful_updates}")

        for configured_symbol in service.symbols:
            print(f"\n{configured_symbol}")
            print("snapshots:", dict(service.get_snapshots(configured_symbol)))
            print("health:", dict(service.get_venue_health(configured_symbol)))
            print("funding:", service.get_funding_divergence(configured_symbol))
            print("lead-lag 1m:", service.get_lead_lag(configured_symbol, "1m"))
            print("bias:", service.get_directional_bias(configured_symbol))
    finally:
        service.stop()
