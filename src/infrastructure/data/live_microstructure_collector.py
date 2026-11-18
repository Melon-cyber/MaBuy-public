"""Live Binance USD-M futures microstructure collector.

This module retrieves public L2 depth, aggregate trades, funding, mark price,
and open-interest data from Binance USD-M futures REST endpoints, transforms
the data through ``MicrostructureFeatureEngine``, and exposes a thread-safe,
TTL-cached microstructure snapshot suitable for rapid portfolio scan loops.
"""

from __future__ import annotations

import logging
import math
import os
import random
import re
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import requests

from src.domain.services.microstructure_feature_engine import (
    MicrostructureFeatureEngine,
    compute_top_of_book_features,
    compute_cvd_features,
)

LOGGER = logging.getLogger(__name__)


class MicrostructureCollectionError(RuntimeError):
    """Raised when live market microstructure data cannot be collected."""


@dataclass(frozen=True)
class MicrostructureSnapshot:
    """Immutable live microstructure feature snapshot for a futures symbol.

    Notes:
        ``microprice_delta`` is expressed in native quote-price units:
        ``microprice - midprice``. ``cvd_delta`` is expressed in native traded
        quantity units over the fetched aggregate-trade sample.
        ``funding_velocity`` is funding-rate change per hour, using an 8-hour
        reference where available and a 24-hour reference as fallback.
        ``oi_velocity`` is percentage open-interest change per hour, using an
        8-hour reference where available and a 24-hour reference as fallback.
    """

    symbol: str
    timestamp: str
    mark_price: float
    ofi_norm: float
    microprice_delta: float
    cvd_delta: float
    aggressor_buyer_ratio: float
    funding_rate: float
    funding_velocity: float
    oi: float
    oi_velocity: float
    composite_flow_bias: float
    flow_veto: bool
    veto_reason: Optional[str]


@dataclass(frozen=True)
class _Observation:
    """Locally retained observation used as a resilient velocity fallback."""

    timestamp_ms: int
    funding_rate: float
    open_interest: float


class LiveMicrostructureCollector:
    """Collect and transform Binance USD-M futures microstructure data.

    The collector uses Binance public REST endpoints directly rather than
    requiring API keys or authenticated exchange sessions. Requests are
    retried on transient network failures, rate-limit responses, and upstream
    5xx failures using bounded exponential backoff with jitter.

    Args:
        base_url: Binance USD-M futures REST base URL.
        cache_ttl_seconds: Per-symbol snapshot cache TTL.
        request_timeout_seconds: Per-request timeout.
        max_retries: Number of retry attempts after the initial HTTP request.
        backoff_base_seconds: Initial exponential-backoff duration.
        max_backoff_seconds: Maximum delay between retried requests.
        flow_veto_threshold: Absolute flow-bias threshold triggering a veto
            against an opposing proposed trade direction.
        feature_engine: Optional preconstructed feature engine. Supplying this
            argument is useful for dependency injection and testing.
        session: Optional requests session. If omitted, a managed session is
            created internally.

    Raises:
        ValueError: If configuration values are outside safe bounds.
    """

    _BASE_URL_DEFAULT = "https://fapi.binance.com"
    _SYMBOL_PATTERN = re.compile(r"^[A-Z0-9]{5,24}$")
    _RETRYABLE_STATUS_CODES = {418, 429, 500, 502, 503, 504}
    _RATE_LIMIT_ERROR_CODES = {-1003, -1015}

    def __init__(
        self,
        base_url: str = _BASE_URL_DEFAULT,
        cache_ttl_seconds: float = 5.0,
        request_timeout_seconds: float = 8.0,
        max_retries: int = 4,
        backoff_base_seconds: float = 0.35,
        max_backoff_seconds: float = 8.0,
        flow_veto_threshold: float = 0.45,
        feature_engine: Optional[MicrostructureFeatureEngine] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not base_url or not base_url.strip():
            raise ValueError("base_url must be a non-empty URL.")
        if cache_ttl_seconds <= 0.0:
            raise ValueError("cache_ttl_seconds must be greater than zero.")
        if request_timeout_seconds <= 0.0:
            raise ValueError("request_timeout_seconds must be greater than zero.")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if backoff_base_seconds <= 0.0:
            raise ValueError("backoff_base_seconds must be greater than zero.")
        if max_backoff_seconds < backoff_base_seconds:
            raise ValueError(
                "max_backoff_seconds must be at least backoff_base_seconds."
            )
        if not 0.0 < flow_veto_threshold <= 1.0:
            raise ValueError("flow_veto_threshold must be in the interval (0, 1].")

        self._base_url = base_url.rstrip("/")
        self._cache_ttl_seconds = float(cache_ttl_seconds)
        self._request_timeout_seconds = float(request_timeout_seconds)
        self._max_retries = int(max_retries)
        self._backoff_base_seconds = float(backoff_base_seconds)
        self._max_backoff_seconds = float(max_backoff_seconds)
        self._flow_veto_threshold = float(flow_veto_threshold)

        self._feature_engine = feature_engine or MicrostructureFeatureEngine()
        self._session = session or requests.Session()
        self._owns_session = session is None
        self._closed = False

        if hasattr(self._session, "headers"):
            self._session.headers.update(
                {
                    "Accept": "application/json",
                    "User-Agent": (
                        "Astra-LiveMicrostructureCollector/1.0 "
                        "(institutional-market-data-client)"
                    ),
                }
            )

        self._cache: Dict[str, Tuple[float, MicrostructureSnapshot]] = {}
        self._observations: Dict[str, List[_Observation]] = {}
        self._inflight_symbols: set[str] = set()

        self._cache_condition = threading.Condition(threading.RLock())
        self._http_lock = threading.RLock()

    def __enter__(self) -> LiveMicrostructureCollector:
        """Return this collector for context-manager usage."""
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Any],
    ) -> None:
        """Close owned network resources."""
        self.close()

    def close(self) -> None:
        """Close the internally managed HTTP session, if applicable."""
        with self._http_lock:
            if self._closed:
                return
            self._closed = True
            if self._owns_session:
                self._session.close()

    def fetch_orderbook_snapshot(
        self,
        symbol: str,
        limit: int = 50,
    ) -> pd.DataFrame:
        """Fetch an L2 orderbook snapshot from Binance USD-M futures.

        Args:
            symbol: Futures symbol in Binance or slash notation, for example
                ``BTCUSDT`` or ``BTC/USDT``.
            limit: Requested depth. Binance supports values up to 1000.

        Returns:
            A level-ranked dataframe with columns:
            ``['bid_px', 'bid_qty', 'ask_px', 'ask_qty']``.

            Bid rows are sorted descending by price and ask rows are sorted
            ascending by price. Rows are aligned by book level rather than
            price, with missing side levels represented as ``NaN``.

        Raises:
            ValueError: If limit is outside Binance-supported bounds.
            MicrostructureCollectionError: If Binance cannot be queried.
        """
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000 inclusive.")

        normalized_symbol = self._normalize_symbol(symbol)
        payload = self._request_json(
            endpoint="/fapi/v1/depth",
            params={"symbol": normalized_symbol, "limit": int(limit)},
        )

        if not isinstance(payload, Mapping):
            raise MicrostructureCollectionError(
                f"Unexpected depth payload type for {normalized_symbol}."
            )

        raw_bids = payload.get("bids")
        raw_asks = payload.get("asks")
        if not isinstance(raw_bids, list) or not isinstance(raw_asks, list):
            raise MicrostructureCollectionError(
                f"Malformed depth payload for {normalized_symbol}: missing bids or asks."
            )

        bids = self._parse_book_side(raw_bids, descending=True)
        asks = self._parse_book_side(raw_asks, descending=False)

        if not bids or not asks:
            raise MicrostructureCollectionError(
                f"Received an empty bid or ask side for {normalized_symbol}."
            )

        rows: List[Dict[str, Optional[float]]] = []
        max_levels = max(len(bids), len(asks))
        for index in range(max_levels):
            bid_px: Optional[float] = None
            bid_qty: Optional[float] = None
            ask_px: Optional[float] = None
            ask_qty: Optional[float] = None

            if index < len(bids):
                bid_px, bid_qty = bids[index]
            if index < len(asks):
                ask_px, ask_qty = asks[index]

            rows.append(
                {
                    "bid_px": bid_px,
                    "bid_qty": bid_qty,
                    "ask_px": ask_px,
                    "ask_qty": ask_qty,
                }
            )

        return pd.DataFrame(
            rows,
            columns=["bid_px", "bid_qty", "ask_px", "ask_qty"],
        )

    def fetch_recent_agg_trades(
        self,
        symbol: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch recent Binance aggregate trades for a symbol.

        Args:
            symbol: Futures symbol in Binance or slash notation.
            limit: Maximum aggregate trades to fetch, up to 1000.

        Returns:
            A timestamp-ascending dataframe with columns:
            ``['timestamp', 'price', 'quantity', 'is_buyer_maker']``.

            ``is_buyer_maker=True`` means the resting buyer was the maker,
            indicating seller aggression into bids. ``False`` indicates buyer
            aggression into asks.

        Raises:
            ValueError: If limit is outside Binance-supported bounds.
            MicrostructureCollectionError: If Binance cannot be queried.
        """
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000 inclusive.")

        normalized_symbol = self._normalize_symbol(symbol)
        payload = self._request_json(
            endpoint="/fapi/v1/aggTrades",
            params={"symbol": normalized_symbol, "limit": int(limit)},
        )

        if not isinstance(payload, list):
            raise MicrostructureCollectionError(
                f"Unexpected aggregate-trade payload type for {normalized_symbol}."
            )

        records: List[Dict[str, Any]] = []
        for trade in payload:
            if not isinstance(trade, Mapping):
                continue

            try:
                timestamp_ms = int(trade["T"])
                price = self._finite_float(trade["p"], "aggregate trade price")
                quantity = self._finite_float(trade["q"], "aggregate trade quantity")
                is_buyer_maker = bool(trade["m"])
            except (KeyError, TypeError, ValueError) as exc:
                LOGGER.warning(
                    "Skipping malformed aggregate trade for %s: %s",
                    normalized_symbol,
                    exc,
                )
                continue

            if quantity < 0.0 or price <= 0.0:
                continue

            records.append(
                {
                    "timestamp": pd.to_datetime(timestamp_ms, unit="ms", utc=True),
                    "price": price,
                    "quantity": quantity,
                    "is_buyer_maker": is_buyer_maker,
                }
            )

        trades = pd.DataFrame(
            records,
            columns=["timestamp", "price", "quantity", "is_buyer_maker"],
        )

        if not trades.empty:
            trades = trades.sort_values("timestamp", kind="stable").reset_index(
                drop=True
            )
            trades["price"] = trades["price"].astype(float)
            trades["quantity"] = trades["quantity"].astype(float)
            trades["is_buyer_maker"] = trades["is_buyer_maker"].astype(bool)

        return trades

    def fetch_funding_and_oi(self, symbol: str) -> Tuple[float, float, float]:
        """Fetch current funding rate, mark price, and open interest.

        Args:
            symbol: Futures symbol in Binance or slash notation.

        Returns:
            Tuple of ``(funding_rate, mark_price, open_interest)``.

        Raises:
            MicrostructureCollectionError: If required data is unavailable or
                malformed.
        """
        normalized_symbol = self._normalize_symbol(symbol)

        premium_payload = self._request_json(
            endpoint="/fapi/v1/premiumIndex",
            params={"symbol": normalized_symbol},
        )
        oi_payload = self._request_json(
            endpoint="/fapi/v1/openInterest",
            params={"symbol": normalized_symbol},
        )

        if not isinstance(premium_payload, Mapping):
            raise MicrostructureCollectionError(
                f"Malformed premium-index payload for {normalized_symbol}."
            )
        if not isinstance(oi_payload, Mapping):
            raise MicrostructureCollectionError(
                f"Malformed open-interest payload for {normalized_symbol}."
            )

        try:
            funding_rate = self._finite_float(
                premium_payload["lastFundingRate"],
                "last funding rate",
            )
            mark_price = self._finite_float(
                premium_payload["markPrice"],
                "mark price",
            )
            open_interest = self._finite_float(
                oi_payload["openInterest"],
                "open interest",
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise MicrostructureCollectionError(
                f"Malformed funding/open-interest data for {normalized_symbol}: {exc}"
            ) from exc

        if mark_price <= 0.0:
            raise MicrostructureCollectionError(
                f"Non-positive mark price received for {normalized_symbol}: {mark_price}."
            )
        if open_interest < 0.0:
            raise MicrostructureCollectionError(
                f"Negative open interest received for {normalized_symbol}: {open_interest}."
            )

        return funding_rate, mark_price, open_interest

    def get_live_microstructure(
        self,
        symbol: str,
        proposed_direction: Optional[str] = None,
    ) -> MicrostructureSnapshot:
        """Fetch, transform, cache, and evaluate live microstructure.

        A cached base snapshot is reused within the configured TTL. Veto logic
        is always evaluated per invocation, allowing callers with different
        proposed directions to share market-data cache entries safely.

        Args:
            symbol: Futures symbol in Binance or slash notation.
            proposed_direction: Optional proposed trade direction. Accepted
                values are ``long``, ``buy``, ``short``, and ``sell``.

        Returns:
            A complete immutable ``MicrostructureSnapshot``.

        Raises:
            ValueError: If the proposed direction is invalid.
            MicrostructureCollectionError: If live required data cannot be
                collected.
        """
        normalized_symbol = self._normalize_symbol(symbol)
        base_snapshot = self._get_cached_or_fetch(normalized_symbol)
        return self._apply_flow_veto(base_snapshot, proposed_direction)

    def _get_cached_or_fetch(self, normalized_symbol: str) -> MicrostructureSnapshot:
        """Return a fresh cache entry or fetch exactly one entry per symbol."""
        with self._cache_condition:
            while True:
                now_monotonic = time.monotonic()
                cached = self._cache.get(normalized_symbol)
                if cached is not None:
                    created_at, snapshot = cached
                    if now_monotonic - created_at < self._cache_ttl_seconds:
                        return snapshot
                    self._cache.pop(normalized_symbol, None)

                if normalized_symbol not in self._inflight_symbols:
                    self._inflight_symbols.add(normalized_symbol)
                    break

                self._cache_condition.wait(timeout=self._request_timeout_seconds)

        try:
            snapshot = self._build_live_snapshot(normalized_symbol)
        finally:
            with self._cache_condition:
                self._inflight_symbols.discard(normalized_symbol)
                self._cache_condition.notify_all()

        with self._cache_condition:
            self._cache[normalized_symbol] = (time.monotonic(), snapshot)

        return snapshot

    def _build_live_snapshot(self, normalized_symbol: str) -> MicrostructureSnapshot:
        """Perform uncached market-data collection and feature construction."""
        orderbook = self.fetch_orderbook_snapshot(normalized_symbol)
        trades = self.fetch_recent_agg_trades(normalized_symbol)
        funding_rate, mark_price, open_interest = self.fetch_funding_and_oi(
            normalized_symbol
        )

        ofi_norm, microprice_delta = self._compute_top_of_book_features(orderbook)
        cvd_delta, aggressor_buyer_ratio = self._compute_cvd_features(trades)

        now_ms = int(time.time() * 1000)
        funding_history = self._fetch_funding_history(
            normalized_symbol,
            current_timestamp_ms=now_ms,
            current_funding_rate=funding_rate,
        )
        oi_history = self._fetch_oi_history(
            normalized_symbol,
            current_timestamp_ms=now_ms,
            current_open_interest=open_interest,
        )

        local_observations = self._observations.get(normalized_symbol, [])
        for observation in local_observations:
            funding_history.append((observation.timestamp_ms, observation.funding_rate))
            oi_history.append((observation.timestamp_ms, observation.open_interest))

        funding_velocity = self._compute_velocity(
            current_value=funding_rate,
            current_timestamp_ms=now_ms,
            history=funding_history,
        )
        oi_velocity = self._compute_velocity(
            current_value=open_interest,
            current_timestamp_ms=now_ms,
            history=oi_history,
            percentage_change=True,
        )

        self._record_observation(
            normalized_symbol,
            _Observation(
                timestamp_ms=now_ms,
                funding_rate=funding_rate,
                open_interest=open_interest,
            ),
        )

        composite_flow_bias = self._compute_composite_flow_bias(
            mark_price=mark_price,
            ofi_norm=ofi_norm,
            microprice_delta=microprice_delta,
            cvd_delta=cvd_delta,
            trades=trades,
            aggressor_buyer_ratio=aggressor_buyer_ratio,
            funding_rate=funding_rate,
            funding_velocity=funding_velocity,
            oi_velocity=oi_velocity,
        )

        return MicrostructureSnapshot(
            symbol=normalized_symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            mark_price=mark_price,
            ofi_norm=ofi_norm,
            microprice_delta=microprice_delta,
            cvd_delta=cvd_delta,
            aggressor_buyer_ratio=aggressor_buyer_ratio,
            funding_rate=funding_rate,
            funding_velocity=funding_velocity,
            oi=open_interest,
            oi_velocity=oi_velocity,
            composite_flow_bias=composite_flow_bias,
            flow_veto=False,
            veto_reason=None,
        )

    def _compute_top_of_book_features(
        self,
        orderbook: pd.DataFrame,
    ) -> Tuple[float, float]:
        """Compute OFI and microprice displacement through the feature engine."""
        fallback_ofi, fallback_delta = self._fallback_top_of_book_features(orderbook)

        try:
            if hasattr(self._feature_engine, "compute_top_of_book_features"):
                result = self._feature_engine.compute_top_of_book_features(orderbook)
            else:
                ob_calc = orderbook.copy()
                if "ts" not in ob_calc.columns:
                    ob_calc["ts"] = pd.Timestamp.now(tz=timezone.utc)
                result = compute_top_of_book_features(ob_calc)
        except Exception as exc:
            LOGGER.debug(
                "MicrostructureFeatureEngine top-of-book computation fallback: %s",
                exc,
            )
            return fallback_ofi, fallback_delta

        if isinstance(result, Sequence) and not isinstance(
            result, (str, bytes, bytearray, pd.DataFrame, pd.Series)
        ):
            if len(result) >= 2:
                candidate_ofi = self._coerce_optional_finite(result[0])
                candidate_delta = self._coerce_optional_finite(result[1])
                if candidate_ofi is not None and candidate_delta is not None:
                    return (
                        self._clip(candidate_ofi, -1.0, 1.0),
                        candidate_delta,
                    )

        ofi_norm = self._extract_feature_value(
            result,
            (
                "ofi_norm",
                "normalized_ofi",
                "order_flow_imbalance_normalized",
                "order_flow_imbalance",
                "ofi",
            ),
        )
        microprice_delta = self._extract_feature_value(
            result,
            (
                "microprice_delta",
                "delta_mu",
                "microprice_displacement",
                "delta_microprice",
            ),
        )

        if microprice_delta is None:
            microprice = self._extract_feature_value(
                result,
                ("microprice", "micro_price"),
            )
            if microprice is not None:
                best_bid = float(orderbook["bid_px"].iloc[0])
                best_ask = float(orderbook["ask_px"].iloc[0])
                microprice_delta = microprice - ((best_bid + best_ask) / 2.0)

        return (
            self._clip(ofi_norm, -1.0, 1.0)
            if ofi_norm is not None
            else fallback_ofi,
            microprice_delta if microprice_delta is not None else fallback_delta,
        )

    def _compute_cvd_features(self, trades: pd.DataFrame) -> Tuple[float, float]:
        """Compute CVD and buyer aggressor ratio through the feature engine."""
        fallback_cvd, fallback_ratio = self._fallback_cvd_features(trades)

        try:
            if hasattr(self._feature_engine, "compute_cvd_features"):
                result = self._feature_engine.compute_cvd_features(trades)
            else:
                tr_calc = trades.copy()
                if "volume" not in tr_calc.columns and "quantity" in tr_calc.columns:
                    tr_calc["volume"] = tr_calc["quantity"]
                if "buyer_is_maker" not in tr_calc.columns and "is_buyer_maker" in tr_calc.columns:
                    tr_calc["buyer_is_maker"] = tr_calc["is_buyer_maker"]
                result = compute_cvd_features(tr_calc)
        except Exception as exc:
            LOGGER.debug(
                "MicrostructureFeatureEngine CVD computation fallback: %s",
                exc,
            )
            return fallback_cvd, fallback_ratio

        if isinstance(result, Sequence) and not isinstance(
            result, (str, bytes, bytearray, pd.DataFrame, pd.Series)
        ):
            if len(result) >= 2:
                candidate_cvd = self._coerce_optional_finite(result[0])
                candidate_ratio = self._coerce_optional_finite(result[1])
                if candidate_cvd is not None and candidate_ratio is not None:
                    return candidate_cvd, self._clip(candidate_ratio, 0.0, 1.0)

        cvd_delta = self._extract_feature_value(
            result,
            (
                "cvd_delta",
                "cumulative_volume_delta_change",
                "volume_delta",
                "delta_cvd",
            ),
        )
        buyer_ratio = self._extract_feature_value(
            result,
            (
                "aggressor_buyer_ratio",
                "buyer_aggressor_ratio",
                "buyer_ratio",
                "buy_ratio",
            ),
        )

        if cvd_delta is None and isinstance(result, pd.DataFrame):
            for column in result.columns:
                if self._canonical_name(column) in {
                    "cvd",
                    "cumulativevolumedelta",
                    "cumulative_delta",
                }:
                    series = pd.to_numeric(result[column], errors="coerce").dropna()
                    if not series.empty:
                        cvd_delta = float(series.iloc[-1] - series.iloc[0])
                    break

        return (
            cvd_delta if cvd_delta is not None else fallback_cvd,
            self._clip(buyer_ratio, 0.0, 1.0)
            if buyer_ratio is not None
            else fallback_ratio,
        )

    @staticmethod
    def _fallback_top_of_book_features(orderbook: pd.DataFrame) -> Tuple[float, float]:
        """Calculate normalized top-level OFI and microprice displacement."""
        required_columns = {"bid_px", "bid_qty", "ask_px", "ask_qty"}
        if not required_columns.issubset(orderbook.columns) or orderbook.empty:
            raise MicrostructureCollectionError(
                "Orderbook snapshot lacks valid top-of-book columns."
            )

        try:
            best_bid = float(orderbook["bid_px"].iloc[0])
            best_bid_qty = float(orderbook["bid_qty"].iloc[0])
            best_ask = float(orderbook["ask_px"].iloc[0])
            best_ask_qty = float(orderbook["ask_qty"].iloc[0])
        except (TypeError, ValueError, IndexError) as exc:
            raise MicrostructureCollectionError(
                "Unable to parse top-of-book price or quantity."
            ) from exc

        denominator = best_bid_qty + best_ask_qty
        if (
            not all(
                math.isfinite(value)
                for value in (best_bid, best_bid_qty, best_ask, best_ask_qty)
            )
            or best_bid <= 0.0
            or best_ask <= 0.0
            or best_bid_qty < 0.0
            or best_ask_qty < 0.0
            or denominator <= 0.0
        ):
            raise MicrostructureCollectionError(
                "Invalid top-of-book values received from exchange."
            )

        ofi_norm = (best_bid_qty - best_ask_qty) / denominator
        mid_price = (best_bid + best_ask) / 2.0
        microprice = (
            (best_ask * best_bid_qty) + (best_bid * best_ask_qty)
        ) / denominator
        microprice_delta = microprice - mid_price

        return max(-1.0, min(1.0, ofi_norm)), microprice_delta

    @staticmethod
    def _fallback_cvd_features(trades: pd.DataFrame) -> Tuple[float, float]:
        """Calculate quantity-weighted CVD and buyer aggressor ratio."""
        if trades.empty:
            return 0.0, 0.5

        required_columns = {"quantity", "is_buyer_maker"}
        if not required_columns.issubset(trades.columns):
            raise MicrostructureCollectionError(
                "Aggregate-trade dataframe lacks required CVD columns."
            )

        quantities = pd.to_numeric(trades["quantity"], errors="coerce").fillna(0.0)
        quantities = quantities.clip(lower=0.0)
        seller_aggressor = trades["is_buyer_maker"].astype(bool)

        buyer_volume = float(quantities.loc[~seller_aggressor].sum())
        seller_volume = float(quantities.loc[seller_aggressor].sum())
        total_volume = buyer_volume + seller_volume

        if total_volume <= 0.0:
            return 0.0, 0.5

        cvd_delta = buyer_volume - seller_volume
        buyer_ratio = buyer_volume / total_volume
        return cvd_delta, buyer_ratio

    def _fetch_funding_history(
        self,
        normalized_symbol: str,
        current_timestamp_ms: int,
        current_funding_rate: float,
    ) -> List[Tuple[int, float]]:
        """Fetch settled funding history, retaining local fallback on failure."""
        history: List[Tuple[int, float]] = [
            (current_timestamp_ms, current_funding_rate)
        ]

        try:
            payload = self._request_json(
                endpoint="/fapi/v1/fundingRate",
                params={"symbol": normalized_symbol, "limit": 10},
            )
            if not isinstance(payload, list):
                raise MicrostructureCollectionError(
                    f"Malformed funding history payload for {normalized_symbol}."
                )

            for record in payload:
                if not isinstance(record, Mapping):
                    continue
                timestamp_ms = int(record["fundingTime"])
                funding_rate = self._finite_float(
                    record["fundingRate"],
                    "historical funding rate",
                )
                history.append((timestamp_ms, funding_rate))
        except Exception as exc:
            LOGGER.warning(
                "Could not retrieve funding history for %s; falling back to "
                "locally retained observations: %s",
                normalized_symbol,
                exc,
            )

        return history

    def _fetch_oi_history(
        self,
        normalized_symbol: str,
        current_timestamp_ms: int,
        current_open_interest: float,
    ) -> List[Tuple[int, float]]:
        """Fetch hourly OI history, retaining local fallback on failure."""
        history: List[Tuple[int, float]] = [
            (current_timestamp_ms, current_open_interest)
        ]

        try:
            payload = self._request_json(
                endpoint="/futures/data/openInterestHist",
                params={
                    "symbol": normalized_symbol,
                    "period": "1h",
                    "limit": 30,
                },
            )
            if not isinstance(payload, list):
                raise MicrostructureCollectionError(
                    f"Malformed OI history payload for {normalized_symbol}."
                )

            for record in payload:
                if not isinstance(record, Mapping):
                    continue
                timestamp_ms = int(record["timestamp"])
                open_interest = self._finite_float(
                    record["sumOpenInterest"],
                    "historical open interest",
                )
                if open_interest >= 0.0:
                    history.append((timestamp_ms, open_interest))
        except Exception as exc:
            LOGGER.warning(
                "Could not retrieve OI history for %s; falling back to locally "
                "retained observations: %s",
                normalized_symbol,
                exc,
            )

        return history

    @staticmethod
    def _compute_velocity(
        current_value: float,
        current_timestamp_ms: int,
        history: Sequence[Tuple[int, float]],
        percentage_change: bool = False,
    ) -> float:
        """Compute a per-hour 8h velocity, falling back to a 24h reference."""
        if not math.isfinite(current_value):
            return 0.0

        deduplicated: Dict[int, float] = {}
        for timestamp_ms, value in history:
            if (
                isinstance(timestamp_ms, int)
                and timestamp_ms > 0
                and math.isfinite(value)
            ):
                deduplicated[timestamp_ms] = value

        for horizon_hours in (8.0, 24.0):
            target_timestamp_ms = current_timestamp_ms - int(
                horizon_hours * 3_600_000
            )
            eligible = [
                (timestamp_ms, value)
                for timestamp_ms, value in deduplicated.items()
                if timestamp_ms <= target_timestamp_ms
            ]

            if not eligible:
                continue

            reference_timestamp_ms, reference_value = max(
                eligible,
                key=lambda item: item[0],
            )
            elapsed_hours = (
                current_timestamp_ms - reference_timestamp_ms
            ) / 3_600_000.0

            if elapsed_hours <= 0.0:
                continue

            if percentage_change:
                if abs(reference_value) <= 1e-12:
                    continue
                return ((current_value - reference_value) / reference_value) / elapsed_hours

            return (current_value - reference_value) / elapsed_hours

        return 0.0

    def _record_observation(
        self,
        normalized_symbol: str,
        observation: _Observation,
    ) -> None:
        """Retain bounded local observations for auxiliary-history resilience."""
        retention_ms = 72 * 3_600_000

        with self._cache_condition:
            observations = self._observations.setdefault(normalized_symbol, [])
            observations.append(observation)

            cutoff_ms = observation.timestamp_ms - retention_ms
            self._observations[normalized_symbol] = [
                item
                for item in observations
                if item.timestamp_ms >= cutoff_ms
            ][-512:]

    @staticmethod
    def _compute_composite_flow_bias(
        mark_price: float,
        ofi_norm: float,
        microprice_delta: float,
        cvd_delta: float,
        trades: pd.DataFrame,
        aggressor_buyer_ratio: float,
        funding_rate: float,
        funding_velocity: float,
        oi_velocity: float,
    ) -> float:
        """Combine liquidity, aggressor, funding, and OI features into [-1, 1]."""
        total_trade_volume = 0.0
        if not trades.empty and "quantity" in trades.columns:
            total_trade_volume = float(
                pd.to_numeric(trades["quantity"], errors="coerce")
                .fillna(0.0)
                .clip(lower=0.0)
                .sum()
            )

        cvd_signal = (
            math.tanh(cvd_delta / max(total_trade_volume, 1e-12))
            if total_trade_volume > 0.0
            else 0.0
        )
        microprice_signal = math.tanh(
            (microprice_delta / max(mark_price, 1e-12)) / 0.00025
        )
        aggressor_signal = (2.0 * max(0.0, min(1.0, aggressor_buyer_ratio))) - 1.0

        funding_crowding_signal = math.tanh(
            (funding_rate / 0.0005) + (funding_velocity / 0.0001)
        )
        oi_confirmation_signal = math.copysign(
            math.tanh(abs(oi_velocity) / 0.01),
            cvd_signal,
        ) if abs(cvd_signal) > 1e-12 else 0.0

        score = (
            (0.30 * max(-1.0, min(1.0, ofi_norm)))
            + (0.15 * microprice_signal)
            + (0.25 * cvd_signal)
            + (0.15 * aggressor_signal)
            + (0.10 * oi_confirmation_signal)
            - (0.05 * funding_crowding_signal)
        )

        return max(-1.0, min(1.0, score))

    def _apply_flow_veto(
        self,
        snapshot: MicrostructureSnapshot,
        proposed_direction: Optional[str],
    ) -> MicrostructureSnapshot:
        """Apply direction-specific flow veto without mutating cached data."""
        if proposed_direction is None:
            return replace(snapshot, flow_veto=False, veto_reason=None)

        direction = proposed_direction.strip().lower()
        direction_aliases = {
            "long": "long",
            "buy": "long",
            "bullish": "long",
            "short": "short",
            "sell": "short",
            "bearish": "short",
        }

        if direction not in direction_aliases:
            raise ValueError(
                "proposed_direction must be one of: long, buy, short, sell, "
                "bullish, bearish."
            )

        normalized_direction = direction_aliases[direction]
        bias = snapshot.composite_flow_bias

        if normalized_direction == "long" and bias <= -self._flow_veto_threshold:
            return replace(
                snapshot,
                flow_veto=True,
                veto_reason=(
                    f"Strong bearish microstructure bias ({bias:+.3f}) "
                    "contradicts proposed long exposure."
                ),
            )

        if normalized_direction == "short" and bias >= self._flow_veto_threshold:
            return replace(
                snapshot,
                flow_veto=True,
                veto_reason=(
                    f"Strong bullish microstructure bias ({bias:+.3f}) "
                    "contradicts proposed short exposure."
                ),
            )

        return replace(snapshot, flow_veto=False, veto_reason=None)

    def _request_json(
        self,
        endpoint: str,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Issue a rate-limit-aware REST request and return decoded JSON."""
        if self._closed:
            raise MicrostructureCollectionError("Collector is closed.")

        url = f"{self._base_url}{endpoint}"
        request_params = dict(params or {})
        last_error: Optional[BaseException] = None

        for attempt in range(self._max_retries + 1):
            response: Optional[requests.Response] = None

            try:
                with self._http_lock:
                    response = self._session.get(
                        url,
                        params=request_params,
                        timeout=self._request_timeout_seconds,
                    )
            except requests.RequestException as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    break
                self._sleep_before_retry(attempt, None)
                continue

            status_code = int(response.status_code)
            retry_after = self._retry_after_seconds(response)

            if status_code in self._RETRYABLE_STATUS_CODES:
                last_error = MicrostructureCollectionError(
                    f"Binance returned retryable HTTP {status_code} for {endpoint}."
                )
                if attempt >= self._max_retries:
                    break
                self._sleep_before_retry(attempt, retry_after)
                continue

            if status_code < 200 or status_code >= 300:
                response_text = getattr(response, "text", "")
                raise MicrostructureCollectionError(
                    f"Binance HTTP {status_code} for {endpoint}; "
                    f"response={response_text[:500]!r}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise MicrostructureCollectionError(
                    f"Binance returned invalid JSON for {endpoint}."
                ) from exc

            if isinstance(payload, Mapping) and "code" in payload:
                try:
                    api_code = int(payload["code"])
                except (TypeError, ValueError):
                    api_code = 0

                if api_code < 0:
                    message = str(payload.get("msg", "Unknown Binance API error"))
                    if api_code in self._RATE_LIMIT_ERROR_CODES:
                        last_error = MicrostructureCollectionError(
                            f"Binance rate-limit error {api_code} for {endpoint}: "
                            f"{message}"
                        )
                        if attempt >= self._max_retries:
                            break
                        self._sleep_before_retry(attempt, retry_after)
                        continue

                    raise MicrostructureCollectionError(
                        f"Binance API error {api_code} for {endpoint}: {message}"
                    )

            return payload

        raise MicrostructureCollectionError(
            f"Failed to retrieve {endpoint} after {self._max_retries + 1} attempts."
        ) from last_error

    def _sleep_before_retry(
        self,
        attempt: int,
        retry_after_seconds: Optional[float],
    ) -> None:
        """Sleep using bounded exponential backoff and bounded random jitter."""
        if retry_after_seconds is not None and retry_after_seconds > 0.0:
            delay = min(retry_after_seconds, self._max_backoff_seconds)
        else:
            exponential_delay = min(
                self._backoff_base_seconds * (2.0**attempt),
                self._max_backoff_seconds,
            )
            jitter = random.uniform(0.0, exponential_delay * 0.25)
            delay = min(exponential_delay + jitter, self._max_backoff_seconds)

        LOGGER.warning("Retrying Binance request in %.3f seconds.", delay)
        time.sleep(delay)

    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> Optional[float]:
        """Extract a valid Retry-After header value, if supplied."""
        headers = getattr(response, "headers", {})
        raw_value = headers.get("Retry-After") if headers is not None else None
        if raw_value is None:
            return None

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None

        return value if value > 0.0 and math.isfinite(value) else None

    @staticmethod
    def _parse_book_side(
        raw_levels: Sequence[Any],
        descending: bool,
    ) -> List[Tuple[float, float]]:
        """Parse and sort a Binance depth side."""
        levels: List[Tuple[float, float]] = []

        for raw_level in raw_levels:
            if not isinstance(raw_level, Sequence) or isinstance(
                raw_level, (str, bytes, bytearray)
            ):
                continue
            if len(raw_level) < 2:
                continue

            try:
                price = LiveMicrostructureCollector._finite_float(
                    raw_level[0],
                    "book price",
                )
                quantity = LiveMicrostructureCollector._finite_float(
                    raw_level[1],
                    "book quantity",
                )
            except (TypeError, ValueError):
                continue

            if price > 0.0 and quantity >= 0.0:
                levels.append((price, quantity))

        return sorted(levels, key=lambda level: level[0], reverse=descending)

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize slash, colon, and Binance futures symbol notation."""
        if not isinstance(symbol, str):
            raise ValueError("symbol must be a string.")

        normalized = symbol.strip().upper()
        if ":" in normalized:
            normalized = normalized.split(":", maxsplit=1)[0]
        normalized = normalized.replace("/", "").replace("-", "").replace("_", "")

        if not LiveMicrostructureCollector._SYMBOL_PATTERN.fullmatch(normalized):
            raise ValueError(
                f"Invalid futures symbol {symbol!r}; expected e.g. BTCUSDT or BTC/USDT."
            )

        return normalized

    @staticmethod
    def _finite_float(value: Any, field_name: str) -> float:
        """Convert a value to a finite float or raise ValueError."""
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{field_name} must be finite, received {value!r}.")
        return converted

    @staticmethod
    def _coerce_optional_finite(value: Any) -> Optional[float]:
        """Convert a candidate feature value to a finite float where possible."""
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted if math.isfinite(converted) else None

    @classmethod
    def _extract_feature_value(
        cls,
        result: Any,
        candidate_names: Sequence[str],
    ) -> Optional[float]:
        """Extract a numeric value from common feature-engine return structures."""
        canonical_candidates = {cls._canonical_name(name) for name in candidate_names}

        if isinstance(result, pd.DataFrame):
            for column in result.columns:
                if cls._canonical_name(column) in canonical_candidates:
                    values = pd.to_numeric(result[column], errors="coerce").dropna()
                    if not values.empty:
                        return cls._coerce_optional_finite(values.iloc[-1])
            return None

        if isinstance(result, pd.Series):
            for index, value in result.items():
                if cls._canonical_name(index) in canonical_candidates:
                    return cls._coerce_optional_finite(value)
            return None

        if isinstance(result, Mapping):
            for key, value in result.items():
                if cls._canonical_name(key) in canonical_candidates:
                    return cls._coerce_optional_finite(value)
            return None

        for name in candidate_names:
            if hasattr(result, name):
                return cls._coerce_optional_finite(getattr(result, name))

        return None

    @staticmethod
    def _canonical_name(value: Any) -> str:
        """Normalize names for resilient feature-result field matching."""
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    @staticmethod
    def _clip(value: Optional[float], lower: float, upper: float) -> float:
        """Clamp a finite numeric value into a closed interval."""
        if value is None or not math.isfinite(value):
            return 0.0
        return max(lower, min(upper, value))


if __name__ == "__main__":
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    requested_symbol = os.getenv("MICROSTRUCTURE_SYMBOL", "SOL/USDT")
    requested_direction = os.getenv("PROPOSED_DIRECTION")

    try:
        with LiveMicrostructureCollector() as collector:
            live_snapshot = collector.get_live_microstructure(
                symbol=requested_symbol,
                proposed_direction=requested_direction,
            )

        print("=" * 72)
        print(f"LIVE MICROSTRUCTURE REPORT: {live_snapshot.symbol}")
        print("=" * 72)
        print(f"Timestamp                 : {live_snapshot.timestamp}")
        print(f"Mark Price                : {live_snapshot.mark_price:,.8f}")
        print(f"Normalized OFI            : {live_snapshot.ofi_norm:+.6f}")
        print(f"Microprice Delta          : {live_snapshot.microprice_delta:+.10f}")
        print(f"CVD Delta                 : {live_snapshot.cvd_delta:+,.6f}")
        print(
            "Buyer Aggressor Ratio     : "
            f"{live_snapshot.aggressor_buyer_ratio:.2%}"
        )
        print(f"Funding Rate              : {live_snapshot.funding_rate:+.8%}")
        print(
            "Funding Velocity / Hour   : "
            f"{live_snapshot.funding_velocity:+.10f}"
        )
        print(f"Open Interest             : {live_snapshot.oi:,.6f}")
        print(f"OI Velocity / Hour        : {live_snapshot.oi_velocity:+.6%}")
        print(
            "Composite Flow Bias       : "
            f"{live_snapshot.composite_flow_bias:+.4f}"
        )
        print(f"Flow Veto                 : {live_snapshot.flow_veto}")
        print(f"Veto Reason               : {live_snapshot.veto_reason or 'None'}")
        print("=" * 72)
    except Exception as exc:
        LOGGER.exception("Unable to produce live microstructure report: %s", exc)
        raise SystemExit(1) from exc