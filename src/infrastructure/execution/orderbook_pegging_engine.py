from __future__ import annotations

import asyncio
import inspect
import logging
import math
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Sequence, Tuple


LOGGER = logging.getLogger(__name__)


class PeggingEngineError(RuntimeError):
    """Base exception raised by the orderbook pegging engine."""


class ExchangeUnavailableError(PeggingEngineError):
    """Raised when an exchange operation cannot be completed."""


class InvalidOrderStateError(PeggingEngineError):
    """Raised when an operation is invalid for an order state."""


@dataclass(frozen=True)
class FeeTier:
    """Exchange fee tier applicable above a rolling thirty-day volume threshold."""

    minimum_30d_volume: float
    maker_fee: float
    taker_fee: float

    def __post_init__(self) -> None:
        if self.minimum_30d_volume < 0:
            raise ValueError("minimum_30d_volume must be non-negative")
        if self.maker_fee < 0 or self.taker_fee < 0:
            raise ValueError("fee rates must be non-negative")


@dataclass
class PeggingConfig:
    """Configuration for maker-first orderbook pegging and adverse-flow protection."""

    order_timeout_seconds: float = 30.0
    cancellation_timeout_seconds: float = 5.0
    market_exit_timeout_seconds: float = 8.0
    quote_refresh_seconds: float = 1.0

    requote_threshold_bps: float = 1.0
    retreat_bps: float = 2.0
    max_adverse_price_drift_bps: float = 8.0
    max_adverse_events: int = 2
    adverse_flow_window_seconds: float = 30.0

    toxic_notional_threshold: float = 100_000.0
    max_requotes: int = 100
    post_only_retries: int = 2

    maker_fee: float = 0.0002
    taker_fee: float = 0.0005
    fee_tiers: Tuple[FeeTier, ...] = field(default_factory=tuple)

    price_tick_size: Optional[float] = None
    queue_price_tolerance_bps: float = 0.5
    position_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.order_timeout_seconds <= 0:
            raise ValueError("order_timeout_seconds must be positive")
        if self.cancellation_timeout_seconds <= 0:
            raise ValueError("cancellation_timeout_seconds must be positive")
        if self.market_exit_timeout_seconds <= 0:
            raise ValueError("market_exit_timeout_seconds must be positive")
        if self.quote_refresh_seconds <= 0:
            raise ValueError("quote_refresh_seconds must be positive")
        if self.requote_threshold_bps < 0:
            raise ValueError("requote_threshold_bps must be non-negative")
        if self.retreat_bps < 0:
            raise ValueError("retreat_bps must be non-negative")
        if self.max_adverse_price_drift_bps < 0:
            raise ValueError("max_adverse_price_drift_bps must be non-negative")
        if self.max_adverse_events < 1:
            raise ValueError("max_adverse_events must be at least one")
        if self.adverse_flow_window_seconds <= 0:
            raise ValueError("adverse_flow_window_seconds must be positive")
        if self.toxic_notional_threshold <= 0:
            raise ValueError("toxic_notional_threshold must be positive")
        if self.max_requotes < 0:
            raise ValueError("max_requotes must be non-negative")
        if self.post_only_retries < 0:
            raise ValueError("post_only_retries must be non-negative")
        if self.maker_fee < 0 or self.taker_fee < 0:
            raise ValueError("maker_fee and taker_fee must be non-negative")
        if self.price_tick_size is not None and self.price_tick_size <= 0:
            raise ValueError("price_tick_size must be positive when specified")

        normalized_tiers = tuple(
            sorted(
                (
                    tier
                    if isinstance(tier, FeeTier)
                    else FeeTier(
                        minimum_30d_volume=float(tier["minimum_30d_volume"]),
                        maker_fee=float(tier["maker_fee"]),
                        taker_fee=float(tier["taker_fee"]),
                    )
                    for tier in self.fee_tiers
                ),
                key=lambda tier: tier.minimum_30d_volume,
            )
        )
        self.fee_tiers = normalized_tiers

    def fee_rates(self, rolling_30d_volume: float = 0.0) -> Tuple[float, float]:
        """Return maker and taker fees for the supplied rolling thirty-day volume."""
        selected_maker = self.maker_fee
        selected_taker = self.taker_fee

        for tier in self.fee_tiers:
            if rolling_30d_volume >= tier.minimum_30d_volume:
                selected_maker = tier.maker_fee
                selected_taker = tier.taker_fee
            else:
                break

        return selected_maker, selected_taker


@dataclass
class BookSnapshot:
    """Best bid and ask snapshot used for maker-first pegging."""

    best_bid: float
    best_ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    timestamp: float = field(default_factory=time.time)

    @property
    def mid_price(self) -> float:
        """Return the midpoint between the best bid and best ask."""
        return (self.best_bid + self.best_ask) / 2.0


@dataclass
class QueueEstimate:
    """Queue-ahead estimate for a passive order resting at the top of book."""

    initial_queue_ahead: float = 0.0
    queue_ahead: float = 0.0
    printed_at_level: float = 0.0
    estimated_queue_depletion_ratio: float = 0.0

    def register_trade(self, amount: float) -> None:
        """Apply an aggressive trade print to the estimated queue ahead."""
        if amount <= 0:
            return

        self.printed_at_level += amount
        self.queue_ahead = max(0.0, self.queue_ahead - amount)

        if self.initial_queue_ahead > 0:
            self.estimated_queue_depletion_ratio = min(
                1.0,
                self.printed_at_level / self.initial_queue_ahead,
            )
        else:
            self.estimated_queue_depletion_ratio = 1.0


@dataclass
class PeggedOrderState:
    """Mutable state associated with a live maker-first pegged order."""

    symbol: str
    side: str
    amount: float
    client_order_id: str
    created_monotonic: float
    maker_fee_rate: float
    taker_fee_rate: float

    order_id: Optional[str] = None
    status: str = "pending"
    working_price: Optional[float] = None
    last_quote_monotonic: float = field(default_factory=time.monotonic)
    requote_count: int = 0

    filled_amount: float = 0.0
    exposure_amount: float = 0.0
    average_fill_price: Optional[float] = None
    estimated_maker_fee: float = 0.0
    estimated_taker_exit_fee: float = 0.0

    queue: QueueEstimate = field(default_factory=QueueEstimate)
    adverse_events: Deque[float] = field(default_factory=deque)
    toxic_events: int = 0

    known_order_ids: set[str] = field(default_factory=set)
    order_fills: Dict[str, float] = field(default_factory=dict)
    order_average_prices: Dict[str, float] = field(default_factory=dict)

    last_error: Optional[str] = None
    exit_attempted: bool = False
    exit_order_id: Optional[str] = None

    @property
    def fill_ratio(self) -> float:
        """Return cumulative filled quantity as a fraction of requested quantity."""
        if self.amount <= 0:
            return 0.0
        return min(1.0, self.filled_amount / self.amount)

    @property
    def is_active(self) -> bool:
        """Return whether the strategy should continue managing this order."""
        return self.status in {"pending", "open", "replacing"}

    @property
    def slippage_bps(self) -> float:
        """Return realized fill slippage against the current or original quote price."""
        reference = self.working_price
        if reference is None or reference <= 0 or self.average_fill_price is None:
            return 0.0

        if self.side == "buy":
            return ((self.average_fill_price - reference) / reference) * 10_000.0

        return ((reference - self.average_fill_price) / reference) * 10_000.0


class OrderbookPeggingEngine:
    """
    Maker-first, post-only orderbook pegging engine for CCXT-compatible exchanges.

    The engine pegs BUY orders to the best bid and SELL orders to the best ask.
    It tracks estimated queue depletion from aggressive top-of-book prints, detects
    adverse taker flow, cancels and retreats after toxic sweeps, and sends a
    reduce-only market order when a filled position exceeds configured risk limits.

    The supplied exchange may be either a synchronous CCXT exchange or an
    ``ccxt.async_support`` exchange instance.
    """

    def __init__(
        self,
        exchange: Optional[Any] = None,
        config: Optional[PeggingConfig] = None,
        *,
        rolling_30d_volume: float = 0.0,
    ) -> None:
        self.exchange = exchange
        self.config = config or PeggingConfig()
        self.rolling_30d_volume = float(rolling_30d_volume)

        self._books: Dict[str, BookSnapshot] = {}
        self._orders: Dict[Tuple[str, str], PeggedOrderState] = {}
        self._async_lock = asyncio.Lock()
        self._thread_lock = threading.RLock()
        self._closed = False

    @classmethod
    async def create_binance_usdm(
        cls,
        api_key: str,
        secret: str,
        *,
        config: Optional[PeggingConfig] = None,
        password: Optional[str] = None,
        sandbox: bool = False,
        exchange_options: Optional[Mapping[str, Any]] = None,
        rolling_30d_volume: float = 0.0,
    ) -> "OrderbookPeggingEngine":
        """
        Create an engine backed by CCXT Binance USD-M futures.

        The method imports ``ccxt.async_support`` lazily so that using this module
        for deterministic tests or externally-managed exchange instances does not
        require CCXT at import time.
        """
        try:
            import ccxt.async_support as ccxt_async  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ExchangeUnavailableError(
                "ccxt.async_support is required to create a Binance USD-M client"
            ) from exc

        options: Dict[str, Any] = {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "future"},
        }
        if password:
            options["password"] = password
        if exchange_options:
            options.update(dict(exchange_options))

        exchange = ccxt_async.binanceusdm(options)
        if sandbox:
            exchange.set_sandbox_mode(True)

        return cls(
            exchange=exchange,
            config=config,
            rolling_30d_volume=rolling_30d_volume,
        )

    async def close(self) -> None:
        """Close the underlying CCXT exchange client when it exposes ``close``."""
        if self._closed:
            return

        self._closed = True
        if self.exchange is not None and hasattr(self.exchange, "close"):
            try:
                await self._exchange_call("close")
            except Exception:
                LOGGER.exception("Failed to close exchange client cleanly")

    async def __aenter__(self) -> "OrderbookPeggingEngine":
        """Return the engine for async context-manager usage."""
        return self

    async def __aexit__(self, *_: Any) -> None:
        """Close the engine when leaving an async context."""
        await self.close()

    async def refresh_orderbook(self, symbol: str) -> BookSnapshot:
        """Fetch and store a fresh top-of-book snapshot from the exchange."""
        if self.exchange is None:
            with self._thread_lock:
                book = self._books.get(symbol)
            if book is None:
                raise ExchangeUnavailableError(
                    f"No local orderbook is available for {symbol}"
                )
            return book

        try:
            response = await self._exchange_call("fetch_order_book", symbol, 5)
        except Exception as exc:
            raise ExchangeUnavailableError(
                f"Unable to fetch orderbook for {symbol}: {exc}"
            ) from exc

        return await self.ingest_order_book(symbol, response)

    async def ingest_order_book(
        self,
        symbol: str,
        order_book: Mapping[str, Any],
    ) -> BookSnapshot:
        """Ingest a CCXT orderbook payload and retain its best bid and ask."""
        bids = order_book.get("bids") or ()
        asks = order_book.get("asks") or ()

        if not bids or not asks:
            raise ValueError(f"Orderbook for {symbol} does not contain both bid and ask")

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        bid_size = float(bids[0][1]) if len(bids[0]) > 1 else 0.0
        ask_size = float(asks[0][1]) if len(asks[0]) > 1 else 0.0

        timestamp_value = order_book.get("timestamp")
        timestamp = (
            float(timestamp_value) / 1000.0
            if timestamp_value is not None and float(timestamp_value) > 1_000_000_000_000
            else float(timestamp_value or time.time())
        )

        return await self.update_orderbook(
            symbol,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_size=bid_size,
            ask_size=ask_size,
            timestamp=timestamp,
        )

    async def update_orderbook(
        self,
        symbol: str,
        best_bid: float,
        best_ask: float,
        bid_size: float = 0.0,
        ask_size: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> BookSnapshot:
        """Update the local best bid/ask snapshot for a trading symbol."""
        if best_bid <= 0 or best_ask <= 0:
            raise ValueError("best_bid and best_ask must be positive")
        if best_bid >= best_ask:
            raise ValueError("best_bid must be strictly lower than best_ask")

        book = BookSnapshot(
            best_bid=float(best_bid),
            best_ask=float(best_ask),
            bid_size=max(0.0, float(bid_size)),
            ask_size=max(0.0, float(ask_size)),
            timestamp=float(timestamp or time.time()),
        )

        async with self._async_lock:
            with self._thread_lock:
                self._books[symbol] = book

        return book

    async def on_orderbook(
        self,
        symbol: str,
        order_book: Mapping[str, Any],
    ) -> BookSnapshot:
        """Compatibility alias for ingesting an incoming CCXT orderbook payload."""
        return await self.ingest_order_book(symbol, order_book)

    async def place_pegged_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        *,
        client_order_id: Optional[str] = None,
    ) -> str:
        """
        Place a maker-first post-only pegged order.

        BUY orders are placed at the current best bid. SELL orders are placed at
        the current best ask. An existing active peg for the same symbol and side
        is cancelled before the new order is submitted.
        """
        self._ensure_open()
        normalized_side = self._normalize_side(side)
        amount = float(amount)
        if amount <= 0:
            raise ValueError("amount must be positive")

        key = (symbol, normalized_side)
        existing = self._get_order(key)
        if existing is not None and existing.is_active:
            await self.cancel_pegged_order(symbol, normalized_side, reason="replaced")

        book = await self._get_or_refresh_book(symbol)
        maker_fee, taker_fee = self.config.fee_rates(self.rolling_30d_volume)

        state = PeggedOrderState(
            symbol=symbol,
            side=normalized_side,
            amount=amount,
            client_order_id=client_order_id or self._new_client_order_id(symbol, normalized_side),
            created_monotonic=time.monotonic(),
            maker_fee_rate=maker_fee,
            taker_fee_rate=taker_fee,
        )

        async with self._async_lock:
            with self._thread_lock:
                self._orders[key] = state

        try:
            await self._submit_state(state, book=book, retreat=False)
        except Exception:
            async with self._async_lock:
                with self._thread_lock:
                    state.status = "error"
                    state.last_error = "initial post-only submission failed"
            raise

        if state.order_id is None:
            raise PeggingEngineError("Exchange did not return an order identifier")

        return state.order_id

    async def submit(
        self,
        symbol: str,
        side: str,
        amount: float,
        *,
        client_order_id: Optional[str] = None,
    ) -> str:
        """Compatibility alias for :meth:`place_pegged_order`."""
        return await self.place_pegged_order(
            symbol,
            side,
            amount,
            client_order_id=client_order_id,
        )

    async def cancel_pegged_order(
        self,
        symbol: str,
        side: str,
        *,
        reason: str = "cancelled",
    ) -> bool:
        """Cancel a live pegged order and stop managing it."""
        normalized_side = self._normalize_side(side)
        state = self._get_order((symbol, normalized_side))
        if state is None:
            return False

        cancelled = await self._cancel_current_order(state)
        async with self._async_lock:
            with self._thread_lock:
                if state.status not in {"exited", "exit_failed"}:
                    state.status = reason
        return cancelled

    async def stop_pegging(self, symbol: str, side: str) -> bool:
        """Stop an active peg by cancelling its current passive order."""
        return await self.cancel_pegged_order(symbol, side, reason="stopped")

    async def on_trade(
        self,
        symbol: str,
        trade: Mapping[str, Any] | float,
        amount: Optional[float] = None,
        side: Optional[str] = None,
        *,
        timestamp: Optional[float] = None,
        is_sweep: Optional[bool] = None,
    ) -> None:
        """
        Process an aggressive trade print for queue and toxic-flow estimation.

        For a resting BUY, aggressive SELL prints at the bid decrement estimated
        queue ahead. For a resting SELL, aggressive BUY prints at the ask decrement
        queue ahead. Large adverse prints above the configured USD notional threshold
        trigger immediate cancellation and a retreat quote. Repeated toxic prints or
        excessive adverse drift trigger a reduce-only market exit for filled exposure.
        """
        parsed = self._parse_trade(
            trade=trade,
            amount=amount,
            side=side,
            timestamp=timestamp,
            is_sweep=is_sweep,
        )
        if parsed is None:
            return

        price, quantity, aggressor_side, trade_timestamp, explicit_sweep = parsed
        if quantity <= 0 or price <= 0:
            return

        notional = price * quantity
        actions: list[Tuple[str, Tuple[str, str]]] = []

        async with self._async_lock:
            with self._thread_lock:
                book = self._books.get(symbol)
                candidates = [
                    (key, state)
                    for key, state in self._orders.items()
                    if key[0] == symbol and state.is_active
                ]

                for key, state in candidates:
                    if not self._is_adverse_aggressor(state.side, aggressor_side):
                        continue

                    if self._trade_hits_passive_level(state, book, price):
                        state.queue.register_trade(quantity)

                    sweep = explicit_sweep
                    if sweep is None:
                        sweep = self._trade_sweeps_quote(state, book, price)

                    if not sweep or notional < self.config.toxic_notional_threshold:
                        continue

                    state.toxic_events += 1
                    state.adverse_events.append(trade_timestamp)
                    self._prune_adverse_events(state, trade_timestamp)

                    adverse_drift = self._adverse_drift_bps(state, book)
                    adverse_limit_breached = (
                        len(state.adverse_events) >= self.config.max_adverse_events
                        or adverse_drift >= self.config.max_adverse_price_drift_bps
                    )

                    if adverse_limit_breached:
                        actions.append(("exit", key))
                    else:
                        actions.append(("retreat", key))

        for action, key in actions:
            state = self._get_order(key)
            if state is None:
                continue

            if action == "exit":
                await self._risk_exit(state, reason="toxic_flow_risk_limit")
            else:
                await self._retreat_order(state, reason="toxic_flow")

    async def process_trade(
        self,
        symbol: str,
        trade: Mapping[str, Any] | float,
        amount: Optional[float] = None,
        side: Optional[str] = None,
        *,
        timestamp: Optional[float] = None,
        is_sweep: Optional[bool] = None,
    ) -> None:
        """Compatibility alias for :meth:`on_trade`."""
        await self.on_trade(
            symbol,
            trade,
            amount=amount,
            side=side,
            timestamp=timestamp,
            is_sweep=is_sweep,
        )

    async def on_order_update(
        self,
        symbol: str,
        side: str,
        order_update: Mapping[str, Any] | str,
        *,
        filled: Optional[float] = None,
        average: Optional[float] = None,
        status: Optional[str] = None,
    ) -> None:
        """
        Apply a CCXT order update and maintain fill ratio, fees, and slippage.

        ``filled`` is interpreted as cumulative quantity for the exchange order ID.
        Updates for replaced historical order IDs are retained and incorporated into
        cumulative strategy exposure.
        """
        normalized_side = self._normalize_side(side)
        key = (symbol, normalized_side)
        state = self._get_order(key)
        if state is None:
            return

        if isinstance(order_update, Mapping):
            order_id = str(order_update.get("id") or order_update.get("orderId") or "")
            cumulative_filled = float(order_update.get("filled") or 0.0)
            average_price = order_update.get("average")
            order_status = order_update.get("status")
        else:
            order_id = str(order_update)
            cumulative_filled = float(filled or 0.0)
            average_price = average
            order_status = status

        if not order_id:
            return

        if average_price is not None:
            average_price = float(average_price)

        async with self._async_lock:
            with self._thread_lock:
                if order_id not in state.known_order_ids:
                    return

                previous_filled = state.order_fills.get(order_id, 0.0)
                cumulative_filled = max(previous_filled, cumulative_filled)
                delta = cumulative_filled - previous_filled

                state.order_fills[order_id] = cumulative_filled
                if average_price is not None and average_price > 0:
                    state.order_average_prices[order_id] = average_price

                if delta > self.config.position_epsilon:
                    fill_price = state.order_average_prices.get(
                        order_id,
                        state.working_price or 0.0,
                    )
                    previous_total = state.filled_amount
                    state.filled_amount += delta
                    state.exposure_amount += delta

                    if fill_price > 0:
                        if state.average_fill_price is None or previous_total <= 0:
                            state.average_fill_price = fill_price
                        else:
                            state.average_fill_price = (
                                (state.average_fill_price * previous_total)
                                + (fill_price * delta)
                            ) / state.filled_amount

                    if fill_price > 0:
                        state.estimated_maker_fee += delta * fill_price * state.maker_fee_rate
                        state.estimated_taker_exit_fee += (
                            delta * fill_price * state.taker_fee_rate
                        )

                normalized_status = str(order_status or "").lower()
                if order_id == state.order_id and normalized_status:
                    if normalized_status in {"open", "new", "partially_filled"}:
                        state.status = "open"
                    elif normalized_status in {"closed", "filled"}:
                        state.status = "filled"
                        state.order_id = None
                    elif normalized_status in {"canceled", "cancelled", "rejected", "expired"}:
                        if state.status != "replacing":
                            state.status = normalized_status
                        state.order_id = None

    async def process_order_update(
        self,
        symbol: str,
        side: str,
        order_update: Mapping[str, Any] | str,
        *,
        filled: Optional[float] = None,
        average: Optional[float] = None,
        status: Optional[str] = None,
    ) -> None:
        """Compatibility alias for :meth:`on_order_update`."""
        await self.on_order_update(
            symbol,
            side,
            order_update,
            filled=filled,
            average=average,
            status=status,
        )

    async def manage_once(self, *, refresh_orderbooks: bool = True) -> None:
        """
        Run one management cycle.

        The cycle optionally refreshes each active symbol's book, re-quotes when
        top-of-book movement exceeds the configured threshold, refreshes expired
        orders, and exits any filled exposure whose adverse drift exceeds limits.
        """
        self._ensure_open()

        with self._thread_lock:
            active_states = [state for state in self._orders.values() if state.is_active]
            active_symbols = sorted({state.symbol for state in active_states})

        if refresh_orderbooks:
            for symbol in active_symbols:
                try:
                    await self.refresh_orderbook(symbol)
                except Exception:
                    LOGGER.warning("Orderbook refresh failed for %s", symbol, exc_info=True)

        now = time.monotonic()
        actions: list[Tuple[str, Tuple[str, str]]] = []

        async with self._async_lock:
            with self._thread_lock:
                for key, state in self._orders.items():
                    if not state.is_active:
                        continue

                    book = self._books.get(state.symbol)
                    if book is None:
                        continue

                    adverse_drift = self._adverse_drift_bps(state, book)
                    if adverse_drift >= self.config.max_adverse_price_drift_bps:
                        actions.append(("exit", key))
                        continue

                    desired_price = self._peg_price(state.side, book, retreat=False)
                    price_delta = self._price_distance_bps(
                        state.working_price,
                        desired_price,
                    )
                    quote_age = now - state.last_quote_monotonic

                    if state.requote_count >= self.config.max_requotes:
                        actions.append(("cancel", key))
                    elif (
                        price_delta >= self.config.requote_threshold_bps
                        or quote_age >= self.config.order_timeout_seconds
                    ):
                        actions.append(("requote", key))

        for action, key in actions:
            state = self._get_order(key)
            if state is None:
                continue

            if action == "exit":
                await self._risk_exit(state, reason="adverse_price_drift")
            elif action == "cancel":
                await self.cancel_pegged_order(
                    state.symbol,
                    state.side,
                    reason="max_requotes_reached",
                )
            elif action == "requote":
                await self._replace_order(state, retreat=False, reason="requote")

    async def emergency_exit(
        self,
        symbol: str,
        side: str,
        amount: Optional[float] = None,
    ) -> bool:
        """
        Cancel a passive order and attempt a Binance-compatible reduce-only market exit.

        If ``amount`` is omitted, current measured filled exposure is used.
        """
        normalized_side = self._normalize_side(side)
        state = self._get_order((symbol, normalized_side))

        if state is None:
            if amount is None or amount <= self.config.position_epsilon:
                return False

            maker_fee, taker_fee = self.config.fee_rates(self.rolling_30d_volume)
            state = PeggedOrderState(
                symbol=symbol,
                side=normalized_side,
                amount=float(amount),
                client_order_id=self._new_client_order_id(symbol, normalized_side),
                created_monotonic=time.monotonic(),
                maker_fee_rate=maker_fee,
                taker_fee_rate=taker_fee,
                filled_amount=float(amount),
                exposure_amount=float(amount),
                status="open",
            )

        if amount is not None:
            state.exposure_amount = max(0.0, float(amount))

        return await self._risk_exit(state, reason="manual_emergency_exit")

    def snapshot(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Return a thread-safe serializable snapshot of engine metrics and state."""
        with self._thread_lock:
            orders: Dict[str, Any] = {}

            for (order_symbol, side), state in self._orders.items():
                if symbol is not None and order_symbol != symbol:
                    continue

                orders[f"{order_symbol}:{side}"] = {
                    "symbol": state.symbol,
                    "side": state.side,
                    "order_id": state.order_id,
                    "client_order_id": state.client_order_id,
                    "status": state.status,
                    "amount": state.amount,
                    "working_price": state.working_price,
                    "filled_amount": state.filled_amount,
                    "exposure_amount": state.exposure_amount,
                    "fill_ratio": state.fill_ratio,
                    "average_fill_price": state.average_fill_price,
                    "slippage_bps": state.slippage_bps,
                    "estimated_maker_fee": state.estimated_maker_fee,
                    "estimated_taker_exit_fee": state.estimated_taker_exit_fee,
                    "queue_initial_ahead": state.queue.initial_queue_ahead,
                    "queue_ahead": state.queue.queue_ahead,
                    "queue_printed_at_level": state.queue.printed_at_level,
                    "queue_depletion_ratio": state.queue.estimated_queue_depletion_ratio,
                    "toxic_events": state.toxic_events,
                    "adverse_events_in_window": len(state.adverse_events),
                    "requote_count": state.requote_count,
                    "exit_attempted": state.exit_attempted,
                    "exit_order_id": state.exit_order_id,
                    "last_error": state.last_error,
                }

            books = {
                book_symbol: {
                    "best_bid": book.best_bid,
                    "best_ask": book.best_ask,
                    "bid_size": book.bid_size,
                    "ask_size": book.ask_size,
                    "timestamp": book.timestamp,
                }
                for book_symbol, book in self._books.items()
                if symbol is None or book_symbol == symbol
            }

        maker_fee, taker_fee = self.config.fee_rates(self.rolling_30d_volume)
        return {
            "maker_fee_rate": maker_fee,
            "taker_fee_rate": taker_fee,
            "rolling_30d_volume": self.rolling_30d_volume,
            "books": books,
            "orders": orders,
        }

    def get_order_state(
        self,
        symbol: str,
        side: str,
    ) -> Optional[PeggedOrderState]:
        """Return the live state object for a symbol and side, if one exists."""
        return self._get_order((symbol, self._normalize_side(side)))

    async def _submit_state(
        self,
        state: PeggedOrderState,
        *,
        book: BookSnapshot,
        retreat: bool,
    ) -> None:
        retries = self.config.post_only_retries
        last_exception: Optional[BaseException] = None

        for attempt in range(retries + 1):
            current_book = book
            if attempt > 0:
                current_book = await self._get_or_refresh_book(state.symbol)

            raw_price = self._peg_price(state.side, current_book, retreat=retreat)
            price = await self._normalize_price(state.symbol, state.side, raw_price)
            queue_ahead = (
                current_book.bid_size if state.side == "buy" else current_book.ask_size
            )

            try:
                order = await self._create_post_only_order(
                    state.symbol,
                    state.side,
                    state.amount - state.filled_amount,
                    price,
                    state.client_order_id,
                )
            except Exception as exc:
                last_exception = exc
                if attempt < retries and self._is_post_only_rejection(exc):
                    continue
                raise ExchangeUnavailableError(
                    f"Unable to submit post-only order for {state.symbol}: {exc}"
                ) from exc

            order_id = str(order.get("id") or order.get("orderId") or "")
            if not order_id:
                raise PeggingEngineError(
                    f"Exchange returned no order ID for {state.symbol} {state.side}"
                )

            async with self._async_lock:
                with self._thread_lock:
                    state.order_id = order_id
                    state.known_order_ids.add(order_id)
                    state.order_fills.setdefault(order_id, 0.0)
                    state.working_price = price
                    state.last_quote_monotonic = time.monotonic()
                    state.status = "open"
                    state.last_error = None
                    state.queue = QueueEstimate(
                        initial_queue_ahead=queue_ahead,
                        queue_ahead=queue_ahead,
                    )
                    if state.requote_count > 0 or retreat:
                        state.requote_count += 1
            return

        raise ExchangeUnavailableError(
            f"Unable to submit post-only order for {state.symbol}: {last_exception}"
        )

    async def _replace_order(
        self,
        state: PeggedOrderState,
        *,
        retreat: bool,
        reason: str,
    ) -> None:
        if not state.is_active:
            return

        async with self._async_lock:
            with self._thread_lock:
                state.status = "replacing"

        cancelled = await self._cancel_current_order(state)
        if not cancelled:
            async with self._async_lock:
                with self._thread_lock:
                    state.status = "open"
                    state.last_error = f"{reason}: cancellation failed"
            return

        remaining = state.amount - state.filled_amount
        if remaining <= self.config.position_epsilon:
            async with self._async_lock:
                with self._thread_lock:
                    state.status = "filled"
            return

        try:
            book = await self._get_or_refresh_book(state.symbol)
            await self._submit_state(state, book=book, retreat=retreat)
        except Exception as exc:
            async with self._async_lock:
                with self._thread_lock:
                    state.status = "error"
                    state.last_error = f"{reason}: {exc}"
            LOGGER.exception(
                "Unable to replace pegged %s %s order after %s",
                state.symbol,
                state.side,
                reason,
            )

    async def _retreat_order(self, state: PeggedOrderState, *, reason: str) -> None:
        """Cancel and re-enter away from the touch after toxic adverse flow."""
        await self._replace_order(state, retreat=True, reason=reason)

    async def _risk_exit(self, state: PeggedOrderState, *, reason: str) -> bool:
        """
        Cancel passive liquidity and execute a reduce-only market exit if exposed.

        The market exit remains reduction-only so a stale or duplicate callback cannot
        unintentionally increase exchange-side exposure.
        """
        async with self._async_lock:
            with self._thread_lock:
                if state.status == "exited":
                    return True
                state.exit_attempted = True

        await self._cancel_current_order(state)

        exposure = max(0.0, state.exposure_amount)
        if exposure <= self.config.position_epsilon:
            async with self._async_lock:
                with self._thread_lock:
                    state.status = "risk_cancelled"
                    state.last_error = reason
            return True

        exit_side = "sell" if state.side == "buy" else "buy"

        try:
            order = await asyncio.wait_for(
                self._create_reduce_only_market_order(
                    state.symbol,
                    exit_side,
                    exposure,
                ),
                timeout=self.config.market_exit_timeout_seconds,
            )
            order_id = str(order.get("id") or order.get("orderId") or "") or None

            async with self._async_lock:
                with self._thread_lock:
                    state.exit_order_id = order_id
                    state.exposure_amount = 0.0
                    state.status = "exited"
                    state.last_error = reason
            return True
        except Exception as exc:
            async with self._async_lock:
                with self._thread_lock:
                    state.status = "exit_failed"
                    state.last_error = f"{reason}: reduce-only market exit failed: {exc}"

            LOGGER.critical(
                "Reduce-only market exit failed for %s %s exposure=%s: %s",
                state.symbol,
                state.side,
                exposure,
                exc,
            )
            return False

    async def _cancel_current_order(self, state: PeggedOrderState) -> bool:
        order_id = state.order_id
        if not order_id:
            return True

        if self.exchange is None:
            async with self._async_lock:
                with self._thread_lock:
                    if state.order_id == order_id:
                        state.order_id = None
            return True

        try:
            await asyncio.wait_for(
                self._exchange_call("cancel_order", order_id, state.symbol),
                timeout=self.config.cancellation_timeout_seconds,
            )
        except Exception as exc:
            if not self._is_already_closed_error(exc):
                LOGGER.warning(
                    "Cancellation failed for %s %s: %s",
                    state.symbol,
                    order_id,
                    exc,
                )
                return False

        async with self._async_lock:
            with self._thread_lock:
                if state.order_id == order_id:
                    state.order_id = None
        return True

    async def _create_post_only_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        client_order_id: str,
    ) -> Mapping[str, Any]:
        if amount <= self.config.position_epsilon:
            raise InvalidOrderStateError("Cannot create an order with zero remaining amount")

        if self.exchange is None:
            return {
                "id": f"local-{uuid.uuid4().hex}",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "price": price,
                "status": "open",
            }

        params = {
            "postOnly": True,
            "timeInForce": "GTX",
            "newClientOrderId": client_order_id,
        }
        result = await self._exchange_call(
            "create_order",
            symbol,
            "limit",
            side,
            amount,
            price,
            params,
        )

        if not isinstance(result, Mapping):
            raise PeggingEngineError("Exchange create_order response is not a mapping")

        return result

    async def _create_reduce_only_market_order(
        self,
        symbol: str,
        side: str,
        amount: float,
    ) -> Mapping[str, Any]:
        if self.exchange is None:
            return {
                "id": f"local-exit-{uuid.uuid4().hex}",
                "symbol": symbol,
                "side": side,
                "amount": amount,
                "status": "closed",
            }

        params = {"reduceOnly": True}
        result = await self._exchange_call(
            "create_order",
            symbol,
            "market",
            side,
            amount,
            None,
            params,
        )

        if not isinstance(result, Mapping):
            raise PeggingEngineError("Exchange market exit response is not a mapping")

        return result

    async def _get_or_refresh_book(self, symbol: str) -> BookSnapshot:
        with self._thread_lock:
            existing = self._books.get(symbol)

        if existing is not None:
            return existing

        return await self.refresh_orderbook(symbol)

    async def _normalize_price(self, symbol: str, side: str, price: float) -> float:
        if price <= 0:
            raise ValueError("price must be positive")

        normalized = float(price)

        if self.config.price_tick_size:
            tick = self.config.price_tick_size
            if side == "buy":
                normalized = math.floor(normalized / tick) * tick
            else:
                normalized = math.ceil(normalized / tick) * tick

        if self.exchange is not None and hasattr(self.exchange, "price_to_precision"):
            try:
                formatted = await self._exchange_call(
                    "price_to_precision",
                    symbol,
                    normalized,
                )
                normalized = float(formatted)
            except Exception:
                LOGGER.debug(
                    "Unable to apply exchange price precision for %s",
                    symbol,
                    exc_info=True,
                )

        if normalized <= 0:
            raise ValueError("normalized price must be positive")

        return normalized

    async def _exchange_call(self, method_name: str, *args: Any) -> Any:
        if self.exchange is None:
            raise ExchangeUnavailableError("No exchange instance has been configured")

        method = getattr(self.exchange, method_name, None)
        if method is None:
            raise ExchangeUnavailableError(
                f"Exchange does not support required method {method_name}"
            )

        if inspect.iscoroutinefunction(method):
            return await method(*args)

        result = await asyncio.to_thread(method, *args)
        if inspect.isawaitable(result):
            return await result
        return result

    def _parse_trade(
        self,
        *,
        trade: Mapping[str, Any] | float,
        amount: Optional[float],
        side: Optional[str],
        timestamp: Optional[float],
        is_sweep: Optional[bool],
    ) -> Optional[Tuple[float, float, Optional[str], float, Optional[bool]]]:
        if isinstance(trade, Mapping):
            price_raw = trade.get("price")
            quantity_raw = (
                trade.get("amount")
                or trade.get("quantity")
                or trade.get("qty")
                or trade.get("volume")
            )

            if price_raw is None or quantity_raw is None:
                return None

            price = float(price_raw)
            quantity = float(quantity_raw)

            trade_side = (
                trade.get("takerSide")
                or trade.get("aggressorSide")
                or trade.get("side")
            )
            aggressor_side = self._infer_aggressor_side(trade, trade_side)

            trade_timestamp = trade.get("timestamp") or trade.get("time") or timestamp
            explicit_sweep = (
                trade.get("isSweep")
                if "isSweep" in trade
                else trade.get("sweep", is_sweep)
            )
        else:
            if amount is None:
                raise ValueError("amount is required when trade is supplied as a price")
            price = float(trade)
            quantity = float(amount)
            aggressor_side = self._normalize_side(side) if side else None
            trade_timestamp = timestamp
            explicit_sweep = is_sweep

        parsed_timestamp = float(trade_timestamp or time.time())
        if parsed_timestamp > 1_000_000_000_000:
            parsed_timestamp /= 1000.0

        return (
            price,
            quantity,
            aggressor_side,
            parsed_timestamp,
            bool(explicit_sweep) if explicit_sweep is not None else None,
        )

    def _infer_aggressor_side(
        self,
        trade: Mapping[str, Any],
        side_value: Any,
    ) -> Optional[str]:
        if side_value is not None:
            try:
                return self._normalize_side(str(side_value))
            except ValueError:
                pass

        info = trade.get("info")
        source: Mapping[str, Any] = info if isinstance(info, Mapping) else trade

        buyer_is_maker = source.get("isBuyerMaker")
        if buyer_is_maker is None:
            buyer_is_maker = source.get("m")

        if buyer_is_maker is not None:
            return "sell" if self._as_bool(buyer_is_maker) else "buy"

        return None

    def _trade_hits_passive_level(
        self,
        state: PeggedOrderState,
        book: Optional[BookSnapshot],
        trade_price: float,
    ) -> bool:
        reference = state.working_price
        if reference is None:
            return False

        tolerance = reference * self.config.queue_price_tolerance_bps / 10_000.0
        if state.side == "buy":
            top_price = book.best_bid if book is not None else reference
            return trade_price <= max(reference, top_price) + tolerance

        top_price = book.best_ask if book is not None else reference
        return trade_price >= min(reference, top_price) - tolerance

    def _trade_sweeps_quote(
        self,
        state: PeggedOrderState,
        book: Optional[BookSnapshot],
        trade_price: float,
    ) -> bool:
        return self._trade_hits_passive_level(state, book, trade_price)

    def _is_adverse_aggressor(
        self,
        passive_side: str,
        aggressor_side: Optional[str],
    ) -> bool:
        if aggressor_side is None:
            return False
        return passive_side != aggressor_side

    def _adverse_drift_bps(
        self,
        state: PeggedOrderState,
        book: Optional[BookSnapshot],
    ) -> float:
        if book is None:
            return 0.0

        reference = state.average_fill_price or state.working_price
        if reference is None or reference <= 0:
            return 0.0

        mid = book.mid_price
        if state.side == "buy":
            return max(0.0, ((reference - mid) / reference) * 10_000.0)

        return max(0.0, ((mid - reference) / reference) * 10_000.0)

    def _peg_price(
        self,
        side: str,
        book: BookSnapshot,
        *,
        retreat: bool,
    ) -> float:
        if side == "buy":
            price = book.best_bid
            if retreat:
                price *= 1.0 - (self.config.retreat_bps / 10_000.0)
            return price

        price = book.best_ask
        if retreat:
            price *= 1.0 + (self.config.retreat_bps / 10_000.0)
        return price

    @staticmethod
    def _price_distance_bps(
        current_price: Optional[float],
        desired_price: float,
    ) -> float:
        if current_price is None or current_price <= 0:
            return float("inf")
        return abs(desired_price - current_price) / current_price * 10_000.0

    def _prune_adverse_events(
        self,
        state: PeggedOrderState,
        now_timestamp: float,
    ) -> None:
        cutoff = now_timestamp - self.config.adverse_flow_window_seconds
        while state.adverse_events and state.adverse_events[0] < cutoff:
            state.adverse_events.popleft()

    def _get_order(self, key: Tuple[str, str]) -> Optional[PeggedOrderState]:
        with self._thread_lock:
            return self._orders.get(key)

    def _ensure_open(self) -> None:
        if self._closed:
            raise PeggingEngineError("Engine has been closed")

    @staticmethod
    def _normalize_side(side: Optional[str]) -> str:
        normalized = str(side or "").strip().lower()
        if normalized not in {"buy", "sell"}:
            raise ValueError("side must be either 'buy' or 'sell'")
        return normalized

    @staticmethod
    def _new_client_order_id(symbol: str, side: str) -> str:
        compact_symbol = "".join(character for character in symbol if character.isalnum())
        return f"peg-{compact_symbol[:10]}-{side[0]}-{uuid.uuid4().hex[:18]}"

    @staticmethod
    def _is_post_only_rejection(exc: BaseException) -> bool:
        message = str(exc).lower()
        markers = (
            "post only",
            "post-only",
            "would immediately match",
            "immediately match",
            "gtx",
            "maker only",
        )
        return any(marker in message for marker in markers)

    @staticmethod
    def _is_already_closed_error(exc: BaseException) -> bool:
        message = str(exc).lower()
        markers = (
            "unknown order",
            "order does not exist",
            "already closed",
            "not found",
            "cancel rejected",
        )
        return any(marker in message for marker in markers)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    class _FakeBinanceUSDM:
        def __init__(self) -> None:
            self.orders: Dict[str, Dict[str, Any]] = {}
            self.created: list[Dict[str, Any]] = []

        async def create_order(
            self,
            symbol: str,
            order_type: str,
            side: str,
            amount: float,
            price: Optional[float],
            params: Optional[Mapping[str, Any]] = None,
        ) -> Mapping[str, Any]:
            order_id = f"fake-{len(self.created) + 1}"
            order = {
                "id": order_id,
                "symbol": symbol,
                "type": order_type,
                "side": side,
                "amount": amount,
                "price": price,
                "params": dict(params or {}),
                "status": "open" if order_type == "limit" else "closed",
            }
            self.orders[order_id] = order
            self.created.append(order)
            return order

        async def cancel_order(self, order_id: str, symbol: str) -> Mapping[str, Any]:
            order = self.orders[order_id]
            order["status"] = "canceled"
            return order

        async def fetch_order_book(self, symbol: str, limit: int = 5) -> Mapping[str, Any]:
            return {
                "bids": [[100.0, 20.0]],
                "asks": [[101.0, 25.0]],
                "timestamp": int(time.time() * 1000),
            }

    async def _self_test() -> None:
        exchange = _FakeBinanceUSDM()
        config = PeggingConfig(
            toxic_notional_threshold=100_000.0,
            max_adverse_events=1,
            max_adverse_price_drift_bps=100.0,
        )
        engine = OrderbookPeggingEngine(exchange, config)

        await engine.update_orderbook(
            "BTC/USDT:USDT",
            best_bid=100.0,
            best_ask=101.0,
            bid_size=20.0,
            ask_size=25.0,
        )
        order_id = await engine.place_pegged_order("BTC/USDT:USDT", "buy", 2.0)
        assert exchange.orders[order_id]["price"] == 100.0
        assert exchange.orders[order_id]["params"]["postOnly"] is True

        await engine.on_order_update(
            "BTC/USDT:USDT",
            "buy",
            {"id": order_id, "filled": 1.0, "average": 100.0, "status": "open"},
        )
        await engine.on_trade(
            "BTC/USDT:USDT",
            {"price": 100.0, "amount": 1_100.0, "side": "sell", "isSweep": True},
        )

        state = engine.get_order_state("BTC/USDT:USDT", "buy")
        assert state is not None
        assert state.status == "exited"
        assert state.exposure_amount == 0.0
        assert any(order["type"] == "market" for order in exchange.created)

        await engine.close()

    asyncio.run(_self_test())
