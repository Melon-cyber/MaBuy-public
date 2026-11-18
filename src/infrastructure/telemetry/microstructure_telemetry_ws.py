"""Async WebSocket telemetry server for market microstructure data."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import logging
import math
import signal
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import websockets
except ImportError:  # pragma: no cover - dependency availability is environment-specific.
    websockets = None  # type: ignore[assignment]


__all__ = ["MicrostructureTelemetryServer"]

_LOG = logging.getLogger(__name__)


@dataclass
class _PositionState:
    """Internal state for a symbol's active position."""

    quantity: float = 0.0
    average_price: float = 0.0
    realized_pnl: float = 0.0
    mark_price: Optional[float] = None

    @property
    def unrealized_pnl(self) -> float:
        """Return unrealized PnL using the latest mark price."""
        if self.quantity == 0.0 or self.mark_price is None:
            return 0.0
        return (self.mark_price - self.average_price) * self.quantity

    @property
    def total_pnl(self) -> float:
        """Return realized plus unrealized PnL."""
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class _ClientState:
    """Per-WebSocket client state managed exclusively by the server event loop."""

    client_id: str
    websocket: Any
    outbound: asyncio.Queue[dict[str, Any]]
    symbols: Optional[set[str]] = None
    event_types: Optional[set[str]] = None
    coalesced: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    inbound_timestamps: deque[float] = field(default_factory=deque)
    sender_task: Optional[asyncio.Task[None]] = None
    dropped_messages: int = 0
    last_sent_monotonic: float = 0.0


class MicrostructureTelemetryServer:
    """
    Async WebSocket broadcaster for L2, trades, executions, and microstructure metrics.

    The public ``publish_*`` methods are synchronous and thread-safe. They may be
    called from an asyncio task, a market-data callback thread, or a synchronous
    execution engine. Events are retained in an in-memory circular replay buffer
    and asynchronously dispatched to connected WebSocket clients.

    Wire protocol
    -------------
    Clients receive JSON objects with a ``type`` field:

    * ``l2``: top-ten depth, OFI, micro-price, quote spread, effective spread.
    * ``trade``: trade tick, maker classification, aggressor classification, CVD.
    * ``metrics``: rolling one-minute volume, buy/sell volume, and CVD.
    * ``execution``: fill, slippage, liquidity classification, active position PnL.
    * ``replay``: one chunk of replay-buffer events.
    * ``system``: welcome, subscription acknowledgement, errors, or pong replies.

    Optional inbound messages:

    .. code-block:: json

        {"action": "subscribe", "symbols": ["BTCUSDT"], "types": ["l2", "trade"]}
        {"action": "replay", "since": 1234}
        {"action": "ping"}

    ``symbols`` and ``types`` are optional. Omitting either means all values are
    accepted. Reconnects automatically receive replay chunks containing the latest
    retained events that match their current subscription.
    """

    _COALESCABLE_EVENT_TYPES = frozenset({"l2", "metrics"})
    _SYSTEM_EVENT_TYPES = frozenset({"system", "replay"})

    def __init__(
        self,
        *,
        ring_size: int = 1_000,
        client_queue_size: int = 512,
        dispatch_queue_size: int = 10_000,
        max_messages_per_second: float = 200.0,
        max_client_messages_per_second: int = 30,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
        replay_chunk_size: int = 100,
        max_message_size: int = 1_048_576,
    ) -> None:
        """
        Initialize the telemetry server.

        Args:
            ring_size: Number of most recent events retained for replay.
            client_queue_size: Maximum pending outbound messages per client.
            dispatch_queue_size: Maximum global pending broadcast events.
            max_messages_per_second: Per-client outbound message rate limit.
            max_client_messages_per_second: Inbound client command rate limit.
            ping_interval: WebSocket ping interval in seconds.
            ping_timeout: WebSocket pong timeout in seconds.
            replay_chunk_size: Number of retained events included in each replay frame.
            max_message_size: Maximum accepted inbound WebSocket frame size.

        Raises:
            ValueError: If configuration values are invalid.
        """
        if ring_size <= 0:
            raise ValueError("ring_size must be greater than zero")
        if client_queue_size <= 0:
            raise ValueError("client_queue_size must be greater than zero")
        if dispatch_queue_size <= 0:
            raise ValueError("dispatch_queue_size must be greater than zero")
        if max_messages_per_second <= 0:
            raise ValueError("max_messages_per_second must be greater than zero")
        if max_client_messages_per_second <= 0:
            raise ValueError("max_client_messages_per_second must be greater than zero")
        if ping_interval <= 0 or ping_timeout <= 0:
            raise ValueError("ping_interval and ping_timeout must be greater than zero")
        if replay_chunk_size <= 0:
            raise ValueError("replay_chunk_size must be greater than zero")
        if max_message_size <= 0:
            raise ValueError("max_message_size must be greater than zero")

        self._ring: deque[dict[str, Any]] = deque(maxlen=ring_size)
        self._state_lock = threading.RLock()
        self._lifecycle_lock = asyncio.Lock()

        self._client_queue_size = client_queue_size
        self._dispatch_queue_size = dispatch_queue_size
        self._max_messages_per_second = float(max_messages_per_second)
        self._max_client_messages_per_second = max_client_messages_per_second
        self._ping_interval = float(ping_interval)
        self._ping_timeout = float(ping_timeout)
        self._replay_chunk_size = replay_chunk_size
        self._max_message_size = max_message_size

        self._sequence = 0
        self._server: Any = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._dispatch_queue: Optional[asyncio.Queue[dict[str, Any]]] = None
        self._dispatcher_task: Optional[asyncio.Task[None]] = None
        self._dispatch_coalesced: dict[tuple[str, str], dict[str, Any]] = {}
        self._clients: dict[str, _ClientState] = {}
        self._stopping = False

        self._previous_books: dict[str, dict[str, float]] = {}
        self._current_books: dict[str, dict[str, float]] = {}
        self._last_effective_spread: dict[str, float] = {}
        self._last_trade_price: dict[str, float] = {}
        self._last_aggressor_side: dict[str, str] = {}

        self._rolling_trades: dict[str, deque[tuple[float, float, float]]] = defaultdict(deque)
        self._cumulative_volume_delta: dict[str, float] = defaultdict(float)
        self._positions: dict[str, _PositionState] = defaultdict(_PositionState)

        self._dropped_global_events = 0

    async def start(self, host: str = "0.0.0.0", port: int = 8765) -> list[str]:
        """
        Start the WebSocket server.

        Args:
            host: Interface on which to bind the WebSocket server.
            port: TCP port on which to bind the WebSocket server.

        Returns:
            Bound socket addresses, when available.

        Raises:
            RuntimeError: If the ``websockets`` package is not installed.
        """
        if websockets is None:
            raise RuntimeError(
                "The 'websockets' package is required to run "
                "MicrostructureTelemetryServer. Install it with: pip install websockets"
            )

        async with self._lifecycle_lock:
            if self._server is not None:
                return self.addresses

            loop = asyncio.get_running_loop()
            self._stopping = False
            self._loop = loop
            self._dispatch_queue = asyncio.Queue(maxsize=self._dispatch_queue_size)
            self._dispatch_coalesced.clear()
            self._dispatcher_task = asyncio.create_task(
                self._dispatch_loop(),
                name="microstructure-telemetry-dispatcher",
            )

            try:
                self._server = await websockets.serve(
                    self._handle_connection,
                    host,
                    port,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                    max_size=self._max_message_size,
                    compression="deflate",
                )
            except Exception:
                dispatcher = self._dispatcher_task
                self._dispatcher_task = None
                self._dispatch_queue = None
                self._loop = None
                if dispatcher is not None:
                    dispatcher.cancel()
                    await asyncio.gather(dispatcher, return_exceptions=True)
                raise

            _LOG.info(
                "Microstructure telemetry WebSocket server listening on %s",
                ", ".join(self.addresses) or f"{host}:{port}",
            )
            return self.addresses

    async def stop(self) -> None:
        """
        Gracefully stop accepting connections and close all active clients.

        The replay ring buffer and in-memory analytics state are intentionally
        retained, allowing the same server instance to be restarted if required.
        """
        async with self._lifecycle_lock:
            if self._server is None and self._dispatcher_task is None:
                return

            self._stopping = True
            server = self._server
            self._server = None

            if server is not None:
                server.close()

            clients = list(self._clients.values())
            close_tasks = [
                asyncio.create_task(
                    client.websocket.close(code=1001, reason="server shutting down")
                )
                for client in clients
            ]
            if close_tasks:
                await asyncio.gather(*close_tasks, return_exceptions=True)

            if server is not None:
                await server.wait_closed()

            dispatcher = self._dispatcher_task
            self._dispatcher_task = None
            if dispatcher is not None:
                dispatcher.cancel()
                await asyncio.gather(dispatcher, return_exceptions=True)

            self._clients.clear()
            self._dispatch_queue = None
            self._dispatch_coalesced.clear()
            self._loop = None
            _LOG.info("Microstructure telemetry WebSocket server stopped")

    async def serve_forever(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        """
        Start the server and run until this coroutine is cancelled.

        Args:
            host: Interface on which to bind.
            port: TCP port on which to bind.
        """
        await self.start(host, port)
        try:
            await asyncio.Future()
        finally:
            await self.stop()

    @property
    def addresses(self) -> list[str]:
        """Return currently bound server socket addresses."""
        server = self._server
        if server is None:
            return []

        addresses: list[str] = []
        for sock in getattr(server, "sockets", ()) or ():
            try:
                host, port = sock.getsockname()[:2]
                addresses.append(f"{host}:{port}")
            except (AttributeError, OSError, TypeError):
                continue
        return addresses

    @property
    def connected_clients(self) -> int:
        """Return the current number of active WebSocket clients."""
        return len(self._clients)

    @property
    def dropped_global_events(self) -> int:
        """Return the number of non-coalescable events dropped before dispatch."""
        with self._state_lock:
            return self._dropped_global_events

    def publish_l2_snapshot(
        self,
        symbol: str,
        bids: Sequence[Any],
        asks: Sequence[Any],
        *,
        timestamp: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Publish a top-ten L2 order-book snapshot.

        OFI is calculated from changes in the best bid and best ask using the
        standard best-quote order-flow imbalance formulation. Micro-price is
        quantity-weighted using the opposite best-quote size.

        Args:
            symbol: Market symbol, for example ``BTCUSDT``.
            bids: Bid levels as ``[price, quantity]`` pairs or mappings.
            asks: Ask levels as ``[price, quantity]`` pairs or mappings.
            timestamp: Optional exchange timestamp in seconds or milliseconds.

        Returns:
            The emitted L2 event.

        Raises:
            ValueError: If symbol or levels are invalid.
        """
        normalized_symbol = self._normalize_symbol(symbol)
        normalized_bids = self._normalize_levels(bids, reverse=True)
        normalized_asks = self._normalize_levels(asks, reverse=False)
        event_timestamp = self._timestamp_ms(timestamp)

        with self._state_lock:
            best_bid_price = normalized_bids[0]["price"] if normalized_bids else None
            best_bid_quantity = normalized_bids[0]["quantity"] if normalized_bids else None
            best_ask_price = normalized_asks[0]["price"] if normalized_asks else None
            best_ask_quantity = normalized_asks[0]["quantity"] if normalized_asks else None

            previous = self._previous_books.get(normalized_symbol)
            current_best: dict[str, float] = {}
            if best_bid_price is not None and best_bid_quantity is not None:
                current_best["bid_price"] = best_bid_price
                current_best["bid_quantity"] = best_bid_quantity
            if best_ask_price is not None and best_ask_quantity is not None:
                current_best["ask_price"] = best_ask_price
                current_best["ask_quantity"] = best_ask_quantity

            ofi = self._calculate_ofi(previous, current_best)
            microprice = self._calculate_microprice(
                best_bid_price,
                best_bid_quantity,
                best_ask_price,
                best_ask_quantity,
            )
            midprice = self._calculate_midprice(best_bid_price, best_ask_price)
            quote_spread = (
                best_ask_price - best_bid_price
                if best_bid_price is not None and best_ask_price is not None
                else None
            )

            if current_best:
                self._previous_books[normalized_symbol] = dict(current_best)
                self._current_books[normalized_symbol] = dict(current_best)

            mark_price = microprice if microprice is not None else midprice
            if mark_price is not None:
                position = self._positions[normalized_symbol]
                position.mark_price = mark_price

            event = self._record_event_locked(
                {
                    "type": "l2",
                    "symbol": normalized_symbol,
                    "timestamp": event_timestamp,
                    "bids": normalized_bids,
                    "asks": normalized_asks,
                    "best_bid": best_bid_price,
                    "best_ask": best_ask_price,
                    "midprice": midprice,
                    "microprice": microprice,
                    "ofi": ofi,
                    "quote_spread": quote_spread,
                    "effective_spread": self._last_effective_spread.get(normalized_symbol),
                }
            )

        self._schedule_dispatch(event)
        return copy.deepcopy(event)

    def publish_trade(
        self,
        symbol: str,
        price: float,
        quantity: float,
        *,
        is_buyer_maker: Optional[bool] = None,
        maker_side: Optional[str] = None,
        aggressor_side: Optional[str] = None,
        trade_id: Optional[str | int] = None,
        timestamp: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Publish a market trade tick and update rolling volume/CVD statistics.

        Classification precedence is ``is_buyer_maker``, then ``maker_side``, then
        ``aggressor_side``. When no explicit classification is supplied, the server
        infers aggressor direction from the current best quotes or a tick-rule
        fallback. If no direction can be inferred, the trade remains unclassified
        and contributes zero to CVD.

        Args:
            symbol: Market symbol.
            price: Executed trade price.
            quantity: Executed base quantity.
            is_buyer_maker: Exchange-style buyer-maker flag.
            maker_side: ``buyer`` or ``seller``.
            aggressor_side: ``buy`` or ``sell``.
            trade_id: Optional exchange trade identifier.
            timestamp: Optional exchange timestamp in seconds or milliseconds.

        Returns:
            The emitted trade event.

        Raises:
            ValueError: If required numeric inputs are invalid.
        """
        normalized_symbol = self._normalize_symbol(symbol)
        normalized_price = self._positive_number(price, "price")
        normalized_quantity = self._positive_number(quantity, "quantity")
        event_timestamp = self._timestamp_ms(timestamp)

        with self._state_lock:
            maker, aggressor, buyer_maker = self._classify_trade_locked(
                normalized_symbol,
                normalized_price,
                is_buyer_maker=is_buyer_maker,
                maker_side=maker_side,
                aggressor_side=aggressor_side,
            )

            best = self._current_books.get(normalized_symbol, {})
            midprice = self._calculate_midprice(
                best.get("bid_price"),
                best.get("ask_price"),
            )
            effective_spread = (
                2.0 * abs(normalized_price - midprice)
                if midprice is not None
                else None
            )
            effective_spread_bps = (
                (effective_spread / midprice) * 10_000.0
                if effective_spread is not None and midprice > 0.0
                else None
            )
            if effective_spread is not None:
                self._last_effective_spread[normalized_symbol] = effective_spread

            signed_quantity = (
                normalized_quantity
                if aggressor == "buy"
                else -normalized_quantity
                if aggressor == "sell"
                else 0.0
            )
            now_monotonic = time.monotonic()
            self._rolling_trades[normalized_symbol].append(
                (now_monotonic, normalized_quantity, signed_quantity)
            )
            self._cumulative_volume_delta[normalized_symbol] += signed_quantity
            metrics = self._metrics_for_symbol_locked(normalized_symbol, now_monotonic)

            self._last_trade_price[normalized_symbol] = normalized_price
            if aggressor is not None:
                self._last_aggressor_side[normalized_symbol] = aggressor

            position = self._positions[normalized_symbol]
            position.mark_price = midprice if midprice is not None else normalized_price

            event = self._record_event_locked(
                {
                    "type": "trade",
                    "symbol": normalized_symbol,
                    "timestamp": event_timestamp,
                    "trade_id": trade_id,
                    "price": normalized_price,
                    "quantity": normalized_quantity,
                    "notional": normalized_price * normalized_quantity,
                    "maker_side": maker,
                    "aggressor_side": aggressor,
                    "is_buyer_maker": buyer_maker,
                    "midprice": midprice,
                    "effective_spread": effective_spread,
                    "effective_spread_bps": effective_spread_bps,
                    "metrics": metrics,
                }
            )
            metrics_event = self._record_event_locked(
                {
                    "type": "metrics",
                    "symbol": normalized_symbol,
                    "timestamp": event_timestamp,
                    **metrics,
                }
            )

        self._schedule_dispatch(event)
        self._schedule_dispatch(metrics_event)
        return copy.deepcopy(event)

    def publish_execution(
        self,
        symbol: str,
        fill_price: float,
        quantity: float,
        side: str,
        *,
        reference_price: Optional[float] = None,
        is_maker: bool = False,
        execution_id: Optional[str | int] = None,
        order_id: Optional[str | int] = None,
        timestamp: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Publish execution telemetry and update in-memory position PnL.

        Slippage is signed so positive values represent adverse execution:
        ``(fill-reference)/reference`` for buys and
        ``(reference-fill)/reference`` for sells.

        Args:
            symbol: Market symbol.
            fill_price: Executed fill price.
            quantity: Filled base quantity.
            side: ``buy`` or ``sell``.
            reference_price: Arrival, decision, or benchmark price. If omitted,
                the latest micro-price/midprice is used when available.
            is_maker: Whether the fill added liquidity.
            execution_id: Optional execution identifier.
            order_id: Optional parent order identifier.
            timestamp: Optional exchange timestamp in seconds or milliseconds.

        Returns:
            The emitted execution event.

        Raises:
            ValueError: If values are invalid.
        """
        normalized_symbol = self._normalize_symbol(symbol)
        normalized_fill_price = self._positive_number(fill_price, "fill_price")
        normalized_quantity = self._positive_number(quantity, "quantity")
        normalized_side = self._normalize_side(side, field_name="side")
        event_timestamp = self._timestamp_ms(timestamp)

        if reference_price is not None:
            normalized_reference_price = self._positive_number(
                reference_price,
                "reference_price",
            )
        else:
            normalized_reference_price = None

        with self._state_lock:
            if normalized_reference_price is None:
                best = self._current_books.get(normalized_symbol, {})
                normalized_reference_price = self._calculate_microprice(
                    best.get("bid_price"),
                    best.get("bid_quantity"),
                    best.get("ask_price"),
                    best.get("ask_quantity"),
                )
                if normalized_reference_price is None:
                    normalized_reference_price = self._calculate_midprice(
                        best.get("bid_price"),
                        best.get("ask_price"),
                    )

            if normalized_reference_price is None:
                slippage_bps = None
            elif normalized_side == "buy":
                slippage_bps = (
                    (normalized_fill_price - normalized_reference_price)
                    / normalized_reference_price
                    * 10_000.0
                )
            else:
                slippage_bps = (
                    (normalized_reference_price - normalized_fill_price)
                    / normalized_reference_price
                    * 10_000.0
                )

            position = self._positions[normalized_symbol]
            signed_quantity = (
                normalized_quantity if normalized_side == "buy" else -normalized_quantity
            )
            self._apply_fill_to_position_locked(
                position,
                signed_quantity,
                normalized_fill_price,
            )

            best = self._current_books.get(normalized_symbol, {})
            mark_price = self._calculate_microprice(
                best.get("bid_price"),
                best.get("bid_quantity"),
                best.get("ask_price"),
                best.get("ask_quantity"),
            )
            if mark_price is None:
                mark_price = self._calculate_midprice(
                    best.get("bid_price"),
                    best.get("ask_price"),
                )
            position.mark_price = (
                mark_price
                if mark_price is not None
                else self._last_trade_price.get(normalized_symbol, normalized_fill_price)
            )

            event = self._record_event_locked(
                {
                    "type": "execution",
                    "symbol": normalized_symbol,
                    "timestamp": event_timestamp,
                    "execution_id": execution_id,
                    "order_id": order_id,
                    "side": normalized_side,
                    "fill_price": normalized_fill_price,
                    "quantity": normalized_quantity,
                    "notional": normalized_fill_price * normalized_quantity,
                    "reference_price": normalized_reference_price,
                    "slippage_bps": slippage_bps,
                    "is_maker": bool(is_maker),
                    "is_taker": not bool(is_maker),
                    "liquidity": "maker" if is_maker else "taker",
                    "position_quantity": position.quantity,
                    "average_entry_price": position.average_price
                    if position.quantity != 0.0
                    else None,
                    "mark_price": position.mark_price,
                    "realized_pnl": position.realized_pnl,
                    "active_position_pnl": position.unrealized_pnl,
                    "total_pnl": position.total_pnl,
                }
            )

        self._schedule_dispatch(event)
        return copy.deepcopy(event)

    def get_recent_events(
        self,
        *,
        limit: Optional[int] = None,
        since_sequence: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """
        Return a thread-safe copy of retained replay-buffer events.

        Args:
            limit: Maximum events returned, newest events retained when limited.
            since_sequence: Return events with a greater sequence number.

        Returns:
            A list of event dictionaries in chronological sequence order.
        """
        if limit is not None and limit <= 0:
            return []

        with self._state_lock:
            events = list(self._ring)
            if since_sequence is not None:
                events = [
                    event
                    for event in events
                    if int(event.get("sequence", 0)) > int(since_sequence)
                ]
            if limit is not None:
                events = events[-limit:]
            return copy.deepcopy(events)

    def get_metrics(self, symbol: Optional[str] = None) -> dict[str, Any]:
        """
        Return current rolling one-minute volume and CVD metrics.

        Args:
            symbol: Optional symbol. When omitted, metrics for all seen symbols
                are returned keyed by symbol.

        Returns:
            Metrics for one symbol or a symbol-to-metrics mapping.
        """
        with self._state_lock:
            now_monotonic = time.monotonic()
            if symbol is not None:
                normalized_symbol = self._normalize_symbol(symbol)
                return copy.deepcopy(
                    self._metrics_for_symbol_locked(normalized_symbol, now_monotonic)
                )

            symbols = set(self._rolling_trades) | set(self._cumulative_volume_delta)
            return {
                market: copy.deepcopy(self._metrics_for_symbol_locked(market, now_monotonic))
                for market in sorted(symbols)
            }

    async def _handle_connection(self, websocket: Any, path: Optional[str] = None) -> None:
        """Register a client, handle subscription commands, and clean up on disconnect."""
        del path

        if self._stopping:
            await websocket.close(code=1012, reason="server restarting")
            return

        client_id = uuid.uuid4().hex
        client = _ClientState(
            client_id=client_id,
            websocket=websocket,
            outbound=asyncio.Queue(maxsize=self._client_queue_size),
        )
        self._clients[client_id] = client
        client.sender_task = asyncio.create_task(
            self._client_sender_loop(client),
            name=f"microstructure-telemetry-sender-{client_id}",
        )

        self._enqueue_client(
            client,
            {
                "type": "system",
                "event": "welcome",
                "client_id": client_id,
                "timestamp": self._timestamp_ms(None),
                "replay_capacity": self._ring.maxlen,
            },
        )
        self._queue_replay(client)

        remote = getattr(websocket, "remote_address", None)
        _LOG.debug("Telemetry client connected: %s (%s)", client_id, remote)

        try:
            async for raw_message in websocket:
                if not self._allow_inbound_message(client):
                    await websocket.close(code=1008, reason="inbound rate limit exceeded")
                    break
                await self._handle_client_message(client, raw_message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _LOG.debug("Telemetry client %s disconnected with error: %s", client_id, exc)
        finally:
            self._clients.pop(client_id, None)
            sender = client.sender_task
            if sender is not None:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            _LOG.debug("Telemetry client disconnected: %s", client_id)

    async def _handle_client_message(self, client: _ClientState, raw_message: Any) -> None:
        """Process a single JSON command from a client."""
        if isinstance(raw_message, bytes):
            try:
                raw_message = raw_message.decode("utf-8")
            except UnicodeDecodeError:
                self._enqueue_client(
                    client,
                    self._system_error("binary messages must contain UTF-8 JSON"),
                )
                return

        try:
            payload = json.loads(raw_message)
        except (TypeError, json.JSONDecodeError):
            self._enqueue_client(client, self._system_error("invalid JSON message"))
            return

        if not isinstance(payload, Mapping):
            self._enqueue_client(client, self._system_error("message must be a JSON object"))
            return

        action = str(payload.get("action", "")).strip().lower()
        if action == "ping":
            self._enqueue_client(
                client,
                {
                    "type": "system",
                    "event": "pong",
                    "timestamp": self._timestamp_ms(None),
                },
            )
            return

        if action == "subscribe":
            try:
                symbols = self._parse_subscription_values(payload.get("symbols"))
                event_types = self._parse_subscription_values(payload.get("types"))
            except ValueError as exc:
                self._enqueue_client(client, self._system_error(str(exc)))
                return

            client.symbols = symbols
            client.event_types = event_types
            self._enqueue_client(
                client,
                {
                    "type": "system",
                    "event": "subscribed",
                    "symbols": sorted(symbols) if symbols is not None else None,
                    "types": sorted(event_types) if event_types is not None else None,
                    "timestamp": self._timestamp_ms(None),
                },
            )
            return

        if action == "replay":
            since = payload.get("since")
            if since is not None:
                try:
                    since = int(since)
                except (TypeError, ValueError):
                    self._enqueue_client(
                        client,
                        self._system_error("'since' must be an integer sequence number"),
                    )
                    return
            self._queue_replay(client, since_sequence=since)
            return

        self._enqueue_client(
            client,
            self._system_error(
                "unsupported action; expected one of: subscribe, replay, ping"
            ),
        )

    async def _client_sender_loop(self, client: _ClientState) -> None:
        """Serialize, rate-limit, and send queued outbound client messages."""
        min_interval = 1.0 / self._max_messages_per_second

        try:
            while True:
                message = await client.outbound.get()

                elapsed = time.monotonic() - client.last_sent_monotonic
                if client.last_sent_monotonic > 0.0 and elapsed < min_interval:
                    await asyncio.sleep(min_interval - elapsed)

                encoded = json.dumps(
                    message,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                await client.websocket.send(encoded)
                client.last_sent_monotonic = time.monotonic()
                self._flush_client_coalesced(client)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _LOG.debug("Sender failed for telemetry client %s: %s", client.client_id, exc)

    async def _dispatch_loop(self) -> None:
        """Dispatch recorded events to matching clients without blocking publishers."""
        assert self._dispatch_queue is not None
        queue = self._dispatch_queue

        try:
            while True:
                event = await queue.get()
                self._broadcast_to_clients(event)
                self._flush_dispatch_coalesced()
        except asyncio.CancelledError:
            raise

    def _broadcast_to_clients(self, event: dict[str, Any]) -> None:
        """Queue an event for all clients whose subscriptions match it."""
        for client in tuple(self._clients.values()):
            if self._event_matches_client(event, client):
                self._enqueue_client(client, event)

    def _enqueue_client(self, client: _ClientState, event: dict[str, Any]) -> None:
        """Queue an event for one client, coalescing depth and metric overload."""
        if client.websocket.closed if hasattr(client.websocket, "closed") else False:
            return

        try:
            client.outbound.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass

        event_type = str(event.get("type", ""))
        symbol = str(event.get("symbol", ""))
        if event_type in self._COALESCABLE_EVENT_TYPES:
            key = (event_type, symbol)
            if len(client.coalesced) < self._client_queue_size or key in client.coalesced:
                client.coalesced[key] = event
                return

        client.dropped_messages += 1

    def _flush_client_coalesced(self, client: _ClientState) -> None:
        """Move latest coalesced client updates into available queue capacity."""
        while client.coalesced and not client.outbound.full():
            _, event = client.coalesced.popitem()
            try:
                client.outbound.put_nowait(event)
            except asyncio.QueueFull:
                break

    def _schedule_dispatch(self, event: dict[str, Any]) -> None:
        """Schedule an event into the asyncio dispatcher from any thread."""
        with self._state_lock:
            loop = self._loop

        if loop is None or loop.is_closed() or self._stopping:
            return

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        try:
            if running_loop is loop:
                self._enqueue_dispatch(event)
            else:
                loop.call_soon_threadsafe(self._enqueue_dispatch, event)
        except RuntimeError:
            # The loop may have been closed between the checks above.
            return

    def _enqueue_dispatch(self, event: dict[str, Any]) -> None:
        """Insert an event into the global dispatcher queue on the server loop."""
        queue = self._dispatch_queue
        if queue is None or self._stopping:
            return

        try:
            queue.put_nowait(event)
            return
        except asyncio.QueueFull:
            pass

        event_type = str(event.get("type", ""))
        symbol = str(event.get("symbol", ""))
        if event_type in self._COALESCABLE_EVENT_TYPES:
            key = (event_type, symbol)
            if len(self._dispatch_coalesced) < self._dispatch_queue_size or key in self._dispatch_coalesced:
                self._dispatch_coalesced[key] = event
                return

        with self._state_lock:
            self._dropped_global_events += 1

    def _flush_dispatch_coalesced(self) -> None:
        """Move coalesced global events into newly available dispatch capacity."""
        queue = self._dispatch_queue
        if queue is None:
            return

        while self._dispatch_coalesced and not queue.full():
            _, event = self._dispatch_coalesced.popitem()
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                break

    def _queue_replay(
        self,
        client: _ClientState,
        *,
        since_sequence: Optional[int] = None,
    ) -> None:
        """Queue replay-buffer events in bounded JSON-friendly chunks."""
        with self._state_lock:
            events = [
                copy.deepcopy(event)
                for event in self._ring
                if (
                    since_sequence is None
                    or int(event.get("sequence", 0)) > since_sequence
                )
            ]

        events = [
            event for event in events if self._event_matches_client(event, client)
        ]

        if not events:
            self._enqueue_client(
                client,
                {
                    "type": "replay",
                    "timestamp": self._timestamp_ms(None),
                    "events": [],
                    "complete": True,
                },
            )
            return

        for start in range(0, len(events), self._replay_chunk_size):
            chunk = events[start : start + self._replay_chunk_size]
            self._enqueue_client(
                client,
                {
                    "type": "replay",
                    "timestamp": self._timestamp_ms(None),
                    "events": chunk,
                    "complete": start + self._replay_chunk_size >= len(events),
                },
            )

    def _event_matches_client(
        self,
        event: Mapping[str, Any],
        client: _ClientState,
    ) -> bool:
        """Return whether an event passes a client's symbol and type filters."""
        event_type = str(event.get("type", ""))
        if event_type in self._SYSTEM_EVENT_TYPES:
            return True

        if client.event_types is not None and event_type not in client.event_types:
            return False

        symbol = event.get("symbol")
        if client.symbols is not None and symbol is not None:
            return str(symbol) in client.symbols

        return True

    def _allow_inbound_message(self, client: _ClientState) -> bool:
        """Apply a rolling one-second inbound command rate limit."""
        now = time.monotonic()
        timestamps = client.inbound_timestamps
        while timestamps and now - timestamps[0] >= 1.0:
            timestamps.popleft()
        if len(timestamps) >= self._max_client_messages_per_second:
            return False
        timestamps.append(now)
        return True

    def _record_event_locked(self, event: dict[str, Any]) -> dict[str, Any]:
        """Assign an event sequence and append an isolated copy to the replay ring."""
        self._sequence += 1
        event["sequence"] = self._sequence
        self._ring.append(copy.deepcopy(event))
        return event

    def _metrics_for_symbol_locked(
        self,
        symbol: str,
        now_monotonic: float,
    ) -> dict[str, float]:
        """Prune a symbol's rolling window and calculate its metrics."""
        trades = self._rolling_trades[symbol]
        cutoff = now_monotonic - 60.0
        while trades and trades[0][0] < cutoff:
            trades.popleft()

        buy_volume = 0.0
        sell_volume = 0.0
        for _, quantity, signed_quantity in trades:
            if signed_quantity > 0.0:
                buy_volume += quantity
            elif signed_quantity < 0.0:
                sell_volume += quantity

        return {
            "rolling_1m_volume": buy_volume + sell_volume,
            "rolling_1m_buy_volume": buy_volume,
            "rolling_1m_sell_volume": sell_volume,
            "rolling_1m_volume_delta": buy_volume - sell_volume,
            "cvd": self._cumulative_volume_delta[symbol],
        }

    def _classify_trade_locked(
        self,
        symbol: str,
        price: float,
        *,
        is_buyer_maker: Optional[bool],
        maker_side: Optional[str],
        aggressor_side: Optional[str],
    ) -> tuple[Optional[str], Optional[str], Optional[bool]]:
        """Resolve buyer/seller maker and aggressive-side trade classification."""
        if is_buyer_maker is not None:
            buyer_maker = bool(is_buyer_maker)
            return (
                "buyer" if buyer_maker else "seller",
                "sell" if buyer_maker else "buy",
                buyer_maker,
            )

        if maker_side is not None:
            maker = self._normalize_maker_side(maker_side)
            return (
                maker,
                "sell" if maker == "buyer" else "buy",
                maker == "buyer",
            )

        if aggressor_side is not None:
            aggressor = self._normalize_side(aggressor_side, field_name="aggressor_side")
            maker = "seller" if aggressor == "buy" else "buyer"
            return maker, aggressor, maker == "buyer"

        book = self._current_books.get(symbol, {})
        bid = book.get("bid_price")
        ask = book.get("ask_price")
        inferred_aggressor: Optional[str] = None
        if ask is not None and price >= ask:
            inferred_aggressor = "buy"
        elif bid is not None and price <= bid:
            inferred_aggressor = "sell"
        else:
            previous_price = self._last_trade_price.get(symbol)
            if previous_price is not None:
                if price > previous_price:
                    inferred_aggressor = "buy"
                elif price < previous_price:
                    inferred_aggressor = "sell"
                else:
                    inferred_aggressor = self._last_aggressor_side.get(symbol)

        if inferred_aggressor is None:
            return None, None, None

        maker = "seller" if inferred_aggressor == "buy" else "buyer"
        return maker, inferred_aggressor, maker == "buyer"

    @staticmethod
    def _apply_fill_to_position_locked(
        position: _PositionState,
        signed_quantity: float,
        fill_price: float,
    ) -> None:
        """Update average entry and realized PnL for a signed position change."""
        current_quantity = position.quantity

        if current_quantity == 0.0 or current_quantity * signed_quantity > 0.0:
            combined_quantity = current_quantity + signed_quantity
            position.average_price = (
                (
                    abs(current_quantity) * position.average_price
                    + abs(signed_quantity) * fill_price
                )
                / abs(combined_quantity)
            )
            position.quantity = combined_quantity
            return

        closed_quantity = min(abs(current_quantity), abs(signed_quantity))
        position.realized_pnl += (
            closed_quantity
            * (fill_price - position.average_price)
            * (1.0 if current_quantity > 0.0 else -1.0)
        )

        new_quantity = current_quantity + signed_quantity
        if new_quantity == 0.0:
            position.quantity = 0.0
            position.average_price = 0.0
        elif current_quantity * new_quantity < 0.0:
            position.quantity = new_quantity
            position.average_price = fill_price
        else:
            position.quantity = new_quantity

    @staticmethod
    def _calculate_ofi(
        previous: Optional[Mapping[str, float]],
        current: Mapping[str, float],
    ) -> float:
        """Calculate best-quote Order Flow Imbalance for consecutive book states."""
        if not previous:
            return 0.0

        required = (
            "bid_price",
            "bid_quantity",
            "ask_price",
            "ask_quantity",
        )
        if any(key not in previous or key not in current for key in required):
            return 0.0

        previous_bid_price = previous["bid_price"]
        previous_bid_quantity = previous["bid_quantity"]
        previous_ask_price = previous["ask_price"]
        previous_ask_quantity = previous["ask_quantity"]
        bid_price = current["bid_price"]
        bid_quantity = current["bid_quantity"]
        ask_price = current["ask_price"]
        ask_quantity = current["ask_quantity"]

        if bid_price > previous_bid_price:
            bid_component = bid_quantity
        elif bid_price == previous_bid_price:
            bid_component = bid_quantity - previous_bid_quantity
        else:
            bid_component = -previous_bid_quantity

        if ask_price < previous_ask_price:
            ask_component = -ask_quantity
        elif ask_price == previous_ask_price:
            ask_component = previous_ask_quantity - ask_quantity
        else:
            ask_component = previous_ask_quantity

        return bid_component + ask_component

    @staticmethod
    def _calculate_microprice(
        bid_price: Optional[float],
        bid_quantity: Optional[float],
        ask_price: Optional[float],
        ask_quantity: Optional[float],
    ) -> Optional[float]:
        """Calculate quantity-weighted micro-price from top-of-book quotes."""
        if (
            bid_price is None
            or bid_quantity is None
            or ask_price is None
            or ask_quantity is None
        ):
            return None

        denominator = bid_quantity + ask_quantity
        if denominator <= 0.0:
            return None
        return (ask_price * bid_quantity + bid_price * ask_quantity) / denominator

    @staticmethod
    def _calculate_midprice(
        bid_price: Optional[float],
        ask_price: Optional[float],
    ) -> Optional[float]:
        """Calculate the top-of-book midpoint."""
        if bid_price is None or ask_price is None:
            return None
        return (bid_price + ask_price) / 2.0

    @staticmethod
    def _normalize_levels(levels: Sequence[Any], *, reverse: bool) -> list[dict[str, float]]:
        """Normalize, filter, sort, and truncate depth levels to ten entries."""
        if isinstance(levels, (str, bytes)) or not isinstance(levels, Sequence):
            raise ValueError("depth levels must be a sequence")

        normalized: list[dict[str, float]] = []
        for level in levels:
            if isinstance(level, Mapping):
                raw_price = level.get("price", level.get("p"))
                raw_quantity = level.get(
                    "quantity",
                    level.get("qty", level.get("size", level.get("q"))),
                )
            elif (
                isinstance(level, Sequence)
                and not isinstance(level, (str, bytes))
                and len(level) >= 2
            ):
                raw_price = level[0]
                raw_quantity = level[1]
            else:
                raise ValueError("each depth level must be a mapping or [price, quantity]")

            price = MicrostructureTelemetryServer._positive_number(raw_price, "level price")
            quantity = MicrostructureTelemetryServer._nonnegative_number(
                raw_quantity,
                "level quantity",
            )
            if quantity > 0.0:
                normalized.append({"price": price, "quantity": quantity})

        normalized.sort(key=lambda level: level["price"], reverse=reverse)
        return normalized[:10]

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Validate and normalize a market symbol."""
        if not isinstance(symbol, str):
            raise ValueError("symbol must be a string")
        normalized = symbol.strip().upper()
        if not normalized:
            raise ValueError("symbol must not be empty")
        if len(normalized) > 128:
            raise ValueError("symbol is too long")
        return normalized

    @staticmethod
    def _normalize_side(side: str, *, field_name: str) -> str:
        """Normalize a buy/sell side value."""
        if not isinstance(side, str):
            raise ValueError(f"{field_name} must be 'buy' or 'sell'")
        normalized = side.strip().lower()
        aliases = {
            "buy": "buy",
            "buyer": "buy",
            "b": "buy",
            "sell": "sell",
            "seller": "sell",
            "s": "sell",
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError(f"{field_name} must be 'buy' or 'sell'") from exc

    @staticmethod
    def _normalize_maker_side(side: str) -> str:
        """Normalize a buyer/seller maker-side value."""
        if not isinstance(side, str):
            raise ValueError("maker_side must be 'buyer' or 'seller'")
        normalized = side.strip().lower()
        aliases = {
            "buyer": "buyer",
            "buy": "buyer",
            "b": "buyer",
            "seller": "seller",
            "sell": "seller",
            "s": "seller",
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            raise ValueError("maker_side must be 'buyer' or 'seller'") from exc

    @staticmethod
    def _positive_number(value: Any, field_name: str) -> float:
        """Validate a finite strictly positive numeric value."""
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be numeric") from exc
        if not math.isfinite(normalized) or normalized <= 0.0:
            raise ValueError(f"{field_name} must be finite and greater than zero")
        return normalized

    @staticmethod
    def _nonnegative_number(value: Any, field_name: str) -> float:
        """Validate a finite non-negative numeric value."""
        try:
            normalized = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be numeric") from exc
        if not math.isfinite(normalized) or normalized < 0.0:
            raise ValueError(f"{field_name} must be finite and non-negative")
        return normalized

    @staticmethod
    def _timestamp_ms(timestamp: Optional[float]) -> int:
        """Convert an optional seconds/milliseconds timestamp to Unix milliseconds."""
        if timestamp is None:
            return int(time.time() * 1_000)

        try:
            normalized = float(timestamp)
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp must be numeric") from exc

        if not math.isfinite(normalized) or normalized < 0.0:
            raise ValueError("timestamp must be a finite non-negative value")

        if normalized < 100_000_000_000:
            normalized *= 1_000.0
        return int(normalized)

    @staticmethod
    def _parse_subscription_values(value: Any) -> Optional[set[str]]:
        """Parse optional subscription arrays with bounded cardinality."""
        if value is None:
            return None
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError("subscription values must be an array of strings")
        if len(value) > 500:
            raise ValueError("subscription arrays may contain at most 500 entries")

        parsed: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("subscription values must contain only strings")
            normalized = item.strip()
            if not normalized:
                raise ValueError("subscription values must not contain empty strings")
            if len(normalized) > 128:
                raise ValueError("subscription value is too long")
            parsed.add(normalized.upper() if normalized != normalized.lower() else normalized)
        return parsed

    @staticmethod
    def _system_error(message: str) -> dict[str, Any]:
        """Create a consistent protocol error response."""
        return {
            "type": "system",
            "event": "error",
            "message": message,
            "timestamp": int(time.time() * 1_000),
        }


async def _run_from_cli(host: str, port: int, log_level: str) -> None:
    """Run the standalone telemetry WebSocket server until a termination signal."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    server = MicrostructureTelemetryServer()
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def request_shutdown() -> None:
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_shutdown)
        except (NotImplementedError, RuntimeError):
            # Windows Proactor loops do not support add_signal_handler.
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(request_shutdown))

    await server.start(host, port)
    try:
        await shutdown_event.wait()
    finally:
        await server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the microstructure telemetry WebSocket server."
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind TCP port")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging level",
    )
    arguments = parser.parse_args()
    asyncio.run(_run_from_cli(arguments.host, arguments.port, arguments.log_level))