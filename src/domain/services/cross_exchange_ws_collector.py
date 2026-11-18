"""Lightweight Event-Driven Cross-Exchange Market Data Core (WebSocket Streams).

Maintains persistent sub-second WebSocket connections to:
  1. Binance Futures: wss://fstream.binance.com/ws (@bookTicker)
  2. Bybit Linear: wss://stream.bybit.com/v5/public/linear (tickers.{symbol})
  3. OKX Swap: wss://ws.okx.com:8443/ws/v5/public (tickers)

Conforms to Astra's (gpt-6-astra) Master Quant Blueprint:
  - Single aiohttp ClientSession shared across all 3 venues
  - Transport ping/pong + exchange application heartbeats
  - Normalized QuoteUpdate publishing into CrossExchangeState (O(1) in-memory)
  - Automatic dual-mode REST fallback if WebSocket quotes are silent > 10s
  - Full-jitter exponential backoff on reconnection (capped at 30s)
  - Thread-safe background execution inside MarketDataThread
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import aiohttp

from src.domain.services.cross_exchange_state import CrossExchangeState, QuoteUpdate

logger = logging.getLogger(__name__)

BINANCE_WS_URL = "wss://fstream.binance.com/ws"
BYBIT_WS_URL = "wss://stream.bybit.com/v5/public/linear"
OKX_WS_URL = "wss://ws.okx.com:8443/ws/v5/public"

BINANCE_REST_URL = "https://fapi.binance.com/fapi/v1/ticker/bookTicker"
BYBIT_REST_URL = "https://api.bybit.com/v5/market/tickers"
OKX_REST_URL = "https://www.okx.com/api/v5/market/ticker"

DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

OKX_SYMBOL_MAP = {
    "BTCUSDT": "BTC-USDT-SWAP",
    "ETHUSDT": "ETH-USDT-SWAP",
    "SOLUSDT": "SOL-USDT-SWAP",
}

OKX_REVERSE_SYMBOL_MAP = {v: k for k, v in OKX_SYMBOL_MAP.items()}


@dataclass(frozen=True, slots=True)
class ExchangeSpec:
    name: str
    ws_url: str


BINANCE = ExchangeSpec("binance", BINANCE_WS_URL)
BYBIT = ExchangeSpec("bybit", BYBIT_WS_URL)
OKX = ExchangeSpec("okx", OKX_WS_URL)


class CrossExchangeWSCollector:
    """Event-driven real-time market data collector managing Binance, Bybit, and OKX WebSocket streams."""

    def __init__(
        self,
        state: CrossExchangeState,
        symbols: Sequence[str] = DEFAULT_SYMBOLS,
        ws_stale_after_s: float = 10.0,
        ws_reconnect_after_s: float = 30.0,
        rest_fallback_interval_s: float = 5.0,
        ws_heartbeat_s: float = 15.0,
        app_heartbeat_s: float = 20.0,
        connect_timeout_s: float = 10.0,
        rest_timeout_s: float = 4.0,
    ) -> None:
        if ws_reconnect_after_s <= ws_stale_after_s:
            raise ValueError("ws_reconnect_after_s must be greater than ws_stale_after_s")

        self._state = state
        self._symbols = tuple(s.upper() for s in symbols)
        self._ws_stale_after_s = ws_stale_after_s
        self._ws_reconnect_after_s = ws_reconnect_after_s
        self._rest_fallback_interval_s = rest_fallback_interval_s
        self._ws_heartbeat_s = ws_heartbeat_s
        self._app_heartbeat_s = app_heartbeat_s
        self._connect_timeout_s = connect_timeout_s
        self._rest_timeout_s = rest_timeout_s

        self._stop_event = threading.Event()
        self._async_stop_event: Optional[asyncio.Event] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._thread: Optional[threading.Thread] = None

        self._fallback_tasks: Dict[Tuple[str, str], asyncio.Task[None]] = {}
        self._last_rest_attempt: Dict[Tuple[str, str], float] = {}

    def start_background(self) -> threading.Thread:
        """Starts the collector in a dedicated daemon thread running an asyncio event loop."""
        if self._thread and self._thread.is_alive():
            return self._thread

        self._stop_event.clear()

        def _runner() -> None:
            asyncio.run(self.run())

        self._thread = threading.Thread(target=_runner, name="MarketDataWSThread", daemon=True)
        self._thread.start()
        logger.info("🛰️ [CECL WS CORE] Запущен высокоскоростной WebSocket-коллектор (Binance/Bybit/OKX).")
        return self._thread

    def stop(self, join_timeout_s: float = 3.0) -> None:
        """Signals shutdown and joins the background thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=join_timeout_s)
        logger.info("🛰️ [CECL WS CORE] WebSocket-коллектор остановлен.")

    async def run(self) -> None:
        """Main coroutine managing the ClientSession and 3 exchange tasks."""
        self._async_stop_event = asyncio.Event()

        connector = aiohttp.TCPConnector(
            limit=8,
            limit_per_host=4,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=self._connect_timeout_s,
            sock_connect=self._connect_timeout_s,
            sock_read=None,
        )

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            raise_for_status=False,
            headers={"User-Agent": "MaBuy-MarketDataCore/2.0"},
        ) as session:
            self._session = session

            tasks = [
                asyncio.create_task(self._run_exchange(BINANCE), name="market-ws-binance"),
                asyncio.create_task(self._run_exchange(BYBIT), name="market-ws-bybit"),
                asyncio.create_task(self._run_exchange(OKX), name="market-ws-okx"),
            ]

            try:
                # Wait until thread stop event is set
                while not self._stop_event.is_set():
                    await asyncio.sleep(0.5)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

                fallback_tasks = list(self._fallback_tasks.values())
                for fb_task in fallback_tasks:
                    fb_task.cancel()
                if fallback_tasks:
                    await asyncio.gather(*fallback_tasks, return_exceptions=True)

                self._session = None

    async def _run_exchange(self, spec: ExchangeSpec) -> None:
        """Supervisor loop for a single exchange with exponential backoff on failure."""
        attempt = 0
        while not self._stop_event.is_set():
            connected_at = time.monotonic()
            try:
                await self._run_connected_exchange(spec)
                attempt = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # If connection stayed healthy for >= 30s, reset attempt counter
                if time.monotonic() - connected_at >= 30.0:
                    attempt = 0
                attempt += 1
                delay_s = self._backoff_delay_s(attempt)
                logger.warning(
                    f"WS {spec.name} disconnected (attempt {attempt}, reconnect in {delay_s:.2f}s): {exc}"
                )
                await self._sleep_interruptible(delay_s)

    @staticmethod
    def _backoff_delay_s(attempt: int) -> float:
        """Full-jitter exponential backoff: delay ~ Uniform(0, min(30, 0.5 * 2^attempt))."""
        cap_s = min(30.0, 0.5 * (2 ** attempt))
        return random.uniform(0.0, cap_s)

    async def _sleep_interruptible(self, seconds: float) -> None:
        remaining = seconds
        while remaining > 0.0 and not self._stop_event.is_set():
            step = min(0.25, remaining)
            await asyncio.sleep(step)
            remaining -= step

    async def _run_connected_exchange(self, spec: ExchangeSpec) -> None:
        """Manages one active WebSocket connection and message loop."""
        session = self._require_session()
        logger.info(f"🛰️ [WS CONNECT] Подключение к {spec.name.upper()} ({spec.ws_url})...")

        # Binance Futures WS server sends PING frames every 3m and expects client PONG (handled by autoping=True).
        # Binance does NOT respond to unsolicited client PING frames, so client-side heartbeat must be None.
        ws_heartbeat = None if spec.name == "binance" else self._ws_heartbeat_s

        async with session.ws_connect(
            spec.ws_url,
            heartbeat=ws_heartbeat,
            autoping=True,
            receive_timeout=None,
            max_msg_size=2 * 1024 * 1024,
        ) as ws:
            await self._subscribe(ws, spec)

            connected_at = time.monotonic()
            last_app_ping_at = connected_at
            last_ws_quote_at = {symbol: connected_at for symbol in self._symbols}
            bybit_ticker_cache: Dict[str, Dict[str, Any]] = {}

            logger.info(f"✅ [WS CONNECTED] {spec.name.upper()} онлайн. Символы: {', '.join(self._symbols)}")

            while not self._stop_event.is_set():
                now = time.monotonic()

                # App heartbeat (Bybit/OKX)
                if now - last_app_ping_at >= self._app_heartbeat_s:
                    await self._send_application_ping(ws, spec)
                    last_app_ping_at = now

                try:
                    message = await ws.receive(timeout=1.0)
                except asyncio.TimeoutError:
                    await self._monitor_staleness(spec=spec, last_ws_quote_at=last_ws_quote_at)
                    continue

                if message.type == aiohttp.WSMsgType.TEXT:
                    quotes = self._parse_message(
                        spec=spec,
                        raw_text=message.data,
                        bybit_ticker_cache=bybit_ticker_cache,
                    )
                    for q in quotes:
                        last_ws_quote_at[q.symbol] = time.monotonic()
                        self._publish_quote(q)

                    await self._monitor_staleness(spec=spec, last_ws_quote_at=last_ws_quote_at)
                    continue

                if message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSE):
                    raise ConnectionError(f"{spec.name} websocket closed cleanly by server")

                if message.type == aiohttp.WSMsgType.ERROR:
                    raise ConnectionError(f"{spec.name} websocket transport error: {ws.exception()!r}")

    async def _subscribe(self, ws: aiohttp.ClientWebSocketResponse, spec: ExchangeSpec) -> None:
        """Dispatches subscription frames tailored to each exchange."""
        if spec.name == "binance":
            params = [f"{symbol.lower()}@bookTicker" for symbol in self._symbols]
            await ws.send_json({"method": "SUBSCRIBE", "params": params, "id": 1})
            return

        if spec.name == "bybit":
            args = [f"tickers.{symbol}" for symbol in self._symbols]
            await ws.send_json({"op": "subscribe", "args": args})
            return

        if spec.name == "okx":
            args = [{"channel": "tickers", "instId": OKX_SYMBOL_MAP[s]} for s in self._symbols if s in OKX_SYMBOL_MAP]
            await ws.send_json({"op": "subscribe", "args": args})
            return

    async def _send_application_ping(self, ws: aiohttp.ClientWebSocketResponse, spec: ExchangeSpec) -> None:
        """Sends application-level heartbeat frames."""
        try:
            if spec.name == "bybit":
                await ws.send_json({"op": "ping"})
            elif spec.name == "okx":
                await ws.send_str("ping")
        except Exception as ping_err:
            logger.debug(f"Ping send error on {spec.name}: {ping_err}")

    def _parse_message(
        self,
        spec: ExchangeSpec,
        raw_text: str,
        bybit_ticker_cache: Dict[str, Dict[str, Any]],
    ) -> List[QuoteUpdate]:
        """Routes text frames to venue-specific parsers."""
        if raw_text == "pong":
            return []

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            return []

        if spec.name == "binance":
            q = self._parse_binance(payload)
            return [q] if q else []

        if spec.name == "bybit":
            q = self._parse_bybit(payload, bybit_ticker_cache)
            return [q] if q else []

        if spec.name == "okx":
            return self._parse_okx(payload)

        return []

    def _parse_binance(self, payload: Dict[str, Any]) -> Optional[QuoteUpdate]:
        if payload.get("e") != "bookTicker":
            return None
        symbol = payload.get("s")
        if symbol not in self._symbols:
            return None
        return self._make_quote(
            exchange="binance",
            symbol=symbol,
            bid=payload.get("b"),
            ask=payload.get("a"),
            exchange_ts_ms=self._to_int(payload.get("T")),
            source="ws",
        )

    def _parse_bybit(self, payload: Dict[str, Any], cache: Dict[str, Dict[str, Any]]) -> Optional[QuoteUpdate]:
        topic = payload.get("topic")
        if not isinstance(topic, str) or not topic.startswith("tickers."):
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            return None

        symbol = data.get("symbol") or topic.removeprefix("tickers.")
        if symbol not in self._symbols:
            return None

        current = cache.setdefault(symbol, {})
        current.update(data)

        bid = current.get("bid1Price")
        ask = current.get("ask1Price")
        if bid is None or ask is None:
            return None

        return self._make_quote(
            exchange="bybit",
            symbol=symbol,
            bid=bid,
            ask=ask,
            exchange_ts_ms=self._to_int(payload.get("ts")),
            source="ws",
        )

    def _parse_okx(self, payload: Dict[str, Any]) -> List[QuoteUpdate]:
        arg = payload.get("arg")
        rows = payload.get("data")
        if not isinstance(arg, dict) or arg.get("channel") != "tickers" or not isinstance(rows, list):
            return []

        result: List[QuoteUpdate] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            okx_sym = row.get("instId")
            symbol = OKX_REVERSE_SYMBOL_MAP.get(okx_sym)
            if symbol is None:
                continue

            q = self._make_quote(
                exchange="okx",
                symbol=symbol,
                bid=row.get("bidPx"),
                ask=row.get("askPx"),
                exchange_ts_ms=self._to_int(row.get("ts")),
                source="ws",
            )
            if q:
                result.append(q)
        return result

    def _make_quote(
        self,
        exchange: str,
        symbol: str,
        bid: Any,
        ask: Any,
        exchange_ts_ms: Optional[int],
        source: str,
    ) -> Optional[QuoteUpdate]:
        try:
            bid_f = float(bid)
            ask_f = float(ask)
        except (TypeError, ValueError):
            return None

        if bid_f <= 0.0 or ask_f <= 0.0 or ask_f < bid_f:
            return None

        return QuoteUpdate(
            exchange=exchange,
            symbol=symbol,
            bid=bid_f,
            ask=ask_f,
            exchange_ts_ms=exchange_ts_ms,
            received_monotonic_ns=time.monotonic_ns(),
            source=source,
        )

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _publish_quote(self, quote: QuoteUpdate) -> None:
        self._state.update_quote_update(quote)

    async def _monitor_staleness(
        self,
        spec: ExchangeSpec,
        last_ws_quote_at: Dict[str, float],
    ) -> None:
        """Detects quiet or stalled WebSocket streams and dispatches REST fallbacks."""
        now = time.monotonic()
        for symbol, last_seen in last_ws_quote_at.items():
            stale_s = now - last_seen
            if stale_s >= self._ws_stale_after_s:
                self._schedule_rest_fallback(exchange=spec.name, symbol=symbol)
            if stale_s >= self._ws_reconnect_after_s:
                raise ConnectionError(f"{spec.name}:{symbol} stream silent for {stale_s:.1f}s, forcing reconnect")

    def _schedule_rest_fallback(self, exchange: str, symbol: str) -> None:
        """Schedules non-blocking REST quote request if throttle interval has elapsed."""
        key = (exchange, symbol)
        now = time.monotonic()
        task = self._fallback_tasks.get(key)
        if task and not task.done():
            return

        if now - self._last_rest_attempt.get(key, 0.0) < self._rest_fallback_interval_s:
            return

        self._last_rest_attempt[key] = now
        new_task = asyncio.create_task(
            self._fetch_rest_fallback(exchange, symbol),
            name=f"rest-fb-{exchange}-{symbol}",
        )
        self._fallback_tasks[key] = new_task

        def _cleanup(_: asyncio.Task[None]) -> None:
            if self._fallback_tasks.get(key) is new_task:
                self._fallback_tasks.pop(key, None)

        new_task.add_done_callback(_cleanup)

    async def _fetch_rest_fallback(self, exchange: str, symbol: str) -> None:
        try:
            quote = await self._fetch_rest_quote(exchange, symbol)
            if quote:
                self._publish_quote(quote)
                logger.info(f"🔄 [REST FALLBACK] Получена котировка {exchange.upper()} {symbol}: ${quote.mid_price:,.2f}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(f"REST fallback error for {exchange} {symbol}: {exc}")

    async def _fetch_rest_quote(self, exchange: str, symbol: str) -> Optional[QuoteUpdate]:
        session = self._require_session()
        timeout = aiohttp.ClientTimeout(total=self._rest_timeout_s)

        if exchange == "binance":
            async with session.get(BINANCE_REST_URL, params={"symbol": symbol}, timeout=timeout) as resp:
                resp.raise_for_status()
                data = await resp.json()
            return self._make_quote(
                exchange="binance",
                symbol=symbol,
                bid=data.get("bidPrice"),
                ask=data.get("askPrice"),
                exchange_ts_ms=self._to_int(data.get("time")),
                source="rest",
            )

        if exchange == "bybit":
            async with session.get(BYBIT_REST_URL, params={"category": "linear", "symbol": symbol}, timeout=timeout) as resp:
                resp.raise_for_status()
                data = await resp.json()
            rows = data.get("result", {}).get("list", [])
            if not rows:
                return None
            row = rows[0]
            return self._make_quote(
                exchange="bybit",
                symbol=symbol,
                bid=row.get("bid1Price"),
                ask=row.get("ask1Price"),
                exchange_ts_ms=self._to_int(data.get("time")),
                source="rest",
            )

        if exchange == "okx":
            okx_sym = OKX_SYMBOL_MAP.get(symbol, f"{symbol}-SWAP")
            async with session.get(OKX_REST_URL, params={"instId": okx_sym}, timeout=timeout) as resp:
                resp.raise_for_status()
                data = await resp.json()
            rows = data.get("data", [])
            if not rows:
                return None
            row = rows[0]
            return self._make_quote(
                exchange="okx",
                symbol=symbol,
                bid=row.get("bidPx"),
                ask=row.get("askPx"),
                exchange_ts_ms=self._to_int(row.get("ts")),
                source="rest",
            )
        return None

    def _require_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            raise RuntimeError("ClientSession is not open")
        return self._session
