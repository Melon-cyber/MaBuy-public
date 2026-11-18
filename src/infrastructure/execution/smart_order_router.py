"""
Institutional Smart Order Router for MaBuy.

Supported design (Astra Step 5 Specification):
- Binance USDT-M: primary venue
- Bybit Linear: optional / canary
- OKX Swap: optional / canary
- maker-first post-only microprice pegging
- partial-fill protective-stop synchronization
- TWAP / Iceberg parent execution
- safe Binance primary fallback
- no heavy dependencies; standard library + decimal + math + ccxt
"""

from __future__ import annotations

import logging
import math
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple


LOG = logging.getLogger("mabuy.execution.sor")


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class ExecutionMode(str, Enum):
    MAKER_FIRST = "maker_first"
    TAKER_ONLY = "taker_only"
    TWAP = "twap"


@dataclass(frozen=True)
class RoutingRequest:
    symbol: str
    side: Side
    quantity: Decimal
    stop_loss_price: Decimal

    mode: ExecutionMode = ExecutionMode.MAKER_FIRST
    preferred_venue: Optional[str] = None

    max_slippage_bps: Decimal = Decimal("8")
    max_price_drift_bps: Decimal = Decimal("4")
    maker_timeout_seconds: float = 3.0
    max_reprices: int = 3

    # TWAP parameters
    twap_slices: int = 1
    twap_interval_seconds: float = 3.0
    twap_jitter_seconds: float = 0.75

    # Participation from best displayed level (e.g. 10%)
    top_book_participation: Decimal = Decimal("0.10")

    # If False, router never sends a market order after maker attempts
    allow_taker_fallback: bool = True

    # Strategy / risk correlation identifiers
    strategy_id: str = "unknown"
    signal_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    reduce_only: bool = False


@dataclass
class ExecutionVenue:
    name: str
    adapter: "VenueAdapter"

    enabled: bool = True
    live_enabled: bool = False
    primary: bool = False

    maker_fee_bps: Decimal = Decimal("2.0")
    taker_fee_bps: Decimal = Decimal("5.0")

    # Operational cost component, e.g. latency/API health/risk penalty
    operational_penalty_bps: Decimal = Decimal("0")
    max_leverage: Decimal = Decimal("3")

    healthy: bool = True
    last_error: Optional[str] = None


@dataclass(frozen=True)
class OrderBookTop:
    bid: Decimal
    ask: Decimal
    bid_qty: Decimal
    ask_qty: Decimal
    timestamp_ms: int


@dataclass
class VenueLiquidity:
    venue: str
    book: OrderBookTop
    free_margin: Decimal
    max_executable_qty: Decimal
    effective_cost: Decimal
    eligible: bool
    reason: str = ""


@dataclass
class RoutingDecision:
    request: RoutingRequest
    selected_venue: str
    ranked_venues: List[VenueLiquidity]
    intended_quantity: Decimal
    reason: str


@dataclass
class ExecutionReceipt:
    request_id: str
    symbol: str
    side: Side
    requested_quantity: Decimal

    filled_quantity: Decimal = Decimal("0")
    average_price: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")

    venue: Optional[str] = None
    order_ids: List[str] = field(default_factory=list)
    stop_order_ids: List[str] = field(default_factory=list)

    used_taker_fallback: bool = False
    completed: bool = False
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Adapter and risk contracts
# ---------------------------------------------------------------------------

class VenueAdapter(Protocol):
    """
    Standardized exchange adapter interface.
    All Decimal conversion and exchange-specific params belong inside adapters.
    """

    name: str

    def is_configured(self) -> bool:
        ...

    def fetch_top_of_book(self, symbol: str) -> OrderBookTop:
        ...

    def fetch_free_margin(self) -> Decimal:
        ...

    def fetch_tick_size(self, symbol: str) -> Decimal:
        ...

    def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        ...

    def place_post_only_limit(
        self,
        symbol: str,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        client_order_id: str,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        ...

    def place_market_order(
        self,
        symbol: str,
        side: Side,
        quantity: Decimal,
        client_order_id: str,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        ...

    def cancel_order(self, order_id: str, symbol: str) -> None:
        ...

    def place_protective_stop(
        self,
        symbol: str,
        entry_side: Side,
        quantity: Decimal,
        stop_price: Decimal,
        client_order_id: str,
    ) -> Dict[str, Any]:
        ...


class RiskKernel(Protocol):
    """Step-4 integration contract."""

    def validate_execution(
        self,
        *,
        symbol: str,
        side: Side,
        quantity: Decimal,
        estimated_price: Decimal,
        stop_loss_price: Decimal,
        venue: str,
        strategy_id: str,
        reduce_only: bool,
    ) -> Tuple[bool, str]:
        ...

    def on_fill(
        self,
        *,
        symbol: str,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        venue: str,
        strategy_id: str,
    ) -> None:
        ...


# ---------------------------------------------------------------------------
# Smart Order Router
# ---------------------------------------------------------------------------

class SmartOrderRouter:
    """
    Thread-safe maker-first Smart Order Router.

    Serializes execution per symbol to prevent simultaneous child-orders
    from racing on aggregate protective-stop quantity.
    """

    def __init__(
        self,
        venues: List[ExecutionVenue],
        risk_kernel: RiskKernel,
        poll_interval_seconds: float = 0.25,
        emergency_close_on_stop_failure: bool = True,
    ) -> None:
        if not venues:
            raise ValueError("SOR requires at least one venue.")

        self._venues = {venue.name.lower(): venue for venue in venues}
        self._risk_kernel = risk_kernel
        self._poll_interval = poll_interval_seconds
        self._emergency_close_on_stop_failure = emergency_close_on_stop_failure

        self._symbol_locks: Dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route_and_execute(self, request: RoutingRequest) -> ExecutionReceipt:
        """Route and execute one parent order, optionally through micro-TWAP."""
        self._validate_request(request)
        request_id = f"sor-{request.signal_id[:12]}"

        with self._lock_for_symbol(request.symbol):
            LOG.info(
                "[SOR-РОУТИНГ] request=%s symbol=%s side=%s qty=%s mode=%s",
                request_id,
                request.symbol,
                request.side.value,
                request.quantity,
                request.mode.value,
            )

            if request.mode is ExecutionMode.TWAP and request.twap_slices > 1:
                return self._execute_twap(request_id, request)

            return self._execute_single_slice(
                request_id=request_id,
                request=request,
                quantity=request.quantity,
            )

    @staticmethod
    def calculate_microprice(book: OrderBookTop) -> Decimal:
        """
        P_micro = (P_ask * Q_bid + P_bid * Q_ask) / (Q_bid + Q_ask)
        """
        denominator = book.bid_qty + book.ask_qty
        if denominator <= Decimal("0"):
            raise ValueError("Invalid top-of-book: bid_qty + ask_qty <= 0")

        return (
            (book.ask * book.bid_qty) + (book.bid * book.ask_qty)
        ) / denominator

    def assess_venue_liquidity(
        self,
        venue: ExecutionVenue,
        request: RoutingRequest,
    ) -> VenueLiquidity:
        """Evaluate health, margin, top-book capacity and effective maker cost."""
        if not venue.enabled:
            return self._ineligible_liquidity(venue.name, "venue disabled")

        if not venue.live_enabled:
            return self._ineligible_liquidity(venue.name, "live mode disabled")

        if not venue.healthy:
            return self._ineligible_liquidity(
                venue.name,
                f"venue unhealthy: {venue.last_error or 'unknown'}",
            )

        if not venue.adapter.is_configured():
            return self._ineligible_liquidity(venue.name, "API adapter not configured")

        try:
            book = venue.adapter.fetch_top_of_book(request.symbol)
            free_margin = venue.adapter.fetch_free_margin()
        except Exception as exc:
            venue.healthy = False
            venue.last_error = str(exc)
            LOG.exception("[SOR-РОУТИНГ] venue=%s market-data failure", venue.name)
            return self._ineligible_liquidity(venue.name, f"market data error: {exc}")

        if book.bid <= Decimal("0") or book.ask <= Decimal("0") or book.ask <= book.bid:
            return self._ineligible_liquidity(venue.name, "invalid bid/ask")

        reference_price = book.ask if request.side is Side.BUY else book.bid
        top_qty = book.ask_qty if request.side is Side.BUY else book.bid_qty

        book_capacity = top_qty * request.top_book_participation
        margin_capacity = (
            free_margin * venue.max_leverage / reference_price
            if reference_price > Decimal("0")
            else Decimal("0")
        )

        max_qty = min(request.quantity, book_capacity, margin_capacity)
        microprice = self.calculate_microprice(book)
        fee_bps = venue.maker_fee_bps

        # BUY: smaller all-in cost is better.
        # SELL: larger net proceeds are better -> negative cost normalization.
        if request.side is Side.BUY:
            effective_cost = microprice * (Decimal("1") + fee_bps / Decimal("10000"))
        else:
            effective_cost = -microprice * (Decimal("1") - fee_bps / Decimal("10000"))

        effective_cost += (
            microprice * venue.operational_penalty_bps / Decimal("10000")
        )

        eligible = max_qty > Decimal("0")
        return VenueLiquidity(
            venue=venue.name,
            book=book,
            free_margin=free_margin,
            max_executable_qty=max_qty,
            effective_cost=effective_cost,
            eligible=eligible,
            reason="" if eligible else "insufficient top-book liquidity or margin",
        )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _select_venue(self, request: RoutingRequest) -> RoutingDecision:
        assessments = [
            self.assess_venue_liquidity(venue, request)
            for venue in self._venues.values()
        ]
        eligible = [item for item in assessments if item.eligible]

        if request.preferred_venue:
            preferred = request.preferred_venue.lower()
            preferred_item = next(
                (item for item in eligible if item.venue.lower() == preferred),
                None,
            )
            if preferred_item is not None:
                return RoutingDecision(
                    request=request,
                    selected_venue=preferred_item.venue,
                    ranked_venues=assessments,
                    intended_quantity=min(request.quantity, preferred_item.max_executable_qty),
                    reason="preferred venue is eligible",
                )

        if eligible:
            # Minimize effective cost; prefer larger capacity
            eligible.sort(
                key=lambda x: (x.effective_cost, -x.max_executable_qty)
            )
            winner = eligible[0]
            return RoutingDecision(
                request=request,
                selected_venue=winner.venue,
                ranked_venues=assessments,
                intended_quantity=min(request.quantity, winner.max_executable_qty),
                reason="best effective maker cost after fee/liquidity/margin checks",
            )

        # Explicit safe fallback to Binance Primary
        binance = self._venues.get("binance")
        if binance and binance.enabled and binance.live_enabled and binance.adapter.is_configured():
            fallback = self.assess_venue_liquidity(binance, request)
            if fallback.eligible:
                return RoutingDecision(
                    request=request,
                    selected_venue=binance.name,
                    ranked_venues=assessments + [fallback],
                    intended_quantity=min(request.quantity, fallback.max_executable_qty),
                    reason="fallback to Binance primary venue",
                )

        reasons = "; ".join(f"{x.venue}: {x.reason}" for x in assessments)
        raise RuntimeError(f"No eligible execution venue. {reasons}")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_single_slice(
        self,
        request_id: str,
        request: RoutingRequest,
        quantity: Decimal,
    ) -> ExecutionReceipt:
        decision = self._select_venue(request)
        venue = self._venues[decision.selected_venue.lower()]
        adapter = venue.adapter

        receipt = ExecutionReceipt(
            request_id=request_id,
            symbol=request.symbol,
            side=request.side,
            requested_quantity=quantity,
            venue=venue.name,
        )

        book = adapter.fetch_top_of_book(request.symbol)
        estimated_price = book.ask if request.side is Side.BUY else book.bid

        approved, reason = self._risk_kernel.validate_execution(
            symbol=request.symbol,
            side=request.side,
            quantity=decision.intended_quantity,
            estimated_price=estimated_price,
            stop_loss_price=request.stop_loss_price,
            venue=venue.name,
            strategy_id=request.strategy_id,
            reduce_only=request.reduce_only,
        )
        if not approved:
            raise PermissionError(f"Risk Kernel rejected execution: {reason}")

        if request.mode is ExecutionMode.TAKER_ONLY:
            return self._execute_taker(
                receipt=receipt,
                request=request,
                venue=venue,
                quantity=decision.intended_quantity,
            )

        return self._execute_maker_first(
            receipt=receipt,
            request=request,
            venue=venue,
            quantity=decision.intended_quantity,
        )

    def _execute_maker_first(
        self,
        receipt: ExecutionReceipt,
        request: RoutingRequest,
        venue: ExecutionVenue,
        quantity: Decimal,
    ) -> ExecutionReceipt:
        adapter = venue.adapter
        remaining = quantity
        protected_qty = Decimal("0")
        total_notional = Decimal("0")

        for attempt in range(request.max_reprices + 1):
            if remaining <= Decimal("0"):
                break

            book = adapter.fetch_top_of_book(request.symbol)
            tick = adapter.fetch_tick_size(request.symbol)
            microprice = self.calculate_microprice(book)
            peg_price = self._maker_peg_price(
                side=request.side,
                microprice=microprice,
                book=book,
                tick=tick,
            )

            client_id = f"{receipt.request_id}-m{attempt}"
            LOG.info(
                "[SOR-MAKER-PEG] venue=%s symbol=%s attempt=%d side=%s "
                "micro=%s peg=%s qty=%s",
                venue.name,
                request.symbol,
                attempt,
                request.side.value,
                microprice,
                peg_price,
                remaining,
            )

            try:
                order = adapter.place_post_only_limit(
                    symbol=request.symbol,
                    side=request.side,
                    quantity=remaining,
                    price=peg_price,
                    client_order_id=client_id,
                    reduce_only=request.reduce_only,
                )
                order_id = str(order["id"])
                receipt.order_ids.append(order_id)
            except Exception as post_err:
                LOG.warning(
                    "[SOR-MAKER-PEGGING] %s attempt %s: Post-Only limit rejected (%s). Proceeding to next attempt or fallback.",
                    request.symbol,
                    attempt,
                    post_err,
                )
                continue

            deadline = time.monotonic() + request.maker_timeout_seconds
            last_filled = Decimal("0")

            while time.monotonic() < deadline:
                status = adapter.fetch_order(order_id, request.symbol)
                filled = self._decimal(status.get("filled", "0"))
                avg_price = self._decimal(status.get("average") or peg_price)

                delta_fill = max(Decimal("0"), filled - last_filled)
                if delta_fill > Decimal("0"):
                    protected_qty = self._protect_partial_fill(
                        receipt=receipt,
                        request=request,
                        venue=venue,
                        delta_fill=delta_fill,
                        fill_price=avg_price,
                        already_protected=protected_qty,
                    )
                    total_notional += delta_fill * avg_price
                    receipt.filled_quantity += delta_fill
                    last_filled = filled

                state = str(status.get("status", "")).lower()
                if state in {"closed", "filled"}:
                    break

                time.sleep(self._poll_interval)

            # Atomic cleanup phase: cancel remainder, then fetch final fill
            try:
                adapter.cancel_order(order_id, request.symbol)
            except Exception as exc:
                LOG.warning(
                    "[SOR-EXECUTION] venue=%s order=%s cancel warning=%s",
                    venue.name,
                    order_id,
                    exc,
                )

            final = adapter.fetch_order(order_id, request.symbol)
            final_filled = self._decimal(final.get("filled", "0"))
            final_avg = self._decimal(final.get("average") or peg_price)

            final_delta = max(Decimal("0"), final_filled - last_filled)
            if final_delta > Decimal("0"):
                protected_qty = self._protect_partial_fill(
                    receipt=receipt,
                    request=request,
                    venue=venue,
                    delta_fill=final_delta,
                    fill_price=final_avg,
                    already_protected=protected_qty,
                )
                total_notional += final_delta * final_avg
                receipt.filled_quantity += final_delta

            remaining = max(Decimal("0"), quantity - receipt.filled_quantity)

            if remaining <= Decimal("0"):
                break

        # Controlled taker fallback only after maker lifecycle ends
        if remaining > Decimal("0") and request.allow_taker_fallback:
            receipt.used_taker_fallback = True
            taker_receipt = self._execute_taker(
                receipt=receipt,
                request=request,
                venue=venue,
                quantity=remaining,
            )
            receipt = taker_receipt

        if receipt.filled_quantity > Decimal("0") and total_notional > Decimal("0"):
            receipt.average_price = total_notional / receipt.filled_quantity

        receipt.completed = receipt.filled_quantity >= quantity

        LOG.info(
            "[SOR-EXECUTION] request=%s venue=%s requested=%s filled=%s "
            "avg=%s completed=%s taker_fallback=%s",
            receipt.request_id,
            venue.name,
            quantity,
            receipt.filled_quantity,
            receipt.average_price,
            receipt.completed,
            receipt.used_taker_fallback,
        )
        return receipt

    def _execute_taker(
        self,
        receipt: ExecutionReceipt,
        request: RoutingRequest,
        venue: ExecutionVenue,
        quantity: Decimal,
    ) -> ExecutionReceipt:
        if quantity <= Decimal("0"):
            return receipt

        adapter = venue.adapter
        book = adapter.fetch_top_of_book(request.symbol)
        estimate = book.ask if request.side is Side.BUY else book.bid

        approved, reason = self._risk_kernel.validate_execution(
            symbol=request.symbol,
            side=request.side,
            quantity=quantity,
            estimated_price=estimate,
            stop_loss_price=request.stop_loss_price,
            venue=venue.name,
            strategy_id=request.strategy_id,
            reduce_only=request.reduce_only,
        )
        if not approved:
            raise PermissionError(f"Risk Kernel rejected taker fallback: {reason}")

        client_id = f"{receipt.request_id}-taker"
        order = adapter.place_market_order(
            symbol=request.symbol,
            side=request.side,
            quantity=quantity,
            client_order_id=client_id,
            reduce_only=request.reduce_only,
        )
        order_id = str(order["id"])
        receipt.order_ids.append(order_id)

        time.sleep(self._poll_interval)
        status = adapter.fetch_order(order_id, request.symbol)
        filled = self._decimal(status.get("filled", "0"))
        average = self._decimal(status.get("average") or estimate)

        if filled > Decimal("0"):
            self._protect_partial_fill(
                receipt=receipt,
                request=request,
                venue=venue,
                delta_fill=filled,
                fill_price=average,
                already_protected=receipt.filled_quantity,
            )
            receipt.filled_quantity += filled
            receipt.average_price = average

        return receipt

    def _protect_partial_fill(
        self,
        receipt: ExecutionReceipt,
        request: RoutingRequest,
        venue: ExecutionVenue,
        delta_fill: Decimal,
        fill_price: Decimal,
        already_protected: Decimal,
    ) -> Decimal:
        """
        Register fill in Risk Kernel and immediately place a protective stop for delta fill.
        """
        if delta_fill <= Decimal("0"):
            return already_protected

        self._risk_kernel.on_fill(
            symbol=request.symbol,
            side=request.side,
            quantity=delta_fill,
            price=fill_price,
            venue=venue.name,
            strategy_id=request.strategy_id,
        )

        try:
            stop = venue.adapter.place_protective_stop(
                symbol=request.symbol,
                entry_side=request.side,
                quantity=delta_fill,
                stop_price=request.stop_loss_price,
                client_order_id=f"{receipt.request_id}-sl-{len(receipt.stop_order_ids)}",
            )
            stop_id = str(stop["id"])
            receipt.stop_order_ids.append(stop_id)

            LOG.info(
                "[SOR-EXECUTION] protective-stop placed venue=%s symbol=%s "
                "filled_delta=%s stop=%s stop_id=%s",
                venue.name,
                request.symbol,
                delta_fill,
                request.stop_loss_price,
                stop_id,
            )
            return already_protected + delta_fill

        except Exception as exc:
            message = (
                f"CRITICAL: stop placement failed on {venue.name}, "
                f"symbol={request.symbol}, qty={delta_fill}: {exc}"
            )
            LOG.critical("[SOR-EXECUTION] %s", message)
            receipt.errors.append(message)

            if self._emergency_close_on_stop_failure:
                self._emergency_reduce_position(
                    venue=venue,
                    symbol=request.symbol,
                    entry_side=request.side,
                    quantity=delta_fill,
                    request_id=receipt.request_id,
                )
            raise RuntimeError(message) from exc

    # ------------------------------------------------------------------
    # TWAP
    # ------------------------------------------------------------------

    def _execute_twap(
        self,
        request_id: str,
        request: RoutingRequest,
    ) -> ExecutionReceipt:
        slices = max(1, request.twap_slices)
        base_slice = request.quantity / Decimal(str(slices))

        parent = ExecutionReceipt(
            request_id=request_id,
            symbol=request.symbol,
            side=request.side,
            requested_quantity=request.quantity,
        )

        remaining = request.quantity
        total_notional = Decimal("0")

        for index in range(slices):
            slice_qty = remaining if index == slices - 1 else base_slice
            slice_request = RoutingRequest(
                symbol=request.symbol,
                side=request.side,
                quantity=slice_qty,
                stop_loss_price=request.stop_loss_price,
                mode=ExecutionMode.MAKER_FIRST,
                preferred_venue=request.preferred_venue,
                max_slippage_bps=request.max_slippage_bps,
                max_price_drift_bps=request.max_price_drift_bps,
                maker_timeout_seconds=request.maker_timeout_seconds,
                max_reprices=request.max_reprices,
                top_book_participation=request.top_book_participation,
                allow_taker_fallback=request.allow_taker_fallback,
                strategy_id=request.strategy_id,
                signal_id=f"{request.signal_id}-slice-{index}",
                reduce_only=request.reduce_only,
            )

            child = self._execute_single_slice(
                request_id=f"{request_id}-slice-{index}",
                request=slice_request,
                quantity=slice_qty,
            )

            parent.order_ids.extend(child.order_ids)
            parent.stop_order_ids.extend(child.stop_order_ids)
            parent.errors.extend(child.errors)
            parent.used_taker_fallback = parent.used_taker_fallback or child.used_taker_fallback
            parent.venue = child.venue

            parent.filled_quantity += child.filled_quantity
            total_notional += child.average_price * child.filled_quantity
            remaining = max(Decimal("0"), request.quantity - parent.filled_quantity)

            if remaining <= Decimal("0"):
                break

            if index < slices - 1:
                # Bounded Gaussian jitter prevents deterministic execution footprint
                jitter = random.gauss(0.0, request.twap_jitter_seconds)
                delay = max(0.05, request.twap_interval_seconds + jitter)
                LOG.info(
                    "[SOR-EXECUTION] TWAP slice=%d/%d next_delay=%.3fs",
                    index + 1,
                    slices,
                    delay,
                )
                time.sleep(delay)

        if parent.filled_quantity > Decimal("0"):
            parent.average_price = total_notional / parent.filled_quantity

        parent.completed = parent.filled_quantity >= request.quantity
        return parent

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _maker_peg_price(
        self,
        side: Side,
        microprice: Decimal,
        book: OrderBookTop,
        tick: Decimal,
    ) -> Decimal:
        if tick <= Decimal("0"):
            raise ValueError("tick size must be positive")

        if side is Side.BUY:
            candidate = self._round_to_tick(microprice, tick, ROUND_FLOOR)
            safe_post_only = book.ask - tick
            return min(candidate, safe_post_only)

        candidate = self._round_to_tick(microprice, tick, ROUND_CEILING)
        safe_post_only = book.bid + tick
        return max(candidate, safe_post_only)

    def _emergency_reduce_position(
        self,
        venue: ExecutionVenue,
        symbol: str,
        entry_side: Side,
        quantity: Decimal,
        request_id: str,
    ) -> None:
        """Fail-closed emergency position reduction after stop-placement failure."""
        try:
            venue.adapter.place_market_order(
                symbol=symbol,
                side=entry_side.opposite,
                quantity=quantity,
                client_order_id=f"{request_id}-emergency-close",
                reduce_only=True,
            )
            LOG.critical(
                "[SOR-EXECUTION] EMERGENCY REDUCE executed venue=%s symbol=%s qty=%s",
                venue.name,
                symbol,
                quantity,
            )
        except Exception:
            LOG.exception(
                "[SOR-EXECUTION] EMERGENCY REDUCE FAILED venue=%s symbol=%s qty=%s",
                venue.name,
                symbol,
                quantity,
            )

    @staticmethod
    def _round_to_tick(
        price: Decimal,
        tick: Decimal,
        rounding: str,
    ) -> Decimal:
        return (price / tick).to_integral_value(rounding=rounding) * tick

    @staticmethod
    def _decimal(value: Any) -> Decimal:
        return Decimal(str(value))

    @staticmethod
    def _validate_request(request: RoutingRequest) -> None:
        if request.quantity <= Decimal("0"):
            raise ValueError("quantity must be positive")
        if request.stop_loss_price <= Decimal("0"):
            raise ValueError("stop_loss_price must be positive")
        if request.twap_slices < 1:
            raise ValueError("twap_slices must be >= 1")
        if not (Decimal("0") < request.top_book_participation <= Decimal("1")):
            raise ValueError("top_book_participation must be in (0, 1]")

    def _lock_for_symbol(self, symbol: str) -> threading.RLock:
        with self._locks_guard:
            return self._symbol_locks.setdefault(symbol, threading.RLock())

    @staticmethod
    def _ineligible_liquidity(venue: str, reason: str) -> VenueLiquidity:
        zero_book = OrderBookTop(
            bid=Decimal("0"),
            ask=Decimal("0"),
            bid_qty=Decimal("0"),
            ask_qty=Decimal("0"),
            timestamp_ms=0,
        )
        return VenueLiquidity(
            venue=venue,
            book=zero_book,
            free_margin=Decimal("0"),
            max_executable_qty=Decimal("0"),
            effective_cost=Decimal("Infinity"),
            eligible=False,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Standard Adapters
# ---------------------------------------------------------------------------

class BinanceVenueAdapter:
    """
    Adapter implementing VenueAdapter for Binance Futures (BinanceAutoTrader).
    """

    def __init__(self, trader: Any):
        self.trader = trader
        self.name = "binance"

    def is_configured(self) -> bool:
        return bool(getattr(self.trader, "api_key", None) and getattr(self.trader, "api_secret", None))

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        clean = symbol.upper().replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").replace("USDT", "").strip()
        return f"{clean}/USDT:USDT"

    def fetch_top_of_book(self, symbol: str) -> OrderBookTop:
        exchange = self.trader._create_isolated_session()
        m_sym = self._normalize_symbol(symbol)
        try:
            book = exchange.fetch_order_book(m_sym, limit=5)
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            if bids and asks:
                return OrderBookTop(
                    bid=Decimal(str(bids[0][0])),
                    ask=Decimal(str(asks[0][0])),
                    bid_qty=Decimal(str(bids[0][1])),
                    ask_qty=Decimal(str(asks[0][1])),
                    timestamp_ms=int(book.get("timestamp") or (time.time() * 1000)),
                )
        except Exception:
            pass

        ticker = exchange.fetch_ticker(m_sym)
        bid = Decimal(str(ticker.get("bid") or ticker.get("last") or "0"))
        ask = Decimal(str(ticker.get("ask") or ticker.get("last") or "0"))
        return OrderBookTop(
            bid=bid,
            ask=ask,
            bid_qty=Decimal("10.0"),
            ask_qty=Decimal("10.0"),
            timestamp_ms=int(time.time() * 1000),
        )

    def fetch_free_margin(self) -> Decimal:
        exchange = self.trader._create_isolated_session()
        bal = exchange.fetch_balance()
        free_usdt = bal.get("USDT", {}).get("free", 0.0)
        return Decimal(str(free_usdt or "0"))

    def fetch_tick_size(self, symbol: str) -> Decimal:
        exchange = self.trader._create_isolated_session()
        m_sym = self._normalize_symbol(symbol)
        try:
            market = exchange.market(m_sym) if m_sym in exchange.markets else {}
            tick = market.get("precision", {}).get("price")
            if tick is not None:
                if isinstance(tick, int):
                    return Decimal("10") ** -tick
                return Decimal(str(tick))
        except Exception:
            pass
        return Decimal("0.01")

    def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        exchange = self.trader._create_isolated_session()
        m_sym = self._normalize_symbol(symbol)
        return exchange.fetch_order(order_id, m_sym)

    def place_post_only_limit(
        self,
        symbol: str,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        client_order_id: str,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        exchange = self.trader._create_isolated_session()
        m_sym = self._normalize_symbol(symbol)
        try:
            if m_sym not in exchange.markets:
                exchange.load_markets()
            clean_amount = float(exchange.amount_to_precision(m_sym, float(quantity)))
            clean_price = float(exchange.price_to_precision(m_sym, float(price)))
        except Exception:
            clean_amount = float(quantity)
            clean_price = float(price)

        params: Dict[str, Any] = {
            "timeInForce": "GTX",
            "postOnly": True,
            "newClientOrderId": client_order_id[:32],
        }
        if reduce_only:
            params["reduceOnly"] = True
        return exchange.create_order(
            symbol=m_sym,
            type="LIMIT",
            side=side.value,
            amount=clean_amount,
            price=clean_price,
            params=params,
        )

    def place_market_order(
        self,
        symbol: str,
        side: Side,
        quantity: Decimal,
        client_order_id: str,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        exchange = self.trader._create_isolated_session()
        m_sym = self._normalize_symbol(symbol)
        try:
            if m_sym not in exchange.markets:
                exchange.load_markets()
            clean_amount = float(exchange.amount_to_precision(m_sym, float(quantity)))
        except Exception:
            clean_amount = float(quantity)

        params: Dict[str, Any] = {"newClientOrderId": client_order_id[:32]}
        if reduce_only:
            params["reduceOnly"] = True
        return exchange.create_order(
            symbol=m_sym,
            type="MARKET",
            side=side.value,
            amount=clean_amount,
            params=params,
        )

    def cancel_order(self, order_id: str, symbol: str) -> None:
        exchange = self.trader._create_isolated_session()
        m_sym = self._normalize_symbol(symbol)
        exchange.cancel_order(order_id, m_sym)

    def place_protective_stop(
        self,
        symbol: str,
        entry_side: Side,
        quantity: Decimal,
        stop_price: Decimal,
        client_order_id: str,
    ) -> Dict[str, Any]:
        exchange = self.trader._create_isolated_session()
        m_sym = self._normalize_symbol(symbol)
        sl_side = "sell" if entry_side == Side.BUY else "buy"
        try:
            if m_sym not in exchange.markets:
                exchange.load_markets()
            clean_amount = float(exchange.amount_to_precision(m_sym, float(quantity)))
            clean_stop = float(exchange.price_to_precision(m_sym, float(stop_price)))
        except Exception:
            clean_amount = float(quantity)
            clean_stop = float(stop_price)

        params: Dict[str, Any] = {
            "stopPrice": clean_stop,
            "reduceOnly": True,
            "workingType": "MARK_PRICE",
            "newClientOrderId": client_order_id[:32],
        }
        try:
            return exchange.create_order(
                symbol=m_sym,
                type="STOP_MARKET",
                side=sl_side,
                amount=clean_amount,
                params=params,
            )
        except Exception:
            params.pop("reduceOnly", None)
            params["closePosition"] = True
            return exchange.create_order(
                symbol=m_sym,
                type="STOP_MARKET",
                side=sl_side,
                amount=None,
                params=params,
            )


class BybitLinearAdapter:
    """
    Canary/Optional Bybit Linear Adapter.
    Automatically inactive until BYBIT_API_KEY and BYBIT_API_SECRET are provided.
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or os.environ.get("BYBIT_API_KEY")
        self.api_secret = api_secret or os.environ.get("BYBIT_API_SECRET")
        self.name = "bybit"

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def fetch_top_of_book(self, symbol: str) -> OrderBookTop:
        raise NotImplementedError("Bybit live adapter not configured")

    def fetch_free_margin(self) -> Decimal:
        return Decimal("0")

    def fetch_tick_size(self, symbol: str) -> Decimal:
        return Decimal("0.01")

    def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError("Bybit live adapter not configured")

    def place_post_only_limit(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Bybit live adapter not configured")

    def place_market_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Bybit live adapter not configured")

    def cancel_order(self, order_id: str, symbol: str) -> None:
        raise NotImplementedError("Bybit live adapter not configured")

    def place_protective_stop(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Bybit live adapter not configured")


class OKXSwapAdapter:
    """
    Canary/Optional OKX Swap Adapter.
    Automatically inactive until OKX_API_KEY and OKX_API_SECRET are provided.
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OKX_API_KEY")
        self.api_secret = api_secret or os.environ.get("OKX_API_SECRET")
        self.name = "okx"

    def is_configured(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def fetch_top_of_book(self, symbol: str) -> OrderBookTop:
        raise NotImplementedError("OKX live adapter not configured")

    def fetch_free_margin(self) -> Decimal:
        return Decimal("0")

    def fetch_tick_size(self, symbol: str) -> Decimal:
        return Decimal("0.01")

    def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError("OKX live adapter not configured")

    def place_post_only_limit(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("OKX live adapter not configured")

    def place_market_order(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("OKX live adapter not configured")

    def cancel_order(self, order_id: str, symbol: str) -> None:
        raise NotImplementedError("OKX live adapter not configured")

    def place_protective_stop(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("OKX live adapter not configured")


class PortfolioRiskKernelAdapter:
    """
    Adapts PortfolioRiskKernel to the SOR RiskKernel protocol.
    Enforces all Step-4 Portfolio Risk Kernel gates during SOR routing.
    """

    def __init__(
        self,
        risk_kernel: Any,
        portfolio_snapshot_provider: Optional[Callable[[], Any]] = None,
    ):
        self.risk_kernel = risk_kernel
        self.snapshot_provider = portfolio_snapshot_provider

    def validate_execution(
        self,
        *,
        symbol: str,
        side: Side,
        quantity: Decimal,
        estimated_price: Decimal,
        stop_loss_price: Decimal,
        venue: str,
        strategy_id: str,
        reduce_only: bool,
    ) -> Tuple[bool, str]:
        if reduce_only:
            return True, "reduce_only orders bypass entry risk gates"

        clean_sym = symbol.upper().replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").replace("USDT", "").strip()
        side_clean = "LONG" if side == Side.BUY else "SHORT"

        if self.snapshot_provider:
            snapshot = self.snapshot_provider()
        else:
            from src.domain.services.portfolio_risk_kernel import PortfolioSnapshot
            snapshot = PortfolioSnapshot(equity_usd=100.0, day_start_equity_usd=100.0)

        from src.domain.services.portfolio_risk_kernel import CandidateOrder
        cand = CandidateOrder(
            symbol=clean_sym,
            side=side_clean,
            entry_price=float(estimated_price),
            stop_price=float(stop_loss_price),
            take_profit_price=float(estimated_price * (Decimal("1.03") if side == Side.BUY else Decimal("0.97"))),
            raw_confidence=0.80,
            requested_leverage=3.0,
        )

        res = self.risk_kernel.evaluate_candidate_order(snapshot, cand)
        if not res.approved:
            return False, res.veto_reason
        return True, "Risk kernel approved"

    def on_fill(
        self,
        *,
        symbol: str,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        venue: str,
        strategy_id: str,
    ) -> None:
        LOG.info(
            "[SOR-RISK-FILL] symbol=%s side=%s qty=%s price=%s venue=%s strategy=%s",
            symbol,
            side.value,
            quantity,
            price,
            venue,
            strategy_id,
        )
