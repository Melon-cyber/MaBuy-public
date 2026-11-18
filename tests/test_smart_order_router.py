"""
Unit tests for Institutional Smart Order Router (SOR) - Astra Step 5.
"""

import threading
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
import pytest

from src.infrastructure.execution.smart_order_router import (
    ExecutionMode,
    ExecutionReceipt,
    ExecutionVenue,
    OrderBookTop,
    RoutingDecision,
    RoutingRequest,
    Side,
    SmartOrderRouter,
    VenueAdapter,
    VenueLiquidity,
    BinanceVenueAdapter,
    PortfolioRiskKernelAdapter,
)


class MockVenueAdapter:
    """Mock venue adapter for deterministic unit testing."""

    def __init__(
        self,
        name: str = "mock_venue",
        bid: Decimal = Decimal("100.00"),
        ask: Decimal = Decimal("100.10"),
        bid_qty: Decimal = Decimal("100.0"),
        ask_qty: Decimal = Decimal("100.0"),
        free_margin: Decimal = Decimal("1000.0"),
        tick_size: Decimal = Decimal("0.01"),
        fill_on_poll: bool = True,
        fail_stop_placement: bool = False,
    ):
        self.name = name
        self.bid = bid
        self.ask = ask
        self.bid_qty = bid_qty
        self.ask_qty = ask_qty
        self.free_margin = free_margin
        self.tick_size = tick_size
        self.fill_on_poll = fill_on_poll
        self.fail_stop_placement = fail_stop_placement

        self.configured = True
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.stop_orders: Dict[str, Dict[str, Any]] = {}
        self.cancelled_orders: List[str] = []
        self.market_orders: List[Dict[str, Any]] = []

    def is_configured(self) -> bool:
        return self.configured

    def fetch_top_of_book(self, symbol: str) -> OrderBookTop:
        return OrderBookTop(
            bid=self.bid,
            ask=self.ask,
            bid_qty=self.bid_qty,
            ask_qty=self.ask_qty,
            timestamp_ms=int(time.time() * 1000),
        )

    def fetch_free_margin(self) -> Decimal:
        return self.free_margin

    def fetch_tick_size(self, symbol: str) -> Decimal:
        return self.tick_size

    def fetch_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        order = self.orders.get(order_id)
        if not order:
            return {"id": order_id, "status": "unknown"}
        if self.fill_on_poll and order["status"] == "open":
            order["status"] = "closed"
            order["filled"] = str(order["amount"])
            order["average"] = str(order["price"])
        return order

    def place_post_only_limit(
        self,
        symbol: str,
        side: Side,
        quantity: Decimal,
        price: Decimal,
        client_order_id: str,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        order_id = f"ord-{len(self.orders)+1}"
        status = "closed" if self.fill_on_poll else "open"
        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side.value,
            "amount": float(quantity),
            "price": float(price),
            "filled": str(quantity) if status == "closed" else "0",
            "average": str(price),
            "status": status,
            "clientOrderId": client_order_id,
            "reduceOnly": reduce_only,
        }
        self.orders[order_id] = order
        return order

    def place_market_order(
        self,
        symbol: str,
        side: Side,
        quantity: Decimal,
        client_order_id: str,
        reduce_only: bool = False,
    ) -> Dict[str, Any]:
        order_id = f"mkt-{len(self.market_orders)+1}"
        order = {
            "id": order_id,
            "symbol": symbol,
            "side": side.value,
            "amount": float(quantity),
            "filled": str(quantity),
            "average": str(self.ask if side == Side.BUY else self.bid),
            "status": "closed",
            "clientOrderId": client_order_id,
            "reduceOnly": reduce_only,
        }
        self.market_orders.append(order)
        return order

    def cancel_order(self, order_id: str, symbol: str) -> None:
        self.cancelled_orders.append(order_id)
        if order_id in self.orders:
            if self.orders[order_id]["status"] != "closed":
                self.orders[order_id]["status"] = "canceled"

    def place_protective_stop(
        self,
        symbol: str,
        entry_side: Side,
        quantity: Decimal,
        stop_price: Decimal,
        client_order_id: str,
    ) -> Dict[str, Any]:
        if self.fail_stop_placement:
            raise ConnectionError("Simulated exchange stop placement failure")
        stop_id = f"stop-{len(self.stop_orders)+1}"
        stop = {
            "id": stop_id,
            "symbol": symbol,
            "side": "sell" if entry_side == Side.BUY else "buy",
            "amount": float(quantity),
            "stopPrice": float(stop_price),
            "clientOrderId": client_order_id,
            "reduceOnly": True,
        }
        self.stop_orders[stop_id] = stop
        return stop


class MockRiskKernel:
    """Mock risk kernel that tracks calls and allows approvals or vetoes."""

    def __init__(self, allow_all: bool = True, veto_reason: str = "Risk limit exceeded"):
        self.allow_all = allow_all
        self.veto_reason = veto_reason
        self.validations: List[Dict[str, Any]] = []
        self.fills: List[Dict[str, Any]] = []

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
        self.validations.append({
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "estimated_price": estimated_price,
            "stop_loss_price": stop_loss_price,
            "venue": venue,
            "strategy_id": strategy_id,
            "reduce_only": reduce_only,
        })
        if self.allow_all or reduce_only:
            return True, "Approved"
        return False, self.veto_reason

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
        self.fills.append({
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "venue": venue,
            "strategy_id": strategy_id,
        })


# ===========================================================================
# 1. Microprice Tests
# ===========================================================================

def test_microprice_symmetric():
    book = OrderBookTop(
        bid=Decimal("100.00"),
        ask=Decimal("102.00"),
        bid_qty=Decimal("10.0"),
        ask_qty=Decimal("10.0"),
        timestamp_ms=1000,
    )
    # (102 * 10 + 100 * 10) / 20 = 2020 / 20 = 101.00
    micro = SmartOrderRouter.calculate_microprice(book)
    assert micro == Decimal("101.00")


def test_microprice_asymmetric_bids_pull_toward_ask():
    book = OrderBookTop(
        bid=Decimal("100.00"),
        ask=Decimal("102.00"),
        bid_qty=Decimal("30.0"),  # Heavy buyer pressure
        ask_qty=Decimal("10.0"),
        timestamp_ms=1000,
    )
    # (102 * 30 + 100 * 10) / 40 = (3060 + 1000) / 40 = 4060 / 40 = 101.50
    micro = SmartOrderRouter.calculate_microprice(book)
    assert micro == Decimal("101.50")


def test_microprice_asymmetric_asks_pull_toward_bid():
    book = OrderBookTop(
        bid=Decimal("100.00"),
        ask=Decimal("102.00"),
        bid_qty=Decimal("10.0"),
        ask_qty=Decimal("30.0"),  # Heavy seller pressure
        timestamp_ms=1000,
    )
    # (102 * 10 + 100 * 30) / 40 = (1020 + 3000) / 40 = 4020 / 40 = 100.50
    micro = SmartOrderRouter.calculate_microprice(book)
    assert micro == Decimal("100.50")


# ===========================================================================
# 2. Maker Peg Price Bounding Tests
# ===========================================================================

def test_maker_peg_price_buy_strictly_below_ask():
    adapter = MockVenueAdapter()
    router = SmartOrderRouter(
        venues=[ExecutionVenue(name="binance", adapter=adapter, enabled=True, live_enabled=True, primary=True)],
        risk_kernel=MockRiskKernel(),
    )
    book = OrderBookTop(
        bid=Decimal("100.00"),
        ask=Decimal("100.05"),
        bid_qty=Decimal("10.0"),
        ask_qty=Decimal("10.0"),
        timestamp_ms=1000,
    )
    tick = Decimal("0.01")
    # Microprice is 100.025. Candidate rounded floor is 100.02.
    # Safe post-only cap is ask - tick = 100.04.
    peg = router._maker_peg_price(Side.BUY, Decimal("100.025"), book, tick)
    assert peg == Decimal("100.02")
    assert peg < book.ask

    # If microprice was >= ask (e.g. 100.06 due to fast book movement):
    peg_drift = router._maker_peg_price(Side.BUY, Decimal("100.06"), book, tick)
    assert peg_drift == Decimal("100.04")  # Capped at ask - tick!
    assert peg_drift < book.ask


def test_maker_peg_price_sell_strictly_above_bid():
    adapter = MockVenueAdapter()
    router = SmartOrderRouter(
        venues=[ExecutionVenue(name="binance", adapter=adapter, enabled=True, live_enabled=True, primary=True)],
        risk_kernel=MockRiskKernel(),
    )
    book = OrderBookTop(
        bid=Decimal("100.00"),
        ask=Decimal("100.05"),
        bid_qty=Decimal("10.0"),
        ask_qty=Decimal("10.0"),
        timestamp_ms=1000,
    )
    tick = Decimal("0.01")
    # Microprice is 100.025. Candidate rounded ceil is 100.03.
    # Safe post-only floor is bid + tick = 100.01.
    peg = router._maker_peg_price(Side.SELL, Decimal("100.025"), book, tick)
    assert peg == Decimal("100.03")
    assert peg > book.bid

    # If microprice was <= bid (e.g. 99.98):
    peg_drift = router._maker_peg_price(Side.SELL, Decimal("99.98"), book, tick)
    assert peg_drift == Decimal("100.01")  # Floored at bid + tick!
    assert peg_drift > book.bid


# ===========================================================================
# 3. Venue Ranking and Selection Tests
# ===========================================================================

def test_venue_cost_ranking():
    adapter_binance = MockVenueAdapter(name="binance", bid=Decimal("100.0"), ask=Decimal("100.1"))
    adapter_bybit = MockVenueAdapter(name="bybit", bid=Decimal("100.0"), ask=Decimal("100.1"))

    venue_binance = ExecutionVenue(
        name="binance",
        adapter=adapter_binance,
        enabled=True,
        live_enabled=True,
        primary=True,
        maker_fee_bps=Decimal("2.0"),
    )
    venue_bybit = ExecutionVenue(
        name="bybit",
        adapter=adapter_bybit,
        enabled=True,
        live_enabled=True,
        primary=False,
        maker_fee_bps=Decimal("1.0"),  # Lower fee -> cheaper effective cost for BUY
    )

    router = SmartOrderRouter(
        venues=[venue_binance, venue_bybit],
        risk_kernel=MockRiskKernel(),
    )

    req = RoutingRequest(
        symbol="SOL/USDT:USDT",
        side=Side.BUY,
        quantity=Decimal("1.0"),
        stop_loss_price=Decimal("95.0"),
    )

    decision = router._select_venue(req)
    # Bybit has lower maker fee, so it should be ranked first for BUY
    assert decision.selected_venue.lower() == "bybit"


def test_fallback_to_binance_when_optional_disabled():
    adapter_binance = MockVenueAdapter(name="binance")
    adapter_bybit = MockVenueAdapter(name="bybit")

    venue_binance = ExecutionVenue(
        name="binance",
        adapter=adapter_binance,
        enabled=True,
        live_enabled=True,
        primary=True,
    )
    venue_bybit = ExecutionVenue(
        name="bybit",
        adapter=adapter_bybit,
        enabled=True,
        live_enabled=False,  # Disabled / canary
        primary=False,
    )

    router = SmartOrderRouter(
        venues=[venue_binance, venue_bybit],
        risk_kernel=MockRiskKernel(),
    )

    req = RoutingRequest(
        symbol="SOL/USDT:USDT",
        side=Side.BUY,
        quantity=Decimal("1.0"),
        stop_loss_price=Decimal("95.0"),
    )

    decision = router._select_venue(req)
    assert decision.selected_venue.lower() == "binance"
    assert decision.reason == "best effective maker cost after fee/liquidity/margin checks"


# ===========================================================================
# 4. Maker-First Execution & Protective Stop Tests
# ===========================================================================

def test_maker_first_execution_success():
    adapter = MockVenueAdapter(name="binance", fill_on_poll=True)
    risk_kernel = MockRiskKernel(allow_all=True)

    venue = ExecutionVenue(
        name="binance",
        adapter=adapter,
        enabled=True,
        live_enabled=True,
        primary=True,
    )

    router = SmartOrderRouter(
        venues=[venue],
        risk_kernel=risk_kernel,
        poll_interval_seconds=0.01,
    )

    req = RoutingRequest(
        symbol="SOL/USDT:USDT",
        side=Side.BUY,
        quantity=Decimal("5.0"),
        stop_loss_price=Decimal("95.0"),
        maker_timeout_seconds=0.1,
    )

    receipt = router.route_and_execute(req)
    assert receipt.completed is True
    assert receipt.filled_quantity == Decimal("5.0")
    assert len(receipt.order_ids) >= 1
    assert len(receipt.stop_order_ids) >= 1

    # Verify risk kernel was notified of fill and protective stop placed
    assert len(risk_kernel.fills) == 1
    assert risk_kernel.fills[0]["quantity"] == Decimal("5.0")
    assert len(adapter.stop_orders) == 1
    stop = list(adapter.stop_orders.values())[0]
    assert stop["amount"] == 5.0
    assert stop["stopPrice"] == 95.0
    assert stop["side"] == "sell"


def test_protective_stop_failure_triggers_emergency_reduce():
    # Adapter where place_protective_stop throws an exception
    adapter = MockVenueAdapter(name="binance", fill_on_poll=True, fail_stop_placement=True)
    risk_kernel = MockRiskKernel(allow_all=True)

    venue = ExecutionVenue(
        name="binance",
        adapter=adapter,
        enabled=True,
        live_enabled=True,
        primary=True,
    )

    router = SmartOrderRouter(
        venues=[venue],
        risk_kernel=risk_kernel,
        poll_interval_seconds=0.01,
        emergency_close_on_stop_failure=True,
    )

    req = RoutingRequest(
        symbol="SOL/USDT:USDT",
        side=Side.BUY,
        quantity=Decimal("2.0"),
        stop_loss_price=Decimal("95.0"),
        maker_timeout_seconds=0.1,
    )

    with pytest.raises(RuntimeError) as exc_info:
        router.route_and_execute(req)

    assert "stop placement failed" in str(exc_info.value)
    # Fail-closed: emergency reduce order must have been dispatched to flatten the position
    assert len(adapter.market_orders) == 1
    emergency_order = adapter.market_orders[0]
    assert emergency_order["side"] == "sell"  # opposite of BUY
    assert emergency_order["amount"] == 2.0
    assert emergency_order["reduceOnly"] is True


# ===========================================================================
# 5. TWAP Micro-Slicing Tests
# ===========================================================================

def test_twap_execution_slices():
    adapter = MockVenueAdapter(name="binance", fill_on_poll=True)
    risk_kernel = MockRiskKernel(allow_all=True)

    venue = ExecutionVenue(
        name="binance",
        adapter=adapter,
        enabled=True,
        live_enabled=True,
        primary=True,
    )

    router = SmartOrderRouter(
        venues=[venue],
        risk_kernel=risk_kernel,
        poll_interval_seconds=0.01,
    )

    req = RoutingRequest(
        symbol="BTC/USDT:USDT",
        side=Side.BUY,
        quantity=Decimal("3.0"),
        stop_loss_price=Decimal("90000.0"),
        mode=ExecutionMode.TWAP,
        twap_slices=3,
        twap_interval_seconds=0.01,
        twap_jitter_seconds=0.005,
        maker_timeout_seconds=0.05,
    )

    receipt = router.route_and_execute(req)
    assert receipt.completed is True
    assert receipt.filled_quantity == Decimal("3.0")
    # 3 slices should produce 3 child orders and 3 protective stops
    assert len(receipt.order_ids) == 3
    assert len(receipt.stop_order_ids) == 3


# ===========================================================================
# 6. Risk Kernel Rejection Tests
# ===========================================================================

def test_risk_kernel_veto_blocks_execution():
    adapter = MockVenueAdapter(name="binance")
    risk_kernel = MockRiskKernel(allow_all=False, veto_reason="Daily drawdown limit hit")

    venue = ExecutionVenue(
        name="binance",
        adapter=adapter,
        enabled=True,
        live_enabled=True,
        primary=True,
    )

    router = SmartOrderRouter(
        venues=[venue],
        risk_kernel=risk_kernel,
    )

    req = RoutingRequest(
        symbol="SOL/USDT:USDT",
        side=Side.BUY,
        quantity=Decimal("1.0"),
        stop_loss_price=Decimal("95.0"),
    )

    with pytest.raises(PermissionError) as exc_info:
        router.route_and_execute(req)

    assert "Risk Kernel rejected execution" in str(exc_info.value)
    assert "Daily drawdown limit hit" in str(exc_info.value)
    # Zero orders sent to exchange
    assert len(adapter.orders) == 0


def test_portfolio_risk_kernel_adapter_integration():
    from src.domain.services.portfolio_risk_kernel import (
        PortfolioRiskKernel,
        RiskKernelConfig,
        PortfolioSnapshot,
    )

    kernel = PortfolioRiskKernel(RiskKernelConfig())
    # Snapshot with high equity and clean slate
    def snap_provider():
        return PortfolioSnapshot(equity_usd=100.0, day_start_equity_usd=100.0)

    adapter_risk = PortfolioRiskKernelAdapter(kernel, portfolio_snapshot_provider=snap_provider)

    venue_adapter = MockVenueAdapter(name="binance", fill_on_poll=True)
    venue = ExecutionVenue(
        name="binance",
        adapter=venue_adapter,
        enabled=True,
        live_enabled=True,
        primary=True,
    )
    router = SmartOrderRouter(venues=[venue], risk_kernel=adapter_risk, poll_interval_seconds=0.01)

    req = RoutingRequest(
        symbol="SOL/USDT:USDT",
        side=Side.BUY,
        quantity=Decimal("1.0"),
        stop_loss_price=Decimal("98.0"),  # 2% stop distance
        maker_timeout_seconds=0.05,
    )

    receipt = router.route_and_execute(req)
    assert receipt.completed is True
    assert receipt.filled_quantity == Decimal("1.0")

