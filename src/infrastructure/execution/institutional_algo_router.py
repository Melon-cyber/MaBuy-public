from __future__ import annotations

import asyncio
import logging
import math
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


LOGGER = logging.getLogger(__name__)


class Side(str, Enum):
    """Trading side for a parent or child order."""

    BUY = "buy"
    SELL = "sell"


class ExecutionAlgorithm(str, Enum):
    """Supported institutional execution algorithms."""

    ALMGREN_CHRISS = "almgren_chriss"
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"


class OrderStatus(str, Enum):
    """Venue-level order state."""

    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    UNKNOWN = "unknown"


class ParentOrderStatus(str, Enum):
    """Terminal state for a parent execution instruction."""

    COMPLETED = "completed"
    PARTIALLY_FILLED = "partially_filled"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    HALTED = "halted"


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """
    Current top-of-book market data required by the router.

    ``recent_volume`` must represent market volume observed since the previous
    market-data update. POV uses this value once per distinct timestamp and
    never submits more than ``pov_rate * recent_volume`` for that update.

    ``order_flow_imbalance`` is expected to be in [-1, 1], where:
      * Positive OFI indicates bid-side dominance.
      * Negative OFI indicates ask-side dominance.
    """

    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    recent_volume: float
    order_flow_imbalance: float
    timestamp: float = field(default_factory=time.time)
    last_price: Optional[float] = None

    @property
    def mid_price(self) -> float:
        """Return the top-of-book midpoint."""
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        """Return the absolute quoted spread."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        """Return spread width in basis points of midpoint."""
        midpoint = self.mid_price
        if midpoint <= 0.0:
            return math.inf
        return self.spread / midpoint * 10_000.0


@dataclass(frozen=True, slots=True)
class VenueOrderState:
    """
    Normalized state returned by a venue adapter.

    The adapter passed to :class:`InstitutionalAlgoRouter` must expose these
    asynchronous methods:

    * ``get_market_snapshot(symbol) -> MarketSnapshot``
    * ``submit_limit_order(symbol, side, quantity, price, post_only,
      client_order_id) -> str | Mapping``
    * ``get_order(order_id) -> VenueOrderState | Mapping``
    * ``cancel_order(order_id) -> VenueOrderState | Mapping | None``

    A submission mapping must include ``order_id``. State mappings must include
    at least ``status`` and may include ``filled_quantity`` and
    ``average_fill_price``.
    """

    order_id: str
    status: OrderStatus
    filled_quantity: float = 0.0
    average_fill_price: Optional[float] = None
    updated_at: float = field(default_factory=time.time)
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ParentOrder:
    """
    Parent execution instruction.

    ``quantity`` must be an exact multiple of ``lot_size``. ``horizon_seconds``
    is measured from the moment :meth:`InstitutionalAlgoRouter.execute` starts.
    """

    symbol: str
    side: Side
    quantity: float
    horizon_seconds: float
    algorithm: ExecutionAlgorithm
    number_of_slices: int = 20
    client_order_id: Optional[str] = None
    volume_profile: Optional[Sequence[float]] = None
    pov_rate: float = 0.05
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """
    Risk, microstructure, and market-impact configuration.

    Market-impact units use the standard continuous-time Almgren-Chriss form:

    * ``temporary_impact`` (eta): temporary impact coefficient.
    * ``permanent_impact`` (gamma): permanent impact coefficient.
    * ``risk_aversion`` (lambda): inventory risk-aversion parameter.
    * ``volatility_per_sqrt_second`` (sigma): volatility expressed per
      square-root second.

    For deterministic liquidation, the permanent impact term contributes a
    fixed expected cost for a fully completed parent and therefore does not
    alter the classical AC trajectory's kappa. It is still included in receipt
    cost estimates.
    """

    max_spread_bps: float = 12.0
    max_adverse_ofi: float = 0.35
    max_quote_age_seconds: float = 2.0
    post_only_timeout_seconds: float = 1.0
    poll_interval_seconds: float = 0.10
    lot_size: float = 0.000001
    tick_size: float = 0.01
    twap_jitter_fraction: float = 0.25
    risk_aversion: float = 1.0e-6
    volatility_per_sqrt_second: float = 0.0005
    temporary_impact: float = 1.0e-7
    permanent_impact: float = 1.0e-8
    random_seed: Optional[int] = None


@dataclass(frozen=True, slots=True)
class ChildExecution:
    """Immutable execution record for one submitted child order."""

    child_order_id: str
    venue_order_id: Optional[str]
    symbol: str
    side: Side
    requested_quantity: float
    filled_quantity: float
    limit_price: Optional[float]
    average_fill_price: Optional[float]
    status: OrderStatus
    submitted_at: float
    completed_at: float
    note: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionReceipt:
    """Terminal parent-order execution receipt with realized slippage metrics."""

    parent_order_id: str
    symbol: str
    side: Side
    algorithm: ExecutionAlgorithm
    requested_quantity: float
    filled_quantity: float
    remaining_quantity: float
    arrival_mid_price: Optional[float]
    average_fill_price: Optional[float]
    signed_slippage_bps: Optional[float]
    estimated_temporary_impact_price: float
    estimated_permanent_impact_price: float
    status: ParentOrderStatus
    started_at: float
    completed_at: float
    child_executions: tuple[ChildExecution, ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the receipt into built-in Python types."""
        payload = asdict(self)
        payload["side"] = self.side.value
        payload["algorithm"] = self.algorithm.value
        payload["status"] = self.status.value
        payload["child_executions"] = [
            {
                **asdict(child),
                "side": child.side.value,
                "status": child.status.value,
            }
            for child in self.child_executions
        ]
        return payload


@dataclass(frozen=True, slots=True)
class _ChildOutcome:
    """Internal child execution result."""

    execution: ChildExecution
    halt_parent: bool = False


class InstitutionalAlgoRouter:
    """
    Institutional execution router implementing AC, TWAP, VWAP, and POV.

    The router is designed for asynchronous venue adapters and serializes
    execution for each symbol. This prevents independent parent orders from
    competing for the same top-of-book liquidity through the same router.

    The supplied venue adapter is intentionally duck-typed to avoid coupling
    this module to a specific exchange SDK. It must implement the methods
    described by :class:`VenueOrderState`.

    Example
    -------
    >>> router = InstitutionalAlgoRouter(venue)
    >>> receipt = await router.execute(
    ...     ParentOrder(
    ...         symbol="BTC-USD",
    ...         side=Side.SELL,
    ...         quantity=0.25,
    ...         horizon_seconds=30.0,
    ...         algorithm=ExecutionAlgorithm.TWAP,
    ...     )
    ... )
    """

    def __init__(self, venue: Any, config: Optional[RouterConfig] = None) -> None:
        """
        Initialize the router.

        Parameters
        ----------
        venue:
            Asynchronous venue adapter implementing market-data and order
            lifecycle methods documented in :class:`VenueOrderState`.
        config:
            Router risk and market-impact configuration.
        """
        required_methods = (
            "get_market_snapshot",
            "submit_limit_order",
            "get_order",
            "cancel_order",
        )
        missing = [name for name in required_methods if not callable(getattr(venue, name, None))]
        if missing:
            raise TypeError(
                "venue adapter is missing required asynchronous methods: "
                + ", ".join(missing)
            )

        self._venue = venue
        self._config = config or RouterConfig()
        self._validate_config(self._config)

        self._symbol_locks: dict[str, asyncio.Lock] = {}
        self._symbol_locks_guard = asyncio.Lock()
        self._receipts: dict[str, ExecutionReceipt] = {}
        self._receipts_lock = asyncio.Lock()

        self._rng: random.Random
        if self._config.random_seed is None:
            self._rng = random.SystemRandom()
        else:
            self._rng = random.Random(self._config.random_seed)

    @property
    def config(self) -> RouterConfig:
        """Return the immutable router configuration."""
        return self._config

    async def execute(
        self,
        order: ParentOrder,
        cancel_event: Optional[asyncio.Event] = None,
    ) -> ExecutionReceipt:
        """
        Execute a parent order and return a final execution receipt.

        The router evaluates spread and OFI toxicity immediately before every
        child dispatch. When market conditions are toxic, dispatch pauses until
        conditions normalize, the parent horizon expires, or ``cancel_event``
        is set.

        Post-only children are placed at the best bid for buys and best ask for
        sells. Any unfilled remainder is cancelled after
        ``post_only_timeout_seconds``.

        Parameters
        ----------
        order:
            Parent instruction to execute.
        cancel_event:
            Optional asynchronous cancellation signal. Existing live child
            orders are cancelled before the receipt is produced.

        Returns
        -------
        ExecutionReceipt
            Immutable aggregate execution report.
        """
        self._validate_order(order)
        parent_id = order.client_order_id or f"iar-{uuid.uuid4().hex}"
        symbol_lock = await self._get_symbol_lock(order.symbol)

        async with symbol_lock:
            started_wall = time.time()
            started_mono = time.monotonic()
            deadline = started_mono + order.horizon_seconds
            child_executions: list[ChildExecution] = []
            halted = False
            cancelled = False

            arrival_snapshot = await self._try_get_snapshot(order.symbol)
            arrival_mid = (
                arrival_snapshot.mid_price if arrival_snapshot is not None else None
            )

            LOGGER.info(
                "parent_execution_started parent_id=%s symbol=%s side=%s algo=%s qty=%.10f horizon=%.3fs",
                parent_id,
                order.symbol,
                order.side.value,
                order.algorithm.value,
                order.quantity,
                order.horizon_seconds,
            )

            if self._is_cancelled(cancel_event):
                cancelled = True
            elif order.algorithm is ExecutionAlgorithm.POV:
                child_executions, halted, cancelled = await self._execute_pov(
                    parent_id=parent_id,
                    order=order,
                    deadline=deadline,
                    cancel_event=cancel_event,
                )
            else:
                plan = self._build_scheduled_plan(order)
                child_executions, halted, cancelled = await self._execute_scheduled_plan(
                    parent_id=parent_id,
                    order=order,
                    plan=plan,
                    started_mono=started_mono,
                    deadline=deadline,
                    cancel_event=cancel_event,
                )

            receipt = self._build_receipt(
                parent_id=parent_id,
                order=order,
                arrival_mid=arrival_mid,
                started_at=started_wall,
                children=child_executions,
                halted=halted,
                cancelled=cancelled,
            )
            await self._store_receipt(receipt)

            LOGGER.info(
                "parent_execution_finished parent_id=%s status=%s filled=%.10f remaining=%.10f avg_fill=%s slippage_bps=%s",
                receipt.parent_order_id,
                receipt.status.value,
                receipt.filled_quantity,
                receipt.remaining_quantity,
                (
                    f"{receipt.average_fill_price:.8f}"
                    if receipt.average_fill_price is not None
                    else "none"
                ),
                (
                    f"{receipt.signed_slippage_bps:.4f}"
                    if receipt.signed_slippage_bps is not None
                    else "none"
                ),
            )
            return receipt

    async def get_receipt(self, parent_order_id: str) -> Optional[ExecutionReceipt]:
        """Return a previously generated execution receipt, if available."""
        async with self._receipts_lock:
            return self._receipts.get(parent_order_id)

    async def _get_symbol_lock(self, symbol: str) -> asyncio.Lock:
        """Get or atomically create the serial execution lock for a symbol."""
        async with self._symbol_locks_guard:
            lock = self._symbol_locks.get(symbol)
            if lock is None:
                lock = asyncio.Lock()
                self._symbol_locks[symbol] = lock
            return lock

    async def _store_receipt(self, receipt: ExecutionReceipt) -> None:
        """Store a completed receipt atomically."""
        async with self._receipts_lock:
            self._receipts[receipt.parent_order_id] = receipt

    async def _execute_scheduled_plan(
        self,
        parent_id: str,
        order: ParentOrder,
        plan: Sequence[tuple[float, float]],
        started_mono: float,
        deadline: float,
        cancel_event: Optional[asyncio.Event],
    ) -> tuple[list[ChildExecution], bool, bool]:
        """Execute a deterministic time-indexed AC, TWAP, or VWAP schedule."""
        children: list[ChildExecution] = []
        halted = False
        cancelled = False

        for sequence, (offset_seconds, quantity) in enumerate(plan, start=1):
            if quantity < self._config.lot_size * 0.5:
                continue

            target_time = started_mono + offset_seconds
            wait_result = await self._sleep_until(target_time, cancel_event)
            if wait_result:
                cancelled = True
                break

            if time.monotonic() >= deadline:
                break

            snapshot = await self._wait_for_safe_market(
                order=order,
                deadline=deadline,
                cancel_event=cancel_event,
            )
            if snapshot is None:
                cancelled = self._is_cancelled(cancel_event)
                break

            outcome = await self._execute_post_only_child(
                parent_id=parent_id,
                sequence=sequence,
                order=order,
                quantity=quantity,
                snapshot=snapshot,
                deadline=deadline,
            )
            children.append(outcome.execution)
            if outcome.halt_parent:
                halted = True
                break

        return children, halted, cancelled

    async def _execute_pov(
        self,
        parent_id: str,
        order: ParentOrder,
        deadline: float,
        cancel_event: Optional[asyncio.Event],
    ) -> tuple[list[ChildExecution], bool, bool]:
        """
        Execute a Percentage-of-Volume strategy.

        A quote update's recent market volume is consumed only once. This
        enforces the participation cap even if the router's polling loop runs
        faster than the market-data feed updates.
        """
        children: list[ChildExecution] = []
        filled_quantity = 0.0
        sequence = 0
        last_volume_timestamp: Optional[float] = None
        halted = False
        cancelled = False

        while time.monotonic() < deadline:
            if self._is_cancelled(cancel_event):
                cancelled = True
                break

            remaining = max(0.0, order.quantity - filled_quantity)
            if remaining < self._config.lot_size * 0.5:
                break

            snapshot = await self._wait_for_safe_market(
                order=order,
                deadline=deadline,
                cancel_event=cancel_event,
            )
            if snapshot is None:
                cancelled = self._is_cancelled(cancel_event)
                break

            if (
                last_volume_timestamp is not None
                and snapshot.timestamp <= last_volume_timestamp
            ):
                if await self._sleep_for(
                    self._config.poll_interval_seconds,
                    cancel_event,
                ):
                    cancelled = True
                    break
                continue

            last_volume_timestamp = snapshot.timestamp
            participation_quantity = self._round_down_quantity(
                max(0.0, order.pov_rate * snapshot.recent_volume)
            )

            if participation_quantity < self._config.lot_size * 0.5:
                LOGGER.debug(
                    "pov_no_available_volume parent_id=%s symbol=%s recent_volume=%.10f pov_rate=%.6f",
                    parent_id,
                    order.symbol,
                    snapshot.recent_volume,
                    order.pov_rate,
                )
                if await self._sleep_for(
                    self._config.poll_interval_seconds,
                    cancel_event,
                ):
                    cancelled = True
                    break
                continue

            quantity = min(remaining, participation_quantity)
            quantity = self._round_down_quantity(quantity)
            if quantity < self._config.lot_size * 0.5:
                break

            sequence += 1
            outcome = await self._execute_post_only_child(
                parent_id=parent_id,
                sequence=sequence,
                order=order,
                quantity=quantity,
                snapshot=snapshot,
                deadline=deadline,
            )
            children.append(outcome.execution)
            filled_quantity += outcome.execution.filled_quantity

            if outcome.halt_parent:
                halted = True
                break

        return children, halted, cancelled

    def _build_scheduled_plan(self, order: ParentOrder) -> list[tuple[float, float]]:
        """Build time offsets and child quantities for AC, TWAP, or VWAP."""
        if order.algorithm is ExecutionAlgorithm.ALMGREN_CHRISS:
            weights = self._almgren_chriss_weights(
                number_of_slices=order.number_of_slices,
                horizon_seconds=order.horizon_seconds,
            )
            quantities = self._allocate_quantities(order.quantity, weights)
            interval = order.horizon_seconds / order.number_of_slices
            return [
                (index * interval, quantity)
                for index, quantity in enumerate(quantities)
            ]

        if order.algorithm is ExecutionAlgorithm.TWAP:
            weights = [1.0] * order.number_of_slices
            quantities = self._allocate_quantities(order.quantity, weights)
            offsets = self._twap_offsets(
                number_of_slices=order.number_of_slices,
                horizon_seconds=order.horizon_seconds,
            )
            return list(zip(offsets, quantities))

        if order.algorithm is ExecutionAlgorithm.VWAP:
            profile = self._volume_profile(order)
            quantities = self._allocate_quantities(order.quantity, profile)
            interval = order.horizon_seconds / len(profile)
            offsets = [
                min(
                    order.horizon_seconds - 0.001,
                    max(0.0, (index + 0.5) * interval),
                )
                for index in range(len(profile))
            ]
            return list(zip(offsets, quantities))

        raise ValueError(f"scheduled plan unsupported for algorithm {order.algorithm!r}")

    def _almgren_chriss_weights(
        self,
        number_of_slices: int,
        horizon_seconds: float,
    ) -> list[float]:
        """
        Return Almgren-Chriss child-order weights.

        The remaining inventory follows:

        ``x_j = X sinh(kappa * (T - t_j)) / sinh(kappa * T)``

        where ``kappa = sqrt(lambda * sigma^2 / eta)``. For negligible kappa,
        the stable limiting solution is TWAP.
        """
        eta = self._config.temporary_impact
        lam = self._config.risk_aversion
        sigma = self._config.volatility_per_sqrt_second

        kappa_squared = lam * sigma * sigma / eta
        kappa = math.sqrt(max(0.0, kappa_squared))
        if kappa * horizon_seconds < 1.0e-7:
            return [1.0] * number_of_slices

        denominator = math.sinh(kappa * horizon_seconds)
        remaining_fractions = [1.0]
        for index in range(1, number_of_slices + 1):
            t_j = horizon_seconds * index / number_of_slices
            remaining = math.sinh(kappa * (horizon_seconds - t_j)) / denominator
            remaining_fractions.append(max(0.0, remaining))

        weights = [
            max(0.0, remaining_fractions[index] - remaining_fractions[index + 1])
            for index in range(number_of_slices)
        ]
        if sum(weights) <= 0.0:
            return [1.0] * number_of_slices
        return weights

    def _twap_offsets(
        self,
        number_of_slices: int,
        horizon_seconds: float,
    ) -> list[float]:
        """
        Build monotonic, randomized TWAP dispatch offsets.

        Jitter is bounded to a fraction of one nominal slice interval and
        prevents an externally visible, perfectly periodic child-order cadence.
        """
        interval = horizon_seconds / number_of_slices
        max_jitter = interval * self._config.twap_jitter_fraction
        offsets: list[float] = []

        for index in range(number_of_slices):
            base = (index + 0.5) * interval
            jitter = self._rng.uniform(-max_jitter, max_jitter)
            candidate = min(
                horizon_seconds - 0.001,
                max(0.0, base + jitter),
            )
            if offsets:
                candidate = max(candidate, offsets[-1] + 0.0001)
            offsets.append(min(candidate, horizon_seconds - 0.0001))

        return offsets

    def _volume_profile(self, order: ParentOrder) -> list[float]:
        """
        Return normalized historical volume weights.

        If a profile is not supplied, this uses a smooth U-shaped crypto
        intraday profile with elevated activity near the start and end of the
        execution horizon.
        """
        if order.volume_profile is not None:
            profile = [float(value) for value in order.volume_profile]
            if not profile or any(value < 0.0 or not math.isfinite(value) for value in profile):
                raise ValueError("volume_profile must contain finite non-negative values")
            if sum(profile) <= 0.0:
                raise ValueError("volume_profile must contain at least one positive value")
            return profile

        profile: list[float] = []
        count = order.number_of_slices
        for index in range(count):
            normalized_time = (index + 0.5) / count
            edge_distance = abs(2.0 * normalized_time - 1.0)
            profile.append(0.65 + 1.35 * edge_distance**1.6)
        return profile

    def _allocate_quantities(
        self,
        total_quantity: float,
        weights: Sequence[float],
    ) -> list[float]:
        """Allocate an exact parent quantity across weights in lot-size units."""
        if not weights or sum(weights) <= 0.0:
            raise ValueError("allocation weights must contain positive total weight")

        lot = self._config.lot_size
        total_lots = int(round(total_quantity / lot))
        raw_lots = [total_lots * max(0.0, weight) / sum(weights) for weight in weights]
        allocated_lots = [int(math.floor(value)) for value in raw_lots]
        remainder = total_lots - sum(allocated_lots)

        ranked_fractional = sorted(
            range(len(weights)),
            key=lambda index: raw_lots[index] - allocated_lots[index],
            reverse=True,
        )
        for index in ranked_fractional[:remainder]:
            allocated_lots[index] += 1

        return [lots * lot for lots in allocated_lots]

    async def _wait_for_safe_market(
        self,
        order: ParentOrder,
        deadline: float,
        cancel_event: Optional[asyncio.Event],
    ) -> Optional[MarketSnapshot]:
        """
        Poll market data until a quote passes toxicity checks or deadline passes.

        Toxic conditions pause dispatch rather than crossing the spread,
        preserving alpha when adverse selection risk is elevated.
        """
        while time.monotonic() < deadline:
            if self._is_cancelled(cancel_event):
                return None

            snapshot = await self._try_get_snapshot(order.symbol)
            if snapshot is not None:
                reason = self._toxicity_reason(order.side, snapshot)
                if reason is None:
                    return snapshot

                LOGGER.info(
                    "child_dispatch_paused symbol=%s side=%s reason=%s spread_bps=%.4f ofi=%.4f",
                    order.symbol,
                    order.side.value,
                    reason,
                    snapshot.spread_bps,
                    snapshot.order_flow_imbalance,
                )

            if await self._sleep_for(self._config.poll_interval_seconds, cancel_event):
                return None

        return None

    def _toxicity_reason(
        self,
        side: Side,
        snapshot: MarketSnapshot,
    ) -> Optional[str]:
        """Return a human-readable toxicity reason or ``None`` if safe."""
        if not self._valid_snapshot(snapshot):
            return "invalid_or_stale_quote"

        if snapshot.spread_bps > self._config.max_spread_bps:
            return "spread_wider_than_threshold"

        if (
            side is Side.BUY
            and snapshot.order_flow_imbalance <= -self._config.max_adverse_ofi
        ):
            return "adverse_sell_order_flow_imbalance"

        if (
            side is Side.SELL
            and snapshot.order_flow_imbalance >= self._config.max_adverse_ofi
        ):
            return "adverse_buy_order_flow_imbalance"

        return None

    def _valid_snapshot(self, snapshot: MarketSnapshot) -> bool:
        """Validate quote integrity and market-data freshness."""
        values = (
            snapshot.bid,
            snapshot.ask,
            snapshot.bid_size,
            snapshot.ask_size,
            snapshot.recent_volume,
            snapshot.order_flow_imbalance,
            snapshot.timestamp,
        )
        if any(not math.isfinite(value) for value in values):
            return False
        if snapshot.bid <= 0.0 or snapshot.ask <= 0.0 or snapshot.ask < snapshot.bid:
            return False
        if snapshot.bid_size < 0.0 or snapshot.ask_size < 0.0:
            return False
        if snapshot.recent_volume < 0.0:
            return False
        if not -1.0 <= snapshot.order_flow_imbalance <= 1.0:
            return False
        return (time.time() - snapshot.timestamp) <= self._config.max_quote_age_seconds

    async def _execute_post_only_child(
        self,
        parent_id: str,
        sequence: int,
        order: ParentOrder,
        quantity: float,
        snapshot: MarketSnapshot,
        deadline: float,
    ) -> _ChildOutcome:
        """
        Submit a maker-first child at the best quote and manage its lifecycle.

        Any adapter uncertainty after a successful submit is treated as a
        parent-level safety halt. The router does not continue dispatching
        exposure when an outstanding order cannot be reliably reconciled.
        """
        now_wall = time.time()
        child_id = f"{parent_id}-{sequence:04d}"
        limit_price = self._post_only_price(order.side, snapshot)

        try:
            response = await self._venue.submit_limit_order(
                symbol=order.symbol,
                side=order.side.value,
                quantity=quantity,
                price=limit_price,
                post_only=True,
                client_order_id=child_id,
            )
            venue_order_id = self._extract_order_id(response)
        except Exception as exc:
            LOGGER.exception(
                "child_submit_uncertain parent_id=%s child_id=%s symbol=%s",
                parent_id,
                child_id,
                order.symbol,
            )
            execution = ChildExecution(
                child_order_id=child_id,
                venue_order_id=None,
                symbol=order.symbol,
                side=order.side,
                requested_quantity=quantity,
                filled_quantity=0.0,
                limit_price=limit_price,
                average_fill_price=None,
                status=OrderStatus.UNKNOWN,
                submitted_at=now_wall,
                completed_at=time.time(),
                note=f"submit exception: {exc!r}",
            )
            return _ChildOutcome(execution=execution, halt_parent=True)

        LOGGER.debug(
            "post_only_child_submitted parent_id=%s child_id=%s venue_order_id=%s side=%s qty=%.10f price=%.10f",
            parent_id,
            child_id,
            venue_order_id,
            order.side.value,
            quantity,
            limit_price,
        )

        timeout_deadline = min(
            deadline,
            time.monotonic() + self._config.post_only_timeout_seconds,
        )
        final_state: Optional[VenueOrderState] = None

        while time.monotonic() < timeout_deadline:
            try:
                state = self._coerce_venue_state(
                    await self._venue.get_order(venue_order_id),
                    venue_order_id,
                )
            except Exception as exc:
                LOGGER.exception(
                    "child_state_uncertain parent_id=%s child_id=%s venue_order_id=%s",
                    parent_id,
                    child_id,
                    venue_order_id,
                )
                await self._best_effort_cancel(venue_order_id)
                execution = ChildExecution(
                    child_order_id=child_id,
                    venue_order_id=venue_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    requested_quantity=quantity,
                    filled_quantity=0.0,
                    limit_price=limit_price,
                    average_fill_price=None,
                    status=OrderStatus.UNKNOWN,
                    submitted_at=now_wall,
                    completed_at=time.time(),
                    note=f"state reconciliation exception: {exc!r}",
                )
                return _ChildOutcome(execution=execution, halt_parent=True)

            final_state = state
            if self._is_terminal_order_status(state.status):
                break

            await asyncio.sleep(
                min(
                    self._config.poll_interval_seconds,
                    max(0.001, timeout_deadline - time.monotonic()),
                )
            )

        if final_state is None or not self._is_terminal_order_status(final_state.status):
            try:
                cancel_response = await self._venue.cancel_order(venue_order_id)
                if cancel_response is not None:
                    final_state = self._coerce_venue_state(cancel_response, venue_order_id)
                else:
                    final_state = self._coerce_venue_state(
                        await self._venue.get_order(venue_order_id),
                        venue_order_id,
                    )
            except Exception as exc:
                LOGGER.exception(
                    "child_cancel_uncertain parent_id=%s child_id=%s venue_order_id=%s",
                    parent_id,
                    child_id,
                    venue_order_id,
                )
                execution = ChildExecution(
                    child_order_id=child_id,
                    venue_order_id=venue_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    requested_quantity=quantity,
                    filled_quantity=0.0,
                    limit_price=limit_price,
                    average_fill_price=None,
                    status=OrderStatus.UNKNOWN,
                    submitted_at=now_wall,
                    completed_at=time.time(),
                    note=f"cancel reconciliation exception: {exc!r}",
                )
                return _ChildOutcome(execution=execution, halt_parent=True)

        assert final_state is not None
        filled_quantity = min(quantity, max(0.0, final_state.filled_quantity))
        average_fill_price = final_state.average_fill_price
        if filled_quantity <= 0.0:
            average_fill_price = None

        execution = ChildExecution(
            child_order_id=child_id,
            venue_order_id=venue_order_id,
            symbol=order.symbol,
            side=order.side,
            requested_quantity=quantity,
            filled_quantity=filled_quantity,
            limit_price=limit_price,
            average_fill_price=average_fill_price,
            status=final_state.status,
            submitted_at=now_wall,
            completed_at=time.time(),
            note=final_state.reason,
        )

        LOGGER.debug(
            "post_only_child_completed parent_id=%s child_id=%s status=%s filled=%.10f requested=%.10f",
            parent_id,
            child_id,
            execution.status.value,
            execution.filled_quantity,
            execution.requested_quantity,
        )
        return _ChildOutcome(execution=execution, halt_parent=False)

    async def _best_effort_cancel(self, venue_order_id: str) -> None:
        """Attempt cancellation without masking a prior reconciliation failure."""
        try:
            await self._venue.cancel_order(venue_order_id)
        except Exception:
            LOGGER.exception(
                "best_effort_cancel_failed venue_order_id=%s",
                venue_order_id,
            )

    def _post_only_price(self, side: Side, snapshot: MarketSnapshot) -> float:
        """Return a tick-normalized best quote that remains post-only."""
        tick = self._config.tick_size
        if side is Side.BUY:
            return math.floor((snapshot.bid + 1.0e-12) / tick) * tick
        return math.ceil((snapshot.ask - 1.0e-12) / tick) * tick

    async def _try_get_snapshot(self, symbol: str) -> Optional[MarketSnapshot]:
        """Fetch and normalize a snapshot, returning ``None`` on transient errors."""
        try:
            raw_snapshot = await self._venue.get_market_snapshot(symbol)
            return self._coerce_snapshot(raw_snapshot, symbol)
        except Exception:
            LOGGER.exception("market_snapshot_fetch_failed symbol=%s", symbol)
            return None

    @staticmethod
    def _coerce_snapshot(raw: Any, symbol: str) -> MarketSnapshot:
        """Normalize a venue snapshot object or mapping."""
        if isinstance(raw, MarketSnapshot):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError("market snapshot must be MarketSnapshot or Mapping")

        return MarketSnapshot(
            symbol=str(raw.get("symbol", symbol)),
            bid=float(raw["bid"]),
            ask=float(raw["ask"]),
            bid_size=float(raw.get("bid_size", 0.0)),
            ask_size=float(raw.get("ask_size", 0.0)),
            recent_volume=float(raw.get("recent_volume", 0.0)),
            order_flow_imbalance=float(raw.get("order_flow_imbalance", 0.0)),
            timestamp=float(raw.get("timestamp", time.time())),
            last_price=(
                float(raw["last_price"])
                if raw.get("last_price") is not None
                else None
            ),
        )

    @staticmethod
    def _extract_order_id(response: Any) -> str:
        """Extract a non-empty venue order identifier from submit response."""
        if isinstance(response, str) and response:
            return response
        if isinstance(response, Mapping):
            order_id = response.get("order_id")
            if order_id:
                return str(order_id)
        raise ValueError("submit_limit_order response does not contain order_id")

    @staticmethod
    def _coerce_venue_state(raw: Any, fallback_order_id: str) -> VenueOrderState:
        """Normalize a venue order state object or mapping."""
        if isinstance(raw, VenueOrderState):
            return raw
        if not isinstance(raw, Mapping):
            raise TypeError("venue order state must be VenueOrderState or Mapping")

        raw_status = str(raw.get("status", OrderStatus.UNKNOWN.value)).lower()
        try:
            status = OrderStatus(raw_status)
        except ValueError:
            status = OrderStatus.UNKNOWN

        average_fill_price = raw.get("average_fill_price")
        return VenueOrderState(
            order_id=str(raw.get("order_id", fallback_order_id)),
            status=status,
            filled_quantity=float(raw.get("filled_quantity", 0.0)),
            average_fill_price=(
                float(average_fill_price)
                if average_fill_price is not None
                else None
            ),
            updated_at=float(raw.get("updated_at", time.time())),
            reason=str(raw.get("reason", "")),
        )

    @staticmethod
    def _is_terminal_order_status(status: OrderStatus) -> bool:
        """Return whether a venue order state is terminal."""
        return status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.UNKNOWN,
        }

    def _build_receipt(
        self,
        parent_id: str,
        order: ParentOrder,
        arrival_mid: Optional[float],
        started_at: float,
        children: Sequence[ChildExecution],
        halted: bool,
        cancelled: bool,
    ) -> ExecutionReceipt:
        """Aggregate child fills, estimated impact, and realized slippage."""
        filled_quantity = sum(child.filled_quantity for child in children)
        filled_quantity = min(order.quantity, max(0.0, filled_quantity))
        remaining_quantity = max(0.0, order.quantity - filled_quantity)

        fill_notional = sum(
            child.filled_quantity * child.average_fill_price
            for child in children
            if child.average_fill_price is not None
        )
        average_fill_price = (
            fill_notional / filled_quantity if filled_quantity > 0.0 else None
        )

        signed_slippage_bps: Optional[float] = None
        if arrival_mid is not None and average_fill_price is not None and arrival_mid > 0.0:
            direction = 1.0 if order.side is Side.BUY else -1.0
            signed_slippage_bps = (
                direction
                * (average_fill_price - arrival_mid)
                / arrival_mid
                * 10_000.0
            )

        duration = max(0.001, time.time() - started_at)
        estimated_temporary_impact = 0.0
        for child in children:
            if child.filled_quantity > 0.0:
                child_rate = child.filled_quantity / duration
                estimated_temporary_impact += (
                    self._config.temporary_impact * child_rate
                )

        estimated_permanent_impact = (
            self._config.permanent_impact * filled_quantity / 2.0
        )

        if halted:
            status = ParentOrderStatus.HALTED
        elif cancelled:
            status = ParentOrderStatus.CANCELLED
        elif remaining_quantity < self._config.lot_size * 0.5:
            status = ParentOrderStatus.COMPLETED
        elif filled_quantity > 0.0:
            status = ParentOrderStatus.PARTIALLY_FILLED
        else:
            status = ParentOrderStatus.EXPIRED

        return ExecutionReceipt(
            parent_order_id=parent_id,
            symbol=order.symbol,
            side=order.side,
            algorithm=order.algorithm,
            requested_quantity=order.quantity,
            filled_quantity=filled_quantity,
            remaining_quantity=remaining_quantity,
            arrival_mid_price=arrival_mid,
            average_fill_price=average_fill_price,
            signed_slippage_bps=signed_slippage_bps,
            estimated_temporary_impact_price=estimated_temporary_impact,
            estimated_permanent_impact_price=estimated_permanent_impact,
            status=status,
            started_at=started_at,
            completed_at=time.time(),
            child_executions=tuple(children),
            metadata=dict(order.metadata),
        )

    async def _sleep_until(
        self,
        target_monotonic: float,
        cancel_event: Optional[asyncio.Event],
    ) -> bool:
        """Sleep until target time; return True when cancellation is requested."""
        return await self._sleep_for(
            max(0.0, target_monotonic - time.monotonic()),
            cancel_event,
        )

    @staticmethod
    async def _sleep_for(
        duration_seconds: float,
        cancel_event: Optional[asyncio.Event],
    ) -> bool:
        """Sleep interruptibly; return True if cancellation event fired."""
        if duration_seconds <= 0.0:
            return cancel_event is not None and cancel_event.is_set()

        if cancel_event is None:
            await asyncio.sleep(duration_seconds)
            return False

        if cancel_event.is_set():
            return True

        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=duration_seconds)
            return True
        except asyncio.TimeoutError:
            return False

    @staticmethod
    def _is_cancelled(cancel_event: Optional[asyncio.Event]) -> bool:
        """Return whether a caller requested parent cancellation."""
        return cancel_event is not None and cancel_event.is_set()

    def _round_down_quantity(self, quantity: float) -> float:
        """Round a positive quantity down to configured lot size."""
        if quantity <= 0.0:
            return 0.0
        lot = self._config.lot_size
        return math.floor((quantity + 1.0e-12) / lot) * lot

    def _validate_order(self, order: ParentOrder) -> None:
        """Validate parent-order invariants before any venue interaction."""
        if not isinstance(order.side, Side):
            raise TypeError("order.side must be a Side")
        if not isinstance(order.algorithm, ExecutionAlgorithm):
            raise TypeError("order.algorithm must be an ExecutionAlgorithm")
        if not order.symbol or not order.symbol.strip():
            raise ValueError("order.symbol must be non-empty")
        if not math.isfinite(order.quantity) or order.quantity <= 0.0:
            raise ValueError("order.quantity must be finite and positive")
        if not math.isfinite(order.horizon_seconds) or order.horizon_seconds <= 0.0:
            raise ValueError("order.horizon_seconds must be finite and positive")
        if order.number_of_slices <= 0:
            raise ValueError("order.number_of_slices must be positive")
        if order.algorithm is ExecutionAlgorithm.POV and not 0.0 < order.pov_rate <= 1.0:
            raise ValueError("POV order.pov_rate must be in (0, 1]")

        lots = order.quantity / self._config.lot_size
        if not math.isclose(lots, round(lots), rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError(
                f"order.quantity must be an exact multiple of lot_size={self._config.lot_size}"
            )

        if order.volume_profile is not None and order.algorithm is not ExecutionAlgorithm.VWAP:
            LOGGER.warning(
                "volume_profile_ignored_for_non_vwap_order symbol=%s algorithm=%s",
                order.symbol,
                order.algorithm.value,
            )

    @staticmethod
    def _validate_config(config: RouterConfig) -> None:
        """Validate immutable router configuration."""
        positive_values = {
            "max_spread_bps": config.max_spread_bps,
            "max_quote_age_seconds": config.max_quote_age_seconds,
            "post_only_timeout_seconds": config.post_only_timeout_seconds,
            "poll_interval_seconds": config.poll_interval_seconds,
            "lot_size": config.lot_size,
            "tick_size": config.tick_size,
            "temporary_impact": config.temporary_impact,
        }
        for name, value in positive_values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"config.{name} must be finite and positive")

        if not 0.0 <= config.max_adverse_ofi <= 1.0:
            raise ValueError("config.max_adverse_ofi must be in [0, 1]")
        if not 0.0 <= config.twap_jitter_fraction < 0.5:
            raise ValueError("config.twap_jitter_fraction must be in [0, 0.5)")
        if not math.isfinite(config.risk_aversion) or config.risk_aversion < 0.0:
            raise ValueError("config.risk_aversion must be finite and non-negative")
        if (
            not math.isfinite(config.volatility_per_sqrt_second)
            or config.volatility_per_sqrt_second < 0.0
        ):
            raise ValueError(
                "config.volatility_per_sqrt_second must be finite and non-negative"
            )
        if not math.isfinite(config.permanent_impact) or config.permanent_impact < 0.0:
            raise ValueError("config.permanent_impact must be finite and non-negative")


class _SimulatedVenue:
    """
    Small asynchronous venue used only by the module's executable simulation.

    It fills part of a post-only order after a short queue delay and fills the
    balance after another delay unless the router cancels the order first.
    """

    def __init__(self) -> None:
        self._orders: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._counter = 0

    async def get_market_snapshot(self, symbol: str) -> MarketSnapshot:
        now = time.time()
        phase = now * 2.0
        midpoint = 60_000.0 + math.sin(phase) * 8.0
        spread = 1.50 + abs(math.sin(phase * 0.7)) * 0.5
        return MarketSnapshot(
            symbol=symbol,
            bid=midpoint - spread / 2.0,
            ask=midpoint + spread / 2.0,
            bid_size=4.0,
            ask_size=4.5,
            recent_volume=0.20 + abs(math.sin(phase)) * 0.30,
            order_flow_imbalance=math.sin(phase * 0.3) * 0.15,
            timestamp=now,
            last_price=midpoint,
        )

    async def submit_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        post_only: bool,
        client_order_id: str,
    ) -> str:
        if not post_only:
            raise ValueError("simulation accepts post-only orders only")

        async with self._lock:
            self._counter += 1
            order_id = f"sim-{self._counter}"
            self._orders[order_id] = {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "created": time.monotonic(),
                "cancelled": False,
            }
            return order_id

    async def get_order(self, order_id: str) -> VenueOrderState:
        async with self._lock:
            order = self._orders[order_id]
            elapsed = time.monotonic() - order["created"]
            quantity = order["quantity"]

            if order["cancelled"]:
                filled = quantity * 0.50 if elapsed >= 0.05 else 0.0
                return VenueOrderState(
                    order_id=order_id,
                    status=OrderStatus.CANCELLED,
                    filled_quantity=filled,
                    average_fill_price=order["price"] if filled > 0.0 else None,
                    reason="cancelled by router timeout",
                )

            if elapsed >= 0.16:
                return VenueOrderState(
                    order_id=order_id,
                    status=OrderStatus.FILLED,
                    filled_quantity=quantity,
                    average_fill_price=order["price"],
                )

            if elapsed >= 0.06:
                return VenueOrderState(
                    order_id=order_id,
                    status=OrderStatus.PARTIALLY_FILLED,
                    filled_quantity=quantity * 0.50,
                    average_fill_price=order["price"],
                )

            return VenueOrderState(
                order_id=order_id,
                status=OrderStatus.NEW,
                filled_quantity=0.0,
            )

    async def cancel_order(self, order_id: str) -> VenueOrderState:
        async with self._lock:
            order = self._orders[order_id]
            order["cancelled"] = True
        return await self.get_order(order_id)


async def _simulation() -> None:
    """Run a short standalone router simulation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    venue = _SimulatedVenue()
    router = InstitutionalAlgoRouter(
        venue=venue,
        config=RouterConfig(
            max_spread_bps=10.0,
            post_only_timeout_seconds=0.30,
            poll_interval_seconds=0.03,
            lot_size=0.0001,
            tick_size=0.01,
            random_seed=7,
        ),
    )

    order = ParentOrder(
        symbol="BTC-USD",
        side=Side.SELL,
        quantity=0.0500,
        horizon_seconds=2.0,
        algorithm=ExecutionAlgorithm.TWAP,
        number_of_slices=5,
        metadata={"strategy": "institutional_demo"},
    )

    receipt = await router.execute(order)
    print(receipt.to_dict())


if __name__ == "__main__":
    asyncio.run(_simulation())