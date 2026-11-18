"""
Unified Portfolio Risk Kernel for MaBuy (Astra Step 4 Specification).

This module is dependency-free:
- Python standard library only;
- deterministic;
- no I/O;
- no external exchange network calls;
- suitable for execution before every paper/live order.

Architecture:
    Signal -> Dynamic Sizing (Quarter-Kelly & Platt) -> Correlation Haircut ->
    Portfolio Limits (Drawdown & Margin) -> Liquidation Distance Validation -> Approved Order
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import exp, isfinite, log
from typing import Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

EPSILON = 1e-12


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a finite float into [minimum, maximum]."""
    if not isfinite(value):
        raise ValueError(f"Non-finite value: {value}")
    if minimum > maximum:
        raise ValueError("minimum cannot exceed maximum")
    return max(minimum, min(value, maximum))


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division used only for non-critical telemetry calculations."""
    if abs(denominator) <= EPSILON:
        return default
    return numerator / denominator


def canonical_pair(symbol_a: str, symbol_b: str) -> Tuple[str, str]:
    """Canonical key for symmetric correlation matrix lookup."""
    clean_a = symbol_a.upper().replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").replace("USDT", "").strip()
    clean_b = symbol_b.upper().replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").replace("USDT", "").strip()
    pair = sorted((clean_a, clean_b))
    return (pair[0], pair[1])


def sigmoid(value: float) -> float:
    """Numerically stable sigmoid."""
    if value >= 0.0:
        z = exp(-value)
        return 1.0 / (1.0 + z)
    z = exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True)
class PlattCalibrator:
    """
    Offline-calibrated logistic probability mapping:
        p = sigmoid(a * logit(raw_confidence) + b)
    """

    a: float = 1.20
    b: float = -0.10
    confidence_floor: float = 0.001
    confidence_ceiling: float = 0.999

    def calibrate(self, raw_confidence: float) -> float:
        score = clamp(
            raw_confidence,
            self.confidence_floor,
            self.confidence_ceiling,
        )
        logit_score = log(score / (1.0 - score))
        return clamp(sigmoid(self.a * logit_score + self.b), 0.001, 0.999)


@dataclass(frozen=True)
class RiskKernelConfig:
    """Immutable portfolio risk configuration."""

    # Fractional Kelly
    alpha: float = 0.15
    f_min: float = 0.0
    f_max: float = 0.0125

    # Strict trade and portfolio limits
    max_trade_risk_fraction: float = 0.0125       # 1.25% equity risk cap
    daily_drawdown_limit_fraction: float = 0.02    # 2.0% daily drawdown cap
    max_margin_utilization_fraction: float = 0.40  # 40% initial margin cap
    max_portfolio_leverage: float = 3.0           # 3.0x max portfolio gross leverage
    max_symbol_leverage: float = 5.0              # 5.0x absolute ceiling on single symbol

    # Liquidation safety
    min_liquidation_distance_fraction: float = 0.20  # 20.0% min distance to liquidation
    maintenance_margin_fraction: float = 0.005      # 0.5% standard maintenance margin tier 1
    liquidation_safety_buffer_fraction: float = 0.005 # 0.5% safety buffer
    require_exchange_liquidation_distance: bool = False  # Allows conservative isolated model if True not passed

    # Correlation haircut
    default_missing_correlation: float = 0.75

    # Costs expressed as fractions (e.g. 0.0004 = 4 bps)
    entry_fee_fraction: float = 0.0004
    exit_fee_fraction: float = 0.0004
    entry_slippage_fraction: float = 0.0003
    exit_slippage_fraction: float = 0.0003

    # Hard operational floors
    min_notional_usd: float = 5.0
    minimum_equity_usd: float = 1.0

    def __post_init__(self) -> None:
        if not 0.05 <= self.alpha <= 0.50:
            raise ValueError("alpha must be in [0.05, 0.50]")

        for name, value in (
            ("f_min", self.f_min),
            ("f_max", self.f_max),
            ("max_trade_risk_fraction", self.max_trade_risk_fraction),
            ("daily_drawdown_limit_fraction", self.daily_drawdown_limit_fraction),
            ("max_margin_utilization_fraction", self.max_margin_utilization_fraction),
            ("min_liquidation_distance_fraction", self.min_liquidation_distance_fraction),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")

        if self.f_max > self.max_trade_risk_fraction:
            raise ValueError(
                "f_max cannot exceed max_trade_risk_fraction: "
                "otherwise the Kelly layer can violate absolute risk limits."
            )

        if self.max_portfolio_leverage <= 0.0:
            raise ValueError("max_portfolio_leverage must be positive")

        if self.max_symbol_leverage < 1.0:
            raise ValueError("max_symbol_leverage must be >= 1.0")


@dataclass(frozen=True)
class PositionSnapshot:
    """Current open position state."""

    symbol: str
    signed_notional_usd: float  # Long: positive, Short: negative
    margin_usd: float          # Actual margin allocated
    leverage: float = 1.0


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Single immutable portfolio state used for a risk assessment decision."""

    equity_usd: float
    day_start_equity_usd: float
    positions: Tuple[PositionSnapshot, ...] = ()
    correlations: Mapping[Tuple[str, str], float] = field(default_factory=dict)

    @property
    def gross_notional_usd(self) -> float:
        return sum(abs(position.signed_notional_usd) for position in self.positions)

    @property
    def used_margin_usd(self) -> float:
        return sum(max(0.0, position.margin_usd) for position in self.positions)

    @property
    def daily_drawdown_fraction(self) -> float:
        if self.day_start_equity_usd <= EPSILON:
            return 1.0
        return max(
            0.0,
            (self.day_start_equity_usd - self.equity_usd) / self.day_start_equity_usd,
        )

    @property
    def margin_utilization_fraction(self) -> float:
        if self.equity_usd <= EPSILON:
            return 1.0
        return self.used_margin_usd / self.equity_usd


@dataclass(frozen=True)
class CandidateOrder:
    """Order intent before exchange submission."""

    symbol: str
    side: str  # "LONG" or "SHORT" (or "BULLISH"/"BEARISH")
    entry_price: float
    stop_price: float
    take_profit_price: float

    raw_confidence: float
    signal_quality_multiplier: float = 1.0
    regime_risk_multiplier: float = 1.0
    requested_leverage: float = 3.0

    projected_liquidation_distance_fraction: Optional[float] = None

    entry_fee_fraction: Optional[float] = None
    exit_fee_fraction: Optional[float] = None
    entry_slippage_fraction: Optional[float] = None
    exit_slippage_fraction: Optional[float] = None


@dataclass(frozen=True)
class RiskAssessmentResult:
    approved: bool
    approved_notional_usd: float
    approved_margin_usd: float
    recommended_leverage: float
    veto_reason: str

    calibrated_probability: float = 0.0
    net_payoff_ratio: float = 0.0
    kelly_fraction: float = 0.0
    target_risk_fraction: float = 0.0
    correlation_haircut: float = 0.0
    estimated_trade_risk_usd: float = 0.0
    daily_drawdown_fraction: float = 0.0
    projected_margin_utilization_fraction: float = 0.0
    projected_portfolio_leverage: float = 0.0
    liquidation_distance_fraction: float = 0.0

    @classmethod
    def veto(
        cls,
        reason: str,
        portfolio: PortfolioSnapshot,
        **kwargs: float,
    ) -> "RiskAssessmentResult":
        return cls(
            approved=False,
            approved_notional_usd=0.0,
            approved_margin_usd=0.0,
            recommended_leverage=0.0,
            veto_reason=reason,
            daily_drawdown_fraction=portfolio.daily_drawdown_fraction,
            projected_margin_utilization_fraction=portfolio.margin_utilization_fraction,
            projected_portfolio_leverage=safe_ratio(
                portfolio.gross_notional_usd,
                portfolio.equity_usd,
                default=0.0,
            ),
            **kwargs,
        )

    def format_russian_summary(self, symbol: str, side: str) -> str:
        """Generates concise, human-readable Russian telemetry string."""
        clean_side = "LONG" if side.upper() in ("LONG", "BULLISH", "BUY") else "SHORT"
        if self.approved:
            return (
                f"РИСК-ОДОБРЕНО | symbol={symbol} side={clean_side} notional=${self.approved_notional_usd:.2f} "
                f"margin=${self.approved_margin_usd:.2f} lev={self.recommended_leverage:.2f}x "
                f"risk=${self.estimated_trade_risk_usd:.2f} "
                f"kelly={self.kelly_fraction*100.0:.2f}% target={self.target_risk_fraction*100.0:.2f}% "
                f"corr_haircut={self.correlation_haircut:.3f} dd={self.daily_drawdown_fraction*100.0:.2f}% "
                f"margin_util={self.projected_margin_utilization_fraction*100.0:.2f}% "
                f"portfolio_lev={self.projected_portfolio_leverage:.2f}x liq_dist={self.liquidation_distance_fraction*100.0:.2f}%"
            )
        else:
            return (
                f"РИСК-ВЕТО | symbol={symbol} side={clean_side} reason={self.veto_reason} "
                f"dd={self.daily_drawdown_fraction*100.0:.2f}% "
                f"margin_util={self.projected_margin_utilization_fraction*100.0:.2f}% "
                f"portfolio_lev={self.projected_portfolio_leverage:.2f}x"
            )


class PortfolioRiskKernel:
    """
    Mandatory synchronous risk gate before all order submission paths.

    Complexity:
        O(N) for N open positions (deterministic, sub-millisecond).
    """

    def __init__(
        self,
        config: Optional[RiskKernelConfig] = None,
        calibrator: Optional[PlattCalibrator] = None,
    ) -> None:
        self._config = config or RiskKernelConfig()
        self._calibrator = calibrator or PlattCalibrator()

    @property
    def config(self) -> RiskKernelConfig:
        return self._config

    def evaluate_candidate_order(
        self,
        portfolio: PortfolioSnapshot,
        candidate: CandidateOrder,
    ) -> RiskAssessmentResult:
        """
        Evaluate, resize, approve or veto a candidate order.
        No exchange state is mutated. Pure calculation.
        """
        try:
            return self._evaluate(portfolio, candidate)
        except (ValueError, OverflowError) as exc:
            return RiskAssessmentResult.veto(
                reason=f"RISK_INPUT_INVALID:{exc}",
                portfolio=portfolio,
            )

    def _evaluate(
        self,
        portfolio: PortfolioSnapshot,
        candidate: CandidateOrder,
    ) -> RiskAssessmentResult:
        config = self._config

        self._validate_portfolio(portfolio)
        self._validate_candidate(candidate)

        # 1. Account Equity Minimum Floor
        if portfolio.equity_usd < config.minimum_equity_usd:
            return RiskAssessmentResult.veto(
                "EQUITY_BELOW_MINIMUM",
                portfolio,
            )

        # 2. Daily Drawdown Cap
        if portfolio.daily_drawdown_fraction >= config.daily_drawdown_limit_fraction:
            return RiskAssessmentResult.veto(
                "DAILY_DRAWDOWN_LIMIT_REACHED",
                portfolio,
            )

        # 3. Existing Margin Utilization Cap
        if portfolio.margin_utilization_fraction >= config.max_margin_utilization_fraction:
            return RiskAssessmentResult.veto(
                "MARGIN_UTILIZATION_LIMIT_REACHED",
                portfolio,
            )

        # 4. Stop Loss & Take Profit Distance
        stop_distance = abs(candidate.entry_price - candidate.stop_price) / candidate.entry_price
        take_profit_distance = abs(candidate.take_profit_price - candidate.entry_price) / candidate.entry_price

        if stop_distance <= EPSILON:
            return RiskAssessmentResult.veto("STOP_DISTANCE_INVALID", portfolio)

        if take_profit_distance <= EPSILON:
            return RiskAssessmentResult.veto("TAKE_PROFIT_DISTANCE_INVALID", portfolio)

        self._validate_stop_direction(candidate)

        # 5. Net Payoff & Trading Friction (Fees + Slippage)
        round_trip_cost = self._round_trip_cost(candidate)
        net_loss_fraction = stop_distance + round_trip_cost
        net_reward_fraction = take_profit_distance - round_trip_cost

        if net_reward_fraction <= EPSILON:
            return RiskAssessmentResult.veto(
                "TAKE_PROFIT_DOES_NOT_COVER_COSTS",
                portfolio,
            )

        net_payoff_ratio = net_reward_fraction / net_loss_fraction
        calibrated_probability = self._calibrator.calibrate(candidate.raw_confidence)

        # 6. Net Kelly Criterion
        kelly_fraction = (
            calibrated_probability * net_payoff_ratio - (1.0 - calibrated_probability)
        ) / net_payoff_ratio

        if kelly_fraction <= 0.0:
            return RiskAssessmentResult.veto(
                "NON_POSITIVE_NET_KELLY",
                portfolio,
                calibrated_probability=calibrated_probability,
                net_payoff_ratio=net_payoff_ratio,
                kelly_fraction=kelly_fraction,
            )

        # 7. Modified Fractional Kelly Target Risk
        signal_quality = clamp(candidate.signal_quality_multiplier, 0.0, 1.0)
        regime_multiplier = clamp(candidate.regime_risk_multiplier, 0.0, 1.0)

        target_risk_fraction = clamp(
            config.alpha * kelly_fraction * signal_quality * regime_multiplier,
            config.f_min,
            config.f_max,
        )

        # Strict absolute trade risk cap
        target_risk_fraction = min(target_risk_fraction, config.max_trade_risk_fraction)

        if target_risk_fraction <= EPSILON:
            return RiskAssessmentResult.veto(
                "TARGET_RISK_FRACTION_ZERO",
                portfolio,
                calibrated_probability=calibrated_probability,
                net_payoff_ratio=net_payoff_ratio,
                kelly_fraction=kelly_fraction,
            )

        # 8. Inverse Sizing: Notional from Stop Loss Distance
        raw_notional = (portfolio.equity_usd * target_risk_fraction) / net_loss_fraction

        # Mathematical hard risk cap: Notional * net_loss <= Equity * max_trade_risk
        hard_risk_notional_cap = (
            portfolio.equity_usd * config.max_trade_risk_fraction
        ) / net_loss_fraction
        raw_notional = min(raw_notional, hard_risk_notional_cap)

        # 9. Directional Multi-Asset Correlation Haircut
        correlation_haircut = self._correlation_haircut(portfolio, candidate)
        correlation_adjusted_notional = raw_notional * correlation_haircut

        # 10. Leverage Recommendation & Liquidation Distance Check
        recommended_leverage = self._recommended_leverage(candidate)

        liquidation_distance = self._resolve_liquidation_distance(
            candidate=candidate,
            leverage=recommended_leverage,
        )

        if liquidation_distance < config.min_liquidation_distance_fraction:
            return RiskAssessmentResult.veto(
                "LIQUIDATION_DISTANCE_TOO_SMALL",
                portfolio,
                calibrated_probability=calibrated_probability,
                net_payoff_ratio=net_payoff_ratio,
                kelly_fraction=kelly_fraction,
                target_risk_fraction=target_risk_fraction,
                correlation_haircut=correlation_haircut,
                liquidation_distance_fraction=liquidation_distance,
            )

        # 11. Portfolio Gross Leverage Capacity Cap
        leverage_notional_capacity = max(
            0.0,
            portfolio.equity_usd * config.max_portfolio_leverage - portfolio.gross_notional_usd,
        )

        # 12. Portfolio Margin Capacity Cap
        allowed_total_margin = portfolio.equity_usd * config.max_margin_utilization_fraction
        available_margin = max(0.0, allowed_total_margin - portfolio.used_margin_usd)
        margin_notional_capacity = available_margin * recommended_leverage

        approved_notional = min(
            correlation_adjusted_notional,
            leverage_notional_capacity,
            margin_notional_capacity,
        )

        # 13. Minimum Exchange Notional Check
        if approved_notional < config.min_notional_usd:
            return RiskAssessmentResult.veto(
                "NOTIONAL_BELOW_EXCHANGE_OR_CONFIG_MINIMUM",
                portfolio,
                calibrated_probability=calibrated_probability,
                net_payoff_ratio=net_payoff_ratio,
                kelly_fraction=kelly_fraction,
                target_risk_fraction=target_risk_fraction,
                correlation_haircut=correlation_haircut,
                liquidation_distance_fraction=liquidation_distance,
            )

        approved_margin = approved_notional / recommended_leverage
        estimated_trade_risk = approved_notional * net_loss_fraction

        # 14. Final Defensive Invariant Assertions
        if estimated_trade_risk > (portfolio.equity_usd * config.max_trade_risk_fraction + EPSILON):
            return RiskAssessmentResult.veto(
                "INTERNAL_TRADE_RISK_INVARIANT_BREACH",
                portfolio,
            )

        projected_margin_utilization = (
            portfolio.used_margin_usd + approved_margin
        ) / portfolio.equity_usd

        if projected_margin_utilization > (config.max_margin_utilization_fraction + EPSILON):
            return RiskAssessmentResult.veto(
                "PROJECTED_MARGIN_UTILIZATION_LIMIT",
                portfolio,
            )

        projected_portfolio_leverage = (
            portfolio.gross_notional_usd + approved_notional
        ) / portfolio.equity_usd

        if projected_portfolio_leverage > (config.max_portfolio_leverage + EPSILON):
            return RiskAssessmentResult.veto(
                "PROJECTED_PORTFOLIO_LEVERAGE_LIMIT",
                portfolio,
            )

        return RiskAssessmentResult(
            approved=True,
            approved_notional_usd=approved_notional,
            approved_margin_usd=approved_margin,
            recommended_leverage=recommended_leverage,
            veto_reason="",
            calibrated_probability=calibrated_probability,
            net_payoff_ratio=net_payoff_ratio,
            kelly_fraction=kelly_fraction,
            target_risk_fraction=target_risk_fraction,
            correlation_haircut=correlation_haircut,
            estimated_trade_risk_usd=estimated_trade_risk,
            daily_drawdown_fraction=portfolio.daily_drawdown_fraction,
            projected_margin_utilization_fraction=projected_margin_utilization,
            projected_portfolio_leverage=projected_portfolio_leverage,
            liquidation_distance_fraction=liquidation_distance,
        )

    def _round_trip_cost(self, candidate: CandidateOrder) -> float:
        config = self._config
        entry_fee = (
            config.entry_fee_fraction
            if candidate.entry_fee_fraction is None
            else candidate.entry_fee_fraction
        )
        exit_fee = (
            config.exit_fee_fraction
            if candidate.exit_fee_fraction is None
            else candidate.exit_fee_fraction
        )
        entry_slippage = (
            config.entry_slippage_fraction
            if candidate.entry_slippage_fraction is None
            else candidate.entry_slippage_fraction
        )
        exit_slippage = (
            config.exit_slippage_fraction
            if candidate.exit_slippage_fraction is None
            else candidate.exit_slippage_fraction
        )

        values = (entry_fee, exit_fee, entry_slippage, exit_slippage)
        if any(value < 0.0 or not isfinite(value) for value in values):
            raise ValueError("Trading costs must be finite and non-negative")

        return sum(values)

    def _correlation_haircut(
        self,
        portfolio: PortfolioSnapshot,
        candidate: CandidateOrder,
    ) -> float:
        """
        Directional correlation haircut:
            O = sum(max(0, sign(new) * sign(existing) * rho * |exposure| / equity))
            haircut = 1 / (1 + O)
        Opposing positions (hedges) do not reduce size.
        """
        c_side = candidate.side.upper()
        candidate_sign = 1.0 if c_side in ("LONG", "BULLISH", "BUY") else -1.0
        overlap = 0.0

        for position in portfolio.positions:
            if abs(position.signed_notional_usd) <= EPSILON:
                continue

            clean_cand = candidate.symbol.upper().replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").replace("USDT", "").strip()
            clean_pos = position.symbol.upper().replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").replace("USDT", "").strip()

            if clean_cand == clean_pos:
                correlation = 1.0
            else:
                correlation = self._get_correlation(
                    portfolio.correlations,
                    clean_cand,
                    clean_pos,
                )

            position_sign = 1.0 if position.signed_notional_usd > 0.0 else -1.0
            directional_correlation = candidate_sign * position_sign * correlation

            if directional_correlation > 0.0:
                overlap += (
                    directional_correlation
                    * abs(position.signed_notional_usd)
                    / portfolio.equity_usd
                )

        return clamp(1.0 / (1.0 + overlap), 0.05, 1.0)

    def _get_correlation(
        self,
        correlations: Mapping[Tuple[str, str], float],
        symbol_a: str,
        symbol_b: str,
    ) -> float:
        key = canonical_pair(symbol_a, symbol_b)
        correlation = correlations.get(key, self._config.default_missing_correlation)
        return clamp(correlation, -1.0, 1.0)

    def _recommended_leverage(self, candidate: CandidateOrder) -> float:
        """
        Select leverage bounded by requested leverage, max symbol leverage,
        and liquidation distance safety ceiling:
            L <= 1 / (D_min + MM + buffer)
        """
        config = self._config
        liquidation_safe_max_leverage = 1.0 / (
            config.min_liquidation_distance_fraction
            + config.maintenance_margin_fraction
            + config.liquidation_safety_buffer_fraction
        )

        leverage = min(
            candidate.requested_leverage,
            config.max_symbol_leverage,
            liquidation_safe_max_leverage,
        )
        return max(1.0, leverage)

    def _resolve_liquidation_distance(
        self,
        candidate: CandidateOrder,
        leverage: float,
    ) -> float:
        config = self._config

        if candidate.projected_liquidation_distance_fraction is not None:
            value = candidate.projected_liquidation_distance_fraction
            if not isfinite(value) or value < 0.0:
                raise ValueError("Invalid projected liquidation distance")
            return value

        if config.require_exchange_liquidation_distance:
            raise ValueError(
                "Exchange-projected liquidation distance is required for live execution"
            )

        # Conservative isolated margin fallback
        return max(
            0.0,
            (1.0 / leverage)
            - config.maintenance_margin_fraction
            - config.liquidation_safety_buffer_fraction,
        )

    @staticmethod
    def _validate_portfolio(portfolio: PortfolioSnapshot) -> None:
        if not isfinite(portfolio.equity_usd) or portfolio.equity_usd <= 0.0:
            raise ValueError("equity_usd must be finite and positive")

        if (
            not isfinite(portfolio.day_start_equity_usd)
            or portfolio.day_start_equity_usd <= 0.0
        ):
            raise ValueError("day_start_equity_usd must be finite and positive")

        for position in portfolio.positions:
            if not position.symbol:
                raise ValueError("Position symbol cannot be empty")
            if not isfinite(position.signed_notional_usd):
                raise ValueError("Position notional must be finite")
            if not isfinite(position.margin_usd) or position.margin_usd < 0.0:
                raise ValueError("Position margin must be finite and non-negative")

    @staticmethod
    def _validate_candidate(candidate: CandidateOrder) -> None:
        side = candidate.side.upper()
        if side not in {"LONG", "SHORT", "BULLISH", "BEARISH", "BUY", "SELL"}:
            raise ValueError(f"Invalid side: {candidate.side}")

        if not candidate.symbol:
            raise ValueError("symbol cannot be empty")

        for name, value in (
            ("entry_price", candidate.entry_price),
            ("stop_price", candidate.stop_price),
            ("take_profit_price", candidate.take_profit_price),
            ("raw_confidence", candidate.raw_confidence),
            ("signal_quality_multiplier", candidate.signal_quality_multiplier),
            ("regime_risk_multiplier", candidate.regime_risk_multiplier),
            ("requested_leverage", candidate.requested_leverage),
        ):
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")

        if candidate.entry_price <= 0.0:
            raise ValueError("entry_price must be positive")

        if candidate.stop_price <= 0.0:
            raise ValueError("stop_price must be positive")

        if candidate.take_profit_price <= 0.0:
            raise ValueError("take_profit_price must be positive")

        if candidate.requested_leverage < 1.0:
            raise ValueError("requested_leverage must be >= 1.0")

    @staticmethod
    def _validate_stop_direction(candidate: CandidateOrder) -> None:
        side = candidate.side.upper()

        if side in ("LONG", "BULLISH", "BUY"):
            if candidate.stop_price >= candidate.entry_price:
                raise ValueError("LONG stop must be below entry")
            if candidate.take_profit_price <= candidate.entry_price:
                raise ValueError("LONG take-profit must be above entry")

        elif side in ("SHORT", "BEARISH", "SELL"):
            if candidate.stop_price <= candidate.entry_price:
                raise ValueError("SHORT stop must be above entry")
            if candidate.take_profit_price >= candidate.entry_price:
                raise ValueError("SHORT take-profit must be below entry")


def log_risk_assessment(
    logger_instance: logging.Logger,
    symbol: str,
    side: str,
    result: RiskAssessmentResult,
) -> None:
    """Standardized logging helper for trading daemons."""
    summary = result.format_russian_summary(symbol=symbol, side=side)
    if result.approved:
        logger_instance.info(summary)
    else:
        logger_instance.info(summary)
