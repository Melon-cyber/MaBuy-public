from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any


@dataclass(frozen=True)
class RiskEngineConfiguration:
    """
    All percentage values are decimal fractions.

    Examples:
        1.20% = 0.0120
        20 basis points = 0.0020
        5 basis points = 0.0005
    """

    maker_fee_rate: float = 0.00020
    taker_fee_rate: float = 0.00050
    additional_breakeven_buffer_pct: float = 0.00200

    minimum_hard_loss_cut_pct: float = 0.01000
    maximum_hard_loss_cut_pct: float = 0.01800
    atr_hard_loss_multiple: float = 2.0

    stalled_momentum_bars: int = 3
    stalled_momentum_hurst_threshold: float = 0.55
    stalled_momentum_loss_cut_pct: float = 0.01200

    trending_hurst_threshold: float = 0.55
    runner_target_atr_multiple: float = 2.5
    minimum_runner_target_pct: float = 0.03500
    maximum_runner_target_pct: float = 0.04500

    non_trending_target_atr_multiple: float = 1.2
    minimum_non_trending_target_pct: float = 0.00800
    maximum_non_trending_target_pct: float = 0.01500

    minimum_candidate_confidence: float = 0.80


@dataclass
class AsymmetricRiskEngine:
    """
    Deterministic risk engine for asymmetric exit management.

    Important limitation:
    evaluate_early_loss_cut receives no actual momentum slope, no volume
    confirmation and no stalled-bar counter. Therefore, for the required API,
    Hurst <= threshold after the configured number of bars is used as a
    conservative non-trending proxy. A production signal layer should provide
    true momentum-stall state before calling this engine.

    active_positions accepted by check_correlation_scale may contain mappings
    or objects with at least:
        - symbol
        - direction or side
        - optional status

    candidate confidence is supplied by candidate_confidence_provider or by
    symbol_confidence. If a same-direction second position exists and confidence
    is unavailable, the engine fails closed and returns 0.0.
    """

    configuration: RiskEngineConfiguration = field(
        default_factory=RiskEngineConfiguration
    )
    candidate_confidence_provider: Callable[[str, str], float] | None = None
    symbol_confidence: Mapping[str, float] = field(default_factory=dict)
    last_correlation_decision_reason: str = field(
        default="No correlation decision evaluated.",
        init=False,
    )

    def evaluate_early_loss_cut(
        self,
        symbol: str,
        current_pnl_pct: float,
        bars_held: int,
        hurst: float,
        atr_pct: float,
    ) -> tuple[bool, str]:
        """
        Returns:
            (should_close_position, decision_reason)

        current_pnl_pct and atr_pct are decimal fractions:
            -1.2% -> -0.012
             1.5% ->  0.015
        """
        self._validate_symbol(symbol)
        self._validate_hurst(hurst)
        self._validate_non_negative_integer(bars_held, "bars_held")
        self._validate_positive_percentage(atr_pct, "atr_pct")

        hard_loss_cut_pct = min(
            self.configuration.maximum_hard_loss_cut_pct,
            max(
                self.configuration.minimum_hard_loss_cut_pct,
                self.configuration.atr_hard_loss_multiple * atr_pct,
            ),
        )

        non_trending_proxy_confirmed = (
            bars_held >= self.configuration.stalled_momentum_bars
            and hurst <= self.configuration.stalled_momentum_hurst_threshold
        )

        if non_trending_proxy_confirmed:
            applicable_loss_cut_pct = min(
                hard_loss_cut_pct,
                self.configuration.stalled_momentum_loss_cut_pct,
            )

            if current_pnl_pct <= -applicable_loss_cut_pct:
                return (
                    True,
                    (
                        f"{symbol}: early loss cut. "
                        f"Three-bar non-trending proxy confirmed "
                        f"(hurst={hurst:.4f}); "
                        f"pnl={current_pnl_pct:.4%} is below "
                        f"the tightened loss limit of {-applicable_loss_cut_pct:.4%}."
                    ),
                )

        if current_pnl_pct <= -hard_loss_cut_pct:
            return (
                True,
                (
                    f"{symbol}: volatility hard loss cut. "
                    f"pnl={current_pnl_pct:.4%} is below "
                    f"the ATR-based maximum loss limit of {-hard_loss_cut_pct:.4%}."
                ),
            )

        return (
            False,
            (
                f"{symbol}: hold. pnl={current_pnl_pct:.4%}; "
                f"current ATR-based hard loss limit is {-hard_loss_cut_pct:.4%}."
            ),
        )

    def get_dynamic_take_profit_target(
        self,
        symbol: str,
        hurst: float,
        atr_pct: float,
    ) -> float:
        """
        Returns the target move as a decimal fraction.

        In a trending regime, this is runner activation, not a mandatory
        market-close level. The execution layer must use an ATR trailing floor
        after runner activation.
        """
        self._validate_symbol(symbol)
        self._validate_hurst(hurst)
        self._validate_positive_percentage(atr_pct, "atr_pct")

        if hurst > self.configuration.trending_hurst_threshold:
            return self._clip(
                self.configuration.runner_target_atr_multiple * atr_pct,
                self.configuration.minimum_runner_target_pct,
                self.configuration.maximum_runner_target_pct,
            )

        return self._clip(
            self.configuration.non_trending_target_atr_multiple * atr_pct,
            self.configuration.minimum_non_trending_target_pct,
            self.configuration.maximum_non_trending_target_pct,
        )

    def calculate_fee_proof_breakeven(
        self,
        entry_price: float,
        direction: str,
        is_maker: bool,
        spread_pct: float,
    ) -> float:
        """
        Calculates a fee-proof stop level.

        spread_pct must be expressed as a decimal fraction.
        For example, 3 basis points is 0.0003.

        The formula intentionally includes:
            two-sided exchange fees + spread estimate + 20 bps safety buffer.
        """
        normalized_direction = self._normalize_direction(direction)
        self._validate_positive_number(entry_price, "entry_price")
        self._validate_non_negative_percentage(spread_pct, "spread_pct")

        exchange_fee_rate = (
            self.configuration.maker_fee_rate
            if is_maker
            else self.configuration.taker_fee_rate
        )

        required_price_buffer_pct = (
            2.0 * exchange_fee_rate
            + spread_pct
            + self.configuration.additional_breakeven_buffer_pct
        )

        entry_price_decimal = self._to_decimal(entry_price)
        price_buffer_decimal = self._to_decimal(required_price_buffer_pct)

        if normalized_direction == "BULLISH":
            breakeven_price = entry_price_decimal * (
                Decimal("1") + price_buffer_decimal
            )
        else:
            breakeven_price = entry_price_decimal * (
                Decimal("1") - price_buffer_decimal
            )

        if breakeven_price <= Decimal("0"):
            raise ValueError(
                "Calculated fee-proof breakeven price must be strictly positive."
            )

        return float(breakeven_price)

    def check_correlation_scale(
        self,
        active_positions: Iterable[Any],
        candidate_symbol: str,
        candidate_direction: str,
    ) -> float:
        """
        Returns:
            1.0 : candidate may use normal margin
            0.5 : one same-direction position exists and confidence >= 80%
            0.0 : veto

        This method implements a strict same-direction beta-exposure freeze.
        It deliberately fails closed if candidate confidence is unavailable.
        """
        self._validate_symbol(candidate_symbol)
        normalized_candidate_direction = self._normalize_direction(
            candidate_direction
        )

        same_direction_position_count = 0

        for active_position in active_positions:
            if not self._position_is_active(active_position):
                continue

            position_symbol = self._extract_position_field(
                active_position,
                "symbol",
            )
            position_direction = self._extract_position_field(
                active_position,
                "direction",
                fallback_field_name="side",
            )

            if position_symbol is None or position_direction is None:
                self.last_correlation_decision_reason = (
                    "Correlation veto: active position has no readable symbol "
                    "or direction."
                )
                return 0.0

            normalized_position_direction = self._normalize_direction(
                str(position_direction)
            )

            if str(position_symbol).upper() == candidate_symbol.upper():
                self.last_correlation_decision_reason = (
                    "Correlation veto: a position in the candidate symbol "
                    "is already active."
                )
                return 0.0

            if normalized_position_direction == normalized_candidate_direction:
                same_direction_position_count += 1

        if same_direction_position_count == 0:
            self.last_correlation_decision_reason = (
                "No same-direction active beta exposure. Full scale allowed."
            )
            return 1.0

        if same_direction_position_count >= 2:
            self.last_correlation_decision_reason = (
                "Correlation veto: two or more same-direction active positions "
                "already exist."
            )
            return 0.0

        candidate_confidence = self._get_candidate_confidence(
            candidate_symbol,
            normalized_candidate_direction,
        )

        if candidate_confidence is None:
            self.last_correlation_decision_reason = (
                "Correlation veto: second same-direction position requires "
                "candidate confidence, but no confidence was supplied."
            )
            return 0.0

        if candidate_confidence < self.configuration.minimum_candidate_confidence:
            self.last_correlation_decision_reason = (
                "Correlation veto: second same-direction position confidence "
                f"{candidate_confidence:.4f} is below required "
                f"{self.configuration.minimum_candidate_confidence:.4f}."
            )
            return 0.0

        self.last_correlation_decision_reason = (
            "One same-direction active position exists; candidate confidence "
            f"{candidate_confidence:.4f} meets the threshold. "
            "Half margin scale required."
        )
        return 0.5

    def _get_candidate_confidence(
        self,
        candidate_symbol: str,
        candidate_direction: str,
    ) -> float | None:
        if self.candidate_confidence_provider is not None:
            confidence = self.candidate_confidence_provider(
                candidate_symbol,
                candidate_direction,
            )
            self._validate_confidence(confidence)
            return confidence

        symbol_key = candidate_symbol.upper()
        if symbol_key in self.symbol_confidence:
            confidence = self.symbol_confidence[symbol_key]
            self._validate_confidence(confidence)
            return confidence

        return None

    @staticmethod
    def _position_is_active(active_position: Any) -> bool:
        status = AsymmetricRiskEngine._extract_position_field(
            active_position,
            "status",
        )

        if status is None:
            return True

        normalized_status = str(status).strip().upper()
        return normalized_status not in {
            "CLOSED",
            "CANCELLED",
            "CANCELED",
            "LIQUIDATED",
            "REJECTED",
        }

    @staticmethod
    def _extract_position_field(
        active_position: Any,
        field_name: str,
        fallback_field_name: str | None = None,
    ) -> Any | None:
        if isinstance(active_position, Mapping):
            if field_name in active_position:
                return active_position[field_name]
            if fallback_field_name and fallback_field_name in active_position:
                return active_position[fallback_field_name]
            return None

        primary_value = getattr(active_position, field_name, None)
        if primary_value is not None:
            return primary_value

        if fallback_field_name is not None:
            return getattr(active_position, fallback_field_name, None)

        return None

    @staticmethod
    def _clip(value: float, lower_bound: float, upper_bound: float) -> float:
        return min(max(value, lower_bound), upper_bound)

    @staticmethod
    def _normalize_direction(direction: str) -> str:
        normalized_direction = direction.strip().upper()

        if normalized_direction in {"BULLISH", "LONG", "BUY"}:
            return "BULLISH"

        if normalized_direction in {"BEARISH", "SHORT", "SELL"}:
            return "BEARISH"

        raise ValueError(
            "direction must be one of: BULLISH, LONG, BUY, BEARISH, SHORT, SELL."
        )

    @staticmethod
    def _validate_symbol(symbol: str) -> None:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string.")

    @staticmethod
    def _validate_hurst(hurst: float) -> None:
        if not 0.0 <= hurst <= 1.0:
            raise ValueError("hurst must be within the closed interval [0.0, 1.0].")

    @staticmethod
    def _validate_positive_percentage(value: float, field_name: str) -> None:
        if not 0.0 < value < 1.0:
            raise ValueError(f"{field_name} must be strictly between 0.0 and 1.0.")

    @staticmethod
    def _validate_non_negative_percentage(value: float, field_name: str) -> None:
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{field_name} must be within [0.0, 1.0).")

    @staticmethod
    def _validate_positive_number(value: float, field_name: str) -> None:
        if value <= 0.0:
            raise ValueError(f"{field_name} must be strictly positive.")

    @staticmethod
    def _validate_non_negative_integer(value: int, field_name: str) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer.")

    @staticmethod
    def _validate_confidence(confidence: float) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("candidate confidence must be within [0.0, 1.0].")

    @staticmethod
    def _to_decimal(value: float) -> Decimal:
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError) as exception:
            raise ValueError(f"Cannot convert value {value!r} to Decimal.") from exception
