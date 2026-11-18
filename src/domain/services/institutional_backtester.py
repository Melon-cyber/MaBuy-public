"""Institutional event-driven backtesting engine with microstructure execution modeling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp, isfinite, sqrt
from statistics import NormalDist
from typing import Any, Final

import numpy as np
import pandas as pd
import sys


_NORMAL_DIST: Final[NormalDist] = NormalDist()
_EPSILON: Final[float] = 1.0e-12


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for institutional backtesting, execution, and risk controls."""

    initial_capital: float = 100_000.0
    leverage: float = 3.0
    allocation_fraction: float = 0.25
    maintenance_margin_rate: float = 0.005
    maker_fee_rate: float = 0.0002
    taker_fee_rate: float = 0.0005
    liquidation_fee_rate: float = 0.005
    base_slippage_pct: float = 0.0001
    slippage_slope: float = 0.0010
    max_slippage_pct: float = 0.0500
    maker_fill_kappa: float = 1.0
    queue_depth_fraction: float = 0.10
    maker_offset_bps: float = 1.0
    funding_interval_bars: int = 8
    atr_period: int = 14
    max_holding_bars: int = 48
    random_seed: int = 42

    def validate(self) -> None:
        """Validate configuration invariants."""
        positive_fields: dict[str, float] = {
            "initial_capital": self.initial_capital,
            "leverage": self.leverage,
            "allocation_fraction": self.allocation_fraction,
            "maintenance_margin_rate": self.maintenance_margin_rate,
            "maker_fee_rate": self.maker_fee_rate,
            "taker_fee_rate": self.taker_fee_rate,
            "liquidation_fee_rate": self.liquidation_fee_rate,
            "base_slippage_pct": self.base_slippage_pct,
            "slippage_slope": self.slippage_slope,
            "max_slippage_pct": self.max_slippage_pct,
            "maker_fill_kappa": self.maker_fill_kappa,
            "queue_depth_fraction": self.queue_depth_fraction,
            "funding_interval_bars": float(self.funding_interval_bars),
            "atr_period": float(self.atr_period),
            "max_holding_bars": float(self.max_holding_bars),
        }
        for field_name, value in positive_fields.items():
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and strictly positive.")

        if self.allocation_fraction > 1.0:
            raise ValueError("allocation_fraction must be less than or equal to 1.0.")
        if self.max_slippage_pct > 1.0:
            raise ValueError("max_slippage_pct must be less than or equal to 1.0.")


@dataclass
class PendingOrder:
    """Represents a one-bar maker limit order awaiting a fill."""

    side: int
    quantity: float
    limit_price: float
    submitted_index: int
    submitted_timestamp: pd.Timestamp
    intended_notional: float


@dataclass
class Position:
    """Represents an active perpetual futures position and triple-barrier state."""

    side: int
    quantity: float
    entry_price: float
    entry_timestamp: pd.Timestamp
    entry_index: int
    entry_fee: float
    stop_price: float
    take_profit_price: float
    atr_at_entry: float
    funding_pnl: float = 0.0


@dataclass(frozen=True)
class BacktestResult:
    """Immutable result container for a completed institutional backtest."""

    metrics: dict[str, float]
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    events: pd.DataFrame
    config: BacktestConfig
    initial_capital: float
    final_equity: float
    bar_hours: float

    @property
    def net_pnl(self) -> float:
        return self.metrics.get("net_realized_pnl", 0.0)

    @property
    def sharpe_ratio(self) -> float:
        return self.metrics.get("annualized_sharpe_ratio", 0.0)

    @property
    def profit_factor(self) -> float:
        return self.metrics.get("profit_factor", 0.0)


class InstitutionalBacktester:
    """
    Event-driven institutional futures backtester.

    The engine supports:
    - discrete OHLCV candle iteration;
    - maker/taker execution models;
    - adverse-selection slippage;
    - Binance VIP-0 futures fees;
    - eight-bar funding debits and credits;
    - maintenance margin liquidation checks;
    - ATR-derived triple barriers;
    - performance attribution and López de Prado statistical diagnostics.
    """

    _REQUIRED_COLUMNS: Final[set[str]] = {"open", "high", "low", "close", "volume"}

    def __init__(self, config: BacktestConfig | None = None, **kwargs: Any) -> None:
        """
        Initialize the backtester.

        Args:
            config: Optional immutable backtesting configuration.
            **kwargs: Optional overrides for BacktestConfig fields (initial_capital, maker_fee, taker_fee, etc.)

        Raises:
            ValueError: If the configuration is invalid.
        """
        if config is None:
            cfg_dict: dict[str, Any] = {}
            if "initial_capital" in kwargs:
                cfg_dict["initial_capital"] = float(kwargs.pop("initial_capital"))
            if "maker_fee" in kwargs or "maker_fee_rate" in kwargs:
                cfg_dict["maker_fee_rate"] = float(kwargs.pop("maker_fee", kwargs.pop("maker_fee_rate", 0.0002)))
            if "taker_fee" in kwargs or "taker_fee_rate" in kwargs:
                cfg_dict["taker_fee_rate"] = float(kwargs.pop("taker_fee", kwargs.pop("taker_fee_rate", 0.0005)))
            for k, v in kwargs.items():
                if hasattr(BacktestConfig, k):
                    cfg_dict[k] = v
            self.config = BacktestConfig(**cfg_dict)
        else:
            self.config = config

        self.config.validate()
        self._rng: np.random.Generator = np.random.default_rng(self.config.random_seed)
        self._result: BacktestResult | None = None

    def compute_deflated_sharpe_ratio(
        self,
        returns: pd.Series | np.ndarray,
        n_trials: int = 1,
        benchmark_sharpe: float = 0.0,
    ) -> float:
        """Public utility for computing Deflated Sharpe Ratio (DSR) on returns."""
        if isinstance(returns, np.ndarray):
            ret_s = pd.Series(returns)
        else:
            ret_s = returns
        if len(ret_s) < 2:
            return 0.5
        p_sharpe, _ = self._sharpe_ratios(ret_s, 365.25 * 24)
        skew = self._sample_skewness(ret_s.to_numpy(dtype=float))
        kurt = self._sample_kurtosis(ret_s.to_numpy(dtype=float))
        _, dsr, _, _ = self._statistical_sharpe_metrics(
            period_sharpe=p_sharpe,
            returns_count=len(ret_s),
            skewness=skew,
            kurtosis=kurt,
            bar_hours=1.0,
            number_of_trials=n_trials,
        )
        return float(dsr) if isfinite(dsr) else 0.5

    def run_backtest(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        stop_atr_mult: float,
        tp_atr_mult: float,
        maker_execution: bool = True,
    ) -> BacktestResult:
        """
        Run an event-driven backtest over historical OHLCV data.

        Signals are interpreted as:
        - positive values: long directional intent;
        - negative values: short directional intent;
        - zero or NaN values: no new directional intent.

        Triple barriers are evaluated after entry:
        - stop-loss barrier;
        - take-profit barrier;
        - vertical time barrier configured by ``max_holding_bars``.

        Args:
            df: OHLCV DataFrame indexed by monotonic timestamps.
            signals: Directional signal series aligned to the DataFrame index.
            stop_atr_mult: Stop-loss distance expressed as ATR multiples.
            tp_atr_mult: Take-profit distance expressed as ATR multiples.
            maker_execution: If True, use one-bar maker orders; otherwise use
                immediate taker execution.

        Returns:
            A fully populated BacktestResult.

        Raises:
            TypeError: If inputs have invalid types.
            ValueError: If data, signal, or parameter validation fails.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if not isinstance(signals, pd.Series):
            raise TypeError("signals must be a pandas Series.")
        if not isfinite(stop_atr_mult) or stop_atr_mult <= 0.0:
            raise ValueError("stop_atr_mult must be finite and strictly positive.")
        if not isfinite(tp_atr_mult) or tp_atr_mult <= 0.0:
            raise ValueError("tp_atr_mult must be finite and strictly positive.")

        frame = self._prepare_data(df)
        normalized_signals = self._prepare_signals(signals, frame.index)
        bar_hours = self._infer_bar_hours(frame.index)
        self._rng = np.random.default_rng(self.config.random_seed)

        cash = float(self.config.initial_capital)
        position: Position | None = None
        pending_order: PendingOrder | None = None
        trades: list[dict[str, Any]] = []
        equity_records: list[dict[str, Any]] = []
        event_records: list[dict[str, Any]] = []

        for bar_index, (timestamp, bar) in enumerate(frame.iterrows()):
            event_messages: list[str] = []
            close_price = float(bar["close"])

            if pending_order is not None and position is None:
                if bar_index <= pending_order.submitted_index:
                    raise RuntimeError("Pending maker order was evaluated before its submission bar.")

                maker_filled, fill_probability = self._attempt_maker_fill(
                    order=pending_order,
                    bar=bar,
                )

                if maker_filled:
                    position, entry_event = self._open_position(
                        side=pending_order.side,
                        quantity=pending_order.quantity,
                        entry_price=pending_order.limit_price,
                        timestamp=timestamp,
                        bar_index=bar_index,
                        atr=float(bar["atr"]),
                        stop_atr_mult=stop_atr_mult,
                        tp_atr_mult=tp_atr_mult,
                        fee_rate=self.config.maker_fee_rate,
                        cash=cash,
                        execution_type="maker",
                    )
                    cash -= position.entry_fee
                    event_messages.append(entry_event)
                    event_records.append(
                        {
                            "timestamp": timestamp,
                            "event_type": "maker_fill",
                            "side": position.side,
                            "price": position.entry_price,
                            "quantity": position.quantity,
                            "fill_probability": fill_probability,
                            "details": "Maker limit order filled.",
                        }
                    )
                else:
                    event_messages.append("maker_order_expired")
                    event_records.append(
                        {
                            "timestamp": timestamp,
                            "event_type": "maker_order_expired",
                            "side": pending_order.side,
                            "price": pending_order.limit_price,
                            "quantity": pending_order.quantity,
                            "fill_probability": fill_probability,
                            "details": "Maker limit order did not fill within one eligible bar.",
                        }
                    )
                pending_order = None

            if position is not None:
                if (bar_index + 1) % self.config.funding_interval_bars == 0:
                    funding_rate = float(bar["funding_rate"])
                    mark_notional = abs(position.quantity * close_price)
                    funding_cash_flow = -float(position.side) * mark_notional * funding_rate
                    position.funding_pnl += funding_cash_flow
                    cash += funding_cash_flow
                    event_messages.append("funding_applied")
                    event_records.append(
                        {
                            "timestamp": timestamp,
                            "event_type": "funding",
                            "side": position.side,
                            "price": close_price,
                            "quantity": position.quantity,
                            "fill_probability": np.nan,
                            "details": f"Funding cash flow: {funding_cash_flow:.8f}",
                        }
                    )

                if bar_index > position.entry_index:
                    barrier_reason = self._evaluate_triple_barrier(position, bar, bar_index)

                    if barrier_reason is not None:
                        raw_exit_price = self._barrier_exit_price(position, barrier_reason, close_price)
                        exit_price = self._taker_fill_price(
                            side=-position.side,
                            reference_price=raw_exit_price,
                            quantity=position.quantity,
                            bar=bar,
                        )
                        trade_record = self._close_position(
                            position=position,
                            exit_price=exit_price,
                            exit_timestamp=timestamp,
                            exit_index=bar_index,
                            exit_reason=barrier_reason,
                            fee_rate=self.config.taker_fee_rate,
                            bar_hours=bar_hours,
                        )
                        cash += trade_record["gross_pnl"] - trade_record["exit_fee"]
                        trades.append(trade_record)
                        event_messages.append(f"exit_{barrier_reason}")
                        event_records.append(
                            {
                                "timestamp": timestamp,
                                "event_type": "position_exit",
                                "side": position.side,
                                "price": exit_price,
                                "quantity": position.quantity,
                                "fill_probability": np.nan,
                                "details": barrier_reason,
                            }
                        )
                        position = None

            if position is not None:
                unrealized_pnl = self._unrealized_pnl(position, close_price)
                mark_equity = cash + unrealized_pnl
                maintenance_margin = (
                    abs(position.quantity * close_price) * self.config.maintenance_margin_rate
                )

                if mark_equity <= maintenance_margin:
                    liquidation_price = self._taker_fill_price(
                        side=-position.side,
                        reference_price=close_price,
                        quantity=position.quantity,
                        bar=bar,
                    )
                    trade_record = self._close_position(
                        position=position,
                        exit_price=liquidation_price,
                        exit_timestamp=timestamp,
                        exit_index=bar_index,
                        exit_reason="liquidation",
                        fee_rate=self.config.liquidation_fee_rate,
                        bar_hours=bar_hours,
                    )
                    cash += trade_record["gross_pnl"] - trade_record["exit_fee"]
                    trades.append(trade_record)
                    event_messages.append("liquidation")
                    event_records.append(
                        {
                            "timestamp": timestamp,
                            "event_type": "liquidation",
                            "side": position.side,
                            "price": liquidation_price,
                            "quantity": position.quantity,
                            "fill_probability": np.nan,
                            "details": "Maintenance margin threshold breached.",
                        }
                    )
                    position = None

            signal_side = int(normalized_signals.iloc[bar_index])

            if (
                position is not None
                and signal_side != 0
                and signal_side != position.side
                and bar_index > position.entry_index
            ):
                reversal_exit_price = self._taker_fill_price(
                    side=-position.side,
                    reference_price=close_price,
                    quantity=position.quantity,
                    bar=bar,
                )
                trade_record = self._close_position(
                    position=position,
                    exit_price=reversal_exit_price,
                    exit_timestamp=timestamp,
                    exit_index=bar_index,
                    exit_reason="signal_reversal",
                    fee_rate=self.config.taker_fee_rate,
                    bar_hours=bar_hours,
                )
                cash += trade_record["gross_pnl"] - trade_record["exit_fee"]
                trades.append(trade_record)
                event_messages.append("exit_signal_reversal")
                event_records.append(
                    {
                        "timestamp": timestamp,
                        "event_type": "signal_reversal_exit",
                        "side": position.side,
                        "price": reversal_exit_price,
                        "quantity": position.quantity,
                        "fill_probability": np.nan,
                        "details": "Position closed due to opposite directional signal.",
                    }
                )
                position = None

            if (
                position is None
                and pending_order is None
                and signal_side != 0
                and bar_index < len(frame) - 1
            ):
                current_equity = cash
                if current_equity > 0.0:
                    target_notional = (
                        current_equity
                        * self.config.leverage
                        * self.config.allocation_fraction
                    )
                    reference_price = close_price
                    quantity = target_notional / max(reference_price, _EPSILON)

                    if quantity > _EPSILON:
                        if maker_execution:
                            spread = float(bar["spread"])
                            maker_offset = max(
                                spread / 2.0,
                                reference_price * self.config.maker_offset_bps / 10_000.0,
                            )
                            limit_price = (
                                reference_price - maker_offset
                                if signal_side > 0
                                else reference_price + maker_offset
                            )
                            limit_price = max(limit_price, _EPSILON)
                            pending_order = PendingOrder(
                                side=signal_side,
                                quantity=quantity,
                                limit_price=limit_price,
                                submitted_index=bar_index,
                                submitted_timestamp=timestamp,
                                intended_notional=target_notional,
                            )
                            event_messages.append("maker_order_submitted")
                            event_records.append(
                                {
                                    "timestamp": timestamp,
                                    "event_type": "maker_order_submitted",
                                    "side": signal_side,
                                    "price": limit_price,
                                    "quantity": quantity,
                                    "fill_probability": np.nan,
                                    "details": "One-bar maker limit order submitted.",
                                }
                            )
                        else:
                            entry_price = self._taker_fill_price(
                                side=signal_side,
                                reference_price=reference_price,
                                quantity=quantity,
                                bar=bar,
                            )
                            position, entry_event = self._open_position(
                                side=signal_side,
                                quantity=quantity,
                                entry_price=entry_price,
                                timestamp=timestamp,
                                bar_index=bar_index,
                                atr=float(bar["atr"]),
                                stop_atr_mult=stop_atr_mult,
                                tp_atr_mult=tp_atr_mult,
                                fee_rate=self.config.taker_fee_rate,
                                cash=cash,
                                execution_type="taker",
                            )
                            cash -= position.entry_fee
                            event_messages.append(entry_event)
                            event_records.append(
                                {
                                    "timestamp": timestamp,
                                    "event_type": "taker_fill",
                                    "side": position.side,
                                    "price": position.entry_price,
                                    "quantity": position.quantity,
                                    "fill_probability": 1.0,
                                    "details": "Immediate taker execution.",
                                }
                            )

            if bar_index == len(frame) - 1 and pending_order is not None:
                event_messages.append("final_pending_order_cancelled")
                event_records.append(
                    {
                        "timestamp": timestamp,
                        "event_type": "pending_order_cancelled",
                        "side": pending_order.side,
                        "price": pending_order.limit_price,
                        "quantity": pending_order.quantity,
                        "fill_probability": np.nan,
                        "details": "Pending order cancelled at end of test.",
                    }
                )
                pending_order = None

            if bar_index == len(frame) - 1 and position is not None:
                final_exit_price = self._taker_fill_price(
                    side=-position.side,
                    reference_price=close_price,
                    quantity=position.quantity,
                    bar=bar,
                )
                trade_record = self._close_position(
                    position=position,
                    exit_price=final_exit_price,
                    exit_timestamp=timestamp,
                    exit_index=bar_index,
                    exit_reason="end_of_test",
                    fee_rate=self.config.taker_fee_rate,
                    bar_hours=bar_hours,
                )
                cash += trade_record["gross_pnl"] - trade_record["exit_fee"]
                trades.append(trade_record)
                event_messages.append("exit_end_of_test")
                event_records.append(
                    {
                        "timestamp": timestamp,
                        "event_type": "end_of_test_exit",
                        "side": position.side,
                        "price": final_exit_price,
                        "quantity": position.quantity,
                        "fill_probability": np.nan,
                        "details": "Open position closed at end of test.",
                    }
                )
                position = None

            unrealized_pnl = (
                self._unrealized_pnl(position, close_price) if position is not None else 0.0
            )
            equity = cash + unrealized_pnl

            equity_records.append(
                {
                    "timestamp": timestamp,
                    "cash": cash,
                    "unrealized_pnl": unrealized_pnl,
                    "equity": equity,
                    "close": close_price,
                    "position_side": position.side if position is not None else 0,
                    "position_quantity": position.quantity if position is not None else 0.0,
                    "event": ";".join(event_messages),
                }
            )

        trades_frame = pd.DataFrame(
            trades,
            columns=[
                "entry_timestamp",
                "exit_timestamp",
                "side",
                "quantity",
                "entry_price",
                "exit_price",
                "entry_notional",
                "exit_notional",
                "gross_pnl",
                "entry_fee",
                "exit_fee",
                "funding_pnl",
                "net_pnl",
                "return_on_notional",
                "holding_bars",
                "holding_hours",
                "exit_reason",
                "triple_barrier_label",
            ],
        )
        equity_curve = pd.DataFrame(equity_records).set_index("timestamp")
        events_frame = pd.DataFrame(
            event_records,
            columns=[
                "timestamp",
                "event_type",
                "side",
                "price",
                "quantity",
                "fill_probability",
                "details",
            ],
        )

        metrics = self._calculate_metrics(
            equity_curve=equity_curve,
            trades=trades_frame,
            bar_hours=bar_hours,
        )
        final_equity = float(equity_curve["equity"].iloc[-1])

        self._result = BacktestResult(
            metrics=metrics,
            trades=trades_frame,
            equity_curve=equity_curve,
            events=events_frame,
            config=self.config,
            initial_capital=self.config.initial_capital,
            final_equity=final_equity,
            bar_hours=bar_hours,
        )
        return self._result

    def generate_report(self) -> str:
        """
        Generate an audited institutional performance report.

        Returns:
            A formatted ASCII performance table.

        Raises:
            RuntimeError: If no completed backtest result is available.
        """
        if self._result is None:
            raise RuntimeError("No backtest result is available. Run run_backtest first.")

        result = self._result
        metrics = result.metrics

        metric_rows: list[tuple[str, str]] = [
            ("Initial Capital", self._format_currency(result.initial_capital)),
            ("Final Equity", self._format_currency(result.final_equity)),
            ("Net Realized PnL", self._format_currency(metrics["net_realized_pnl"])),
            ("Total Return", self._format_percent(metrics["total_return_pct"])),
            ("Gross Profit", self._format_currency(metrics["gross_profit"])),
            ("Gross Loss", self._format_currency(metrics["gross_loss"])),
            ("Profit Factor", self._format_number(metrics["profit_factor"])),
            ("Payoff Ratio", self._format_number(metrics["payoff_ratio"])),
            ("Win Rate", self._format_percent(metrics["win_rate_pct"])),
            ("Trade Count", self._format_integer(metrics["trade_count"])),
            ("Annualized Net Sharpe", self._format_number(metrics["annualized_sharpe_ratio"])),
            ("Annualized Sortino", self._format_number(metrics["annualized_sortino_ratio"])),
            ("Maximum Drawdown", self._format_percent(metrics["max_drawdown_pct"])),
            (
                "Maximum Drawdown Duration",
                f"{self._format_number(metrics['max_drawdown_duration_hours'])} hours "
                f"({self._format_number(metrics['max_drawdown_duration_days'])} days)",
            ),
            ("Calmar Ratio", self._format_number(metrics["calmar_ratio"])),
            ("Return Skewness", self._format_number(metrics["return_skewness"])),
            ("Return Kurtosis", self._format_number(metrics["return_kurtosis"])),
            ("Probabilistic Sharpe Ratio", self._format_percent(metrics["psr_pct"])),
            ("Deflated Sharpe Ratio", self._format_percent(metrics["dsr_pct"])),
            ("Expected Max Sharpe", self._format_number(metrics["expected_max_sharpe"])),
            ("Minimum Track Record Length", f"{self._format_number(metrics['min_trl_days'])} days"),
            ("Bar Interval", f"{self._format_number(result.bar_hours)} hours"),
            ("Execution Model", "Maker" if self._is_maker_result(result) else "Mixed/Taker"),
        ]

        label_width = max(len(label) for label, _ in metric_rows)
        value_width = max(len(value) for _, value in metric_rows)
        total_width = label_width + value_width + 7
        border = "+" + "-" * (total_width - 2) + "+"
        title = " INSTITUTIONAL BACKTEST PERFORMANCE AUDIT "
        title_line = "|" + title.center(total_width - 2) + "|"

        lines = [border, title_line, border]
        for label, value in metric_rows:
            lines.append(f"| {label:<{label_width}} | {value:>{value_width}} |")
        lines.append(border)

        config_dict = asdict(result.config)
        lines.append("Execution and Risk Controls:")
        for key, value in config_dict.items():
            lines.append(f"  - {key}: {value}")

        return "\n".join(lines)

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate, normalize, and enrich raw market data."""
        if df.empty:
            raise ValueError("df must not be empty.")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df must use a pandas DatetimeIndex.")
        if df.index.has_duplicates:
            raise ValueError("df index must not contain duplicate timestamps.")
        if not df.index.is_monotonic_increasing:
            raise ValueError("df index must be strictly monotonic increasing.")

        frame = df.copy()
        frame.columns = [str(column).strip().lower() for column in frame.columns]

        missing_columns = self._REQUIRED_COLUMNS.difference(frame.columns)
        if missing_columns:
            raise ValueError(
                f"df is missing required OHLCV columns: {sorted(missing_columns)}."
            )

        for column in self._REQUIRED_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        required_values = frame[list(self._REQUIRED_COLUMNS)]
        if required_values.isna().any().any():
            raise ValueError("Required OHLCV columns contain non-numeric or missing values.")
        if (required_values[["open", "high", "low", "close"]] <= 0.0).any().any():
            raise ValueError("OHLC prices must be strictly positive.")
        if (required_values["volume"] <= 0.0).any():
            raise ValueError("Volume must be strictly positive.")
        if (frame["high"] < frame["low"]).any():
            raise ValueError("Each bar must satisfy high >= low.")
        if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
            raise ValueError("High must be greater than or equal to open and close.")
        if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
            raise ValueError("Low must be less than or equal to open and close.")

        if "spread" in frame.columns:
            frame["spread"] = pd.to_numeric(frame["spread"], errors="coerce")
        elif {"bid", "ask"}.issubset(frame.columns):
            bid = pd.to_numeric(frame["bid"], errors="coerce")
            ask = pd.to_numeric(frame["ask"], errors="coerce")
            frame["spread"] = ask - bid
        else:
            frame["spread"] = frame["close"] * self.config.maker_offset_bps / 10_000.0

        if frame["spread"].isna().any() or (frame["spread"] < 0.0).any():
            raise ValueError("spread must be numeric and non-negative.")

        if "funding_rate" not in frame.columns:
            frame["funding_rate"] = 0.0
        frame["funding_rate"] = pd.to_numeric(frame["funding_rate"], errors="coerce").fillna(0.0)

        if "queue_depth_proxy" in frame.columns:
            frame["queue_depth_proxy"] = pd.to_numeric(
                frame["queue_depth_proxy"], errors="coerce"
            )
        elif "queue_depth" in frame.columns:
            frame["queue_depth_proxy"] = pd.to_numeric(frame["queue_depth"], errors="coerce")
        else:
            frame["queue_depth_proxy"] = (
                frame["volume"] * frame["close"] * self.config.queue_depth_fraction
            )

        if frame["queue_depth_proxy"].isna().any() or (frame["queue_depth_proxy"] <= 0.0).any():
            raise ValueError("queue_depth_proxy must be numeric and strictly positive.")

        previous_close = frame["close"].shift(1)
        true_range = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous_close).abs(),
                (frame["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        frame["atr"] = true_range.rolling(
            window=self.config.atr_period,
            min_periods=1,
        ).mean()
        frame["atr"] = frame["atr"].clip(lower=frame["close"] * 1.0e-8)

        return frame

    @staticmethod
    def _prepare_signals(signals: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
        """Normalize signals to {-1, 0, +1} and align them to market data."""
        if signals.empty:
            raise ValueError("signals must not be empty.")

        if signals.index.equals(index):
            aligned = signals.copy()
        elif len(signals) == len(index):
            aligned = pd.Series(signals.to_numpy(copy=True), index=index, name=signals.name)
        else:
            aligned = signals.reindex(index)

        numeric_signals = pd.to_numeric(aligned, errors="coerce").fillna(0.0)
        return np.sign(numeric_signals).astype(int)

    @staticmethod
    def _infer_bar_hours(index: pd.DatetimeIndex) -> float:
        """Infer the median elapsed time between bars in hours."""
        if len(index) < 2:
            return 1.0

        diffs = (index[1:] - index[:-1]).total_seconds() / 3600.0
        if np.any(diffs <= 0):
            raise ValueError("Datetime index must be strictly increasing.")

        bar_hours = float(np.median(diffs))
        if not isfinite(bar_hours) or bar_hours <= 0.0:
            raise ValueError("Unable to infer a positive bar interval from the DataFrame index.")
        return bar_hours

    def _attempt_maker_fill(
        self,
        order: PendingOrder,
        bar: pd.Series,
    ) -> tuple[bool, float]:
        """Evaluate maker fill probability from range, volume, and queue depth."""
        bar_low = float(bar["low"])
        bar_high = float(bar["high"])
        if not (bar_low <= order.limit_price <= bar_high):
            return False, 0.0

        quote_volume = max(float(bar["volume"]) * float(bar["close"]), _EPSILON)
        queue_depth = max(float(bar["queue_depth_proxy"]), _EPSILON)
        fill_probability = 1.0 - exp(
            -self.config.maker_fill_kappa * (quote_volume / queue_depth)
        )
        fill_probability = float(np.clip(fill_probability, 0.0, 1.0))
        return bool(self._rng.random() < fill_probability), fill_probability

    def _taker_fill_price(
        self,
        side: int,
        reference_price: float,
        quantity: float,
        bar: pd.Series,
    ) -> float:
        """
        Compute immediate taker fill with adverse-selection square-root slippage.

        Slippage model:
            base_slippage + slippage_slope * sqrt(order_notional / bar_quote_volume)
        """
        if side not in (-1, 1):
            raise ValueError("side must be either -1 or +1.")
        if reference_price <= 0.0 or quantity <= 0.0:
            raise ValueError("reference_price and quantity must be strictly positive.")

        order_notional = abs(reference_price * quantity)
        bar_quote_volume = max(float(bar["volume"]) * float(bar["close"]), _EPSILON)
        impact_term = self.config.slippage_slope * sqrt(order_notional / bar_quote_volume)
        slippage_pct = min(
            self.config.base_slippage_pct + impact_term,
            self.config.max_slippage_pct,
        )

        fill_price = reference_price * (1.0 + side * slippage_pct)
        return max(fill_price, _EPSILON)

    def _open_position(
        self,
        side: int,
        quantity: float,
        entry_price: float,
        timestamp: pd.Timestamp,
        bar_index: int,
        atr: float,
        stop_atr_mult: float,
        tp_atr_mult: float,
        fee_rate: float,
        cash: float,
        execution_type: str,
    ) -> tuple[Position, str]:
        """Create a position and calculate entry fees and triple-barrier levels."""
        if side not in (-1, 1):
            raise ValueError("Position side must be either -1 or +1.")
        if quantity <= 0.0 or entry_price <= 0.0 or atr <= 0.0:
            raise ValueError("Position quantity, entry price, and ATR must be positive.")

        entry_notional = quantity * entry_price
        entry_fee = entry_notional * fee_rate
        if cash - entry_fee <= -self.config.initial_capital:
            raise RuntimeError("Entry fee would create an invalid cash balance.")

        stop_distance = atr * stop_atr_mult
        take_profit_distance = atr * tp_atr_mult

        position = Position(
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_timestamp=timestamp,
            entry_index=bar_index,
            entry_fee=entry_fee,
            stop_price=entry_price - side * stop_distance,
            take_profit_price=entry_price + side * take_profit_distance,
            atr_at_entry=atr,
        )
        event = f"{execution_type}_entry_{'long' if side > 0 else 'short'}"
        return position, event

    def _evaluate_triple_barrier(
        self,
        position: Position,
        bar: pd.Series,
        current_index: int,
    ) -> str | None:
        """
        Determine whether a triple barrier has been crossed.

        When stop and target are both touched in an OHLC candle, the engine uses
        conservative stop-first ordering because the intrabar path is unknown.
        """
        high = float(bar["high"])
        low = float(bar["low"])

        if position.side > 0:
            stop_hit = low <= position.stop_price
            take_profit_hit = high >= position.take_profit_price
        else:
            stop_hit = high >= position.stop_price
            take_profit_hit = low <= position.take_profit_price

        if stop_hit:
            return "stop_loss"
        if take_profit_hit:
            return "take_profit"
        if current_index - position.entry_index >= self.config.max_holding_bars:
            return "vertical_barrier"
        return None

    @staticmethod
    def _barrier_exit_price(
        position: Position,
        exit_reason: str,
        close_price: float,
    ) -> float:
        """Select the raw barrier reference price before taker adverse selection."""
        if exit_reason == "stop_loss":
            return position.stop_price
        if exit_reason == "take_profit":
            return position.take_profit_price
        if exit_reason == "vertical_barrier":
            return close_price
        raise ValueError(f"Unsupported barrier exit reason: {exit_reason}.")

    @staticmethod
    def _unrealized_pnl(position: Position, mark_price: float) -> float:
        """Calculate mark-to-market unrealized PnL for an active position."""
        return float(position.side) * position.quantity * (mark_price - position.entry_price)

    def _close_position(
        self,
        position: Position,
        exit_price: float,
        exit_timestamp: pd.Timestamp,
        exit_index: int,
        exit_reason: str,
        fee_rate: float,
        bar_hours: float,
    ) -> dict[str, Any]:
        """Close a position and produce an audited trade record."""
        if exit_price <= 0.0:
            raise ValueError("exit_price must be strictly positive.")
        if exit_index < position.entry_index:
            raise ValueError("exit_index cannot precede entry_index.")

        entry_notional = position.quantity * position.entry_price
        exit_notional = position.quantity * exit_price
        gross_pnl = float(position.side) * position.quantity * (
            exit_price - position.entry_price
        )
        exit_fee = exit_notional * fee_rate
        net_pnl = gross_pnl - position.entry_fee - exit_fee + position.funding_pnl
        holding_bars = exit_index - position.entry_index
        holding_hours = holding_bars * bar_hours

        triple_barrier_label = {
            "take_profit": 1,
            "stop_loss": -1,
            "vertical_barrier": 0,
            "signal_reversal": 0,
            "liquidation": -1,
            "end_of_test": 0,
        }.get(exit_reason, 0)

        return {
            "entry_timestamp": position.entry_timestamp,
            "exit_timestamp": exit_timestamp,
            "side": position.side,
            "quantity": position.quantity,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "entry_notional": entry_notional,
            "exit_notional": exit_notional,
            "gross_pnl": gross_pnl,
            "entry_fee": position.entry_fee,
            "exit_fee": exit_fee,
            "funding_pnl": position.funding_pnl,
            "net_pnl": net_pnl,
            "return_on_notional": net_pnl / max(entry_notional, _EPSILON),
            "holding_bars": holding_bars,
            "holding_hours": holding_hours,
            "exit_reason": exit_reason,
            "triple_barrier_label": triple_barrier_label,
        }

    def _calculate_metrics(
        self,
        equity_curve: pd.DataFrame,
        trades: pd.DataFrame,
        bar_hours: float,
    ) -> dict[str, float]:
        """Compute risk-adjusted performance, drawdown, and statistical metrics."""
        if equity_curve.empty:
            raise ValueError("Cannot calculate metrics for an empty equity curve.")

        equity = equity_curve["equity"].astype(float)
        returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        returns = returns[np.isfinite(returns)]

        trade_count = float(len(trades))
        net_realized_pnl = float(trades["net_pnl"].sum()) if not trades.empty else 0.0
        gross_profit = (
            float(trades.loc[trades["net_pnl"] > 0.0, "net_pnl"].sum())
            if not trades.empty
            else 0.0
        )
        gross_loss = (
            abs(float(trades.loc[trades["net_pnl"] < 0.0, "net_pnl"].sum()))
            if not trades.empty
            else 0.0
        )

        if gross_loss > _EPSILON:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > _EPSILON:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        winners = trades.loc[trades["net_pnl"] > 0.0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
        losers = trades.loc[trades["net_pnl"] < 0.0, "net_pnl"] if not trades.empty else pd.Series(dtype=float)
        win_rate_pct = float(len(winners) / len(trades) * 100.0) if len(trades) > 0 else 0.0

        average_win = float(winners.mean()) if not winners.empty else 0.0
        average_loss = abs(float(losers.mean())) if not losers.empty else 0.0
        if average_loss > _EPSILON:
            payoff_ratio = average_win / average_loss
        elif average_win > _EPSILON:
            payoff_ratio = float("inf")
        else:
            payoff_ratio = 0.0

        periods_per_year = (365.0 * 24.0) / max(bar_hours, _EPSILON)
        period_sharpe, annualized_sharpe = self._sharpe_ratios(returns, periods_per_year)
        annualized_sortino = self._sortino_ratio(returns, periods_per_year)

        running_peak = equity.cummax()
        drawdown = equity / running_peak - 1.0
        max_drawdown_pct = abs(float(drawdown.min())) * 100.0
        max_drawdown_duration_bars = self._maximum_drawdown_duration_bars(drawdown)
        max_drawdown_duration_hours = max_drawdown_duration_bars * bar_hours
        max_drawdown_duration_days = max_drawdown_duration_hours / 24.0

        total_return_pct = (
            (float(equity.iloc[-1]) / max(float(equity.iloc[0]), _EPSILON) - 1.0) * 100.0
        )
        elapsed_years = max((len(equity) - 1) / periods_per_year, _EPSILON)
        try:
            power = min(1.0 / elapsed_years, 100.0)
            ratio = max(float(equity.iloc[-1]), _EPSILON) / max(float(equity.iloc[0]), _EPSILON)
            annualized_return = (ratio ** power) - 1.0
        except (OverflowError, ValueError):
            annualized_return = float("inf") if ratio > 1.0 else -1.0
        calmar_ratio = (
            annualized_return / (max_drawdown_pct / 100.0)
            if max_drawdown_pct > _EPSILON
            else float("nan")
        )

        skewness = self._sample_skewness(returns.to_numpy(dtype=float))
        kurtosis = self._sample_kurtosis(returns.to_numpy(dtype=float))
        psr, dsr, expected_max_sharpe, min_trl_days = self._statistical_sharpe_metrics(
            period_sharpe=period_sharpe,
            returns_count=len(returns),
            skewness=skewness,
            kurtosis=kurtosis,
            bar_hours=bar_hours,
            number_of_trials=max(int(trade_count), 1),
        )

        return {
            "net_realized_pnl": net_realized_pnl,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": profit_factor,
            "payoff_ratio": payoff_ratio,
            "win_rate_pct": win_rate_pct,
            "trade_count": trade_count,
            "total_return_pct": total_return_pct,
            "annualized_return_pct": annualized_return * 100.0,
            "annualized_sharpe_ratio": annualized_sharpe,
            "annualized_sortino_ratio": annualized_sortino,
            "max_drawdown_pct": max_drawdown_pct,
            "max_drawdown_duration_hours": max_drawdown_duration_hours,
            "max_drawdown_duration_days": max_drawdown_duration_days,
            "calmar_ratio": calmar_ratio,
            "return_skewness": skewness,
            "return_kurtosis": kurtosis,
            "psr_pct": psr * 100.0,
            "dsr_pct": dsr * 100.0,
            "expected_max_sharpe": expected_max_sharpe,
            "min_trl_days": min_trl_days,
        }

    @staticmethod
    def _sharpe_ratios(
        returns: pd.Series,
        periods_per_year: float,
    ) -> tuple[float, float]:
        """Calculate period-level and annualized net Sharpe ratios."""
        if len(returns) < 2:
            return float("nan"), float("nan")

        mean_return = float(returns.mean())
        return_std = float(returns.std(ddof=1))
        if return_std <= _EPSILON:
            if mean_return > 0.0:
                return float("inf"), float("inf")
            if mean_return < 0.0:
                return float("-inf"), float("-inf")
            return 0.0, 0.0

        period_sharpe = mean_return / return_std
        annualized_sharpe = period_sharpe * sqrt(periods_per_year)
        return period_sharpe, annualized_sharpe

    @staticmethod
    def _sortino_ratio(returns: pd.Series, periods_per_year: float) -> float:
        """Calculate annualized Sortino ratio using zero as the target return."""
        if len(returns) < 2:
            return float("nan")

        mean_return = float(returns.mean())
        downside = returns[returns < 0.0]
        if downside.empty:
            return float("inf") if mean_return > 0.0 else 0.0

        downside_deviation = float(np.sqrt(np.mean(np.square(downside.to_numpy(dtype=float)))))
        if downside_deviation <= _EPSILON:
            return float("inf") if mean_return > 0.0 else 0.0

        return mean_return / downside_deviation * sqrt(periods_per_year)

    @staticmethod
    def _maximum_drawdown_duration_bars(drawdown: pd.Series) -> int:
        """Calculate the longest uninterrupted duration spent below the high-water mark."""
        longest_duration = 0
        current_duration = 0

        for value in drawdown.to_numpy(dtype=float):
            if value < -_EPSILON:
                current_duration += 1
                longest_duration = max(longest_duration, current_duration)
            else:
                current_duration = 0

        return longest_duration

    @staticmethod
    def _sample_skewness(values: np.ndarray) -> float:
        """Calculate bias-adjusted sample skewness."""
        n = len(values)
        if n < 3:
            return 0.0

        centered = values - np.mean(values)
        sample_std = np.std(values, ddof=1)
        if sample_std <= _EPSILON:
            return 0.0

        standardized_third = np.sum((centered / sample_std) ** 3)
        return float(n / ((n - 1) * (n - 2)) * standardized_third)

    @staticmethod
    def _sample_kurtosis(values: np.ndarray) -> float:
        """Calculate bias-adjusted Pearson sample kurtosis, where Gaussian kurtosis is 3."""
        n = len(values)
        if n < 4:
            return 3.0

        centered = values - np.mean(values)
        sample_std = np.std(values, ddof=1)
        if sample_std <= _EPSILON:
            return 3.0

        standardized_fourth_sum = float(np.sum((centered / sample_std) ** 4))
        excess_kurtosis = (
            n * (n + 1) * standardized_fourth_sum / ((n - 1) * (n - 2) * (n - 3))
            - 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        )
        return excess_kurtosis + 3.0

    def _statistical_sharpe_metrics(
        self,
        period_sharpe: float,
        returns_count: int,
        skewness: float,
        kurtosis: float,
        bar_hours: float,
        number_of_trials: int,
    ) -> tuple[float, float, float, float]:
        """
        Calculate PSR, DSR, expected maximum Sharpe ratio, and MinTRL.

        PSR and DSR use the López de Prado / Bailey non-normal return correction.
        DSR additionally uses an expected maximum Sharpe hurdle induced by multiple
        strategy-selection trials.
        """
        if returns_count < 2 or not isfinite(period_sharpe):
            return float("nan"), float("nan"), float("nan"), float("nan")

        variance_component = (
            1.0
            - skewness * period_sharpe
            + ((kurtosis - 1.0) / 4.0) * period_sharpe * period_sharpe
        )
        variance_component = max(variance_component, _EPSILON)
        sharpe_standard_error = sqrt(variance_component / max(returns_count - 1, 1))

        psr = self._probabilistic_sharpe_ratio(
            observed_sharpe=period_sharpe,
            benchmark_sharpe=0.0,
            returns_count=returns_count,
            skewness=skewness,
            kurtosis=kurtosis,
        )

        if number_of_trials <= 1:
            expected_max_sharpe = 0.0
        else:
            euler_mascheroni = 0.5772156649015329
            p_one = float(np.clip(1.0 - 1.0 / number_of_trials, _EPSILON, 1.0 - _EPSILON))
            p_two = float(
                np.clip(
                    1.0 - 1.0 / (number_of_trials * np.e),
                    _EPSILON,
                    1.0 - _EPSILON,
                )
            )
            expected_max_sharpe = sharpe_standard_error * (
                (1.0 - euler_mascheroni) * _NORMAL_DIST.inv_cdf(p_one)
                + euler_mascheroni * _NORMAL_DIST.inv_cdf(p_two)
            )

        dsr = self._probabilistic_sharpe_ratio(
            observed_sharpe=period_sharpe,
            benchmark_sharpe=expected_max_sharpe,
            returns_count=returns_count,
            skewness=skewness,
            kurtosis=kurtosis,
        )

        alpha = 0.05
        z_alpha = _NORMAL_DIST.inv_cdf(1.0 - alpha)
        if period_sharpe <= _EPSILON:
            min_trl_days = float("inf")
        else:
            min_track_record_periods = 1.0 + variance_component * (
                z_alpha / period_sharpe
            ) ** 2
            min_trl_days = min_track_record_periods * bar_hours / 24.0

        return psr, dsr, expected_max_sharpe, min_trl_days

    @staticmethod
    def _probabilistic_sharpe_ratio(
        observed_sharpe: float,
        benchmark_sharpe: float,
        returns_count: int,
        skewness: float,
        kurtosis: float,
    ) -> float:
        """Calculate the López de Prado probabilistic Sharpe ratio."""
        if returns_count < 2:
            return float("nan")

        denominator_component = (
            1.0
            - skewness * observed_sharpe
            + ((kurtosis - 1.0) / 4.0) * observed_sharpe * observed_sharpe
        )
        denominator = sqrt(max(denominator_component, _EPSILON))
        z_score = (
            (observed_sharpe - benchmark_sharpe)
            * sqrt(returns_count - 1)
            / denominator
        )
        return float(np.clip(_NORMAL_DIST.cdf(z_score), 0.0, 1.0))

    @staticmethod
    def _format_currency(value: float) -> str:
        """Format a numeric value as a USD currency string."""
        if not isfinite(value):
            return "N/A"
        return f"${value:,.2f}"

    @staticmethod
    def _format_percent(value: float) -> str:
        """Format a numeric value as a percentage string."""
        if np.isnan(value):
            return "N/A"
        if np.isposinf(value):
            return "∞"
        if np.isneginf(value):
            return "-∞"
        return f"{value:,.2f}%"

    @staticmethod
    def _format_number(value: float) -> str:
        """Format finite, infinite, and unavailable numeric values safely."""
        if np.isnan(value):
            return "N/A"
        if np.isposinf(value):
            return "∞"
        if np.isneginf(value):
            return "-∞"
        return f"{value:,.4f}"

    @staticmethod
    def _format_integer(value: float) -> str:
        """Format a numeric value as an integer."""
        if not isfinite(value):
            return "N/A"
        return f"{int(round(value)):,}"

    @staticmethod
    def _is_maker_result(result: BacktestResult) -> bool:
        """Determine whether the completed result includes maker execution."""
        if result.events.empty:
            return False
        return bool((result.events["event_type"] == "maker_fill").any())


def _generate_synthetic_market_data(
    periods: int = 2_000,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate deterministic synthetic hourly futures data and directional signals.

    Args:
        periods: Number of hourly candles to generate.
        seed: Random seed.

    Returns:
        Tuple containing OHLCV/funding data and moving-average directional signals.

    Raises:
        ValueError: If periods is insufficient for signal formation.
    """
    if periods < 100:
        raise ValueError("periods must be at least 100.")

    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-01", periods=periods, freq="h", tz="UTC")

    innovations = rng.normal(loc=0.00008, scale=0.012, size=periods)
    close = 25_000.0 * np.exp(np.cumsum(innovations))
    open_prices = np.concatenate(([close[0]], close[:-1]))
    intrabar_range = rng.uniform(0.001, 0.012, size=periods)
    high = np.maximum(open_prices, close) * (1.0 + intrabar_range)
    low = np.minimum(open_prices, close) * (1.0 - intrabar_range)
    volume = rng.lognormal(mean=10.8, sigma=0.45, size=periods)
    spread = close * rng.uniform(0.00002, 0.00015, size=periods)
    funding_rate = rng.normal(loc=0.00001, scale=0.00003, size=periods)
    queue_depth_proxy = volume * close * rng.uniform(0.05, 0.25, size=periods)

    frame = pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "spread": spread,
            "funding_rate": funding_rate,
            "queue_depth_proxy": queue_depth_proxy,
        },
        index=index,
    )

    fast_ma = frame["close"].rolling(20, min_periods=20).mean()
    slow_ma = frame["close"].rolling(60, min_periods=60).mean()
    signals = pd.Series(
        np.where(fast_ma > slow_ma, 1, np.where(fast_ma < slow_ma, -1, 0)),
        index=index,
        name="synthetic_signal",
    )
    return frame, signals


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    market_data, synthetic_signals = _generate_synthetic_market_data()
    engine = InstitutionalBacktester(
        BacktestConfig(
            initial_capital=100_000.0,
            leverage=3.0,
            allocation_fraction=0.25,
            max_holding_bars=48,
            random_seed=42,
        )
    )
    engine.run_backtest(
        df=market_data,
        signals=synthetic_signals,
        stop_atr_mult=2.0,
        tp_atr_mult=3.0,
        maker_execution=True,
    )
    print(engine.generate_report())