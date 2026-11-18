"""Institutional Monte Carlo risk simulator for constrained micro-capital futures portfolios.

The simulator models correlated Student's t-copula returns, flash-crash jump diffusion,
asymmetric execution costs, funding shocks, Binance USD-M contract minimums, and
portfolio-level risk statistics over 30, 90, and 365 trading-day horizons.
"""

from __future__ import annotations

import argparse
import math
import sys
import threading
from dataclasses import dataclass, field
from itertools import product
from typing import Final, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


TRADING_DAYS_PER_YEAR: Final[int] = 252
TRADING_DAYS_PER_MONTH: Final[int] = 21
DEFAULT_HORIZONS: Final[tuple[int, int, int]] = (30, 90, 365)
MIN_SUPPORTED_EQUITY: Final[float] = 0.01
MAX_SUPPORTED_EQUITY: Final[float] = 1_000.00


class SimulationError(RuntimeError):
    """Raised when a Monte Carlo simulation cannot be completed safely."""


@dataclass(frozen=True, slots=True)
class AssetContract:
    """Represents a Binance USD-M futures contract and model parameters."""

    symbol: str
    reference_price: float
    minimum_quantity: float
    target_weight: float
    annual_expected_return: float
    annual_volatility: float
    jump_beta: float = 1.0

    def __post_init__(self) -> None:
        """Validate contract and stochastic-model inputs."""
        if not self.symbol.strip():
            raise ValueError("Asset symbol must be non-empty.")
        if not math.isfinite(self.reference_price) or self.reference_price <= 0.0:
            raise ValueError(f"{self.symbol}: reference_price must be finite and positive.")
        if not math.isfinite(self.minimum_quantity) or self.minimum_quantity <= 0.0:
            raise ValueError(f"{self.symbol}: minimum_quantity must be finite and positive.")
        if not math.isfinite(self.target_weight) or self.target_weight < 0.0:
            raise ValueError(f"{self.symbol}: target_weight must be finite and non-negative.")
        if not math.isfinite(self.annual_expected_return):
            raise ValueError(f"{self.symbol}: annual_expected_return must be finite.")
        if not math.isfinite(self.annual_volatility) or self.annual_volatility <= 0.0:
            raise ValueError(f"{self.symbol}: annual_volatility must be finite and positive.")
        if not math.isfinite(self.jump_beta) or self.jump_beta < 0.0:
            raise ValueError(f"{self.symbol}: jump_beta must be finite and non-negative.")

    @property
    def minimum_notional(self) -> float:
        """Return the approximate USD notional of one minimum contract lot."""
        return self.reference_price * self.minimum_quantity


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Immutable configuration governing the Monte Carlo process."""

    horizons: tuple[int, int, int] = DEFAULT_HORIZONS
    student_t_degrees_of_freedom: float = 3.8
    monthly_jump_intensity: float = 0.02
    jump_lower_bound: float = -0.20
    jump_upper_bound: float = -0.10
    base_daily_funding_rate: float = 0.00020
    funding_shock_probability_daily: float = 0.015
    funding_shock_scale: float = 0.00075
    base_slippage_bps: float = 4.0
    slippage_tail_bps: float = 7.5
    adverse_slippage_multiplier: float = 1.75
    cash_penalty_weight: float = 0.20
    batch_size: int = 2_048
    correlation: tuple[tuple[float, float, float], ...] = (
        (1.00, 0.72, 0.65),
        (0.72, 1.00, 0.70),
        (0.65, 0.70, 1.00),
    )
    assets: tuple[AssetContract, AssetContract, AssetContract] = (
        AssetContract(
            symbol="BTCUSDT",
            reference_price=85_000.00,
            minimum_quantity=0.001,
            target_weight=0.35,
            annual_expected_return=0.12,
            annual_volatility=0.65,
            jump_beta=1.00,
        ),
        AssetContract(
            symbol="ETHUSDT",
            reference_price=3_000.00,
            minimum_quantity=0.01,
            target_weight=0.35,
            annual_expected_return=0.14,
            annual_volatility=0.80,
            jump_beta=0.90,
        ),
        AssetContract(
            symbol="SOLUSDT",
            reference_price=200.00,
            minimum_quantity=0.10,
            target_weight=0.30,
            annual_expected_return=0.16,
            annual_volatility=1.00,
            jump_beta=0.85,
        ),
    )

    def __post_init__(self) -> None:
        """Validate that the configuration is mathematically and operationally sound."""
        if self.horizons != DEFAULT_HORIZS if False else False:
            pass
        if tuple(self.horizons) != DEFAULT_HORIZONS:
            raise ValueError(
                f"Simulation horizons must be exactly {DEFAULT_HORIZONS}; received {self.horizons}."
            )
        if not 3.5 <= self.student_t_degrees_of_freedom <= 4.0:
            raise ValueError(
                "student_t_degrees_of_freedom must be in the institutional range [3.5, 4.0]."
            )
        if not math.isfinite(self.monthly_jump_intensity) or self.monthly_jump_intensity < 0.0:
            raise ValueError("monthly_jump_intensity must be finite and non-negative.")
        if not (
            math.isfinite(self.jump_lower_bound)
            and math.isfinite(self.jump_upper_bound)
            and -1.0 < self.jump_lower_bound < self.jump_upper_bound < 0.0
        ):
            raise ValueError(
                "Jump bounds must satisfy -1.0 < jump_lower_bound < jump_upper_bound < 0.0."
            )
        if not math.isfinite(self.base_daily_funding_rate) or self.base_daily_funding_rate < 0.0:
            raise ValueError("base_daily_funding_rate must be finite and non-negative.")
        if not (
            math.isfinite(self.funding_shock_probability_daily)
            and 0.0 <= self.funding_shock_probability_daily <= 1.0
        ):
            raise ValueError(
                "funding_shock_probability_daily must be finite and in the interval [0, 1]."
            )
        if not math.isfinite(self.funding_shock_scale) or self.funding_shock_scale < 0.0:
            raise ValueError("funding_shock_scale must be finite and non-negative.")
        if not math.isfinite(self.base_slippage_bps) or self.base_slippage_bps < 0.0:
            raise ValueError("base_slippage_bps must be finite and non-negative.")
        if not math.isfinite(self.slippage_tail_bps) or self.slippage_tail_bps < 0.0:
            raise ValueError("slippage_tail_bps must be finite and non-negative.")
        if (
            not math.isfinite(self.adverse_slippage_multiplier)
            or self.adverse_slippage_multiplier < 1.0
        ):
            raise ValueError("adverse_slippage_multiplier must be finite and at least 1.0.")
        if not math.isfinite(self.cash_penalty_weight) or self.cash_penalty_weight < 0.0:
            raise ValueError("cash_penalty_weight must be finite and non-negative.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be strictly positive.")
        if len(self.assets) != 3:
            raise ValueError("This institutional model requires exactly BTC, ETH, and SOL contracts.")
        if len(self.correlation) != len(self.assets) or any(
            len(row) != len(self.assets) for row in self.correlation
        ):
            raise ValueError("Correlation matrix dimensions must equal the number of assets.")

        target_weight_sum = sum(asset.target_weight for asset in self.assets)
        if not math.isclose(target_weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"Asset target weights must sum to 1.0; received {target_weight_sum:.12f}."
            )

        correlation_matrix = np.asarray(self.correlation, dtype=np.float64)
        if not np.all(np.isfinite(correlation_matrix)):
            raise ValueError("Correlation matrix contains non-finite values.")
        if not np.allclose(correlation_matrix, correlation_matrix.T, atol=1e-12):
            raise ValueError("Correlation matrix must be symmetric.")
        if not np.allclose(np.diag(correlation_matrix), 1.0, atol=1e-12):
            raise ValueError("Correlation matrix diagonal entries must equal 1.0.")

        try:
            np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Correlation matrix must be positive definite.") from exc


@dataclass(frozen=True, slots=True)
class PortfolioAllocation:
    """Quantized initial portfolio allocation satisfying contract lot minimums."""

    initial_equity: float
    quantities: Mapping[str, float]
    notionals: Mapping[str, float]
    weights: Mapping[str, float]
    cash_reserve: float

    @property
    def gross_exposure(self) -> float:
        """Return total initial gross futures exposure divided by initial equity."""
        if self.initial_equity <= 0.0:
            return 0.0
        return sum(self.notionals.values()) / self.initial_equity

    @property
    def invested_notional(self) -> float:
        """Return initial futures notional in USD."""
        return sum(self.notionals.values())


@dataclass(frozen=True, slots=True)
class HorizonRiskMetrics:
    """Risk measures generated from one forecast horizon."""

    horizon_days: int
    simulations: int
    probability_of_ruin: float
    probability_capital_breach_5pct: float
    probability_capital_breach_15pct: float
    var_95: float
    var_99: float
    var_999: float
    cvar_95: float
    cvar_99: float
    cvar_999: float
    maximum_drawdown_median: float
    maximum_drawdown_95th_percentile: float
    maximum_drawdown_worst_case: float
    longest_drawdown_duration_median_days: float
    longest_drawdown_duration_95th_percentile_days: float
    longest_drawdown_duration_worst_case_days: int
    terminal_equity_median: float
    terminal_equity_5th_percentile: float
    terminal_equity_worst_case: float
    win_probability: float
    payoff_ratio: float
    full_kelly_fraction: float
    half_kelly_fraction: float
    quarter_kelly_fraction: float


@dataclass(frozen=True, slots=True)
class _SimulationSample:
    """Internal compact representation of simulated path outcomes."""

    final_equity: NDArray[np.float64]
    maximum_drawdown: NDArray[np.float64]
    longest_drawdown_duration: NDArray[np.int64]
    ruined: NDArray[np.bool_]


@dataclass(slots=True)
class MonteCarloRiskSimulator:
    """Simulate fat-tailed, jump-diffusion portfolio equity paths under contract constraints.

    Parameters
    ----------
    equity:
        Initial account equity in USD. The supported micro-capital domain is greater
        than zero and no greater than $1,000.
    trades:
        Expected number of round-trip or turnover events per 21-trading-day month.
        Each event receives asymmetric slippage treatment.
    simulations:
        Number of independent Monte Carlo paths. The institutional default is 50,000.
    config:
        Optional immutable simulation configuration.
    seed:
        Optional NumPy random seed. A supplied seed makes repeated runs reproducible.
    """

    equity: float
    trades: int = 30
    simulations: int = 50_000
    config: SimulationConfig = field(default_factory=SimulationConfig)
    seed: int | None = 17_291
    _allocation: PortfolioAllocation = field(init=False, repr=False)
    _raw_samples: dict[int, _SimulationSample] = field(init=False, default_factory=dict, repr=False)
    _metrics: dict[int, HorizonRiskMetrics] = field(
        init=False, default_factory=dict, repr=False
    )
    _lock: threading.RLock = field(init=False, default_factory=threading.RLock, repr=False)

    def __post_init__(self) -> None:
        """Validate constructor inputs and compute the quantized contract allocation."""
        if not math.isfinite(self.equity) or self.equity < MIN_SUPPORTED_EQUITY:
            raise ValueError(
                f"equity must be finite and at least ${MIN_SUPPORTED_EQUITY:.2f}; "
                f"received {self.equity!r}."
            )
        if self.equity > MAX_SUPPORTED_EQUITY:
            raise ValueError(
                f"equity exceeds the micro-capital mandate maximum of "
                f"${MAX_SUPPORTED_EQUITY:,.2f}."
            )
        if isinstance(self.trades, bool) or not isinstance(self.trades, int) or self.trades < 0:
            raise ValueError("trades must be a non-negative integer.")
        if (
            isinstance(self.simulations, bool)
            or not isinstance(self.simulations, int)
            or self.simulations <= 0
        ):
            raise ValueError("simulations must be a strictly positive integer.")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise ValueError("seed must be an integer or None.")

        self._allocation = self._build_contract_constrained_allocation()

    @property
    def allocation(self) -> PortfolioAllocation:
        """Return the initial Binance-lot-constrained allocation."""
        return self._allocation

    def run_simulation(self) -> dict[int, HorizonRiskMetrics]:
        """Run all institutional horizons and return their risk metrics.

        The method regenerates all independent path samples on every invocation. When
        a seed is supplied, each invocation is deterministic and reproducible.
        """
        with self._lock:
            rng = np.random.default_rng(self.seed)
            generated_samples: dict[int, _SimulationSample] = {}

            try:
                for horizon in self.config.horizons:
                    generated_samples[horizon] = self._simulate_horizon(horizon, rng)
            except (FloatingPointError, MemoryError, ValueError, np.linalg.LinAlgError) as exc:
                raise SimulationError(f"Monte Carlo simulation failed: {exc}") from exc

            self._raw_samples = generated_samples
            self._metrics = self._compute_metrics_from_samples()
            return dict(self._metrics)

    def compute_risk_metrics(self) -> dict[int, HorizonRiskMetrics]:
        """Compute and return risk metrics from simulated samples.

        If no samples currently exist, the required simulations are executed first.
        """
        with self._lock:
            if not self._raw_samples:
                return self.run_simulation()

            self._metrics = self._compute_metrics_from_samples()
            return dict(self._metrics)

    def generate_advisory_report(self) -> str:
        """Generate an institutional ASCII capital-allocation advisory report."""
        with self._lock:
            metrics = self.compute_risk_metrics()
            lines: list[str] = []
            width = 102

            def divider(char: str = "=") -> str:
                return char * width

            lines.append(divider())
            lines.append(" ASTRA INSTITUTIONAL AUDIT | MONTE CARLO CAPITAL ALLOCATION ADVISORY")
            lines.append(divider())
            lines.append(
                f" Account Equity: ${self.equity:,.2f} | Simulations: {self.simulations:,} "
                f"| Expected Monthly Trades: {self.trades:,}"
            )
            lines.append(
                " Distribution: Multivariate Student-t Copula "
                f"(nu={self.config.student_t_degrees_of_freedom:.2f}) | "
                f"Jump Intensity: {self.config.monthly_jump_intensity:.3f}/month"
            )
            lines.append(
                f" Flash-Crash Jumps: {self.config.jump_lower_bound:.0%} to "
                f"{self.config.jump_upper_bound:.0%} | "
                "Costs: Asymmetric Slippage + Adverse Funding Shocks"
            )
            lines.append(divider("-"))
            lines.append(" EXECUTABLE BINANCE USD-M CONTRACT ALLOCATION")
            lines.append(divider("-"))

            if self._allocation.invested_notional <= 0.0:
                lines.append(
                    " NO EXECUTABLE CONTRACT POSITION: equity is below all quantized "
                    "minimum-lot purchase requirements."
                )
            else:
                for asset in self.config.assets:
                    quantity = self._allocation.quantities[asset.symbol]
                    notional = self._allocation.notionals[asset.symbol]
                    if quantity > 0.0:
                        lines.append(
                            f" {asset.symbol:<8} Qty {quantity:>10.6f} | "
                            f"Notional ${notional:>9.2f} | "
                            f"Equity Weight {self._allocation.weights[asset.symbol]:>7.2%} | "
                            f"Minimum Lot ${asset.minimum_notional:>7.2f}"
                        )

            lines.append(
                f" Cash Reserve: ${self._allocation.cash_reserve:,.2f} | "
                f"Gross Futures Exposure: {self._allocation.gross_exposure:.2%}"
            )
            lines.append(divider("-"))
            lines.append(
                " HORIZON |   RUIN | DD >= 5% | DD >= 15% |   VaR95 |   VaR99 | VaR99.9 "
                "| CVaR99 | MED MDD | P95 MDD | P95 UW DAYS"
            )
            lines.append(divider("-"))

            for horizon in self.config.horizons:
                metric = metrics[horizon]
                lines.append(
                    f" {horizon:>6}D | {metric.probability_of_ruin:>6.2%} | "
                    f"{metric.probability_capital_breach_5pct:>7.2%} | "
                    f"{metric.probability_capital_breach_15pct:>8.2%} | "
                    f"{metric.var_95:>7.2%} | {metric.var_99:>7.2%} | "
                    f"{metric.var_999:>7.2%} | {metric.cvar_99:>6.2%} | "
                    f"{metric.maximum_drawdown_median:>7.2%} | "
                    f"{metric.maximum_drawdown_95th_percentile:>7.2%} | "
                    f"{metric.longest_drawdown_duration_95th_percentile_days:>10.1f}"
                )

            lines.append(divider("-"))
            lines.append(" KELLY FRACTIONAL SIZING ANALYSIS")
            lines.append(divider("-"))

            long_horizon = metrics[365]
            lines.append(
                f" 365D Win Probability p: {long_horizon.win_probability:.2%} | "
                f"Payoff Ratio R: {long_horizon.payoff_ratio:.4f}"
            )
            lines.append(
                " f* = (p * (R + 1) - 1) / R | "
                f"Full Kelly: {long_horizon.full_kelly_fraction:.2%} | "
                f"Half Kelly: {long_horizon.half_kelly_fraction:.2%} | "
                f"Quarter Kelly: {long_horizon.quarter_kelly_fraction:.2%}"
            )

            quarter_kelly_bounded = min(
                1.0, max(0.0, long_horizon.quarter_kelly_fraction)
            )
            ninety_day = metrics[90]
            if self._allocation.invested_notional <= 0.0:
                advisory = (
                    "ACTION: DO NOT DEPLOY. Account equity cannot satisfy an executable "
                    "Binance USD-M minimum lot while preserving modeled risk controls."
                )
            elif (
                ninety_day.probability_capital_breach_15pct >= 0.25
                or ninety_day.probability_of_ruin >= 0.01
            ):
                advisory = (
                    "ACTION: CAPITAL PRESERVATION MODE. Do not increase exposure; retain "
                    "cash reserve and require a materially lower-risk signal regime."
                )
            elif quarter_kelly_bounded <= 0.0:
                advisory = (
                    "ACTION: NO POSITIVE KELLY EDGE. Maintain cash and reject additional "
                    "risk allocation until expectancy improves."
                )
            else:
                advisory = (
                    "ACTION: FRACTIONAL-KELLY ONLY. The maximum model-derived risk budget "
                    f"is {quarter_kelly_bounded:.2%} of deployable capital before contract "
                    "minimum-lot rounding."
                )

            lines.append(divider("-"))
            lines.append(" MATHEMATICAL CAPITAL ALLOCATION ADVISORY")
            lines.append(divider("-"))
            lines.append(f" {advisory}")
            lines.append(
                f" 365D Terminal Equity: Median ${long_horizon.terminal_equity_median:,.2f} | "
                f"5th Percentile ${long_horizon.terminal_equity_5th_percentile:,.2f} | "
                f"Worst Case ${long_horizon.terminal_equity_worst_case:,.2f}"
            )
            lines.append(
                f" 365D Maximum Drawdown: Median {long_horizon.maximum_drawdown_median:.2%} | "
                f"95th Percentile {long_horizon.maximum_drawdown_95th_percentile:.2%} | "
                f"Worst Case {long_horizon.maximum_drawdown_worst_case:.2%}"
            )
            lines.append(
                f" 365D Underwater Duration: Median "
                f"{long_horizon.longest_drawdown_duration_median_days:.1f} days | "
                f"95th Percentile {long_horizon.longest_drawdown_duration_95th_percentile_days:.1f} "
                f"days | Worst Case {long_horizon.longest_drawdown_duration_worst_case_days:d} days"
            )
            lines.append(divider("-"))
            lines.append(" MICRO-CAPITAL CONTRACT FEASIBILITY REFERENCE")
            lines.append(divider("-"))

            for scenario_equity in (39.54, 150.00, 200.00):
                scenario_allocation = self._build_contract_constrained_allocation(scenario_equity)
                positions = ", ".join(
                    f"{symbol}:{quantity:.6f}"
                    for symbol, quantity in scenario_allocation.quantities.items()
                    if quantity > 0.0
                )
                if not positions:
                    positions = "CASH ONLY"
                lines.append(
                    f" Equity ${scenario_equity:>7.2f} | Positions {positions:<40} | "
                    f"Cash ${scenario_allocation.cash_reserve:>7.2f} | "
                    f"Exposure {scenario_allocation.gross_exposure:>6.2%}"
                )

            lines.append(divider())
            lines.append(
                " MODEL NOTICE: Monte Carlo outputs are probabilistic stress estimates, not "
                "guarantees. Futures liquidation, exchange outages, and liquidity gaps can "
                "produce losses beyond modeled assumptions."
            )
            lines.append(divider())
            return "\n".join(lines)

    def _build_contract_constrained_allocation(
        self, equity: float | None = None
    ) -> PortfolioAllocation:
        """Optimize a lot-quantized allocation under cash and target-weight constraints."""
        capital = self.equity if equity is None else equity
        if not math.isfinite(capital) or capital <= 0.0:
            raise ValueError("Allocation capital must be finite and strictly positive.")

        assets = self.config.assets
        target_weights = np.asarray(
            [asset.target_weight for asset in assets], dtype=np.float64
        )
        minimum_notionals = np.asarray(
            [asset.minimum_notional for asset in assets], dtype=np.float64
        )

        max_lots = np.floor((capital + 1e-10) / minimum_notionals).astype(np.int64)
        best_lots = np.zeros(len(assets), dtype=np.int64)
        best_score = float("inf")
        best_invested = -1.0

        ranges: Sequence[range] = tuple(range(int(maximum) + 1) for maximum in max_lots)
        for lot_tuple in product(*ranges):
            lots = np.asarray(lot_tuple, dtype=np.int64)
            notionals = lots.astype(np.float64) * minimum_notionals
            invested = float(np.sum(notionals))
            if invested > capital + 1e-8:
                continue

            weights = notionals / capital
            cash_ratio = max(0.0, (capital - invested) / capital)
            score = float(
                np.sum(np.square(weights - target_weights))
                + self.config.cash_penalty_weight * cash_ratio * cash_ratio
            )

            if (
                score < best_score - 1e-14
                or (
                    math.isclose(score, best_score, abs_tol=1e-14)
                    and invested > best_invested + 1e-10
                )
            ):
                best_score = score
                best_invested = invested
                best_lots = lots

        quantities: dict[str, float] = {}
        notionals_by_symbol: dict[str, float] = {}
        weights_by_symbol: dict[str, float] = {}

        for asset, lots in zip(assets, best_lots, strict=True):
            quantity = float(lots) * asset.minimum_quantity
            notional = quantity * asset.reference_price
            quantities[asset.symbol] = quantity
            notionals_by_symbol[asset.symbol] = notional
            weights_by_symbol[asset.symbol] = notional / capital

        invested_notional = sum(notionals_by_symbol.values())
        cash_reserve = max(0.0, capital - invested_notional)

        return PortfolioAllocation(
            initial_equity=capital,
            quantities=quantities,
            notionals=notionals_by_symbol,
            weights=weights_by_symbol,
            cash_reserve=cash_reserve,
        )

    def _simulate_horizon(
        self,
        horizon_days: int,
        rng: np.random.Generator,
    ) -> _SimulationSample:
        """Generate independent Student-t-copula jump-diffusion paths for one horizon."""
        if horizon_days <= 0:
            raise ValueError("horizon_days must be strictly positive.")

        assets = self.config.assets
        asset_count = len(assets)
        correlation = np.asarray(self.config.correlation, dtype=np.float64)
        cholesky = np.linalg.cholesky(correlation)

        annual_returns = np.asarray(
            [asset.annual_expected_return for asset in assets], dtype=np.float64
        )
        annual_volatility = np.asarray(
            [asset.annual_volatility for asset in assets], dtype=np.float64
        )
        jump_betas = np.asarray([asset.jump_beta for asset in assets], dtype=np.float64)
        portfolio_weights = np.asarray(
            [self._allocation.weights[asset.symbol] for asset in assets],
            dtype=np.float64,
        )
        gross_exposure = float(np.sum(portfolio_weights))

        daily_mean = annual_returns / TRADING_DAYS_PER_YEAR
        daily_volatility = annual_volatility / math.sqrt(TRADING_DAYS_PER_YEAR)
        nu = self.config.student_t_degrees_of_freedom
        t_standardization = math.sqrt(nu / (nu - 2.0))
        daily_jump_probability = 1.0 - math.exp(
            -self.config.monthly_jump_intensity / TRADING_DAYS_PER_MONTH
        )
        daily_trade_intensity = self.trades / TRADING_DAYS_PER_MONTH

        final_equity = np.empty(self.simulations, dtype=np.float64)
        maximum_drawdown = np.empty(self.simulations, dtype=np.float64)
        longest_drawdown_duration = np.empty(self.simulations, dtype=np.int64)
        ruined = np.empty(self.simulations, dtype=np.bool_)

        for start_index in range(0, self.simulations, self.config.batch_size):
            stop_index = min(start_index + self.config.batch_size, self.simulations)
            batch_count = stop_index - start_index

            independent_normals = rng.standard_normal(
                size=(batch_count, horizon_days, asset_count)
            )
            correlated_normals = independent_normals @ cholesky.T
            chi_square = rng.chisquare(
                df=nu,
                size=(batch_count, horizon_days, 1),
            )
            student_t_innovations = (
                correlated_normals / np.sqrt(chi_square / nu) / t_standardization
            )

            asset_returns = (
                daily_mean[None, None, :]
                + daily_volatility[None, None, :] * student_t_innovations
            )

            jump_occurs = (
                rng.random(size=(batch_count, horizon_days, 1)) < daily_jump_probability
            )
            jump_sizes = rng.uniform(
                low=self.config.jump_lower_bound,
                high=self.config.jump_upper_bound,
                size=(batch_count, horizon_days, 1),
            )
            asset_returns += np.where(jump_occurs, jump_sizes, 0.0) * jump_betas[
                None, None, :
            ]

            portfolio_returns = np.einsum(
                "bda,a->bd",
                asset_returns,
                portfolio_weights,
                optimize=True,
            )

            if gross_exposure > 0.0:
                funding_shock_occurs = (
                    rng.random(size=(batch_count, horizon_days))
                    < self.config.funding_shock_probability_daily
                )
                funding_shocks = rng.exponential(
                    scale=self.config.funding_shock_scale,
                    size=(batch_count, horizon_days),
                )
                funding_rates = self.config.base_daily_funding_rate + np.where(
                    funding_shock_occurs,
                    funding_shocks,
                    0.0,
                )
                portfolio_returns -= gross_exposure * funding_rates

                trade_events = rng.poisson(
                    lam=daily_trade_intensity,
                    size=(batch_count, horizon_days),
                )
                slippage_per_event = (
                    self.config.base_slippage_bps
                    + rng.exponential(
                        scale=self.config.slippage_tail_bps,
                        size=(batch_count, horizon_days),
                    )
                ) / 10_000.0
                asymmetric_multiplier = np.where(
                    portfolio_returns < 0.0,
                    self.config.adverse_slippage_multiplier,
                    1.0,
                )
                portfolio_returns -= trade_events * slippage_per_event * asymmetric_multiplier

            if not np.all(np.isfinite(portfolio_returns)):
                raise FloatingPointError("Non-finite portfolio return generated.")

            normalized_equity = np.ones(batch_count, dtype=np.float64)
            high_watermark = np.ones(batch_count, dtype=np.float64)
            batch_maximum_drawdown = np.zeros(batch_count, dtype=np.float64)
            current_underwater_duration = np.zeros(batch_count, dtype=np.int64)
            batch_longest_underwater_duration = np.zeros(batch_count, dtype=np.int64)
            batch_ruined = np.zeros(batch_count, dtype=np.bool_)

            for day_index in range(horizon_days):
                daily_multiplier = 1.0 + portfolio_returns[:, day_index]
                invalid_or_ruin = (
                    ~np.isfinite(daily_multiplier)
                    | (daily_multiplier <= 0.0)
                    | batch_ruined
                )

                next_equity = normalized_equity * daily_multiplier
                invalid_or_ruin |= ~np.isfinite(next_equity) | (next_equity <= 0.0)
                batch_ruined |= invalid_or_ruin
                normalized_equity = np.where(batch_ruined, 0.0, next_equity)

                high_watermark = np.maximum(high_watermark, normalized_equity)
                current_drawdown = 1.0 - np.divide(
                    normalized_equity,
                    high_watermark,
                    out=np.ones_like(normalized_equity),
                    where=high_watermark > 0.0,
                )
                current_drawdown = np.clip(current_drawdown, 0.0, 1.0)
                batch_maximum_drawdown = np.maximum(
                    batch_maximum_drawdown,
                    current_drawdown,
                )

                underwater = normalized_equity < high_watermark
                current_underwater_duration = np.where(
                    underwater,
                    current_underwater_duration + 1,
                    0,
                )
                batch_longest_underwater_duration = np.maximum(
                    batch_longest_underwater_duration,
                    current_underwater_duration,
                )

            final_equity[start_index:stop_index] = normalized_equity * self.equity
            maximum_drawdown[start_index:stop_index] = batch_maximum_drawdown
            longest_drawdown_duration[start_index:stop_index] = (
                batch_longest_underwater_duration
            )
            ruined[start_index:stop_index] = batch_ruined

        return _SimulationSample(
            final_equity=final_equity,
            maximum_drawdown=maximum_drawdown,
            longest_drawdown_duration=longest_drawdown_duration,
            ruined=ruined,
        )

    def _compute_metrics_from_samples(self) -> dict[int, HorizonRiskMetrics]:
        """Transform raw path samples into horizon-specific institutional risk metrics."""
        if not self._raw_samples:
            raise SimulationError("No raw simulation samples are available.")

        metrics: dict[int, HorizonRiskMetrics] = {}
        for horizon, sample in self._raw_samples.items():
            if sample.final_equity.size != self.simulations:
                raise SimulationError(
                    f"Sample size mismatch for horizon {horizon}: "
                    f"expected {self.simulations}, got {sample.final_equity.size}."
                )

            terminal_losses = 1.0 - (sample.final_equity / self.equity)
            var_95 = self._value_at_risk(terminal_losses, 0.95)
            var_99 = self._value_at_risk(terminal_losses, 0.99)
            var_999 = self._value_at_risk(terminal_losses, 0.999)

            terminal_returns = (sample.final_equity / self.equity) - 1.0
            winning_returns = terminal_returns[terminal_returns > 0.0]
            losing_returns = terminal_returns[terminal_returns < 0.0]
            win_probability = float(winning_returns.size / terminal_returns.size)
            payoff_ratio, full_kelly = self._kelly_fraction(
                win_probability=win_probability,
                winning_returns=winning_returns,
                losing_returns=losing_returns,
            )

            metrics[horizon] = HorizonRiskMetrics(
                horizon_days=horizon,
                simulations=self.simulations,
                probability_of_ruin=float(np.mean(sample.ruined)),
                probability_capital_breach_5pct=float(
                    np.mean(sample.maximum_drawdown >= 0.05)
                ),
                probability_capital_breach_15pct=float(
                    np.mean(sample.maximum_drawdown >= 0.15)
                ),
                var_95=var_95,
                var_99=var_99,
                var_999=var_999,
                cvar_95=self._conditional_value_at_risk(terminal_losses, var_95),
                cvar_99=self._conditional_value_at_risk(terminal_losses, var_99),
                cvar_999=self._conditional_value_at_risk(terminal_losses, var_999),
                maximum_drawdown_median=float(np.quantile(sample.maximum_drawdown, 0.50)),
                maximum_drawdown_95th_percentile=float(
                    np.quantile(sample.maximum_drawdown, 0.95)
                ),
                maximum_drawdown_worst_case=float(np.max(sample.maximum_drawdown)),
                longest_drawdown_duration_median_days=float(
                    np.quantile(sample.longest_drawdown_duration, 0.50)
                ),
                longest_drawdown_duration_95th_percentile_days=float(
                    np.quantile(sample.longest_drawdown_duration, 0.95)
                ),
                longest_drawdown_duration_worst_case_days=int(
                    np.max(sample.longest_drawdown_duration)
                ),
                terminal_equity_median=float(np.quantile(sample.final_equity, 0.50)),
                terminal_equity_5th_percentile=float(
                    np.quantile(sample.final_equity, 0.05)
                ),
                terminal_equity_worst_case=float(np.min(sample.final_equity)),
                win_probability=win_probability,
                payoff_ratio=payoff_ratio,
                full_kelly_fraction=full_kelly,
                half_kelly_fraction=0.5 * full_kelly,
                quarter_kelly_fraction=0.25 * full_kelly,
            )

        return metrics

    @staticmethod
    def _value_at_risk(losses: NDArray[np.float64], confidence: float) -> float:
        """Return loss-distribution Value at Risk at the requested confidence."""
        if losses.size == 0:
            raise SimulationError("Cannot calculate VaR from an empty loss distribution.")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must lie strictly between zero and one.")
        return float(np.quantile(losses, confidence))

    @staticmethod
    def _conditional_value_at_risk(
        losses: NDArray[np.float64],
        value_at_risk: float,
    ) -> float:
        """Return Expected Shortfall over the loss tail at or above VaR."""
        tail_losses = losses[losses >= value_at_risk]
        if tail_losses.size == 0:
            return float(value_at_risk)
        return float(np.mean(tail_losses))

    @staticmethod
    def _kelly_fraction(
        win_probability: float,
        winning_returns: NDArray[np.float64],
        losing_returns: NDArray[np.float64],
    ) -> tuple[float, float]:
        """Calculate payoff ratio and raw full-Kelly fraction from terminal returns."""
        if winning_returns.size == 0:
            return 0.0, 0.0

        if losing_returns.size == 0:
            return float("inf"), 1.0

        average_win = float(np.mean(winning_returns))
        average_loss = float(abs(np.mean(losing_returns)))

        if average_win <= 0.0 or average_loss <= 0.0:
            return 0.0, 0.0

        payoff_ratio = average_win / average_loss
        kelly_fraction = (
            win_probability * (payoff_ratio + 1.0) - 1.0
        ) / payoff_ratio

        if not math.isfinite(kelly_fraction):
            return payoff_ratio, 0.0

        return payoff_ratio, float(kelly_fraction)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for institutional simulation execution."""
    parser = argparse.ArgumentParser(
        description=(
            "Run an institutional Student-t-copula jump-diffusion Monte Carlo "
            "risk audit for a micro-capital Binance USD-M futures portfolio."
        )
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=39.54,
        help="Initial account equity in USD, from $0.01 to $1,000.00. Default: 39.54.",
    )
    parser.add_argument(
        "--trades",
        type=int,
        default=30,
        help="Expected trade or turnover events per 21-trading-day month. Default: 30.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=50_000,
        help="Independent Monte Carlo paths. Default: 50000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17_291,
        help="Deterministic random seed. Default: 17291.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the simulator CLI and print the institutional advisory report."""
    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        simulator = MonteCarloRiskSimulator(
            equity=arguments.equity,
            trades=arguments.trades,
            simulations=arguments.simulations,
            seed=arguments.seed,
        )
        print(simulator.generate_advisory_report())
        return 0
    except (SimulationError, ValueError, MemoryError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())