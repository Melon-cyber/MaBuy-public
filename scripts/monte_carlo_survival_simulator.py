#!/usr/bin/env python3
"""
MaBuy V5.2 Titan Monte Carlo Survival Simulator.

Implements a regime-switching Merton jump-diffusion simulation with skewed
Student-t innovations, fractional Kelly sizing, volatility targeting, five-tier
risk gates, maker fees, funding compounding, and institutional risk telemetry.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


LOGGER = logging.getLogger("monte_carlo_survival_simulator")


@dataclass
class SimulationConfig:
    paths: int = 10_000
    horizon_days: int = 1_000
    initial_capital: float = 1_000.0
    symbols: tuple[str, ...] = ("BTCUSDT", "SOLUSDT")
    seed: int = 20_250_308
    batch_size: int = 500

    annual_vol_target: float = 0.40
    fractional_kelly: float = 0.25
    max_gross_exposure: float = 1.00
    maker_fee_rate: float = 0.0002
    funding_periods_per_day: int = 3

    student_t_df: float = 5.0
    skew_lambda: float = 1.18
    hurst_window: int = 64
    hurst_recompute_days: int = 5
    flash_shock_threshold: float = -0.075
    flash_cash_days: int = 5
    preservation_drawdown: float = 0.46
    ruin_drawdown: float = 0.50
    max_equity_multiplier: float = 1.0e12


REGIME_NAMES = ("Bull Trend", "Bear Crisis", "Brownian Chop")
GATE_NAMES = (
    "hurst_deadband",
    "volatility_targeting",
    "dynamic_circuit_breaker",
    "flash_shock_cash",
    "capital_preservation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MaBuy V5.2 Titan institutional Monte Carlo survival simulator."
    )
    parser.add_argument("--config", type=Path, help="Optional JSON configuration file.")
    parser.add_argument("--paths", type=int)
    parser.add_argument("--horizon-days", type=int)
    parser.add_argument("--initial-capital", type=float)
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated symbols, for example BTCUSDT,SOLUSDT.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/monte_carlo_survival_telemetry.json"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("models/monte_carlo_survival_report.txt"),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def configure_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.Formatter.converter = time.gmtime


def load_config(args: argparse.Namespace) -> SimulationConfig:
    config_data: dict[str, Any] = {}

    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as handle:
            config_data = json.load(handle)
        if not isinstance(config_data, dict):
            raise ValueError("Configuration file must contain a JSON object.")

    valid_fields = set(SimulationConfig.__dataclass_fields__.keys())
    unknown_fields = set(config_data).difference(valid_fields)
    if unknown_fields:
        raise ValueError(
            f"Unsupported configuration fields: {', '.join(sorted(unknown_fields))}"
        )

    if "symbols" in config_data:
        config_data["symbols"] = tuple(str(item).upper() for item in config_data["symbols"])

    config = SimulationConfig(**config_data)

    overrides: dict[str, Any] = {
        "paths": args.paths,
        "horizon_days": args.horizon_days,
        "initial_capital": args.initial_capital,
        "seed": args.seed,
        "batch_size": args.batch_size,
    }

    if args.symbols is not None:
        overrides["symbols"] = tuple(
            item.strip().upper() for item in args.symbols.split(",") if item.strip()
        )

    for field_name, value in overrides.items():
        if value is not None:
            setattr(config, field_name, value)

    validate_config(config)
    return config


def validate_config(config: SimulationConfig) -> None:
    if config.paths <= 0:
        raise ValueError("paths must be greater than zero.")
    if config.horizon_days <= 0:
        raise ValueError("horizon_days must be greater than zero.")
    if config.initial_capital <= 0.0:
        raise ValueError("initial_capital must be greater than zero.")
    if not config.symbols:
        raise ValueError("At least one symbol is required.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    if config.student_t_df <= 2.0:
        raise ValueError("student_t_df must be greater than 2.0.")
    if config.skew_lambda <= 0.0:
        raise ValueError("skew_lambda must be greater than zero.")
    if not 0.0 < config.fractional_kelly <= 1.0:
        raise ValueError("fractional_kelly must be in the interval (0, 1].")
    if not 0.0 < config.max_gross_exposure <= 1.0:
        raise ValueError("max_gross_exposure must be in the interval (0, 1].")
    if config.hurst_window < 16:
        raise ValueError("hurst_window must be at least 16.")
    if config.hurst_recompute_days <= 0:
        raise ValueError("hurst_recompute_days must be greater than zero.")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def asset_parameters(symbols: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(symbols)
    predefined_volatility = {
        "BTCUSDT": 1.00,
        "ETHUSDT": 1.10,
        "SOLUSDT": 1.28,
        "BNBUSDT": 0.92,
        "XRPUSDT": 1.12,
        "DOGEUSDT": 1.38,
    }

    predefined_jump_beta = {
        "BTCUSDT": 1.00,
        "ETHUSDT": 1.16,
        "SOLUSDT": 1.35,
        "BNBUSDT": 0.88,
        "XRPUSDT": 1.14,
        "DOGEUSDT": 1.44,
    }

    predefined_drift_multiplier = {
        "BTCUSDT": 1.00,
        "ETHUSDT": 1.05,
        "SOLUSDT": 1.13,
        "BNBUSDT": 0.92,
        "XRPUSDT": 0.96,
        "DOGEUSDT": 1.08,
    }

    volatility = np.array(
        [predefined_volatility.get(symbol, 1.0 + 0.08 * index) for index, symbol in enumerate(symbols)],
        dtype=np.float64,
    )
    jump_beta = np.array(
        [predefined_jump_beta.get(symbol, 1.0 + 0.10 * index) for index, symbol in enumerate(symbols)],
        dtype=np.float64,
    )
    drift_multiplier = np.array(
        [
            predefined_drift_multiplier.get(symbol, 1.0 + 0.03 * index)
            for index, symbol in enumerate(symbols)
        ],
        dtype=np.float64,
    )

    if count == 1:
        correlation = np.ones((1, 1), dtype=np.float64)
    else:
        indices = np.arange(count)
        correlation = np.power(0.62, np.abs(indices[:, None] - indices[None, :]))
        np.fill_diagonal(correlation, 1.0)

    return volatility, jump_beta, drift_multiplier, correlation


def skewed_student_t(
    rng: np.random.Generator,
    shape: tuple[int, int],
    degrees_of_freedom: float,
    skew_lambda: float,
) -> np.ndarray:
    innovations = rng.standard_t(degrees_of_freedom, size=shape)
    transformed = np.where(
        innovations >= 0.0,
        innovations * skew_lambda,
        innovations / skew_lambda,
    )

    gamma_ratio = math.gamma((degrees_of_freedom + 1.0) / 2.0) / math.gamma(
        degrees_of_freedom / 2.0
    )
    expected_abs_t = (
        2.0
        * math.sqrt(degrees_of_freedom)
        * gamma_ratio
        / (math.sqrt(math.pi) * (degrees_of_freedom - 1.0))
    )

    transformed_mean = 0.5 * (
        skew_lambda - (1.0 / skew_lambda)
    ) * expected_abs_t
    transformed_second_moment = (
        0.5 * (skew_lambda**2 + skew_lambda**-2)
        * degrees_of_freedom
        / (degrees_of_freedom - 2.0)
    )
    transformed_variance = max(
        transformed_second_moment - transformed_mean**2,
        1.0e-12,
    )

    return (transformed - transformed_mean) / math.sqrt(transformed_variance)


def estimate_hurst_rs(history: np.ndarray) -> np.ndarray:
    observations = history.shape[1]
    centered = history - history.mean(axis=1, keepdims=True)
    cumulative = np.cumsum(centered, axis=1)
    ranges = cumulative.max(axis=1) - cumulative.min(axis=1)
    standard_deviation = history.std(axis=1, ddof=1)
    rescaled_range = ranges / np.maximum(standard_deviation, 1.0e-12)
    hurst = np.log(np.maximum(rescaled_range, 1.0e-12)) / math.log(observations)
    return np.clip(hurst, 0.05, 0.95)


def calculate_metrics(
    terminal_wealth: np.ndarray,
    max_drawdown: np.ndarray,
    cagr: np.ndarray,
    initial_capital: float,
    ruin_drawdown: float,
) -> dict[str, Any]:
    terminal_loss = 1.0 - (terminal_wealth / initial_capital)

    var_95 = float(np.quantile(terminal_loss, 0.95))
    var_99 = float(np.quantile(terminal_loss, 0.99))
    cvar_tail = terminal_loss[terminal_loss >= var_99]
    cvar_99 = float(cvar_tail.mean()) if cvar_tail.size else var_99

    wealth_quantiles = np.quantile(terminal_wealth, [0.01, 0.05, 0.50, 0.95, 0.99])
    drawdown_quantiles = np.quantile(max_drawdown, [0.50, 0.90, 0.99])
    cagr_quantiles = np.quantile(cagr, [0.025, 0.05, 0.50, 0.95, 0.975])

    return {
        "empirical_probability_of_ruin": float(np.mean(max_drawdown > ruin_drawdown)),
        "ruin_definition": f"maximum_drawdown > {ruin_drawdown:.0%}",
        "value_at_risk": {
            "var_95_fraction": var_95,
            "var_95_currency": var_95 * initial_capital,
            "var_99_fraction": var_99,
            "var_99_currency": var_99 * initial_capital,
            "cvar_99_fraction": cvar_99,
            "cvar_99_currency": cvar_99 * initial_capital,
        },
        "maximum_drawdown_distribution": {
            "p50": float(drawdown_quantiles[0]),
            "p90": float(drawdown_quantiles[1]),
            "p99": float(drawdown_quantiles[2]),
            "mean": float(np.mean(max_drawdown)),
            "worst": float(np.max(max_drawdown)),
        },
        "terminal_wealth_distribution": {
            "mean": float(np.mean(terminal_wealth)),
            "standard_deviation": float(np.std(terminal_wealth, ddof=1)),
            "minimum": float(np.min(terminal_wealth)),
            "p01": float(wealth_quantiles[0]),
            "p05": float(wealth_quantiles[1]),
            "median": float(wealth_quantiles[2]),
            "p95": float(wealth_quantiles[3]),
            "p99": float(wealth_quantiles[4]),
            "maximum": float(np.max(terminal_wealth)),
        },
        "cagr_distribution": {
            "mean": float(np.mean(cagr)),
            "p05": float(cagr_quantiles[1]),
            "median": float(cagr_quantiles[2]),
            "p95": float(cagr_quantiles[3]),
            "confidence_interval_95": {
                "lower": float(cagr_quantiles[0]),
                "upper": float(cagr_quantiles[4]),
            },
        },
    }


def simulate_batch(
    config: SimulationConfig,
    path_count: int,
    rng: np.random.Generator,
    volatility_multiplier: np.ndarray,
    jump_beta: np.ndarray,
    drift_multiplier: np.ndarray,
    correlation_cholesky: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    asset_count = len(config.symbols)

    regime_transition_matrix = np.array(
        [
            [0.930, 0.040, 0.030],
            [0.100, 0.820, 0.080],
            [0.170, 0.180, 0.650],
        ],
        dtype=np.float64,
    )
    regime_parameters = np.array(
        [
            [0.00110, 0.0260, 0.0060, -0.085, 0.070, 0.00012],
            [-0.00180, 0.0550, 0.0350, -0.150, 0.125, -0.00004],
            [0.00010, 0.0350, 0.0120, -0.065, 0.085, 0.00003],
        ],
        dtype=np.float64,
    )

    regime = rng.choice(
        3,
        size=path_count,
        p=np.array([0.48, 0.17, 0.35], dtype=np.float64),
    )

    equity = np.full(path_count, config.initial_capital, dtype=np.float64)
    peak_equity = equity.copy()
    max_drawdown = np.zeros(path_count, dtype=np.float64)
    prior_positions = np.zeros((path_count, asset_count), dtype=np.float64)

    ema_mean = np.zeros((path_count, asset_count), dtype=np.float64)
    initial_asset_variance = np.square(0.040 * volatility_multiplier)
    ema_variance = np.broadcast_to(
        initial_asset_variance,
        (path_count, asset_count),
    ).copy()

    history = np.zeros((path_count, config.hurst_window), dtype=np.float64)
    history_index = 0
    hurst = np.full(path_count, 0.50, dtype=np.float64)

    flash_cooldown = np.zeros(path_count, dtype=np.int16)
    preservation_cooldown = np.zeros(path_count, dtype=np.int16)

    gate_counts = np.zeros(len(GATE_NAMES), dtype=np.int64)
    regime_counts = np.zeros(3, dtype=np.int64)
    jump_count_total = 0

    transition_cdf = np.cumsum(regime_transition_matrix, axis=1)
    equal_weights = np.full(asset_count, 1.0 / asset_count, dtype=np.float64)

    for day in range(config.horizon_days):
        regime_counts += np.bincount(regime, minlength=3)

        regime_drift = regime_parameters[regime, 0]
        regime_sigma = regime_parameters[regime, 1]
        jump_intensity = regime_parameters[regime, 2]
        jump_mean = regime_parameters[regime, 3]
        jump_sigma = regime_parameters[regime, 4]
        funding_base = regime_parameters[regime, 5]

        raw_innovations = skewed_student_t(
            rng,
            (path_count, asset_count),
            config.student_t_df,
            config.skew_lambda,
        )
        correlated_innovations = raw_innovations @ correlation_cholesky.T

        jump_occurrences = rng.poisson(jump_intensity)
        jump_count_total += int(jump_occurrences.sum())

        common_jump = (
            jump_occurrences * jump_mean
            + np.sqrt(jump_occurrences) * jump_sigma * rng.standard_normal(path_count)
        )
        idiosyncratic_jump = (
            np.sqrt(jump_occurrences)[:, None]
            * 0.012
            * volatility_multiplier[None, :]
            * rng.standard_normal((path_count, asset_count))
        )

        log_returns = (
            regime_drift[:, None] * drift_multiplier[None, :]
            + regime_sigma[:, None]
            * volatility_multiplier[None, :]
            * correlated_innovations
            + common_jump[:, None] * jump_beta[None, :]
            + idiosyncratic_jump
        )
        log_returns = np.clip(log_returns, -2.9957, 0.7000)
        asset_returns = np.expm1(log_returns)

        current_drawdown = 1.0 - (equity / np.maximum(peak_equity, 1.0e-12))
        current_drawdown = np.clip(current_drawdown, 0.0, 1.0)

        regime_expected_return = regime_drift[:, None] * drift_multiplier[None, :]
        expected_return = 0.55 * ema_mean + 0.45 * regime_expected_return
        asset_kelly = config.fractional_kelly * expected_return / np.maximum(
            ema_variance,
            1.0e-8,
        )
        positive_kelly = np.clip(asset_kelly, 0.0, config.max_gross_exposure)

        kelly_sum = positive_kelly.sum(axis=1)
        base_weights = np.divide(
            positive_kelly,
            kelly_sum[:, None],
            out=np.broadcast_to(equal_weights, (path_count, asset_count)).copy(),
            where=kelly_sum[:, None] > 1.0e-12,
        )
        base_exposure = np.clip(
            positive_kelly.mean(axis=1),
            0.0,
            config.max_gross_exposure,
        )

        weighted_volatility = base_weights * np.sqrt(np.maximum(ema_variance, 1.0e-12))
        portfolio_variance = np.sum(
            (weighted_volatility @ (correlation_cholesky @ correlation_cholesky.T))
            * weighted_volatility,
            axis=1,
        )
        annualized_volatility = np.sqrt(np.maximum(portfolio_variance, 1.0e-12)) * math.sqrt(365.0)
        volatility_scale = np.clip(
            config.annual_vol_target / np.maximum(annualized_volatility, 1.0e-8),
            0.10,
            1.15,
        )

        hurst_multiplier = np.ones(path_count, dtype=np.float64)
        low_persistence = hurst < 0.43
        deadband = (hurst >= 0.47) & (hurst <= 0.53)
        hurst_multiplier[low_persistence] = 0.45
        hurst_multiplier[deadband] = 0.25
        gate_counts[0] += int(np.count_nonzero(hurst_multiplier < 0.999))
        gate_counts[1] += int(np.count_nonzero(np.abs(volatility_scale - 1.0) > 0.01))

        circuit_breaker_scale = np.ones(path_count, dtype=np.float64)
        circuit_breaker_scale[(current_drawdown >= 0.08) & (current_drawdown < 0.15)] = 0.70
        circuit_breaker_scale[(current_drawdown >= 0.15) & (current_drawdown < 0.25)] = 0.40
        circuit_breaker_scale[(current_drawdown >= 0.25) & (current_drawdown < 0.40)] = 0.15
        circuit_breaker_scale[current_drawdown >= 0.40] = 0.00
        circuit_breaker_scale[annualized_volatility >= 0.85] = np.minimum(
            circuit_breaker_scale[annualized_volatility >= 0.85],
            0.35,
        )
        gate_counts[2] += int(np.count_nonzero(circuit_breaker_scale < 0.999))

        cash_from_flash = flash_cooldown > 0
        cash_from_preservation = (
            (preservation_cooldown > 0)
            | (current_drawdown >= config.preservation_drawdown)
        )
        gate_counts[3] += int(np.count_nonzero(cash_from_flash))
        gate_counts[4] += int(np.count_nonzero(cash_from_preservation))

        gross_exposure = np.clip(
            base_exposure * volatility_scale * hurst_multiplier * circuit_breaker_scale,
            0.0,
            config.max_gross_exposure,
        )
        gross_exposure[cash_from_flash | cash_from_preservation] = 0.0

        positions = base_weights * gross_exposure[:, None]
        turnover = np.abs(positions - prior_positions).sum(axis=1)
        maker_fee = config.maker_fee_rate * turnover

        strategy_return = np.sum(positions * asset_returns, axis=1)
        strategy_gross = np.maximum(1.0 + strategy_return, 1.0e-12)

        funding_rate = np.clip(
            funding_base + rng.normal(0.0, 0.00003, size=path_count),
            -0.00050,
            0.00080,
        )
        funding_factor = np.power(
            1.0 + funding_rate,
            -config.funding_periods_per_day * gross_exposure,
        )

        equity *= strategy_gross * np.maximum(1.0 - maker_fee, 1.0e-12) * funding_factor
        equity = np.clip(
            equity,
            config.initial_capital * 1.0e-12,
            config.initial_capital * config.max_equity_multiplier,
        )

        peak_equity = np.maximum(peak_equity, equity)
        realized_drawdown = 1.0 - equity / np.maximum(peak_equity, 1.0e-12)
        max_drawdown = np.maximum(max_drawdown, realized_drawdown)

        market_return = asset_returns.mean(axis=1)
        history[:, history_index] = market_return
        history_index = (history_index + 1) % config.hurst_window

        if (
            day + 1 >= config.hurst_window
            and (day + 1) % config.hurst_recompute_days == 0
        ):
            ordered_history = np.concatenate(
                (
                    history[:, history_index:],
                    history[:, :history_index],
                ),
                axis=1,
            )
            hurst = estimate_hurst_rs(ordered_history)

        flash_cooldown = np.maximum(flash_cooldown - 1, 0)
        preservation_cooldown = np.maximum(preservation_cooldown - 1, 0)

        flash_trigger = market_return <= config.flash_shock_threshold
        preservation_trigger = realized_drawdown >= config.preservation_drawdown

        flash_cooldown = np.maximum(
            flash_cooldown,
            flash_trigger.astype(np.int16) * config.flash_cash_days,
        )
        preservation_cooldown = np.maximum(
            preservation_cooldown,
            preservation_trigger.astype(np.int16) * 20,
        )

        ema_mean = 0.94 * ema_mean + 0.06 * log_returns
        innovation_residual = log_returns - ema_mean
        ema_variance = 0.94 * ema_variance + 0.06 * np.square(innovation_residual)
        ema_variance = np.clip(ema_variance, 1.0e-8, 2.0)

        prior_positions = positions

        uniforms = rng.random(path_count)
        regime = (uniforms[:, None] > transition_cdf[regime]).sum(axis=1)

    years = config.horizon_days / 365.0
    cagr = np.power(
        np.maximum(equity / config.initial_capital, 1.0e-12),
        1.0 / years,
    ) - 1.0

    return equity, max_drawdown, cagr, gate_counts, regime_counts, jump_count_total


def build_report(
    config: SimulationConfig,
    metrics: dict[str, Any],
    risk_gate_telemetry: dict[str, Any],
    regime_telemetry: dict[str, Any],
    runtime_seconds: float,
) -> str:
    ruin_probability = metrics["empirical_probability_of_ruin"]
    var_metrics = metrics["value_at_risk"]
    drawdown_metrics = metrics["maximum_drawdown_distribution"]
    terminal_metrics = metrics["terminal_wealth_distribution"]
    cagr_metrics = metrics["cagr_distribution"]
    cagr_ci = cagr_metrics["confidence_interval_95"]

    lines = [
        "MABUY V5.2 TITAN | INSTITUTIONAL MONTE CARLO SURVIVAL REPORT",
        "=" * 72,
        f"Simulation Paths:                 {config.paths:,}",
        f"Investment Horizon:                {config.horizon_days:,} trading days",
        f"Initial Capital per Path:          ${config.initial_capital:,.2f}",
        f"Assets:                           {', '.join(config.symbols)}",
        f"Random Seed:                      {config.seed}",
        f"Runtime:                          {runtime_seconds:.2f} seconds",
        "",
        "SURVIVAL AND TAIL RISK",
        "-" * 72,
        f"Empirical Probability of Ruin:    {ruin_probability:.2%}",
        f"Ruin Definition:                  {metrics['ruin_definition']}",
        f"VaR 95%:                          {var_metrics['var_95_fraction']:.2%} "
        f"(${var_metrics['var_95_currency']:,.2f})",
        f"VaR 99%:                          {var_metrics['var_99_fraction']:.2%} "
        f"(${var_metrics['var_99_currency']:,.2f})",
        f"CVaR 99% / Expected Shortfall:    {var_metrics['cvar_99_fraction']:.2%} "
        f"(${var_metrics['cvar_99_currency']:,.2f})",
        "",
        "MAXIMUM DRAWDOWN DISTRIBUTION",
        "-" * 72,
        f"Median Maximum Drawdown:          {drawdown_metrics['p50']:.2%}",
        f"90th Percentile Maximum Drawdown: {drawdown_metrics['p90']:.2%}",
        f"99th Percentile Maximum Drawdown: {drawdown_metrics['p99']:.2%}",
        f"Worst Observed Maximum Drawdown:  {drawdown_metrics['worst']:.2%}",
        "",
        "TERMINAL WEALTH DISTRIBUTION",
        "-" * 72,
        f"Mean Terminal Wealth:             ${terminal_metrics['mean']:,.2f}",
        f"Median Terminal Wealth:           ${terminal_metrics['median']:,.2f}",
        f"5th Percentile Terminal Wealth:   ${terminal_metrics['p05']:,.2f}",
        f"95th Percentile Terminal Wealth:  ${terminal_metrics['p95']:,.2f}",
        f"99th Percentile Terminal Wealth:  ${terminal_metrics['p99']:,.2f}",
        "",
        "COMPOUNDING CAGR DISTRIBUTION",
        "-" * 72,
        f"Mean CAGR:                        {cagr_metrics['mean']:.2%}",
        f"Median CAGR:                      {cagr_metrics['median']:.2%}",
        f"95% CAGR Confidence Interval:     [{cagr_ci['lower']:.2%}, {cagr_ci['upper']:.2%}]",
        "",
        "RISK GATE UTILIZATION",
        "-" * 72,
    ]

    for gate_name, telemetry in risk_gate_telemetry.items():
        label = gate_name.replace("_", " ").title()
        lines.append(
            f"{label:<32} {telemetry['activation_rate']:.2%} "
            f"({telemetry['activations']:,} path-days)"
        )

    lines.extend(
        [
            "",
            "REGIME OCCUPANCY",
            "-" * 72,
        ]
    )

    for regime_name, telemetry in regime_telemetry.items():
        lines.append(
            f"{regime_name:<32} {telemetry['occupancy_rate']:.2%} "
            f"({telemetry['path_days']:,} path-days)"
        )

    lines.extend(
        [
            "",
            "MODEL GOVERNANCE",
            "-" * 72,
            "Market dynamics combine Markov regime switching, Merton Poisson jumps,",
            "correlated skewed Student-t innovations, volatility-scaled fractional Kelly",
            "sizing, maker-first execution costs, and 8-hour perpetual funding compounding.",
            "Risk results are empirical simulation estimates and are not investment advice.",
        ]
    )

    return "\n".join(lines) + "\n"


def run_simulation(config: SimulationConfig) -> dict[str, Any]:
    started = time.perf_counter()
    rng = np.random.default_rng(config.seed)

    volatility_multiplier, jump_beta, drift_multiplier, correlation = asset_parameters(
        config.symbols
    )
    correlation_cholesky = np.linalg.cholesky(correlation)

    terminal_wealth = np.empty(config.paths, dtype=np.float64)
    maximum_drawdown = np.empty(config.paths, dtype=np.float64)
    cagr = np.empty(config.paths, dtype=np.float64)

    total_gate_counts = np.zeros(len(GATE_NAMES), dtype=np.int64)
    total_regime_counts = np.zeros(3, dtype=np.int64)
    total_jumps = 0

    completed_paths = 0
    total_batches = math.ceil(config.paths / config.batch_size)

    for batch_index, start in enumerate(range(0, config.paths, config.batch_size), start=1):
        end = min(start + config.batch_size, config.paths)
        current_batch_size = end - start

        (
            batch_terminal,
            batch_drawdown,
            batch_cagr,
            batch_gate_counts,
            batch_regime_counts,
            batch_jumps,
        ) = simulate_batch(
            config=config,
            path_count=current_batch_size,
            rng=rng,
            volatility_multiplier=volatility_multiplier,
            jump_beta=jump_beta,
            drift_multiplier=drift_multiplier,
            correlation_cholesky=correlation_cholesky,
        )

        terminal_wealth[start:end] = batch_terminal
        maximum_drawdown[start:end] = batch_drawdown
        cagr[start:end] = batch_cagr
        total_gate_counts += batch_gate_counts
        total_regime_counts += batch_regime_counts
        total_jumps += batch_jumps
        completed_paths = end

        elapsed = time.perf_counter() - started
        paths_per_second = completed_paths / max(elapsed, 1.0e-9)
        remaining_paths = config.paths - completed_paths
        eta_seconds = remaining_paths / max(paths_per_second, 1.0e-9)

        LOGGER.info(
            "Completed batch %d/%d | paths=%d/%d | elapsed=%.1fs | ETA=%.1fs",
            batch_index,
            total_batches,
            completed_paths,
            config.paths,
            elapsed,
            eta_seconds,
        )

    runtime_seconds = time.perf_counter() - started
    metrics = calculate_metrics(
        terminal_wealth=terminal_wealth,
        max_drawdown=maximum_drawdown,
        cagr=cagr,
        initial_capital=config.initial_capital,
        ruin_drawdown=config.ruin_drawdown,
    )

    total_path_days = config.paths * config.horizon_days
    risk_gate_telemetry = {
        gate_name: {
            "activations": int(total_gate_counts[index]),
            "activation_rate": float(total_gate_counts[index] / total_path_days),
        }
        for index, gate_name in enumerate(GATE_NAMES)
    }

    regime_telemetry = {
        regime_name: {
            "path_days": int(total_regime_counts[index]),
            "occupancy_rate": float(total_regime_counts[index] / total_path_days),
        }
        for index, regime_name in enumerate(REGIME_NAMES)
    }

    report = build_report(
        config=config,
        metrics=metrics,
        risk_gate_telemetry=risk_gate_telemetry,
        regime_telemetry=regime_telemetry,
        runtime_seconds=runtime_seconds,
    )

    return {
        "engine": {
            "name": "MaBuy V5.2 Titan Monte Carlo Survival Simulator",
            "version": "5.2",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "runtime_seconds": runtime_seconds,
            "random_seed": config.seed,
        },
        "configuration": asdict(config),
        "model_specification": {
            "stochastic_process": "Regime-switching correlated Merton jump-diffusion",
            "regimes": list(REGIME_NAMES),
            "student_t_degrees_of_freedom": config.student_t_df,
            "student_t_skew_lambda": config.skew_lambda,
            "market_crash_process": "Poisson-distributed common and idiosyncratic jumps",
            "execution_model": {
                "maker_fee_rate": config.maker_fee_rate,
                "funding_periods_per_day": config.funding_periods_per_day,
                "funding_compounding": "Perpetual funding is compounded every eight hours.",
            },
            "position_sizing": {
                "method": "Fractional Kelly with volatility scaling",
                "fractional_kelly": config.fractional_kelly,
                "annual_volatility_target": config.annual_vol_target,
                "maximum_gross_exposure": config.max_gross_exposure,
            },
            "risk_gates": {
                "hurst_deadband": {
                    "window_days": config.hurst_window,
                    "deadband": [0.47, 0.53],
                    "low_persistence_threshold": 0.43,
                },
                "dynamic_circuit_breaker": {
                    "drawdown_thresholds": [0.08, 0.15, 0.25, 0.40],
                    "capital_preservation_drawdown": config.preservation_drawdown,
                },
                "flash_shock_cash_threshold": {
                    "daily_market_shock_threshold": config.flash_shock_threshold,
                    "cash_days": config.flash_cash_days,
                },
            },
        },
        "risk_metrics": metrics,
        "risk_gate_telemetry": risk_gate_telemetry,
        "regime_telemetry": regime_telemetry,
        "jump_telemetry": {
            "total_poisson_jump_events": int(total_jumps),
            "mean_jump_events_per_path": float(total_jumps / config.paths),
            "mean_jump_events_per_path_day": float(total_jumps / total_path_days),
        },
        "institutional_summary_report": report,
    }


def main() -> int:
    args = parse_args()
    configure_logging(args.quiet)

    try:
        config = load_config(args)
        LOGGER.info(
            "Starting MaBuy V5.2 Titan simulation | paths=%d | horizon=%d days | symbols=%s",
            config.paths,
            config.horizon_days,
            ",".join(config.symbols),
        )

        telemetry = run_simulation(config)
        report = telemetry["institutional_summary_report"]

        atomic_write_text(
            args.output,
            json.dumps(telemetry, indent=2, sort_keys=True, allow_nan=False),
        )
        atomic_write_text(args.report_output, report)

        LOGGER.info("Telemetry saved to %s", args.output)
        LOGGER.info("Institutional report saved to %s", args.report_output)

        if not args.quiet:
            print(report, end="")

        return 0
    except KeyboardInterrupt:
        LOGGER.warning("Simulation interrupted by user.")
        return 130
    except Exception as exc:
        LOGGER.exception("Simulation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
