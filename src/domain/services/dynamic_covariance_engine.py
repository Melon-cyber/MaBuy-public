"""Dynamic covariance estimation, risk parity optimization, and micro-capital sizing."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_FLOOR
from threading import RLock
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DynamicCovarianceConfig:
    """Configuration for dynamic covariance, allocation, and position sizing."""

    ewma_decay: float = 0.94
    minimum_ewma_observations: int = 20
    volatility_window: int = 30
    annualization_factor: int = 365
    covariance_eigenvalue_floor: float = 1e-12
    fallback_daily_volatility: float = 0.05
    max_gross_leverage: float = 1.0
    systemic_correlation_threshold: float = 0.85
    systemic_stress_penalty: float = 0.50
    minimum_equity: float = 39.54
    maximum_equity: float = 1_000.00
    enforce_micro_capital_range: bool = True
    erc_tolerance: float = 1e-10
    erc_max_iterations: int = 10_000
    lot_sizes: Mapping[str, float] = field(
        default_factory=lambda: {
            "BTC": 0.001,
            "ETH": 0.01,
            "SOL": 0.1,
        }
    )

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 < self.ewma_decay < 1.0:
            raise ValueError("ewma_decay must be strictly between 0 and 1.")
        if self.minimum_ewma_observations < 2:
            raise ValueError("minimum_ewma_observations must be at least 2.")
        if self.volatility_window < 2:
            raise ValueError("volatility_window must be at least 2.")
        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive.")
        if self.covariance_eigenvalue_floor <= 0.0:
            raise ValueError("covariance_eigenvalue_floor must be positive.")
        if self.fallback_daily_volatility <= 0.0:
            raise ValueError("fallback_daily_volatility must be positive.")
        if self.max_gross_leverage <= 0.0:
            raise ValueError("max_gross_leverage must be positive.")
        if not 0.0 < self.systemic_correlation_threshold <= 1.0:
            raise ValueError(
                "systemic_correlation_threshold must be in the interval (0, 1]."
            )
        if not 0.0 < self.systemic_stress_penalty <= 1.0:
            raise ValueError(
                "systemic_stress_penalty must be in the interval (0, 1]."
            )
        if self.minimum_equity <= 0.0:
            raise ValueError("minimum_equity must be positive.")
        if self.maximum_equity < self.minimum_equity:
            raise ValueError("maximum_equity must be greater than minimum_equity.")
        if self.erc_tolerance <= 0.0:
            raise ValueError("erc_tolerance must be positive.")
        if self.erc_max_iterations <= 0:
            raise ValueError("erc_max_iterations must be positive.")

        for symbol, lot_size in self.lot_sizes.items():
            if not isinstance(symbol, str) or not symbol.strip():
                raise ValueError("Each lot_sizes symbol must be a non-empty string.")
            if not np.isfinite(lot_size) or lot_size <= 0.0:
                raise ValueError(
                    f"Lot size for symbol '{symbol}' must be a finite positive number."
                )


@dataclass(frozen=True)
class _ClusterNode:
    """Internal node representation for deterministic single-linkage clustering."""

    members: tuple[int, ...]
    left: _ClusterNode | None = None
    right: _ClusterNode | None = None


class DynamicCovarianceEngine:
    """
    Dynamic covariance, correlation, volatility, risk parity, HRP, and lot sizing engine.

    The engine estimates covariance using RiskMetrics-style EWMA whenever sufficient
    complete return observations are available. With limited observations, it applies a
    Ledoit-Wolf-style diagonal shrinkage estimator, falling back to a regularized
    diagonal prior only when there is insufficient information for sample covariance.

    All allocation methods are long-only by construction; trading direction is applied
    independently during discrete position sizing from supplied active signals.
    """

    _LONG_SIGNALS: frozenset[str] = frozenset({"long", "buy", "1", "+1", "bullish"})
    _SHORT_SIGNALS: frozenset[str] = frozenset({"short", "sell", "-1", "bearish"})
    _FLAT_SIGNALS: frozenset[str] = frozenset({"flat", "hold", "none", "0", "", "neutral"})

    def __init__(self, config: DynamicCovarianceConfig | None = None) -> None:
        """
        Initialize the covariance engine.

        Args:
            config: Optional validated engine configuration.

        Raises:
            ValueError: If configuration contains invalid numeric values.
        """
        self._config = config or DynamicCovarianceConfig()
        self._lock = RLock()

        self._returns: pd.DataFrame | None = None
        self._symbols: list[str] = []
        self._covariance: np.ndarray | None = None
        self._correlation: np.ndarray | None = None
        self._annualized_volatility: dict[str, float] = {}
        self._covariance_method: str | None = None

        self._last_weights: dict[str, float] = {}
        self._last_position_sizes: dict[str, float] = {}
        self._last_equity: float | None = None
        self._last_effective_gross_leverage_limit: float | None = None
        self._last_systemic_stress_active: bool = False

    @property
    def config(self) -> DynamicCovarianceConfig:
        """Return the immutable engine configuration."""
        return self._config

    def update_returns(self, returns_df: pd.DataFrame) -> None:
        """
        Replace the active return history and refresh all covariance state.

        The input must contain one numeric return column per asset. Rows containing
        missing or non-finite values are removed as covariance estimation requires
        aligned observations across all assets.

        Args:
            returns_df: DataFrame of periodic arithmetic or log returns. Columns must
                be unique non-empty strings representing tradable symbols.

        Raises:
            TypeError: If returns_df is not a pandas DataFrame.
            ValueError: If the data are empty, malformed, non-numeric, or contain no
                complete finite observations after cleaning.
        """
        if not isinstance(returns_df, pd.DataFrame):
            raise TypeError("returns_df must be a pandas DataFrame.")
        if returns_df.empty:
            raise ValueError("returns_df must not be empty.")
        if returns_df.columns.empty:
            raise ValueError("returns_df must contain at least one asset column.")
        if returns_df.columns.duplicated().any():
            raise ValueError("returns_df columns must be unique.")

        symbols = list(returns_df.columns)
        if any(not isinstance(symbol, str) or not symbol.strip() for symbol in symbols):
            raise ValueError("All returns_df columns must be non-empty string symbols.")

        try:
            numeric_returns = returns_df.astype(float).copy()
        except (TypeError, ValueError) as exc:
            raise ValueError("All returns_df columns must be numeric.") from exc

        numeric_returns = numeric_returns.replace([np.inf, -np.inf], np.nan).dropna(
            axis=0,
            how="any",
        )

        if numeric_returns.empty:
            raise ValueError(
                "No complete finite return observations remain after data cleaning."
            )

        values = numeric_returns.to_numpy(dtype=float, copy=True)
        if not np.isfinite(values).all():
            raise ValueError("Return observations must be finite after cleaning.")

        covariance, method = self._estimate_covariance(values)
        correlation = self._covariance_to_correlation(covariance)
        annualized_volatility = self._calculate_annualized_volatility(
            numeric_returns,
            covariance,
        )

        with self._lock:
            self._returns = numeric_returns
            self._symbols = symbols
            self._covariance = covariance
            self._correlation = correlation
            self._annualized_volatility = annualized_volatility
            self._covariance_method = method

            self._last_weights = {}
            self._last_position_sizes = {}
            self._last_equity = None
            self._last_effective_gross_leverage_limit = None
            self._last_systemic_stress_active = False

    def get_covariance_matrix(self) -> np.ndarray:
        """
        Return a defensive copy of the current dynamic covariance matrix.

        Raises:
            RuntimeError: If returns have not been loaded.
        """
        with self._lock:
            self._require_initialized()
            assert self._covariance is not None
            return self._covariance.copy()

    def get_correlation_matrix(self) -> np.ndarray:
        """
        Return a defensive copy of the current dynamic correlation matrix.

        Raises:
            RuntimeError: If returns have not been loaded.
        """
        with self._lock:
            self._require_initialized()
            assert self._correlation is not None
            return self._correlation.copy()

    def compute_risk_parity_weights(
        self,
        active_symbols: list[str],
    ) -> dict[str, float]:
        """
        Compute long-only Equal Risk Contribution portfolio weights.

        The solver uses cyclic coordinate descent on the convex risk-budgeting
        formulation. Resulting normalized weights have approximately equal absolute
        contribution to portfolio variance.

        Args:
            active_symbols: Ordered list of symbols to include in the allocation.

        Returns:
            Dictionary mapping each active symbol to a normalized long-only weight.

        Raises:
            RuntimeError: If return history has not been loaded.
            ValueError: If active symbols are invalid or the solver cannot converge.
        """
        with self._lock:
            indices = self._validate_active_symbols(active_symbols)
            assert self._covariance is not None

            covariance = self._covariance[np.ix_(indices, indices)]
            weights = self._solve_equal_risk_contribution(covariance)
            result = {
                symbol: float(weight)
                for symbol, weight in zip(active_symbols, weights, strict=True)
            }
            self._last_weights = result.copy()
            return result

    def compute_hrp_weights(self, active_symbols: list[str]) -> dict[str, float]:
        """
        Compute long-only Hierarchical Risk Parity weights.

        Single-linkage clustering is applied to correlation-derived distances. The
        resulting hierarchy is quasi-diagonalized, then allocated by recursive
        bisection using inverse-variance cluster risk estimates.

        Args:
            active_symbols: Ordered list of symbols to include in the allocation.

        Returns:
            Dictionary mapping each active symbol to a normalized HRP weight.

        Raises:
            RuntimeError: If return history has not been loaded.
            ValueError: If active symbols are invalid.
        """
        with self._lock:
            indices = self._validate_active_symbols(active_symbols)
            assert self._covariance is not None
            assert self._correlation is not None

            covariance = self._covariance[np.ix_(indices, indices)]
            correlation = self._correlation[np.ix_(indices, indices)]
            weights = self._solve_hierarchical_risk_parity(covariance, correlation)

            result = {
                symbol: float(weight)
                for symbol, weight in zip(active_symbols, weights, strict=True)
            }
            self._last_weights = result.copy()
            return result

    def calculate_position_sizes(
        self,
        equity: float,
        active_signals: dict[str, str],
        current_prices: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculate signed, lot-rounded, leverage-compliant crypto position quantities.

        Target notionals are risk-parity weighted and direction is determined by
        active_signals. Quantities are rounded toward zero to contract increments,
        ensuring gross leverage cannot increase through rounding. A systemic stress
        haircut is applied whenever the maximum active positive pairwise correlation
        exceeds the configured threshold.

        Args:
            equity: Current account equity in quote currency.
            active_signals: Mapping of symbol to directional signal such as "long",
                "short", "buy", "sell", "flat", or "hold".
            current_prices: Mapping of symbol to strictly positive current price.

        Returns:
            Dictionary mapping every supplied signal symbol to signed contract quantity.

        Raises:
            RuntimeError: If return history has not been loaded.
            ValueError: If equity, signals, prices, lot sizes, or symbols are invalid.
        """
        if not np.isfinite(equity) or equity <= 0.0:
            raise ValueError("equity must be a finite positive number.")
        if not isinstance(active_signals, dict):
            raise TypeError("active_signals must be a dictionary.")
        if not isinstance(current_prices, dict):
            raise TypeError("current_prices must be a dictionary.")

        if self._config.enforce_micro_capital_range and not (
            self._config.minimum_equity <= equity <= self._config.maximum_equity
        ):
            raise ValueError(
                "equity must be within the configured micro-capital range "
                f"[{self._config.minimum_equity:.2f}, "
                f"{self._config.maximum_equity:.2f}]."
            )

        with self._lock:
            self._require_initialized()

            normalized_signals: dict[str, int] = {}
            for symbol, raw_signal in active_signals.items():
                if not isinstance(symbol, str) or not symbol.strip():
                    raise ValueError("active_signals contains an invalid symbol.")
                if not isinstance(raw_signal, str):
                    raise ValueError(
                        f"Signal for '{symbol}' must be a string, received "
                        f"{type(raw_signal).__name__}."
                    )
                normalized_signals[symbol] = self._parse_signal(raw_signal)

            actionable_symbols = [
                symbol
                for symbol, direction in normalized_signals.items()
                if direction != 0
            ]

            result: dict[str, float] = {
                symbol: 0.0 for symbol in normalized_signals
            }

            if not actionable_symbols:
                self._last_weights = {}
                self._last_position_sizes = result.copy()
                self._last_equity = float(equity)
                self._last_effective_gross_leverage_limit = 0.0
                self._last_systemic_stress_active = False
                return result

            self._validate_active_symbols(actionable_symbols)

            missing_prices = [
                symbol for symbol in actionable_symbols if symbol not in current_prices
            ]
            if missing_prices:
                raise ValueError(
                    f"Missing current prices for actionable symbols: {missing_prices}."
                )

            for symbol in actionable_symbols:
                price = current_prices[symbol]
                if not isinstance(price, (int, float, np.number)):
                    raise ValueError(f"Price for '{symbol}' must be numeric.")
                if not np.isfinite(float(price)) or float(price) <= 0.0:
                    raise ValueError(
                        f"Price for '{symbol}' must be finite and strictly positive."
                    )
                self._get_lot_size(symbol)

            weights = self.compute_risk_parity_weights(actionable_symbols)
            gross_leverage_limit, stress_active = self._effective_gross_leverage_limit(
                actionable_symbols
            )

            equity_decimal = Decimal(str(float(equity)))
            gross_limit_decimal = equity_decimal * Decimal(
                str(gross_leverage_limit)
            )

            quantities_decimal: dict[str, Decimal] = {}
            for symbol in actionable_symbols:
                price_decimal = Decimal(str(float(current_prices[symbol])))
                lot_decimal = Decimal(str(self._get_lot_size(symbol)))
                target_notional = (
                    equity_decimal
                    * Decimal(str(weights[symbol]))
                    * Decimal(str(gross_leverage_limit))
                )

                raw_lots = target_notional / price_decimal / lot_decimal
                lot_count = raw_lots.to_integral_value(rounding=ROUND_FLOOR)
                unsigned_quantity = lot_count * lot_decimal

                if normalized_signals[symbol] < 0:
                    quantities_decimal[symbol] = -unsigned_quantity
                else:
                    quantities_decimal[symbol] = unsigned_quantity

            self._enforce_decimal_gross_limit(
                quantities=quantities_decimal,
                prices=current_prices,
                gross_limit=gross_limit_decimal,
            )

            for symbol in actionable_symbols:
                result[symbol] = float(quantities_decimal[symbol])

            self._last_weights = weights.copy()
            self._last_position_sizes = result.copy()
            self._last_equity = float(equity)
            self._last_effective_gross_leverage_limit = gross_leverage_limit
            self._last_systemic_stress_active = stress_active

            return result

    def compute_portfolio_beta(self, weights: Mapping[str, float]) -> float | None:
        """
        Compute portfolio beta against BTC using the current covariance matrix.

        Asset beta is calculated as cov(asset, BTC) / var(BTC), then portfolio beta is
        calculated as w^T beta_mkt. BTC need not itself have a non-zero portfolio
        weight, but it must exist in the return universe.

        Args:
            weights: Portfolio weights keyed by symbols in the loaded universe.

        Returns:
            Portfolio beta against BTC, or None when BTC is unavailable.

        Raises:
            RuntimeError: If return history has not been loaded.
            ValueError: If supplied weights are invalid.
        """
        with self._lock:
            self._require_initialized()
            if "BTC" not in self._symbols:
                return None

            assert self._covariance is not None
            btc_index = self._symbols.index("BTC")
            btc_variance = float(self._covariance[btc_index, btc_index])

            if not np.isfinite(btc_variance) or btc_variance <= 0.0:
                return None

            portfolio_vector = np.zeros(len(self._symbols), dtype=float)
            for symbol, weight in weights.items():
                if symbol not in self._symbols:
                    raise ValueError(
                        f"Cannot calculate beta: symbol '{symbol}' is not in the "
                        "loaded return universe."
                    )
                if not isinstance(weight, (int, float, np.number)) or not np.isfinite(
                    float(weight)
                ):
                    raise ValueError(
                        f"Portfolio weight for '{symbol}' must be finite and numeric."
                    )
                portfolio_vector[self._symbols.index(symbol)] = float(weight)

            asset_betas = self._covariance[:, btc_index] / btc_variance
            return float(portfolio_vector @ asset_betas)

    def get_portfolio_risk_report(self) -> dict[str, Any]:
        """
        Return a comprehensive snapshot of current covariance and portfolio risk state.

        Returns:
            JSON-serializable risk report containing covariance, correlations,
            annualized volatilities, systemic stress state, recent allocation, gross
            leverage information, and BTC-relative portfolio beta.

        Raises:
            RuntimeError: If return history has not been loaded.
        """
        with self._lock:
            self._require_initialized()
            assert self._returns is not None
            assert self._covariance is not None
            assert self._correlation is not None

            maximum_correlation = self._maximum_cross_asset_correlation(
                self._correlation
            )
            portfolio_volatility = self._portfolio_annualized_volatility(
                self._last_weights
            )
            portfolio_beta = self.compute_portfolio_beta(self._last_weights)

            return {
                "symbols": self._symbols.copy(),
                "observations": int(len(self._returns)),
                "covariance_method": self._covariance_method,
                "covariance_matrix": self._covariance.tolist(),
                "correlation_matrix": self._correlation.tolist(),
                "annualized_volatility": self._annualized_volatility.copy(),
                "maximum_cross_asset_correlation": maximum_correlation,
                "systemic_stress_active": self._last_systemic_stress_active,
                "configured_max_gross_leverage": self._config.max_gross_leverage,
                "effective_gross_leverage_limit": (
                    self._last_effective_gross_leverage_limit
                ),
                "portfolio_weights": self._last_weights.copy(),
                "position_sizes": self._last_position_sizes.copy(),
                "equity": self._last_equity,
                "portfolio_annualized_volatility": portfolio_volatility,
                "portfolio_beta_to_btc": portfolio_beta,
            }

    def _estimate_covariance(self, returns: np.ndarray) -> tuple[np.ndarray, str]:
        """Estimate and stabilize covariance using EWMA or limited-sample fallback."""
        if returns.ndim != 2 or returns.shape[0] < 1 or returns.shape[1] < 1:
            raise ValueError("returns must be a non-empty two-dimensional array.")

        observation_count, asset_count = returns.shape

        if observation_count >= self._config.minimum_ewma_observations:
            covariance = self._ewma_covariance(returns)
            return self._stabilize_covariance(covariance), "ewma_riskmetrics"

        if observation_count >= 2:
            try:
                covariance = self._ledoit_wolf_covariance(returns)
                return self._stabilize_covariance(covariance), "ledoit_wolf_shrinkage"
            except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                covariance = np.cov(returns, rowvar=False, ddof=1)
                covariance = np.atleast_2d(covariance)
                if covariance.shape != (asset_count, asset_count):
                    raise ValueError(
                        "Sample covariance shape is inconsistent with return data."
                    )
                return self._stabilize_covariance(covariance), "sample_covariance"

        diagonal_variance = max(
            self._config.fallback_daily_volatility**2,
            self._config.covariance_eigenvalue_floor,
        )
        covariance = np.eye(asset_count, dtype=float) * diagonal_variance
        return covariance, "diagonal_prior"

    def _ewma_covariance(self, returns: np.ndarray) -> np.ndarray:
        """Compute normalized, mean-adjusted RiskMetrics-style EWMA covariance."""
        observation_count = returns.shape[0]
        ages = np.arange(observation_count - 1, -1, -1, dtype=float)
        raw_weights = np.power(self._config.ewma_decay, ages)
        weights = raw_weights / raw_weights.sum()

        weighted_mean = np.sum(returns * weights[:, None], axis=0)
        centered_returns = returns - weighted_mean
        covariance = (centered_returns * weights[:, None]).T @ centered_returns

        if not np.isfinite(covariance).all():
            raise FloatingPointError("EWMA covariance contains non-finite values.")

        return covariance

    def _ledoit_wolf_covariance(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute a Ledoit-Wolf-style shrinkage covariance estimator.

        The sample covariance is shrunk toward a scaled identity target. Shrinkage
        intensity is analytically estimated from the Frobenius variance of individual
        outer-product covariance observations.
        """
        observation_count, asset_count = returns.shape
        if observation_count < 2:
            raise ValueError("Ledoit-Wolf shrinkage requires at least two observations.")

        centered_returns = returns - returns.mean(axis=0, keepdims=True)
        empirical_covariance = (
            centered_returns.T @ centered_returns / float(observation_count)
        )
        empirical_covariance = 0.5 * (
            empirical_covariance + empirical_covariance.T
        )

        target_scale = float(np.trace(empirical_covariance) / asset_count)
        target = np.eye(asset_count, dtype=float) * target_scale

        delta = float(np.sum((empirical_covariance - target) ** 2))
        row_squared_norms = np.sum(centered_returns**2, axis=1)
        beta = (
            float(np.sum(row_squared_norms**2)) / float(observation_count**2)
            - float(np.sum(empirical_covariance**2)) / float(observation_count)
        )
        beta = max(beta, 0.0)

        shrinkage = 1.0 if delta <= 0.0 else min(beta / delta, 1.0)
        covariance = (
            shrinkage * target + (1.0 - shrinkage) * empirical_covariance
        )

        if not np.isfinite(covariance).all():
            raise FloatingPointError(
                "Ledoit-Wolf covariance contains non-finite values."
            )

        return covariance

    def _stabilize_covariance(self, covariance: np.ndarray) -> np.ndarray:
        """Symmetrize and project covariance onto a numerically positive-definite cone."""
        covariance = np.asarray(covariance, dtype=float)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("Covariance matrix must be square.")
        if not np.isfinite(covariance).all():
            raise ValueError("Covariance matrix contains non-finite values.")

        symmetric_covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric_covariance)

        scale = max(
            float(np.max(np.abs(eigenvalues))),
            self._config.covariance_eigenvalue_floor,
        )
        eigenvalue_floor = max(
            self._config.covariance_eigenvalue_floor,
            scale * 1e-10,
        )
        stabilized_eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
        stabilized = (
            eigenvectors * stabilized_eigenvalues
        ) @ eigenvectors.T
        stabilized = 0.5 * (stabilized + stabilized.T)

        if not np.isfinite(stabilized).all():
            raise FloatingPointError("Stabilized covariance contains non-finite values.")

        return stabilized

    def _covariance_to_correlation(self, covariance: np.ndarray) -> np.ndarray:
        """Convert a stabilized covariance matrix into a valid correlation matrix."""
        variances = np.diag(covariance)
        if np.any(variances <= 0.0) or not np.isfinite(variances).all():
            raise ValueError(
                "Covariance diagonal must contain finite positive variances."
            )

        standard_deviations = np.sqrt(variances)
        denominator = np.outer(standard_deviations, standard_deviations)
        correlation = covariance / denominator
        correlation = np.clip(correlation, -1.0, 1.0)
        correlation = 0.5 * (correlation + correlation.T)
        np.fill_diagonal(correlation, 1.0)

        return correlation

    def _calculate_annualized_volatility(
        self,
        returns: pd.DataFrame,
        covariance: np.ndarray,
    ) -> dict[str, float]:
        """Calculate annualized rolling sample volatility for every active symbol."""
        rolling_returns = returns.tail(self._config.volatility_window)
        if len(rolling_returns) >= 2:
            volatility = rolling_returns.std(axis=0, ddof=1) * np.sqrt(
                self._config.annualization_factor
            )
            values = volatility.to_numpy(dtype=float)
        else:
            values = np.sqrt(np.diag(covariance)) * np.sqrt(
                self._config.annualization_factor
            )

        fallback_values = np.sqrt(np.diag(covariance)) * np.sqrt(
            self._config.annualization_factor
        )
        values = np.where(np.isfinite(values) & (values >= 0.0), values, fallback_values)

        return {
            symbol: float(value)
            for symbol, value in zip(self._symbols or list(returns.columns), values, strict=True)
        }

    def _solve_equal_risk_contribution(self, covariance: np.ndarray) -> np.ndarray:
        """Solve the equal-risk-contribution risk budgeting problem by coordinate descent."""
        covariance = self._stabilize_covariance(covariance)
        asset_count = covariance.shape[0]

        if asset_count == 1:
            return np.array([1.0], dtype=float)

        diagonal = np.diag(covariance)
        if np.any(diagonal <= 0.0):
            raise ValueError("ERC requires strictly positive covariance diagonal.")

        risk_budget = np.full(asset_count, 1.0 / asset_count, dtype=float)
        allocation = 1.0 / np.sqrt(diagonal)
        allocation /= allocation.sum()

        for _ in range(self._config.erc_max_iterations):
            previous = allocation.copy()

            for index in range(asset_count):
                covariance_row = covariance[index]
                cross_term = float(covariance_row @ allocation - diagonal[index] * allocation[index])
                discriminant = cross_term * cross_term + 4.0 * diagonal[index] * risk_budget[index]

                if discriminant <= 0.0 or not np.isfinite(discriminant):
                    raise ValueError("ERC solver encountered an invalid discriminant.")

                allocation[index] = (
                    -cross_term + float(np.sqrt(discriminant))
                ) / (2.0 * diagonal[index])

            if np.any(allocation <= 0.0) or not np.isfinite(allocation).all():
                raise ValueError("ERC solver produced invalid non-positive allocations.")

            relative_change = np.max(
                np.abs(allocation - previous) / np.maximum(np.abs(previous), 1e-16)
            )
            if relative_change <= self._config.erc_tolerance:
                weights = allocation / allocation.sum()
                portfolio_variance = float(weights @ covariance @ weights)

                if portfolio_variance <= 0.0 or not np.isfinite(portfolio_variance):
                    raise ValueError("ERC portfolio variance is invalid.")

                risk_contributions = weights * (covariance @ weights)
                target_contribution = portfolio_variance / asset_count
                contribution_error = np.max(
                    np.abs(risk_contributions - target_contribution)
                    / max(target_contribution, 1e-16)
                )

                if contribution_error <= max(self._config.erc_tolerance * 100.0, 1e-8):
                    return weights

        raise ValueError(
            "Equal Risk Contribution solver failed to converge within "
            f"{self._config.erc_max_iterations} iterations."
        )

    def _solve_hierarchical_risk_parity(
        self,
        covariance: np.ndarray,
        correlation: np.ndarray,
    ) -> np.ndarray:
        """Compute HRP weights using deterministic single-linkage clustering."""
        covariance = self._stabilize_covariance(covariance)
        asset_count = covariance.shape[0]

        if asset_count == 1:
            return np.array([1.0], dtype=float)

        clipped_correlation = np.clip(correlation, -1.0, 1.0)
        distances = np.sqrt(np.maximum((1.0 - clipped_correlation) / 2.0, 0.0))
        np.fill_diagonal(distances, 0.0)

        root = self._single_linkage_cluster(distances)
        ordered_indices = self._quasi_diagonalize(root)

        weights = np.ones(asset_count, dtype=float)
        clusters: list[list[int]] = [ordered_indices]

        while clusters:
            next_clusters: list[list[int]] = []

            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                split_index = len(cluster) // 2
                left_cluster = cluster[:split_index]
                right_cluster = cluster[split_index:]

                left_variance = self._cluster_variance(covariance, left_cluster)
                right_variance = self._cluster_variance(covariance, right_cluster)
                combined_variance = left_variance + right_variance

                if combined_variance <= 0.0 or not np.isfinite(combined_variance):
                    left_allocation = 0.5
                else:
                    left_allocation = right_variance / combined_variance

                weights[left_cluster] *= left_allocation
                weights[right_cluster] *= 1.0 - left_allocation

                next_clusters.append(left_cluster)
                next_clusters.append(right_cluster)

            clusters = next_clusters

        if np.any(weights < 0.0) or not np.isfinite(weights).all():
            raise ValueError("HRP solver produced invalid weights.")

        total_weight = float(weights.sum())
        if total_weight <= 0.0:
            raise ValueError("HRP solver produced zero total weight.")

        return weights / total_weight

    def _single_linkage_cluster(self, distances: np.ndarray) -> _ClusterNode:
        """Build a deterministic single-linkage clustering tree from a distance matrix."""
        asset_count = distances.shape[0]
        clusters: list[_ClusterNode] = [
            _ClusterNode(members=(index,)) for index in range(asset_count)
        ]

        while len(clusters) > 1:
            best_pair: tuple[int, int] | None = None
            best_distance = np.inf
            best_tie_breaker: tuple[tuple[int, ...], tuple[int, ...]] | None = None

            for left_index in range(len(clusters) - 1):
                for right_index in range(left_index + 1, len(clusters)):
                    left_members = clusters[left_index].members
                    right_members = clusters[right_index].members
                    cluster_distance = float(
                        np.min(distances[np.ix_(left_members, right_members)])
                    )
                    tie_breaker = (left_members, right_members)

                    if (
                        cluster_distance < best_distance - 1e-15
                        or (
                            abs(cluster_distance - best_distance) <= 1e-15
                            and (
                                best_tie_breaker is None
                                or tie_breaker < best_tie_breaker
                            )
                        )
                    ):
                        best_distance = cluster_distance
                        best_pair = (left_index, right_index)
                        best_tie_breaker = tie_breaker

            if best_pair is None:
                raise ValueError("Single-linkage clustering failed to select a merge.")

            left_index, right_index = best_pair
            left_node = clusters[left_index]
            right_node = clusters[right_index]
            merged_node = _ClusterNode(
                members=left_node.members + right_node.members,
                left=left_node,
                right=right_node,
            )

            for index in sorted(best_pair, reverse=True):
                clusters.pop(index)
            clusters.append(merged_node)

        return clusters[0]

    def _quasi_diagonalize(self, node: _ClusterNode) -> list[int]:
        """Return leaf ordering implied by the hierarchical cluster tree."""
        if node.left is None and node.right is None:
            return list(node.members)
        if node.left is None or node.right is None:
            raise ValueError("Invalid cluster tree: internal node has one child.")

        return self._quasi_diagonalize(node.left) + self._quasi_diagonalize(node.right)

    def _cluster_variance(
        self,
        covariance: np.ndarray,
        cluster_indices: list[int],
    ) -> float:
        """Estimate cluster variance using inverse-variance subportfolio weights."""
        sub_covariance = covariance[np.ix_(cluster_indices, cluster_indices)]
        diagonal = np.diag(sub_covariance)

        if np.any(diagonal <= 0.0) or not np.isfinite(diagonal).all():
            raise ValueError("Cluster covariance has invalid diagonal variances.")

        inverse_variance = 1.0 / diagonal
        inverse_variance /= inverse_variance.sum()
        variance = float(inverse_variance @ sub_covariance @ inverse_variance)

        if not np.isfinite(variance) or variance < 0.0:
            raise ValueError("Cluster variance is invalid.")

        return max(variance, self._config.covariance_eigenvalue_floor)

    def _effective_gross_leverage_limit(
        self,
        active_symbols: list[str],
    ) -> tuple[float, bool]:
        """Return stress-adjusted gross leverage limit for selected symbols."""
        indices = self._validate_active_symbols(active_symbols)
        assert self._correlation is not None

        active_correlation = self._correlation[np.ix_(indices, indices)]
        maximum_correlation = self._maximum_cross_asset_correlation(active_correlation)

        stress_active = (
            len(active_symbols) > 1
            and maximum_correlation > self._config.systemic_correlation_threshold
        )
        effective_limit = self._config.max_gross_leverage

        if stress_active:
            effective_limit *= self._config.systemic_stress_penalty

        return float(effective_limit), stress_active

    def _enforce_decimal_gross_limit(
        self,
        quantities: dict[str, Decimal],
        prices: Mapping[str, float],
        gross_limit: Decimal,
    ) -> None:
        """Defensively reduce lot-rounded positions until Decimal gross notional is valid."""
        if gross_limit < Decimal("0"):
            raise ValueError("gross_limit must not be negative.")

        def calculate_gross_notional() -> Decimal:
            return sum(
                (
                    abs(quantity) * Decimal(str(float(prices[symbol])))
                    for symbol, quantity in quantities.items()
                ),
                start=Decimal("0"),
            )

        gross_notional = calculate_gross_notional()
        while gross_notional > gross_limit:
            reducible_symbols = [
                symbol for symbol, quantity in quantities.items() if quantity != 0
            ]
            if not reducible_symbols:
                break

            symbol_to_reduce = max(
                reducible_symbols,
                key=lambda symbol: abs(quantities[symbol])
                * Decimal(str(float(prices[symbol]))),
            )
            lot_size = Decimal(str(self._get_lot_size(symbol_to_reduce)))

            if quantities[symbol_to_reduce] > 0:
                quantities[symbol_to_reduce] -= lot_size
            else:
                quantities[symbol_to_reduce] += lot_size

            gross_notional = calculate_gross_notional()

        if gross_notional > gross_limit:
            raise ValueError(
                "Unable to satisfy gross leverage constraint after lot rounding."
            )

    def _portfolio_annualized_volatility(
        self,
        weights: Mapping[str, float],
    ) -> float | None:
        """Calculate annualized portfolio volatility from a partial universe weight map."""
        if not weights:
            return None

        assert self._covariance is not None
        weight_vector = np.zeros(len(self._symbols), dtype=float)

        for symbol, weight in weights.items():
            if symbol not in self._symbols:
                continue
            weight_vector[self._symbols.index(symbol)] = float(weight)

        variance = float(weight_vector @ self._covariance @ weight_vector)
        if variance < 0.0 or not np.isfinite(variance):
            return None

        return float(np.sqrt(variance * self._config.annualization_factor))

    @staticmethod
    def _maximum_cross_asset_correlation(correlation: np.ndarray) -> float:
        """Return maximum positive off-diagonal correlation, or zero for one asset."""
        if correlation.shape[0] <= 1:
            return 0.0

        upper_triangle = correlation[np.triu_indices(correlation.shape[0], k=1)]
        if upper_triangle.size == 0:
            return 0.0

        return float(np.max(upper_triangle))

    def _validate_active_symbols(self, active_symbols: list[str]) -> list[int]:
        """Validate active universe membership and return matching covariance indices."""
        self._require_initialized()

        if not isinstance(active_symbols, list) or not active_symbols:
            raise ValueError("active_symbols must be a non-empty list of symbols.")
        if any(not isinstance(symbol, str) or not symbol.strip() for symbol in active_symbols):
            raise ValueError("active_symbols must contain non-empty string symbols.")
        if len(set(active_symbols)) != len(active_symbols):
            raise ValueError("active_symbols must not contain duplicates.")

        unknown_symbols = [
            symbol for symbol in active_symbols if symbol not in self._symbols
        ]
        if unknown_symbols:
            raise ValueError(
                f"Symbols are absent from loaded return data: {unknown_symbols}."
            )

        return [self._symbols.index(symbol) for symbol in active_symbols]

    def _get_lot_size(self, symbol: str) -> float:
        """Return validated configured contract increment for a symbol."""
        if symbol not in self._config.lot_sizes:
            raise ValueError(
                f"No minimum contract increment configured for symbol '{symbol}'."
            )

        lot_size = float(self._config.lot_sizes[symbol])
        if not np.isfinite(lot_size) or lot_size <= 0.0:
            raise ValueError(
                f"Configured minimum contract increment for '{symbol}' is invalid."
            )

        return lot_size

    def _parse_signal(self, signal: str) -> int:
        """Convert a string trading signal to direction: long=1, flat=0, short=-1."""
        normalized_signal = signal.strip().lower()

        if normalized_signal in self._LONG_SIGNALS:
            return 1
        if normalized_signal in self._SHORT_SIGNALS:
            return -1
        if normalized_signal in self._FLAT_SIGNALS:
            return 0

        raise ValueError(
            f"Unsupported signal '{signal}'. Supported values include long, short, "
            "buy, sell, flat, and hold."
        )

    def _require_initialized(self) -> None:
        """Ensure that valid return data and covariance state are available."""
        if (
            self._returns is None
            or self._covariance is None
            or self._correlation is None
            or not self._symbols
        ):
            raise RuntimeError(
                "Covariance engine is not initialized. Call update_returns first."
            )


if __name__ == "__main__":
    rng = np.random.default_rng(seed=42)
    periods = 100

    btc_returns = rng.normal(loc=0.0008, scale=0.035, size=periods)
    eth_returns = 0.72 * btc_returns + rng.normal(
        loc=0.0003,
        scale=0.028,
        size=periods,
    )
    sol_returns = 0.55 * btc_returns + 0.25 * eth_returns + rng.normal(
        loc=0.0005,
        scale=0.045,
        size=periods,
    )

    simulated_returns = pd.DataFrame(
        {
            "BTC": btc_returns,
            "ETH": eth_returns,
            "SOL": sol_returns,
        }
    )

    engine = DynamicCovarianceEngine()
    engine.update_returns(simulated_returns)

    risk_parity_weights = engine.compute_risk_parity_weights(["BTC", "ETH", "SOL"])
    hrp_weights = engine.compute_hrp_weights(["BTC", "ETH", "SOL"])

    positions = engine.calculate_position_sizes(
        equity=1_000.00,
        active_signals={
            "BTC": "long",
            "ETH": "long",
            "SOL": "long",
        },
        current_prices={
            "BTC": 65_000.00,
            "ETH": 3_400.00,
            "SOL": 150.00,
        },
    )

    report = engine.get_portfolio_risk_report()

    print("Equal Risk Contribution Weights:")
    for symbol, weight in risk_parity_weights.items():
        print(f"  {symbol}: {weight:.4%}")

    print("\nHierarchical Risk Parity Weights:")
    for symbol, weight in hrp_weights.items():
        print(f"  {symbol}: {weight:.4%}")

    print("\nDiscrete Position Sizes:")
    for symbol, quantity in positions.items():
        print(f"  {symbol}: {quantity:.8f}")

    print("\nPortfolio Risk Summary:")
    print(f"  BTC Beta: {report['portfolio_beta_to_btc']}")
    print(
        "  Annualized Portfolio Volatility: "
        f"{report['portfolio_annualized_volatility']:.4%}"
    )
    print(
        "  Maximum Cross-Asset Correlation: "
        f"{report['maximum_cross_asset_correlation']:.4f}"
    )
    print(f"  Systemic Stress Active: {report['systemic_stress_active']}")