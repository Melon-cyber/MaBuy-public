"""Volatility surface calibration and risk-regime analytics.

This module provides:
- Raw SVI smile calibration.
- Heston stochastic-volatility pricing and calibration via Carr-Madan quadrature.
- Merton jump-diffusion pricing and calibration.
- Implied-volatility smile generation, 25-delta risk reversals and butterflies.
- Volatility-regime classification and MaBuy Titan stop-loss sizing guidance.

The implementation uses risk-neutral pricing conventions. Rates, dividend yields,
jump intensities, and Heston parameters are annualized and continuously compounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from typing import Iterable, Literal, Mapping, Optional, Sequence, Union
import warnings

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq, least_squares
from scipy.stats import norm, poisson


ArrayLike = Union[Sequence[float], np.ndarray]
ModelName = Literal["svi", "heston", "merton"]


@dataclass(frozen=True)
class SVIParameters:
    """Raw SVI parameterization for a single expiry.

    The total implied variance is:

        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

    where k is log-forward moneyness, k = log(K / F).

    Parameters
    ----------
    a:
        Vertical level of total variance.
    b:
        Positive curvature scale.
    rho:
        Correlation/skew parameter, constrained to (-1, 1).
    m:
        Horizontal smile displacement in log-forward moneyness.
    sigma:
        Positive smoothing parameter.
    maturity:
        Expiry in years for which the SVI parameters were calibrated.
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    maturity: float

    def total_variance(self, log_forward_moneyness: ArrayLike) -> np.ndarray:
        """Return SVI total implied variance for log-forward moneyness."""
        k = np.asarray(log_forward_moneyness, dtype=float)
        x = k - self.m
        return self.a + self.b * (
            self.rho * x + np.sqrt(x * x + self.sigma * self.sigma)
        )

    def implied_volatility(self, log_forward_moneyness: ArrayLike) -> np.ndarray:
        """Return SVI Black implied volatility."""
        if self.maturity <= 0.0:
            raise ValueError("SVI maturity must be strictly positive.")
        variance = np.maximum(self.total_variance(log_forward_moneyness), 0.0)
        return np.sqrt(variance / self.maturity)

    @property
    def minimum_total_variance(self) -> float:
        """Return the analytical minimum of the raw SVI total-variance curve."""
        return self.a + self.b * self.sigma * np.sqrt(max(0.0, 1.0 - self.rho**2))

    @property
    def is_admissible(self) -> bool:
        """Return whether basic raw-SVI positivity and parameter conditions hold."""
        return (
            self.b > 0.0
            and self.sigma > 0.0
            and abs(self.rho) < 1.0
            and self.minimum_total_variance >= -1e-10
        )


@dataclass(frozen=True)
class HestonParameters:
    """Risk-neutral Heston stochastic-volatility parameters.

    Parameters
    ----------
    kappa:
        Mean-reversion speed of instantaneous variance.
    theta:
        Long-run variance.
    sigma_v:
        Volatility of variance, also called vol-of-vol.
    rho:
        Correlation between spot and variance Brownian motions.
    v0:
        Initial instantaneous variance.
    """

    kappa: float
    theta: float
    sigma_v: float
    rho: float
    v0: float

    @property
    def feller_margin(self) -> float:
        """Return 2*kappa*theta - sigma_v^2; positive values satisfy Feller."""
        return 2.0 * self.kappa * self.theta - self.sigma_v**2

    @property
    def satisfies_feller_condition(self) -> bool:
        """Return whether the Feller condition is satisfied."""
        return self.feller_margin >= 0.0


@dataclass(frozen=True)
class MertonJumpParameters:
    """Risk-neutral Merton jump-diffusion parameters.

    Under the risk-neutral measure:

        dS_t = (r - q - lambda * kappa_J) S_t dt
               + sigma S_t dW_t
               + (J - 1) S_t dN_t

    with log(J) normally distributed with mean ``mu_jump`` and standard deviation
    ``sigma_jump``. The jump compensator is:

        kappa_J = E[J - 1] = exp(mu_jump + 0.5 * sigma_jump^2) - 1.

    Parameters
    ----------
    sigma:
        Diffusion volatility.
    lambda_jump:
        Annual Poisson jump intensity.
    mu_jump:
        Mean of log jump size log(J).
    sigma_jump:
        Standard deviation of log jump size log(J).
    """

    sigma: float
    lambda_jump: float
    mu_jump: float
    sigma_jump: float

    @property
    def jump_compensator(self) -> float:
        """Return E[J - 1], the risk-neutral jump drift compensator."""
        return float(np.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1.0)

    @property
    def expected_jump_size(self) -> float:
        """Return E[J]."""
        return float(np.exp(self.mu_jump + 0.5 * self.sigma_jump**2))

    @property
    def crash_probability_one_year(self) -> float:
        """Return the probability of at least one jump over one year."""
        return float(1.0 - np.exp(-self.lambda_jump))


@dataclass
class OptionSurfaceSlice:
    """Option quote slice for one expiry.

    Either ``prices`` or ``implied_vols`` must be supplied. If both are supplied,
    prices are used for model calibration and implied volatilities remain available
    for diagnostics.

    Parameters
    ----------
    maturity:
        Time to expiration in years.
    strikes:
        Strike prices.
    prices:
        Option market prices. Put prices are converted to calls through put-call
        parity before model calibration.
    implied_vols:
        Black-Scholes implied volatilities.
    option_types:
        Either one option type for all quotes or one type per strike. Accepted
        values are "call", "c", "put", and "p".
    weights:
        Optional non-negative calibration weights.
    """

    maturity: float
    strikes: ArrayLike
    prices: Optional[ArrayLike] = None
    implied_vols: Optional[ArrayLike] = None
    option_types: Union[str, Sequence[str], np.ndarray] = "call"
    weights: Optional[ArrayLike] = None

    def __post_init__(self) -> None:
        self.strikes = np.asarray(self.strikes, dtype=float)
        if self.maturity <= 0.0:
            raise ValueError("Surface slice maturity must be strictly positive.")
        if self.strikes.ndim != 1 or len(self.strikes) == 0:
            raise ValueError("strikes must be a non-empty one-dimensional array.")
        if np.any(~np.isfinite(self.strikes)) or np.any(self.strikes <= 0.0):
            raise ValueError("strikes must be finite and strictly positive.")

        if self.prices is None and self.implied_vols is None:
            raise ValueError("Either prices or implied_vols must be provided.")

        if self.prices is not None:
            self.prices = np.asarray(self.prices, dtype=float)
            if self.prices.shape != self.strikes.shape:
                raise ValueError("prices must have the same shape as strikes.")

        if self.implied_vols is not None:
            self.implied_vols = np.asarray(self.implied_vols, dtype=float)
            if self.implied_vols.shape != self.strikes.shape:
                raise ValueError(
                    "implied_vols must have the same shape as strikes."
                )

        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float)
            if self.weights.shape != self.strikes.shape:
                raise ValueError("weights must have the same shape as strikes.")
            if np.any(self.weights < 0.0):
                raise ValueError("weights cannot be negative.")


@dataclass(frozen=True)
class SmileMetrics:
    """25-delta smile diagnostics for one expiry."""

    maturity: float
    forward: float
    atm_strike: float
    put_25_delta_strike: float
    call_25_delta_strike: float
    atm_volatility: float
    put_25_delta_volatility: float
    call_25_delta_volatility: float
    risk_reversal_25: float
    butterfly_25: float

    @property
    def downside_skew_slope(self) -> float:
        """Return call-minus-put skew, negative for conventional downside skew."""
        return self.call_25_delta_volatility - self.put_25_delta_volatility


@dataclass(frozen=True)
class RegimeSignal:
    """Volatility-regime classification and associated diagnostics."""

    regime: str
    severe_downside_crash_risk: bool
    volatility_compression: bool
    jump_intensity: float
    downside_skew: float
    skew_spike: bool
    jump_intensity_elevated: bool
    confidence: float
    reasons: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StopLossAdjustment:
    """MaBuy Titan-oriented stop-loss and position-size adjustment."""

    stop_loss_dilation_multiplier: float
    position_size_multiplier: float
    regime: str
    rationale: str


class VolatilitySurfaceCalibrator:
    """Calibrate volatility models, derive smile metrics, and signal risk regimes.

    Parameters
    ----------
    spot:
        Current underlying spot price.
    risk_free_rate:
        Continuously compounded annual risk-free rate.
    dividend_yield:
        Continuously compounded annual dividend yield or foreign rate.
    integration_limit:
        Upper integration limit for Carr-Madan Heston quadrature.
    quadrature_limit:
        Maximum number of adaptive quadrature subintervals.
    history_limit:
        Maximum number of internally retained regime observations.

    Notes
    -----
    - Heston prices use Carr-Madan damped Fourier inversion evaluated by adaptive
      quadrature. This is generally more stable than direct P1/P2 integration near
      difficult parameter combinations.
    - Merton prices use the exact Poisson-mixture representation of lognormal
      conditional distributions.
    - SVI calibration operates on total implied variance versus log-forward
      moneyness.
    """

    def __init__(
        self,
        spot: float,
        risk_free_rate: float = 0.0,
        dividend_yield: float = 0.0,
        integration_limit: float = 150.0,
        quadrature_limit: int = 250,
        history_limit: int = 252,
    ) -> None:
        if not np.isfinite(spot) or spot <= 0.0:
            raise ValueError("spot must be finite and strictly positive.")
        if integration_limit <= 0.0:
            raise ValueError("integration_limit must be strictly positive.")
        if quadrature_limit < 50:
            raise ValueError("quadrature_limit must be at least 50.")
        if history_limit < 2:
            raise ValueError("history_limit must be at least 2.")

        self.spot = float(spot)
        self.risk_free_rate = float(risk_free_rate)
        self.dividend_yield = float(dividend_yield)
        self.integration_limit = float(integration_limit)
        self.quadrature_limit = int(quadrature_limit)
        self.history_limit = int(history_limit)
        self._regime_history: list[tuple[SmileMetrics, float]] = []

    def discount_factor(self, maturity: float) -> float:
        """Return the domestic discount factor exp(-rT)."""
        self._validate_maturity(maturity)
        return float(np.exp(-self.risk_free_rate * maturity))

    def forward(self, maturity: float) -> float:
        """Return the risk-neutral forward price F = S exp((r-q)T)."""
        self._validate_maturity(maturity)
        return float(
            self.spot
            * np.exp((self.risk_free_rate - self.dividend_yield) * maturity)
        )

    @staticmethod
    def _validate_maturity(maturity: float) -> None:
        if not np.isfinite(maturity) or maturity <= 0.0:
            raise ValueError("maturity must be finite and strictly positive.")

    @staticmethod
    def _normalize_option_types(
        option_types: Union[str, Sequence[str], np.ndarray],
        n: int,
    ) -> np.ndarray:
        """Normalize option-type labels into an array of 'call' or 'put'."""
        if isinstance(option_types, str):
            labels = np.full(n, option_types.lower(), dtype=object)
        else:
            labels = np.asarray(option_types, dtype=object)
            if labels.shape != (n,):
                raise ValueError("option_types must be scalar or match strikes.")

        normalized = np.empty(n, dtype=object)
        for index, label in enumerate(labels):
            value = str(label).strip().lower()
            if value in ("call", "c"):
                normalized[index] = "call"
            elif value in ("put", "p"):
                normalized[index] = "put"
            else:
                raise ValueError(f"Unsupported option type: {label!r}")
        return normalized

    @staticmethod
    def black_scholes_price(
        spot: float,
        strike: ArrayLike,
        maturity: float,
        rate: float,
        dividend_yield: float,
        volatility: ArrayLike,
        option_type: Union[str, Sequence[str], np.ndarray] = "call",
    ) -> np.ndarray:
        """Price European options using Black-Scholes with continuous carry."""
        if spot <= 0.0:
            raise ValueError("spot must be strictly positive.")
        if maturity < 0.0:
            raise ValueError("maturity cannot be negative.")

        strikes = np.asarray(strike, dtype=float)
        vol = np.asarray(volatility, dtype=float)
        if np.any(strikes <= 0.0):
            raise ValueError("strikes must be strictly positive.")
        if np.any(vol < 0.0):
            raise ValueError("volatility cannot be negative.")

        option_types = VolatilitySurfaceCalibrator._normalize_option_types(
            option_type,
            np.broadcast(strikes, vol).size,
        )

        strikes_b, vol_b = np.broadcast_arrays(strikes, vol)
        flat_strikes = strikes_b.ravel()
        flat_vol = vol_b.ravel()

        discounted_spot = spot * np.exp(-dividend_yield * maturity)
        discounted_strike = flat_strikes * np.exp(-rate * maturity)

        result = np.empty_like(flat_strikes, dtype=float)
        intrinsic_call = np.maximum(discounted_spot - discounted_strike, 0.0)
        intrinsic_put = np.maximum(discounted_strike - discounted_spot, 0.0)

        positive = (flat_vol > 1e-14) & (maturity > 1e-14)
        result[~positive] = np.where(
            option_types[~positive] == "call",
            intrinsic_call[~positive],
            intrinsic_put[~positive],
        )

        if np.any(positive):
            sqrt_t = np.sqrt(maturity)
            d1 = (
                np.log(spot / flat_strikes[positive])
                + (rate - dividend_yield + 0.5 * flat_vol[positive] ** 2) * maturity
            ) / (flat_vol[positive] * sqrt_t)
            d2 = d1 - flat_vol[positive] * sqrt_t

            call = (
                discounted_spot * norm.cdf(d1)
                - discounted_strike[positive] * norm.cdf(d2)
            )
            put = (
                discounted_strike[positive] * norm.cdf(-d2)
                - discounted_spot * norm.cdf(-d1)
            )
            result[positive] = np.where(
                option_types[positive] == "call",
                call,
                put,
            )

        return result.reshape(strikes_b.shape)

    def black_scholes_vega(
        self,
        strike: ArrayLike,
        maturity: float,
        volatility: ArrayLike,
    ) -> np.ndarray:
        """Return Black-Scholes vega per one unit of volatility."""
        self._validate_maturity(maturity)
        strikes = np.asarray(strike, dtype=float)
        vol = np.asarray(volatility, dtype=float)
        strikes, vol = np.broadcast_arrays(strikes, vol)

        result = np.zeros_like(strikes, dtype=float)
        valid = (strikes > 0.0) & (vol > 1e-12)
        if np.any(valid):
            d1 = (
                np.log(self.spot / strikes[valid])
                + (
                    self.risk_free_rate
                    - self.dividend_yield
                    + 0.5 * vol[valid] ** 2
                )
                * maturity
            ) / (vol[valid] * np.sqrt(maturity))
            result[valid] = (
                self.spot
                * np.exp(-self.dividend_yield * maturity)
                * norm.pdf(d1)
                * np.sqrt(maturity)
            )
        return result

    def implied_volatility(
        self,
        price: float,
        strike: float,
        maturity: float,
        option_type: str = "call",
        strict: bool = False,
    ) -> float:
        """Solve the Black-Scholes implied volatility for one option price.

        Invalid prices return NaN unless ``strict=True``, in which case a
        ``ValueError`` is raised.
        """
        self._validate_maturity(maturity)
        if strike <= 0.0 or not np.isfinite(strike):
            raise ValueError("strike must be finite and strictly positive.")
        if not np.isfinite(price):
            if strict:
                raise ValueError("Option price must be finite.")
            return float("nan")

        option = self._normalize_option_types(option_type, 1)[0]
        discounted_spot = self.spot * np.exp(-self.dividend_yield * maturity)
        discounted_strike = strike * np.exp(-self.risk_free_rate * maturity)

        if option == "call":
            lower = max(discounted_spot - discounted_strike, 0.0)
            upper = discounted_spot
        else:
            lower = max(discounted_strike - discounted_spot, 0.0)
            upper = discounted_strike

        tolerance = 1e-10 * max(1.0, self.spot)
        if price < lower - tolerance or price > upper + tolerance:
            if strict:
                raise ValueError(
                    f"Price {price:.10g} violates no-arbitrage bounds "
                    f"[{lower:.10g}, {upper:.10g}]."
                )
            return float("nan")

        clipped_price = min(max(price, lower), upper)
        if clipped_price <= lower + tolerance:
            return 0.0
        if clipped_price >= upper - tolerance:
            if strict:
                raise ValueError("Price is at the upper no-arbitrage bound.")
            return float("nan")

        def objective(volatility: float) -> float:
            return float(
                self.black_scholes_price(
                    self.spot,
                    strike,
                    maturity,
                    self.risk_free_rate,
                    self.dividend_yield,
                    volatility,
                    option,
                )
                - clipped_price
            )

        try:
            return float(brentq(objective, 1e-10, 8.0, xtol=1e-10, rtol=1e-10))
        except ValueError:
            if strict:
                raise
            return float("nan")

    def implied_volatility_array(
        self,
        prices: ArrayLike,
        strikes: ArrayLike,
        maturity: float,
        option_types: Union[str, Sequence[str], np.ndarray] = "call",
    ) -> np.ndarray:
        """Return Black implied volatilities for a vector of option prices."""
        prices_array = np.asarray(prices, dtype=float)
        strikes_array = np.asarray(strikes, dtype=float)
        if prices_array.shape != strikes_array.shape:
            raise ValueError("prices and strikes must have identical shapes.")

        types = self._normalize_option_types(option_types, len(strikes_array))
        return np.asarray(
            [
                self.implied_volatility(
                    float(price),
                    float(strike),
                    maturity,
                    str(option_type),
                    strict=False,
                )
                for price, strike, option_type in zip(
                    prices_array,
                    strikes_array,
                    types,
                )
            ],
            dtype=float,
        )

    def _slice_call_prices(self, option_slice: OptionSurfaceSlice) -> np.ndarray:
        """Convert an option slice to undiscounted-vector call price conventions."""
        n = len(option_slice.strikes)
        option_types = self._normalize_option_types(option_slice.option_types, n)
        maturity = option_slice.maturity
        strikes = option_slice.strikes

        if option_slice.prices is not None:
            prices = np.asarray(option_slice.prices, dtype=float)
            calls = prices.copy()
            put_mask = option_types == "put"
            if np.any(put_mask):
                calls[put_mask] = (
                    prices[put_mask]
                    + self.spot * np.exp(-self.dividend_yield * maturity)
                    - strikes[put_mask] * np.exp(-self.risk_free_rate * maturity)
                )
            return calls

        assert option_slice.implied_vols is not None
        return self.black_scholes_price(
            self.spot,
            strikes,
            maturity,
            self.risk_free_rate,
            self.dividend_yield,
            option_slice.implied_vols,
            "call",
        )

    def _slice_implied_vols(self, option_slice: OptionSurfaceSlice) -> np.ndarray:
        """Obtain valid Black implied volatilities from a surface slice."""
        if option_slice.implied_vols is not None:
            return np.asarray(option_slice.implied_vols, dtype=float)

        assert option_slice.prices is not None
        return self.implied_volatility_array(
            option_slice.prices,
            option_slice.strikes,
            option_slice.maturity,
            option_slice.option_types,
        )

    def calibrate_svi(
        self,
        strikes: ArrayLike,
        implied_vols: ArrayLike,
        maturity: float,
        weights: Optional[ArrayLike] = None,
        initial: Optional[SVIParameters] = None,
        max_nfev: int = 2_000,
    ) -> SVIParameters:
        """Calibrate raw SVI parameters to a single implied-volatility smile.

        The fit minimizes weighted total-variance residuals while penalizing a
        negative analytical minimum of the raw SVI curve.
        """
        self._validate_maturity(maturity)
        strikes_array = np.asarray(strikes, dtype=float)
        vols_array = np.asarray(implied_vols, dtype=float)

        if strikes_array.shape != vols_array.shape:
            raise ValueError("strikes and implied_vols must have the same shape.")
        if len(strikes_array) < 5:
            raise ValueError("At least five quotes are required for SVI calibration.")

        if weights is None:
            weights_array = np.ones_like(strikes_array)
        else:
            weights_array = np.asarray(weights, dtype=float)
            if weights_array.shape != strikes_array.shape:
                raise ValueError("weights must have the same shape as strikes.")
            if np.any(weights_array < 0.0):
                raise ValueError("weights cannot be negative.")

        valid = (
            np.isfinite(strikes_array)
            & np.isfinite(vols_array)
            & np.isfinite(weights_array)
            & (strikes_array > 0.0)
            & (vols_array > 0.0)
            & (weights_array > 0.0)
        )
        strikes_array = strikes_array[valid]
        vols_array = vols_array[valid]
        weights_array = weights_array[valid]

        if len(strikes_array) < 5:
            raise ValueError("At least five valid SVI observations are required.")

        forward = self.forward(maturity)
        log_moneyness = np.log(strikes_array / forward)
        target_variance = vols_array**2 * maturity
        sqrt_weights = np.sqrt(weights_array / np.mean(weights_array))

        if initial is None:
            initial_a = max(1e-7, 0.5 * float(np.min(target_variance)))
            initial_b = max(0.02, min(2.0, float(np.ptp(target_variance)) + 0.1))
            initial_rho = -0.30
            initial_m = float(np.median(log_moneyness))
            initial_sigma = 0.20
            x0 = np.array(
                [initial_a, initial_b, initial_rho, initial_m, initial_sigma],
                dtype=float,
            )
        else:
            x0 = np.array(
                [initial.a, initial.b, initial.rho, initial.m, initial.sigma],
                dtype=float,
            )

        lower = np.array([-3.0, 1e-9, -0.999, -5.0, 1e-6], dtype=float)
        upper = np.array([5.0, 20.0, 0.999, 5.0, 5.0], dtype=float)
        x0 = np.minimum(np.maximum(x0, lower + 1e-8), upper - 1e-8)

        def residuals(parameters: np.ndarray) -> np.ndarray:
            a, b, rho, m, sigma = parameters
            x = log_moneyness - m
            model_variance = a + b * (
                rho * x + np.sqrt(x * x + sigma * sigma)
            )
            minimum_variance = a + b * sigma * np.sqrt(max(0.0, 1.0 - rho * rho))
            positivity_penalty = 50.0 * max(0.0, -minimum_variance)
            return np.concatenate(
                [
                    sqrt_weights * (model_variance - target_variance),
                    np.array([positivity_penalty]),
                ]
            )

        result = least_squares(
            residuals,
            x0,
            bounds=(lower, upper),
            method="trf",
            max_nfev=max_nfev,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )

        if not result.success:
            raise RuntimeError(f"SVI calibration failed: {result.message}")

        return SVIParameters(
            a=float(result.x[0]),
            b=float(result.x[1]),
            rho=float(result.x[2]),
            m=float(result.x[3]),
            sigma=float(result.x[4]),
            maturity=float(maturity),
        )

    def calibrate_svi_surface(
        self,
        slices: Sequence[OptionSurfaceSlice],
        max_nfev: int = 2_000,
    ) -> dict[float, SVIParameters]:
        """Calibrate independent raw-SVI smiles across multiple expirations."""
        if not slices:
            raise ValueError("At least one surface slice is required.")

        calibrated: dict[float, SVIParameters] = {}
        for option_slice in sorted(slices, key=lambda item: item.maturity):
            implied_vols = self._slice_implied_vols(option_slice)
            calibrated[option_slice.maturity] = self.calibrate_svi(
                option_slice.strikes,
                implied_vols,
                option_slice.maturity,
                option_slice.weights,
                max_nfev=max_nfev,
            )
        return calibrated

    def heston_characteristic_function(
        self,
        u: Union[complex, np.ndarray],
        maturity: float,
        parameters: HestonParameters,
    ) -> Union[complex, np.ndarray]:
        """Return the characteristic function of log(S_T) under Heston."""
        self._validate_maturity(maturity)
        self._validate_heston_parameters(parameters)

        u_array = np.asarray(u, dtype=complex)
        kappa = parameters.kappa
        theta = parameters.theta
        sigma_v = parameters.sigma_v
        rho = parameters.rho
        v0 = parameters.v0
        imaginary_unit = 1j

        beta = kappa - rho * sigma_v * imaginary_unit * u_array
        discriminant = np.sqrt(
            beta * beta + sigma_v * sigma_v * (u_array * u_array + imaginary_unit * u_array)
        )
        g = (beta - discriminant) / (beta + discriminant)

        exp_dt = np.exp(-discriminant * maturity)
        log_ratio = np.log((1.0 - g * exp_dt) / (1.0 - g))

        c_term = (
            imaginary_unit
            * u_array
            * (
                np.log(self.spot)
                + (self.risk_free_rate - self.dividend_yield) * maturity
            )
            + (kappa * theta / (sigma_v * sigma_v))
            * ((beta - discriminant) * maturity - 2.0 * log_ratio)
        )
        d_term = (
            (beta - discriminant)
            / (sigma_v * sigma_v)
            * ((1.0 - exp_dt) / (1.0 - g * exp_dt))
        )

        return np.exp(c_term + d_term * v0)

    def heston_call_price(
        self,
        strike: float,
        maturity: float,
        parameters: HestonParameters,
        alpha: float = 1.5,
    ) -> float:
        """Price a European call using Carr-Madan damped Fourier quadrature.

        Parameters
        ----------
        strike:
            Option strike.
        maturity:
            Time to expiry in years.
        parameters:
            Heston risk-neutral parameters.
        alpha:
            Carr-Madan damping coefficient. Values around 1.0 to 2.0 are
            generally appropriate for vanilla call pricing.
        """
        self._validate_maturity(maturity)
        self._validate_heston_parameters(parameters)
        if strike <= 0.0:
            raise ValueError("strike must be strictly positive.")
        if alpha <= 0.0:
            raise ValueError("alpha must be strictly positive.")

        log_strike = np.log(strike)
        shifted_imaginary = (alpha + 1.0) * 1j

        def integrand(u: float) -> float:
            shifted_u = u - shifted_imaginary
            numerator = np.exp(-1j * u * log_strike) * self.heston_characteristic_function(
                shifted_u,
                maturity,
                parameters,
            )
            denominator = (
                alpha * alpha
                + alpha
                - u * u
                + 1j * (2.0 * alpha + 1.0) * u
            )
            return float(np.real(numerator / denominator))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            integral, _ = quad(
                integrand,
                0.0,
                self.integration_limit,
                epsabs=1e-7,
                epsrel=1e-6,
                limit=self.quadrature_limit,
            )

        price = np.exp(-self.risk_free_rate * maturity - alpha * log_strike)
        price *= integral / np.pi

        lower = max(
            self.spot * np.exp(-self.dividend_yield * maturity)
            - strike * np.exp(-self.risk_free_rate * maturity),
            0.0,
        )
        upper = self.spot * np.exp(-self.dividend_yield * maturity)
        return float(np.clip(price, lower, upper))

    def heston_option_price(
        self,
        strike: float,
        maturity: float,
        parameters: HestonParameters,
        option_type: str = "call",
    ) -> float:
        """Price a Heston European call or put using put-call parity."""
        option = self._normalize_option_types(option_type, 1)[0]
        call = self.heston_call_price(strike, maturity, parameters)
        if option == "call":
            return call
        return float(
            call
            - self.spot * np.exp(-self.dividend_yield * maturity)
            + strike * np.exp(-self.risk_free_rate * maturity)
        )

    def heston_call_prices(
        self,
        strikes: ArrayLike,
        maturity: float,
        parameters: HestonParameters,
    ) -> np.ndarray:
        """Return Heston call prices for a strike vector."""
        strikes_array = np.asarray(strikes, dtype=float)
        return np.asarray(
            [
                self.heston_call_price(float(strike), maturity, parameters)
                for strike in strikes_array
            ],
            dtype=float,
        )

    def calibrate_heston(
        self,
        slices: Sequence[OptionSurfaceSlice],
        initial: Optional[HestonParameters] = None,
        max_nfev: int = 250,
        enforce_feller_penalty: bool = False,
    ) -> HestonParameters:
        """Calibrate Heston parameters to a multi-expiry option surface.

        The residuals are price residuals normalized by floored Black vegas, which
        approximately weights the fit in implied-volatility units.
        """
        maturities, strikes, market_calls, quote_weights = self._prepare_market_data(
            slices
        )
        if len(strikes) < 5:
            raise ValueError("At least five valid option observations are required.")

        market_vols = np.asarray(
            [
                self.implied_volatility(price, strike, maturity, "call")
                for price, strike, maturity in zip(
                    market_calls,
                    strikes,
                    maturities,
                )
            ],
            dtype=float,
        )
        market_vols = np.where(
            np.isfinite(market_vols) & (market_vols > 1e-6),
            market_vols,
            0.25,
        )

        vegas = np.asarray(
            [
                self.black_scholes_vega(strike, maturity, volatility)
                for strike, maturity, volatility in zip(
                    strikes,
                    maturities,
                    market_vols,
                )
            ],
            dtype=float,
        )
        scales = np.maximum(
            vegas,
            0.01 * self.spot * np.sqrt(np.maximum(maturities, 1e-6)),
        )
        sqrt_weights = np.sqrt(quote_weights / np.mean(quote_weights))

        if initial is None:
            initial = HestonParameters(
                kappa=1.5,
                theta=max(0.01, float(np.nanmedian(market_vols) ** 2)),
                sigma_v=0.50,
                rho=-0.60,
                v0=max(0.01, float(market_vols[0] ** 2)),
            )

        lower = np.array([1e-4, 1e-6, 1e-4, -0.999, 1e-6], dtype=float)
        upper = np.array([20.0, 4.0, 10.0, 0.999, 4.0], dtype=float)
        x0 = np.array(
            [
                initial.kappa,
                initial.theta,
                initial.sigma_v,
                initial.rho,
                initial.v0,
            ],
            dtype=float,
        )
        x0 = np.minimum(np.maximum(x0, lower + 1e-8), upper - 1e-8)

        def residuals(values: np.ndarray) -> np.ndarray:
            parameters = HestonParameters(
                kappa=float(values[0]),
                theta=float(values[1]),
                sigma_v=float(values[2]),
                rho=float(values[3]),
                v0=float(values[4]),
            )
            try:
                model_prices = np.asarray(
                    [
                        self.heston_call_price(strike, maturity, parameters)
                        for strike, maturity in zip(strikes, maturities)
                    ],
                    dtype=float,
                )
                result = sqrt_weights * (model_prices - market_calls) / scales
                if not np.all(np.isfinite(result)):
                    result = np.full_like(market_calls, 1e5)
            except (ArithmeticError, FloatingPointError, ValueError, OverflowError):
                result = np.full_like(market_calls, 1e5)

            if enforce_feller_penalty:
                feller_violation = max(
                    0.0,
                    parameters.sigma_v**2
                    - 2.0 * parameters.kappa * parameters.theta,
                )
                result = np.concatenate([result, np.array([0.25 * feller_violation])])
            return result

        result = least_squares(
            residuals,
            x0,
            bounds=(lower, upper),
            method="trf",
            max_nfev=max_nfev,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        if not result.success:
            raise RuntimeError(f"Heston calibration failed: {result.message}")

        return HestonParameters(
            kappa=float(result.x[0]),
            theta=float(result.x[1]),
            sigma_v=float(result.x[2]),
            rho=float(result.x[3]),
            v0=float(result.x[4]),
        )

    def merton_characteristic_function(
        self,
        u: Union[complex, np.ndarray],
        maturity: float,
        parameters: MertonJumpParameters,
    ) -> Union[complex, np.ndarray]:
        """Return the characteristic function of log(S_T) under Merton."""
        self._validate_maturity(maturity)
        self._validate_merton_parameters(parameters)

        u_array = np.asarray(u, dtype=complex)
        sigma = parameters.sigma
        lambda_jump = parameters.lambda_jump
        mu_jump = parameters.mu_jump
        sigma_jump = parameters.sigma_jump
        compensator = parameters.jump_compensator

        drift = (
            np.log(self.spot)
            + (
                self.risk_free_rate
                - self.dividend_yield
                - lambda_jump * compensator
                - 0.5 * sigma * sigma
            )
            * maturity
        )
        jump_cf = np.exp(
            1j * u_array * mu_jump - 0.5 * sigma_jump * sigma_jump * u_array * u_array
        )

        exponent = (
            1j * u_array * drift
            - 0.5 * sigma * sigma * u_array * u_array * maturity
            + lambda_jump * maturity * (jump_cf - 1.0)
        )
        return np.exp(exponent)

    def merton_call_price(
        self,
        strike: float,
        maturity: float,
        parameters: MertonJumpParameters,
        tail_probability: float = 1e-12,
    ) -> float:
        """Price a Merton jump-diffusion European call via a Poisson mixture."""
        self._validate_maturity(maturity)
        self._validate_merton_parameters(parameters)
        if strike <= 0.0:
            raise ValueError("strike must be strictly positive.")
        if not 0.0 < tail_probability < 1.0:
            raise ValueError("tail_probability must lie strictly between zero and one.")

        sigma = parameters.sigma
        lambda_jump = parameters.lambda_jump
        mu_jump = parameters.mu_jump
        sigma_jump = parameters.sigma_jump
        jump_compensator = parameters.jump_compensator
        poisson_mean = lambda_jump * maturity

        if poisson_mean <= 1e-14:
            max_jumps = 0
        else:
            quantile = poisson.ppf(1.0 - tail_probability, poisson_mean)
            max_jumps = max(10, int(ceil(float(quantile))))

        discount = np.exp(-self.risk_free_rate * maturity)
        price = 0.0

        for jumps in range(max_jumps + 1):
            probability = float(poisson.pmf(jumps, poisson_mean))
            total_variance = sigma * sigma * maturity + jumps * sigma_jump * sigma_jump
            total_stddev = np.sqrt(max(total_variance, 1e-16))

            conditional_forward = self.spot * np.exp(
                (
                    self.risk_free_rate
                    - self.dividend_yield
                    - lambda_jump * jump_compensator
                )
                * maturity
                + jumps * (mu_jump + 0.5 * sigma_jump * sigma_jump)
            )

            d1 = (
                np.log(conditional_forward / strike) + 0.5 * total_variance
            ) / total_stddev
            d2 = d1 - total_stddev
            price += probability * discount * (
                conditional_forward * norm.cdf(d1) - strike * norm.cdf(d2)
            )

        lower = max(
            self.spot * np.exp(-self.dividend_yield * maturity)
            - strike * np.exp(-self.risk_free_rate * maturity),
            0.0,
        )
        upper = self.spot * np.exp(-self.dividend_yield * maturity)
        return float(np.clip(price, lower, upper))

    def merton_option_price(
        self,
        strike: float,
        maturity: float,
        parameters: MertonJumpParameters,
        option_type: str = "call",
    ) -> float:
        """Price a Merton jump-diffusion European call or put."""
        option = self._normalize_option_types(option_type, 1)[0]
        call = self.merton_call_price(strike, maturity, parameters)
        if option == "call":
            return call
        return float(
            call
            - self.spot * np.exp(-self.dividend_yield * maturity)
            + strike * np.exp(-self.risk_free_rate * maturity)
        )

    def merton_call_prices(
        self,
        strikes: ArrayLike,
        maturity: float,
        parameters: MertonJumpParameters,
    ) -> np.ndarray:
        """Return Merton jump-diffusion call prices for a strike vector."""
        strikes_array = np.asarray(strikes, dtype=float)
        return np.asarray(
            [
                self.merton_call_price(float(strike), maturity, parameters)
                for strike in strikes_array
            ],
            dtype=float,
        )

    def calibrate_merton(
        self,
        slices: Sequence[OptionSurfaceSlice],
        initial: Optional[MertonJumpParameters] = None,
        max_nfev: int = 1_000,
    ) -> MertonJumpParameters:
        """Calibrate Merton jump-diffusion parameters to a multi-expiry surface."""
        maturities, strikes, market_calls, quote_weights = self._prepare_market_data(
            slices
        )
        if len(strikes) < 5:
            raise ValueError("At least five valid option observations are required.")

        market_vols = np.asarray(
            [
                self.implied_volatility(price, strike, maturity, "call")
                for price, strike, maturity in zip(
                    market_calls,
                    strikes,
                    maturities,
                )
            ],
            dtype=float,
        )
        market_vols = np.where(
            np.isfinite(market_vols) & (market_vols > 1e-6),
            market_vols,
            0.25,
        )
        vegas = np.asarray(
            [
                self.black_scholes_vega(strike, maturity, volatility)
                for strike, maturity, volatility in zip(
                    strikes,
                    maturities,
                    market_vols,
                )
            ],
            dtype=float,
        )
        scales = np.maximum(
            vegas,
            0.01 * self.spot * np.sqrt(np.maximum(maturities, 1e-6)),
        )
        sqrt_weights = np.sqrt(quote_weights / np.mean(quote_weights))

        if initial is None:
            initial = MertonJumpParameters(
                sigma=max(0.05, float(np.nanmedian(market_vols))),
                lambda_jump=0.20,
                mu_jump=-0.10,
                sigma_jump=0.20,
            )

        lower = np.array([1e-4, 0.0, -2.0, 1e-4], dtype=float)
        upper = np.array([4.0, 5.0, 1.0, 2.0], dtype=float)
        x0 = np.array(
            [
                initial.sigma,
                initial.lambda_jump,
                initial.mu_jump,
                initial.sigma_jump,
            ],
            dtype=float,
        )
        x0 = np.minimum(np.maximum(x0, lower + 1e-8), upper - 1e-8)

        def residuals(values: np.ndarray) -> np.ndarray:
            parameters = MertonJumpParameters(
                sigma=float(values[0]),
                lambda_jump=float(values[1]),
                mu_jump=float(values[2]),
                sigma_jump=float(values[3]),
            )
            try:
                model_prices = np.asarray(
                    [
                        self.merton_call_price(strike, maturity, parameters)
                        for strike, maturity in zip(strikes, maturities)
                    ],
                    dtype=float,
                )
                result = sqrt_weights * (model_prices - market_calls) / scales
                if not np.all(np.isfinite(result)):
                    return np.full_like(market_calls, 1e5)
                return result
            except (ArithmeticError, FloatingPointError, ValueError, OverflowError):
                return np.full_like(market_calls, 1e5)

        result = least_squares(
            residuals,
            x0,
            bounds=(lower, upper),
            method="trf",
            max_nfev=max_nfev,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        if not result.success:
            raise RuntimeError(f"Merton calibration failed: {result.message}")

        return MertonJumpParameters(
            sigma=float(result.x[0]),
            lambda_jump=float(result.x[1]),
            mu_jump=float(result.x[2]),
            sigma_jump=float(result.x[3]),
        )

    def implied_volatility_smile(
        self,
        model: ModelName,
        parameters: Union[
            SVIParameters,
            HestonParameters,
            MertonJumpParameters,
            Mapping[float, SVIParameters],
        ],
        maturity: float,
        strikes: ArrayLike,
    ) -> np.ndarray:
        """Compute a risk-neutral implied-volatility smile for a given model."""
        self._validate_maturity(maturity)
        strikes_array = np.asarray(strikes, dtype=float)
        if np.any(strikes_array <= 0.0):
            raise ValueError("strikes must be strictly positive.")

        normalized_model = model.lower()
        if normalized_model == "svi":
            svi_parameters = self._resolve_svi_parameters(parameters, maturity)
            log_moneyness = np.log(strikes_array / self.forward(maturity))
            return svi_parameters.implied_volatility(log_moneyness)

        if normalized_model == "heston":
            if not isinstance(parameters, HestonParameters):
                raise TypeError("Heston model requires HestonParameters.")
            prices = self.heston_call_prices(strikes_array, maturity, parameters)
        elif normalized_model == "merton":
            if not isinstance(parameters, MertonJumpParameters):
                raise TypeError("Merton model requires MertonJumpParameters.")
            prices = self.merton_call_prices(strikes_array, maturity, parameters)
        else:
            raise ValueError(f"Unsupported model: {model!r}")

        return self.implied_volatility_array(
            prices,
            strikes_array,
            maturity,
            "call",
        )

    def extract_skew_and_butterfly(
        self,
        model: ModelName,
        parameters: Union[
            SVIParameters,
            HestonParameters,
            MertonJumpParameters,
            Mapping[float, SVIParameters],
        ],
        maturity: float,
        delta: float = 0.25,
    ) -> SmileMetrics:
        """Extract ATM volatility, 25-delta risk reversal, and butterfly.

        Definitions
        -----------
        Risk Reversal (RR):
            IV(25-delta put) - IV(25-delta call).

        Butterfly (BF):
            0.5 * [IV(25-delta put) + IV(25-delta call)] - IV(ATM).

        Delta convention:
            Spot Black delta with continuous dividend yield is used.
        """
        self._validate_maturity(maturity)
        if not 0.0 < delta < 0.5:
            raise ValueError("delta must lie strictly between zero and 0.5.")

        forward = self.forward(maturity)
        atm_strike = forward
        atm_iv = self._model_iv_at_strike(model, parameters, maturity, atm_strike)

        call_strike = self._solve_delta_strike(
            model=model,
            parameters=parameters,
            maturity=maturity,
            target_delta=delta,
        )
        put_strike = self._solve_delta_strike(
            model=model,
            parameters=parameters,
            maturity=maturity,
            target_delta=-delta,
        )

        call_iv = self._model_iv_at_strike(
            model,
            parameters,
            maturity,
            call_strike,
        )
        put_iv = self._model_iv_at_strike(
            model,
            parameters,
            maturity,
            put_strike,
        )

        if not all(np.isfinite([atm_iv, call_iv, put_iv])):
            raise RuntimeError("Could not derive finite smile metrics.")

        risk_reversal = put_iv - call_iv
        butterfly = 0.5 * (put_iv + call_iv) - atm_iv

        return SmileMetrics(
            maturity=float(maturity),
            forward=float(forward),
            atm_strike=float(atm_strike),
            put_25_delta_strike=float(put_strike),
            call_25_delta_strike=float(call_strike),
            atm_volatility=float(atm_iv),
            put_25_delta_volatility=float(put_iv),
            call_25_delta_volatility=float(call_iv),
            risk_reversal_25=float(risk_reversal),
            butterfly_25=float(butterfly),
        )

    def assess_regime(
        self,
        metrics: SmileMetrics,
        jump_intensity: float,
        history: Optional[Sequence[tuple[SmileMetrics, float]]] = None,
        record: bool = True,
    ) -> RegimeSignal:
        """Classify downside crash risk and volatility-compression regimes.

        Severe downside crash risk requires:
        - a downside-skew spike, represented by a strongly negative call-minus-put
          skew slope, and
        - annualized Merton jump intensity greater than 0.15.

        Volatility compression is identified from a low ATM volatility and a flat
        butterfly/skew structure relative to historical observations. It is useful
        as a theta-harvesting or short-gamma-safety context, but it is not a
        guarantee against subsequent gap risk.
        """
        if not np.isfinite(jump_intensity) or jump_intensity < 0.0:
            raise ValueError("jump_intensity must be finite and non-negative.")

        observations = list(history) if history is not None else list(
            self._regime_history
        )
        historical_metrics = [item[0] for item in observations]
        historical_lambdas = np.asarray(
            [item[1] for item in observations],
            dtype=float,
        )

        current_rr = metrics.risk_reversal_25
        current_slope = metrics.downside_skew_slope
        current_atm = metrics.atm_volatility
        current_abs_fly = abs(metrics.butterfly_25)

        reasons: list[str] = []
        if len(historical_metrics) >= 5:
            historical_rr = np.asarray(
                [item.risk_reversal_25 for item in historical_metrics],
                dtype=float,
            )
            rr_median = float(np.nanmedian(historical_rr))
            rr_mad = float(np.nanmedian(np.abs(historical_rr - rr_median)))
            rr_scale = max(0.005, 1.4826 * rr_mad)
            skew_spike = bool(current_rr >= rr_median + max(0.015, 1.5 * rr_scale))

            historical_atm = np.asarray(
                [item.atm_volatility for item in historical_metrics],
                dtype=float,
            )
            historical_fly = np.asarray(
                [abs(item.butterfly_25) for item in historical_metrics],
                dtype=float,
            )
            atm_compressed = current_atm <= float(np.nanpercentile(historical_atm, 25))
            fly_flat = current_abs_fly <= max(
                0.005,
                float(np.nanpercentile(historical_fly, 40)),
            )
            volatility_compression = bool(atm_compressed and fly_flat)

            lambda_reference = float(np.nanmedian(historical_lambdas))
            intensity_elevated = bool(
                jump_intensity > 0.15
                and (
                    jump_intensity >= lambda_reference + 0.03
                    or lambda_reference <= 0.15
                )
            )
        else:
            skew_spike = bool(current_rr >= 0.04)
            volatility_compression = bool(
                current_atm <= 0.15
                and current_abs_fly <= 0.01
                and current_rr <= 0.025
            )
            intensity_elevated = bool(jump_intensity > 0.15)

        severe_downside = bool(skew_spike and jump_intensity > 0.15)
        if severe_downside:
            reasons.append(
                "Negative call-minus-put skew spike and jump intensity above 0.15."
            )
        elif skew_spike:
            reasons.append("Downside skew is elevated, but jump intensity is below 0.15.")
        elif jump_intensity > 0.15:
            reasons.append("Jump intensity is elevated, but skew has not spiked.")

        if volatility_compression:
            reasons.append(
                "ATM volatility and butterfly curvature indicate volatility compression."
            )

        if severe_downside:
            regime = "severe_downside_crash_risk"
            confidence = min(
                1.0,
                0.50
                + 1.5 * max(0.0, jump_intensity - 0.15)
                + 5.0 * max(0.0, current_rr - 0.04),
            )
        elif volatility_compression:
            regime = "volatility_compression"
            confidence = 0.70
        elif jump_intensity > 0.15 or current_rr > 0.025:
            regime = "elevated_downside_risk"
            confidence = 0.60
        else:
            regime = "normal"
            confidence = 0.55

        signal = RegimeSignal(
            regime=regime,
            severe_downside_crash_risk=severe_downside,
            volatility_compression=volatility_compression,
            jump_intensity=float(jump_intensity),
            downside_skew=float(current_slope),
            skew_spike=skew_spike,
            jump_intensity_elevated=intensity_elevated,
            confidence=float(confidence),
            reasons=tuple(reasons) if reasons else ("No material regime anomaly.",),
        )

        if record:
            self._regime_history.append((metrics, float(jump_intensity)))
            if len(self._regime_history) > self.history_limit:
                self._regime_history = self._regime_history[-self.history_limit :]

        return signal

    def dynamic_stop_loss_adjustment(
        self,
        signal: RegimeSignal,
        base_multiplier: float = 1.0,
    ) -> StopLossAdjustment:
        """Create a MaBuy Titan stop-loss dilation and position-size recommendation.

        A multiplier above one expands the stop-loss distance. During severe crash
        risk the method simultaneously lowers position size, so increased stop
        distance does not automatically increase monetary risk.
        """
        if not np.isfinite(base_multiplier) or base_multiplier <= 0.0:
            raise ValueError("base_multiplier must be finite and positive.")

        if signal.severe_downside_crash_risk:
            jump_excess = max(0.0, signal.jump_intensity - 0.15)
            skew_excess = max(0.0, -signal.downside_skew - 0.04)
            dilation = base_multiplier * min(
                2.75,
                1.50 + 2.0 * jump_excess + 5.0 * skew_excess,
            )
            position_multiplier = max(0.35, 1.0 / dilation)
            rationale = (
                "Severe downside jump/skew regime: widen stop distance to avoid "
                "noise exits and reduce gross exposure to preserve cash risk."
            )
        elif signal.regime == "elevated_downside_risk":
            dilation = base_multiplier * 1.25
            position_multiplier = 0.80
            rationale = (
                "Elevated downside risk: moderately widen stops and reduce size."
            )
        elif signal.volatility_compression:
            dilation = base_multiplier * 0.90
            position_multiplier = 1.00
            rationale = (
                "Volatility compression: modestly tighter stops are appropriate "
                "for controlled theta-harvesting or short-gamma conditions."
            )
        else:
            dilation = base_multiplier
            position_multiplier = 1.00
            rationale = "Normal volatility regime: retain baseline risk controls."

        return StopLossAdjustment(
            stop_loss_dilation_multiplier=float(dilation),
            position_size_multiplier=float(position_multiplier),
            regime=signal.regime,
            rationale=rationale,
        )

    def titan_risk_payload(
        self,
        signal: RegimeSignal,
        base_stop_loss_multiplier: float = 1.0,
    ) -> dict[str, Union[str, float, bool, tuple[str, ...]]]:
        """Return a serializable MaBuy Titan risk-control payload."""
        adjustment = self.dynamic_stop_loss_adjustment(
            signal,
            base_stop_loss_multiplier,
        )
        return {
            "regime": signal.regime,
            "severe_downside_crash_risk": signal.severe_downside_crash_risk,
            "volatility_compression": signal.volatility_compression,
            "jump_intensity": signal.jump_intensity,
            "downside_skew": signal.downside_skew,
            "confidence": signal.confidence,
            "stop_loss_dilation_multiplier": adjustment.stop_loss_dilation_multiplier,
            "position_size_multiplier": adjustment.position_size_multiplier,
            "rationale": adjustment.rationale,
            "signal_reasons": signal.reasons,
        }

    def clear_regime_history(self) -> None:
        """Clear internally retained regime observations."""
        self._regime_history.clear()

    def _prepare_market_data(
        self,
        slices: Sequence[OptionSurfaceSlice],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Flatten valid multi-expiry market slices into calibration arrays."""
        if not slices:
            raise ValueError("At least one surface slice is required.")

        maturities_list: list[np.ndarray] = []
        strikes_list: list[np.ndarray] = []
        prices_list: list[np.ndarray] = []
        weights_list: list[np.ndarray] = []

        for option_slice in slices:
            calls = self._slice_call_prices(option_slice)
            strikes = option_slice.strikes
            if option_slice.weights is None:
                weights = np.ones_like(strikes)
            else:
                weights = np.asarray(option_slice.weights, dtype=float)

            valid = (
                np.isfinite(calls)
                & np.isfinite(strikes)
                & np.isfinite(weights)
                & (strikes > 0.0)
                & (weights > 0.0)
            )
            if not np.any(valid):
                continue

            maturities_list.append(
                np.full(np.sum(valid), option_slice.maturity, dtype=float)
            )
            strikes_list.append(strikes[valid])
            prices_list.append(calls[valid])
            weights_list.append(weights[valid])

        if not maturities_list:
            raise ValueError("No valid market observations were supplied.")

        return (
            np.concatenate(maturities_list),
            np.concatenate(strikes_list),
            np.concatenate(prices_list),
            np.concatenate(weights_list),
        )

    def _resolve_svi_parameters(
        self,
        parameters: Union[SVIParameters, Mapping[float, SVIParameters]],
        maturity: float,
    ) -> SVIParameters:
        """Resolve a single SVI expiry parameter set."""
        if isinstance(parameters, SVIParameters):
            return parameters
        if isinstance(parameters, Mapping):
            if maturity in parameters:
                return parameters[maturity]
            nearest_maturity = min(parameters.keys(), key=lambda item: abs(item - maturity))
            if abs(nearest_maturity - maturity) > 1e-10:
                raise KeyError(f"No SVI parameters found for maturity {maturity}.")
            return parameters[nearest_maturity]
        raise TypeError("SVI model requires SVIParameters or a maturity mapping.")

    def _model_iv_at_strike(
        self,
        model: ModelName,
        parameters: Union[
            SVIParameters,
            HestonParameters,
            MertonJumpParameters,
            Mapping[float, SVIParameters],
        ],
        maturity: float,
        strike: float,
    ) -> float:
        """Return model-consistent Black implied volatility at one strike."""
        normalized_model = model.lower()
        if normalized_model == "svi":
            svi_parameters = self._resolve_svi_parameters(parameters, maturity)
            return float(
                svi_parameters.implied_volatility(
                    np.log(strike / self.forward(maturity))
                )
            )
        if normalized_model == "heston":
            if not isinstance(parameters, HestonParameters):
                raise TypeError("Heston model requires HestonParameters.")
            price = self.heston_call_price(strike, maturity, parameters)
            return self.implied_volatility(price, strike, maturity, "call", strict=True)
        if normalized_model == "merton":
            if not isinstance(parameters, MertonJumpParameters):
                raise TypeError("Merton model requires MertonJumpParameters.")
            price = self.merton_call_price(strike, maturity, parameters)
            return self.implied_volatility(price, strike, maturity, "call", strict=True)
        raise ValueError(f"Unsupported model: {model!r}")

    def _solve_delta_strike(
        self,
        model: ModelName,
        parameters: Union[
            SVIParameters,
            HestonParameters,
            MertonJumpParameters,
            Mapping[float, SVIParameters],
        ],
        maturity: float,
        target_delta: float,
    ) -> float:
        """Solve for a strike corresponding to a target spot Black delta."""
        if target_delta == 0.0 or abs(target_delta) >= 1.0:
            raise ValueError("target_delta must be strictly between -1 and 1.")

        is_call = target_delta > 0.0
        discount_q = np.exp(-self.dividend_yield * maturity)
        forward = self.forward(maturity)

        def delta_residual(log_strike: float) -> float:
            strike = float(np.exp(log_strike))
            volatility = self._model_iv_at_strike(
                model,
                parameters,
                maturity,
                strike,
            )
            volatility = max(volatility, 1e-8)
            d1 = (
                np.log(self.spot / strike)
                + (
                    self.risk_free_rate
                    - self.dividend_yield
                    + 0.5 * volatility * volatility
                )
                * maturity
            ) / (volatility * np.sqrt(maturity))
            model_delta = (
                discount_q * norm.cdf(d1)
                if is_call
                else -discount_q * norm.cdf(-d1)
            )
            return float(model_delta - target_delta)

        lower = np.log(forward * 0.03)
        upper = np.log(forward * 8.0)

        try:
            return float(np.exp(brentq(delta_residual, lower, upper, xtol=1e-8)))
        except ValueError:
            grid = np.linspace(lower, upper, 61)
            values = np.asarray([delta_residual(value) for value in grid])
            sign_changes = np.where(values[:-1] * values[1:] <= 0.0)[0]
            if len(sign_changes) == 0:
                raise RuntimeError(
                    f"Could not bracket the {target_delta:.2f}-delta strike."
                )
            index = int(sign_changes[0])
            root = brentq(
                delta_residual,
                float(grid[index]),
                float(grid[index + 1]),
                xtol=1e-8,
            )
            return float(np.exp(root))

    @staticmethod
    def _validate_heston_parameters(parameters: HestonParameters) -> None:
        """Validate Heston parameter domains."""
        values = np.array(
            [
                parameters.kappa,
                parameters.theta,
                parameters.sigma_v,
                parameters.rho,
                parameters.v0,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("Heston parameters must be finite.")
        if parameters.kappa <= 0.0:
            raise ValueError("Heston kappa must be positive.")
        if parameters.theta < 0.0:
            raise ValueError("Heston theta cannot be negative.")
        if parameters.sigma_v <= 0.0:
            raise ValueError("Heston sigma_v must be positive.")
        if abs(parameters.rho) >= 1.0:
            raise ValueError("Heston rho must lie strictly between -1 and 1.")
        if parameters.v0 < 0.0:
            raise ValueError("Heston v0 cannot be negative.")

    @staticmethod
    def _validate_merton_parameters(parameters: MertonJumpParameters) -> None:
        """Validate Merton jump-diffusion parameter domains."""
        values = np.array(
            [
                parameters.sigma,
                parameters.lambda_jump,
                parameters.mu_jump,
                parameters.sigma_jump,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("Merton parameters must be finite.")
        if parameters.sigma < 0.0:
            raise ValueError("Merton diffusion sigma cannot be negative.")
        if parameters.lambda_jump < 0.0:
            raise ValueError("Merton jump intensity cannot be negative.")
        if parameters.sigma_jump < 0.0:
            raise ValueError("Merton jump volatility cannot be negative.")


if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)

    spot = 100.0
    calibrator = VolatilitySurfaceCalibrator(
        spot=spot,
        risk_free_rate=0.02,
        dividend_yield=0.01,
    )

    true_merton = MertonJumpParameters(
        sigma=0.17,
        lambda_jump=0.28,
        mu_jump=-0.18,
        sigma_jump=0.24,
    )

    maturities = (0.25, 0.50, 1.00)
    strikes = np.array([70.0, 80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0, 130.0])
    synthetic_slices: list[OptionSurfaceSlice] = []

    for maturity in maturities:
        synthetic_prices = calibrator.merton_call_prices(
            strikes,
            maturity,
            true_merton,
        )
        synthetic_iv = calibrator.implied_volatility_array(
            synthetic_prices,
            strikes,
            maturity,
            "call",
        )
        synthetic_slices.append(
            OptionSurfaceSlice(
                maturity=maturity,
                strikes=strikes,
                prices=synthetic_prices,
                implied_vols=synthetic_iv,
                option_types="call",
            )
        )

    svi_surface = calibrator.calibrate_svi_surface(synthetic_slices)
    calibrated_merton = calibrator.calibrate_merton(
        synthetic_slices,
        initial=MertonJumpParameters(
            sigma=0.20,
            lambda_jump=0.20,
            mu_jump=-0.12,
            sigma_jump=0.20,
        ),
        max_nfev=600,
    )

    test_expiry = 0.50
    test_strikes = np.linspace(75.0, 125.0, 11)
    merton_smile = calibrator.implied_volatility_smile(
        "merton",
        calibrated_merton,
        test_expiry,
        test_strikes,
    )
    svi_smile = calibrator.implied_volatility_smile(
        "svi",
        svi_surface,
        test_expiry,
        test_strikes,
    )

    smile_metrics = calibrator.extract_skew_and_butterfly(
        "merton",
        calibrated_merton,
        test_expiry,
    )
    regime_signal = calibrator.assess_regime(
        smile_metrics,
        calibrated_merton.lambda_jump,
    )
    stop_adjustment = calibrator.dynamic_stop_loss_adjustment(regime_signal)

    sample_heston = HestonParameters(
        kappa=1.8,
        theta=0.04,
        sigma_v=0.55,
        rho=-0.70,
        v0=0.045,
    )
    heston_atm_call = calibrator.heston_call_price(
        calibrator.forward(test_expiry),
        test_expiry,
        sample_heston,
    )

    print("Synthetic Merton calibration")
    print("True Merton parameters:      ", true_merton)
    print("Calibrated Merton parameters:", calibrated_merton)
    print("\nSVI parameters at 0.50Y:     ", svi_surface[test_expiry])
    print("\nMerton smile IVs:", merton_smile)
    print("SVI smile IVs:   ", svi_smile)
    print("\nSmile metrics:", smile_metrics)
    print("Regime signal:", regime_signal)
    print("Titan adjustment:", stop_adjustment)
    print(f"\nSample Heston ATM call price: {heston_atm_call:.6f}")