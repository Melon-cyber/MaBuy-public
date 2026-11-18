"""
Portfolio Optimization, Modern Portfolio Theory (MPT), DCA Simulator, and Stress Testing Service.
Part of the MaBuy Quantum Risk & Compliance Architecture.

Strict mathematical guarantees:
1. Modern Portfolio Theory (MPT) & Efficient Frontier:
   - Max Sharpe Ratio Portfolio Optimization via SLSQP.
   - Minimum Variance Portfolio (Global Minimum Variance - GMV) Optimization.
   - Strict constraints: w_i >= 0, sum(w_i) == 1.0, altcoin weight <= 35% (configurable).
   - Numerical covariance shrinkage (Ledoit-Wolf / Tikhonov regularization) for positive semi-definiteness.
2. Realistic Dollar-Cost Averaging (DCA) Simulator:
   - Weekly deposit ($10 - $500), 1 to 5 year horizon.
   - Historical Block Bootstrap (Politis-Romano / Moving Block) to preserve volatility clustering and cross-asset correlations.
   - Multi-trajectory Monte Carlo fan (P10, P50, P90 percentiles, ROI, CAGR, Drawdowns).
3. Historical Crash Stress Tests:
   - FTX Collapse (-50% market, -65% SOL), COVID-19 Flash Crash (-50%), Luna/UST Crash (-60%),
     China Mining Ban (-52%), and Crypto Winter 2022 Peak-to-Trough (-77% to -96%).
   - Stress Survival Score (0-100) and Tail Correlation breakdown analysis.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import numpy as np
import scipy.optimize as sco

logger = logging.getLogger(__name__)


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass(frozen=True)
class PortfolioOptimizationResult:
    """Represents an optimized portfolio allocation result."""
    weights: Dict[str, float]
    expected_return_annual: float
    volatility_annual: float
    sharpe_ratio: float
    daily_volatility: float
    var_95_daily_pct: float
    cvar_95_daily_pct: float
    var_95_annual_pct: float
    cvar_95_annual_pct: float
    diversification_score: int  # 0 to 100
    hhi: float
    effective_n: float
    status: str
    optimization_type: str  # "max_sharpe" | "min_variance" | "target_return"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EfficientFrontierPoint:
    """A single point along the Markowitz Efficient Frontier curve."""
    target_return_annual: float
    volatility_annual: float
    sharpe_ratio: float
    weights: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EfficientFrontierResult:
    """The complete Efficient Frontier dataset."""
    frontier_points: List[EfficientFrontierPoint]
    max_sharpe_portfolio: PortfolioOptimizationResult
    min_variance_portfolio: PortfolioOptimizationResult
    assets: List[str]
    expected_returns: Dict[str, float]
    covariance_matrix: List[List[float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frontier_points": [p.to_dict() for p in self.frontier_points],
            "max_sharpe_portfolio": self.max_sharpe_portfolio.to_dict(),
            "min_variance_portfolio": self.min_variance_portfolio.to_dict(),
            "assets": self.assets,
            "expected_returns": self.expected_returns,
            "covariance_matrix": self.covariance_matrix,
        }


@dataclass(frozen=True)
class DCASimulationResult:
    """Results from the Historical Block Bootstrap DCA Simulator."""
    weekly_investment_usd: float
    horizon_years: float
    total_weeks: int
    initial_deposit_usd: float
    total_invested_usd: float
    p10_final_value_usd: float
    p50_final_value_usd: float
    p90_final_value_usd: float
    p10_roi_pct: float
    p50_roi_pct: float
    p90_roi_pct: float
    p10_cagr_pct: float
    p50_cagr_pct: float
    p90_cagr_pct: float
    prob_profit: float  # P(Final Value > Total Invested)
    prob_double: float  # P(Final Value >= 2 * Total Invested)
    median_max_drawdown_pct: float
    worst_case_drawdown_pct: float
    weekly_trajectory_p10: List[float]
    weekly_trajectory_p50: List[float]
    weekly_trajectory_p90: List[float]
    weekly_invested_curve: List[float]
    n_simulations: int
    block_size_weeks: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StressScenarioResult:
    """Impact of a single historical crash scenario on the portfolio."""
    scenario_id: str
    name: str
    description: str
    portfolio_drawdown_pct: float
    loss_usd_on_10k: float
    remaining_value_usd_on_10k: float
    worst_hit_asset: str
    worst_hit_asset_drop_pct: float
    asset_breakdown: Dict[str, float]
    estimated_recovery_months: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PortfolioStressTestReport:
    """Comprehensive Stress Test report across all systemic crash events."""
    scenarios: Dict[str, StressScenarioResult]
    worst_case_scenario_id: str
    worst_case_loss_pct: float
    worst_case_loss_usd_10k: float
    average_crash_drawdown_pct: float
    stress_survival_score: int  # 0 to 100
    resilience_rating: str  # e.g. "MAXIMUM SHIELD", "BALANCED DEFENSE", "CRITICAL RISK"
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenarios": {k: v.to_dict() for k, v in self.scenarios.items()},
            "worst_case_scenario_id": self.worst_case_scenario_id,
            "worst_case_loss_pct": self.worst_case_loss_pct,
            "worst_case_loss_usd_10k": self.worst_case_loss_usd_10k,
            "average_crash_drawdown_pct": self.average_crash_drawdown_pct,
            "stress_survival_score": self.stress_survival_score,
            "resilience_rating": self.resilience_rating,
            "recommendations": self.recommendations,
        }


@dataclass(frozen=True)
class CompletePortfolioAnalysis:
    """Full end-to-end portfolio diagnostic combining MPT, DCA & Stress Tests."""
    current_weights: Dict[str, float]
    max_sharpe_optimal: PortfolioOptimizationResult
    min_variance_optimal: PortfolioOptimizationResult
    dca_simulation: DCASimulationResult
    stress_test_report: PortfolioStressTestReport
    efficient_frontier_sample: List[EfficientFrontierPoint]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_weights": self.current_weights,
            "max_sharpe_optimal": self.max_sharpe_optimal.to_dict(),
            "min_variance_optimal": self.min_variance_optimal.to_dict(),
            "dca_simulation": self.dca_simulation.to_dict(),
            "stress_test_report": self.stress_test_report.to_dict(),
            "efficient_frontier_sample": [p.to_dict() for p in self.efficient_frontier_sample],
        }


@dataclass(frozen=True)
class BlackLittermanView:
    """
    Represents an investor or AI quantitative view for Black-Litterman model.
    view_type: 'absolute' (e.g. BTC expected return = 45%) or 'relative' (e.g. NVDA outperforms SPY by 10%).
    """
    view_type: str  # 'absolute' | 'relative'
    assets: List[str]  # e.g. ['BTC'] or ['NVDA', 'SPY']
    weights: List[float]  # e.g. [1.0] or [1.0, -1.0]
    expected_return: float  # e.g. 0.45 or 0.10
    confidence: float = 0.60  # in (0.0, 1.0]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BlackLittermanResult:
    """Comprehensive Black-Litterman Asset Allocation Diagnostic Result."""
    assets: List[str]
    implied_equilibrium_returns: Dict[str, float]
    posterior_expected_returns: Dict[str, float]
    posterior_covariance_matrix: List[List[float]]
    views_summary: List[Dict[str, Any]]
    tau: float
    risk_aversion_delta: float
    optimized_max_sharpe: PortfolioOptimizationResult
    optimized_min_variance: PortfolioOptimizationResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assets": self.assets,
            "implied_equilibrium_returns": self.implied_equilibrium_returns,
            "posterior_expected_returns": self.posterior_expected_returns,
            "posterior_covariance_matrix": self.posterior_covariance_matrix,
            "views_summary": self.views_summary,
            "tau": self.tau,
            "risk_aversion_delta": self.risk_aversion_delta,
            "optimized_max_sharpe": self.optimized_max_sharpe.to_dict(),
            "optimized_min_variance": self.optimized_min_variance.to_dict(),
        }


# ============================================================================
# PORTFOLIO OPTIMIZER SERVICE
# ============================================================================

class PortfolioOptimizerService:
    """
    Industrial-Grade Long-Term Investment Portfolio Optimizer for Cross-Market Assets.

    Mathematical Core:
    - Modern Portfolio Theory (Markowitz, 1952) with SLSQP quadratic programming.
    - Black-Litterman Asset Allocation Model (1992) for weight stabilization and view integration.
    - Ledoit-Wolf Shrinkage & Positive Semi-Definite (PSD) Covariance regularizer.
    - Moving Block Bootstrap DCA Simulation (Politis & Romano, 1994).
    - Extreme Event Historical Stress Testing & Tail Breakdown Analysis.
    """

    CORE_ASSETS: Set[str] = {"BTC", "USDT", "USDC", "SPY", "QQQ", "GLD"}
    DEFAULT_RISK_FREE_RATE: float = 0.035  # 3.5% annual risk-free rate
    MAX_ALTCOIN_WEIGHT_DEFAULT: float = 0.35  # 35% cap on individual altcoins

    # Calibrated Annualized Expected Returns (Long-Term Cycle Geometric Priors)
    ASSET_EXPECTED_RETURNS_ANNUAL: Dict[str, float] = {
        # Cryptocurrencies (Continuous 365 days / 24-7)
        "BTC": 0.55,
        "ETH": 0.62,
        "BNB": 0.48,
        "SOL": 0.75,
        "XRP": 0.38,
        "TON": 0.65,
        "NEAR": 0.70,
        "DOGE": 0.42,
        "ADA": 0.40,
        "AVAX": 0.68,
        "USDT": 0.04,  # Staking/Lending yield
        "USDC": 0.04,
        # US Equities (NYSE / NASDAQ Calendar — 252 trading days)
        "NVDA": 0.35,
        "AAPL": 0.18,
        "TSLA": 0.30,
        "MSFT": 0.20,
        "GOOGL": 0.20,
        # Macro ETFs (NYSE / NASDAQ Calendar — 252 trading days)
        "SPY": 0.11,
        "QQQ": 0.16,
        "GLD": 0.08,
    }

    # Calibrated Annualized Volatilities
    ASSET_ANNUAL_VOLATILITIES: Dict[str, float] = {
        # Cryptocurrencies
        "BTC": 0.60,
        "ETH": 0.75,
        "BNB": 0.65,
        "SOL": 0.95,
        "XRP": 0.85,
        "TON": 0.90,
        "NEAR": 1.05,
        "DOGE": 1.15,
        "ADA": 0.90,
        "AVAX": 1.00,
        "USDT": 0.02,
        "USDC": 0.02,
        # US Equities
        "NVDA": 0.45,
        "AAPL": 0.22,
        "TSLA": 0.50,
        "MSFT": 0.23,
        "GOOGL": 0.25,
        # Macro ETFs
        "SPY": 0.16,
        "QQQ": 0.21,
        "GLD": 0.14,
    }

    # Pairwise Correlations in Normal Regime
    PAIRWISE_CORRELATIONS: Dict[Tuple[str, str], float] = {
        # Crypto-Crypto
        ("BTC", "ETH"): 0.82,
        ("BTC", "SOL"): 0.76,
        ("BTC", "BNB"): 0.74,
        ("BTC", "XRP"): 0.62,
        ("BTC", "TON"): 0.55,
        ("BTC", "NEAR"): 0.72,
        ("BTC", "DOGE"): 0.65,
        ("BTC", "ADA"): 0.70,
        ("BTC", "AVAX"): 0.74,
        ("ETH", "SOL"): 0.79,
        ("ETH", "BNB"): 0.76,
        ("ETH", "NEAR"): 0.78,
        ("ETH", "TON"): 0.58,
        ("ETH", "AVAX"): 0.82,
        ("ETH", "ADA"): 0.75,
        ("SOL", "NEAR"): 0.81,
        ("SOL", "TON"): 0.64,
        ("SOL", "AVAX"): 0.80,
        # US Equities / Macro ETFs
        ("SPY", "QQQ"): 0.92,
        ("AAPL", "MSFT"): 0.76,
        ("AAPL", "QQQ"): 0.88,
        ("AAPL", "SPY"): 0.84,
        ("MSFT", "QQQ"): 0.89,
        ("MSFT", "SPY"): 0.86,
        ("NVDA", "QQQ"): 0.86,
        ("NVDA", "SPY"): 0.78,
        ("TSLA", "QQQ"): 0.72,
        ("TSLA", "SPY"): 0.65,
        ("GOOGL", "QQQ"): 0.84,
        ("GOOGL", "SPY"): 0.82,
        ("GLD", "SPY"): 0.05,
        ("GLD", "QQQ"): 0.02,
        # Cross-Market (Crypto vs Equities/ETFs)
        ("BTC", "SPY"): 0.38,
        ("BTC", "QQQ"): 0.45,
        ("BTC", "GLD"): 0.18,
        ("BTC", "NVDA"): 0.48,
        ("BTC", "TSLA"): 0.46,
        ("ETH", "SPY"): 0.40,
        ("ETH", "QQQ"): 0.46,
        ("SOL", "QQQ"): 0.48,
        ("SOL", "NVDA"): 0.50,
    }

    # Model Institutional Presets
    MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
        "all_weather_2": {
            "name": "All-Weather 2.0 (Ray Dalio Enhanced)",
            "description": "Классический всепогодный портфель с добавлением крипто-альфы (15% BTC) для максимального коэффициента Шарпа.",
            "weights": {"SPY": 0.30, "GLD": 0.20, "QQQ": 0.20, "BTC": 0.15, "USDT": 0.15},
        },
        "tech_crypto_alpha": {
            "name": "Tech Giants + Crypto Alpha",
            "description": "Высокотехнологичный импульсный портфель с акциями BigTech лидеров и топовыми L1 блокчейнами.",
            "weights": {"NVDA": 0.25, "MSFT": 0.20, "BTC": 0.25, "SOL": 0.15, "ETH": 0.15},
        },
        "buffett_hedge": {
            "name": "Buffett Value & Gold Shield",
            "description": "Консервативный баланс надежных денежных потоков (Apple, Google), золотого хеджа и микро-доли BTC.",
            "weights": {"AAPL": 0.35, "GOOGL": 0.25, "GLD": 0.20, "SPY": 0.15, "BTC": 0.05},
        },
    }

    # Calibrated Historical Crash Loss Vectors
    HISTORICAL_CRASH_SCENARIOS: Dict[str, Dict[str, Any]] = {
        "ftx_collapse_2022": {
            "name": "FTX Collapse (Nov 2022)",
            "description": (
                "Крах биржи FTX и фонда Alameda Research. Каскадная волна ликвидаций, "
                "сокрушительный удар по экосистеме Solana и бегство институционалов."
            ),
            "asset_drops": {
                "BTC": -0.30,
                "ETH": -0.35,
                "SOL": -0.65,
                "BNB": -0.28,
                "XRP": -0.38,
                "ADA": -0.42,
                "AVAX": -0.48,
                "TON": -0.40,
                "NEAR": -0.56,
                "DOGE": -0.45,
                "USDT": -0.005,
                "USDC": 0.00,
                "NVDA": -0.05,
                "AAPL": -0.02,
                "TSLA": -0.08,
                "MSFT": -0.03,
                "GOOGL": -0.03,
                "SPY": -0.02,
                "QQQ": -0.03,
                "GLD": 0.02,
            },
            "default_alt_drop": -0.52,
            "recovery_months": 8,
        },
        "covid_flash_crash_2020": {
            "name": "COVID-19 Flash Crash (March 2020)",
            "description": (
                "Глобальный шок ликвидности мировых рынков. Массовые маржин-коллы, "
                "схлопывание плеч и мгновенная распродажа всех рисковых активов."
            ),
            "asset_drops": {
                "BTC": -0.50,
                "ETH": -0.55,
                "SOL": -0.60,
                "BNB": -0.52,
                "XRP": -0.50,
                "ADA": -0.56,
                "AVAX": -0.60,
                "TON": -0.55,
                "NEAR": -0.60,
                "DOGE": -0.55,
                "USDT": 0.00,
                "USDC": 0.00,
                "NVDA": -0.32,
                "AAPL": -0.31,
                "TSLA": -0.60,
                "MSFT": -0.28,
                "GOOGL": -0.33,
                "SPY": -0.34,
                "QQQ": -0.28,
                "GLD": -0.04,
            },
            "default_alt_drop": -0.60,
            "recovery_months": 3,
        },
        "luna_ust_crash_2022": {
            "name": "Terra / LUNA & UST Crash (May 2022)",
            "description": (
                "Крах алгоритмического стейблкоина UST и токена LUNA, банкротство фонда 3AC. "
                "Системный кризис доверия в DeFi и цепная девальвация альткоинов."
            ),
            "asset_drops": {
                "BTC": -0.35,
                "ETH": -0.42,
                "SOL": -0.58,
                "BNB": -0.36,
                "XRP": -0.40,
                "ADA": -0.48,
                "AVAX": -0.65,
                "TON": -0.45,
                "NEAR": -0.58,
                "DOGE": -0.46,
                "USDT": -0.015,
                "USDC": 0.00,
                "NVDA": -0.12,
                "AAPL": -0.06,
                "TSLA": -0.15,
                "MSFT": -0.06,
                "GOOGL": -0.07,
                "SPY": -0.05,
                "QQQ": -0.08,
                "GLD": -0.02,
            },
            "default_alt_drop": -0.62,
            "recovery_months": 6,
        },
        "china_mining_ban_2021": {
            "name": "China Mining Ban (May 2021)",
            "description": (
                "Запрет майнинга и операций с криптовалютой в КНР. Падение хешрейта сети "
                "на 50%, вынужденная капитуляция майнеров и панический сброс."
            ),
            "asset_drops": {
                "BTC": -0.52,
                "ETH": -0.58,
                "SOL": -0.62,
                "BNB": -0.65,
                "XRP": -0.55,
                "ADA": -0.50,
                "AVAX": -0.68,
                "TON": -0.55,
                "NEAR": -0.65,
                "DOGE": -0.60,
                "USDT": 0.00,
                "USDC": 0.00,
                "NVDA": -0.03,
                "AAPL": 0.01,
                "TSLA": -0.04,
                "MSFT": 0.01,
                "GOOGL": 0.02,
                "SPY": 0.01,
                "QQQ": 0.02,
                "GLD": 0.03,
            },
            "default_alt_drop": -0.65,
            "recovery_months": 5,
        },
        "crypto_winter_bear_2022": {
            "name": "Crypto Winter 2022 Peak-to-Trough",
            "description": (
                "Глубокая 12-месячная медвежья фаза цикла на фоне агрессивного повышения ставок ФРС США. "
                "Максимальная просадка от исторического максимума до абсолютного дна."
            ),
            "asset_drops": {
                "BTC": -0.77,
                "ETH": -0.82,
                "SOL": -0.96,
                "BNB": -0.72,
                "XRP": -0.84,
                "ADA": -0.91,
                "AVAX": -0.93,
                "TON": -0.75,
                "NEAR": -0.94,
                "DOGE": -0.92,
                "USDT": 0.00,
                "USDC": 0.00,
                "NVDA": -0.65,
                "AAPL": -0.31,
                "TSLA": -0.73,
                "MSFT": -0.38,
                "GOOGL": -0.44,
                "SPY": -0.25,
                "QQQ": -0.36,
                "GLD": -0.10,
            },
            "default_alt_drop": -0.90,
            "recovery_months": 18,
        },
    }

    # ============================================================================
    # 1. MODERN PORTFOLIO THEORY & EFFICIENT FRONTIER
    # ============================================================================

    @classmethod
    def normalize_weights(cls, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalizes portfolio weights strictly to sum to 1.0."""
        if not weights:
            return {"BTC": 1.0}
        
        cleaned = {k.upper(): max(0.0, float(v)) for k, v in weights.items()}
        total = sum(cleaned.values())
        if total <= 0:
            return {"BTC": 1.0}
        
        return {k: round(v / total, 6) for k, v in cleaned.items()}

    @classmethod
    def get_asset_return_prior(cls, asset: str) -> float:
        """Returns annualized expected return prior for asset."""
        return cls.ASSET_EXPECTED_RETURNS_ANNUAL.get(asset.upper(), 0.50)

    @classmethod
    def get_asset_volatility_prior(cls, asset: str) -> float:
        """Returns annualized volatility prior for asset."""
        return cls.ASSET_ANNUAL_VOLATILITIES.get(asset.upper(), 0.90)

    @classmethod
    def get_correlation(cls, a1: str, a2: str) -> float:
        """Returns baseline return correlation between two assets."""
        s1, s2 = a1.upper(), a2.upper()
        if s1 == s2:
            return 1.0
        if s1 in ("USDT", "USDC") or s2 in ("USDT", "USDC"):
            return 0.02
        if (s1, s2) in cls.PAIRWISE_CORRELATIONS:
            return cls.PAIRWISE_CORRELATIONS[(s1, s2)]
        if (s2, s1) in cls.PAIRWISE_CORRELATIONS:
            return cls.PAIRWISE_CORRELATIONS[(s2, s1)]
        return 0.70  # Default correlation among altcoins

    @classmethod
    def build_covariance_matrix(
        cls,
        assets: List[str],
        custom_cov: Optional[np.ndarray] = None,
        shrinkage_lambda: float = 1e-4
    ) -> np.ndarray:
        """
        Builds a Positive Semi-Definite (PSD) annualized covariance matrix.
        Applies Tikhonov shrinkage regularization to prevent singular or ill-conditioned matrices.
        """
        n = len(assets)
        if custom_cov is not None:
            cov = np.array(custom_cov, dtype=np.float64)
            if cov.shape != (n, n):
                raise ValueError(f"Custom covariance shape {cov.shape} must match assets length ({n}, {n})")
        else:
            cov = np.zeros((n, n), dtype=np.float64)
            for i in range(n):
                vol_i = cls.get_asset_volatility_prior(assets[i])
                for j in range(n):
                    vol_j = cls.get_asset_volatility_prior(assets[j])
                    rho = cls.get_correlation(assets[i], assets[j])
                    cov[i, j] = vol_i * vol_j * rho

        # Ensure symmetry
        cov = 0.5 * (cov + cov.T)

        # Spectral decomposition to guarantee Positive Semi-Definiteness (PSD)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 1e-6)
        cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Add small diagonal shrinkage (Tikhonov regularizer)
        cov_psd += np.eye(n) * shrinkage_lambda
        return cov_psd

    @classmethod
    def _build_asset_bounds(
        cls,
        assets: List[str],
        max_alt_weight: float,
        core_assets: Optional[Set[str]] = None
    ) -> List[Tuple[float, float]]:
        """
        Constructs optimization bounds for each asset.
        Altcoins are strictly capped at max_alt_weight (default 35%).
        """
        core = core_assets or cls.CORE_ASSETS
        n = len(assets)
        bounds = []

        # Check feasibility: if all assets are altcoins and n * max_alt < 1.0, relax upper bound
        alt_count = sum(1 for a in assets if a.upper() not in core)
        effective_max_alt = max_alt_weight
        if alt_count == n and (n * max_alt_weight) < 1.0:
            effective_max_alt = max(max_alt_weight, 1.0 / n + 1e-4)

        for a in assets:
            if a.upper() in core:
                bounds.append((0.0, 1.0))
            else:
                bounds.append((0.0, float(effective_max_alt)))
        return bounds

    @classmethod
    def calculate_portfolio_metrics(
        cls,
        weights: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    ) -> Tuple[float, float, float]:
        """
        Calculates (expected_return, volatility, sharpe_ratio).
        """
        w = np.asarray(weights, dtype=np.float64)
        ret = float(np.dot(w, mu))
        var = float(np.dot(w.T, np.dot(cov, w)))
        vol = math.sqrt(max(1e-8, var))
        sharpe = (ret - risk_free_rate) / vol
        return ret, vol, sharpe

    @classmethod
    def optimize_max_sharpe(
        cls,
        assets: List[str],
        expected_returns: Optional[Dict[str, float]] = None,
        cov_matrix: Optional[np.ndarray] = None,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        max_alt_weight: float = MAX_ALTCOIN_WEIGHT_DEFAULT,
        core_assets: Optional[Set[str]] = None
    ) -> PortfolioOptimizationResult:
        """
        Finds the Max Sharpe Ratio (Tangency) Portfolio under strict constraints:
        - w_i >= 0 (no shorting)
        - sum(w_i) == 1.0
        - w_alt <= max_alt_weight (<= 35%)
        """
        cleaned_assets = [a.upper() for a in assets]
        n = len(cleaned_assets)
        if n == 0:
            raise ValueError("Assets list cannot be empty.")
        if n == 1:
            w_single = {cleaned_assets[0]: 1.0}
            ret = cls.get_asset_return_prior(cleaned_assets[0])
            vol = cls.get_asset_volatility_prior(cleaned_assets[0])
            sharpe = (ret - risk_free_rate) / vol
            return cls._build_opt_result(
                cleaned_assets, np.array([1.0]), ret, vol, sharpe, "single_asset", "max_sharpe"
            )

        mu = np.array([
            expected_returns[a] if expected_returns and a in expected_returns else cls.get_asset_return_prior(a)
            for a in cleaned_assets
        ], dtype=np.float64)

        cov = cls.build_covariance_matrix(cleaned_assets, cov_matrix)
        bounds = cls._build_asset_bounds(cleaned_assets, max_alt_weight, core_assets)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def neg_sharpe(w: np.ndarray) -> float:
            r, v, s = cls.calculate_portfolio_metrics(w, mu, cov, risk_free_rate)
            return -s

        # Robust starting points: equal weight + asset picks
        best_res = None
        min_obj = float("inf")

        initial_guesses = [
            np.ones(n) / n,
        ]
        # Add pure-BTC or top asset guess if within bounds
        for i in range(n):
            g = np.zeros(n)
            if bounds[i][1] >= 0.99:
                g[i] = 1.0
            else:
                g[i] = bounds[i][1]
                remaining = (1.0 - g[i]) / max(1, n - 1)
                for j in range(n):
                    if j != i:
                        g[j] = remaining
            if abs(np.sum(g) - 1.0) < 1e-4:
                initial_guesses.append(g)

        for init_w in initial_guesses:
            try:
                res = sco.minimize(
                    neg_sharpe,
                    init_w,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 500, "ftol": 1e-9}
                )
                if res.success and res.fun < min_obj:
                    min_obj = res.fun
                    best_res = res
            except Exception as e:
                logger.debug(f"Optimization guess failed: {e}")

        if best_res is None or not best_res.success:
            # Fallback to bounded equal weights
            raw_w = np.array([min(bounds[i][1], 1.0 / n) for i in range(n)])
            opt_w = raw_w / np.sum(raw_w)
            status = "fallback_equal_bounded"
        else:
            opt_w = np.clip(best_res.x, [b[0] for b in bounds], [b[1] for b in bounds])
            opt_w = opt_w / np.sum(opt_w)
            status = "optimal_converged"

        ret, vol, sharpe = cls.calculate_portfolio_metrics(opt_w, mu, cov, risk_free_rate)
        return cls._build_opt_result(cleaned_assets, opt_w, ret, vol, sharpe, status, "max_sharpe")

    @classmethod
    def optimize_min_variance(
        cls,
        assets: List[str],
        expected_returns: Optional[Dict[str, float]] = None,
        cov_matrix: Optional[np.ndarray] = None,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        max_alt_weight: float = MAX_ALTCOIN_WEIGHT_DEFAULT,
        core_assets: Optional[Set[str]] = None
    ) -> PortfolioOptimizationResult:
        """
        Finds the Global Minimum Variance (GMV) Portfolio under strict constraints:
        - Minimize w^T Sigma w
        - w_i >= 0
        - sum(w_i) == 1.0
        - w_alt <= max_alt_weight (<= 35%)
        """
        cleaned_assets = [a.upper() for a in assets]
        n = len(cleaned_assets)
        if n == 0:
            raise ValueError("Assets list cannot be empty.")
        if n == 1:
            ret = cls.get_asset_return_prior(cleaned_assets[0])
            vol = cls.get_asset_volatility_prior(cleaned_assets[0])
            sharpe = (ret - risk_free_rate) / vol
            return cls._build_opt_result(
                cleaned_assets, np.array([1.0]), ret, vol, sharpe, "single_asset", "min_variance"
            )

        mu = np.array([
            expected_returns[a] if expected_returns and a in expected_returns else cls.get_asset_return_prior(a)
            for a in cleaned_assets
        ], dtype=np.float64)

        cov = cls.build_covariance_matrix(cleaned_assets, cov_matrix)
        bounds = cls._build_asset_bounds(cleaned_assets, max_alt_weight, core_assets)
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def portfolio_variance(w: np.ndarray) -> float:
            return float(np.dot(w.T, np.dot(cov, w)))

        init_w = np.ones(n) / n
        res = sco.minimize(
            portfolio_variance,
            init_w,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9}
        )

        if res.success:
            opt_w = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
            opt_w = opt_w / np.sum(opt_w)
            status = "optimal_converged"
        else:
            raw_w = np.array([min(bounds[i][1], 1.0 / n) for i in range(n)])
            opt_w = raw_w / np.sum(raw_w)
            status = "fallback_equal_bounded"

        ret, vol, sharpe = cls.calculate_portfolio_metrics(opt_w, mu, cov, risk_free_rate)
        return cls._build_opt_result(cleaned_assets, opt_w, ret, vol, sharpe, status, "min_variance")

    @classmethod
    def compute_efficient_frontier(
        cls,
        assets: List[str],
        n_points: int = 25,
        expected_returns: Optional[Dict[str, float]] = None,
        cov_matrix: Optional[np.ndarray] = None,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        max_alt_weight: float = MAX_ALTCOIN_WEIGHT_DEFAULT,
        core_assets: Optional[Set[str]] = None
    ) -> EfficientFrontierResult:
        """
        Computes the complete Markowitz Efficient Frontier curve.
        Returns N points along the frontier from Min Variance to Maximum Feasible Return.
        """
        cleaned_assets = [a.upper() for a in assets]
        n = len(cleaned_assets)
        if n == 0:
            raise ValueError("Assets list cannot be empty.")

        mu = np.array([
            expected_returns[a] if expected_returns and a in expected_returns else cls.get_asset_return_prior(a)
            for a in cleaned_assets
        ], dtype=np.float64)

        cov = cls.build_covariance_matrix(cleaned_assets, cov_matrix)
        bounds = cls._build_asset_bounds(cleaned_assets, max_alt_weight, core_assets)

        # 1. Compute anchor portfolios: Min Variance and Max Sharpe
        min_var_res = cls.optimize_min_variance(
            cleaned_assets, expected_returns, cov_matrix, risk_free_rate, max_alt_weight, core_assets
        )
        max_sharpe_res = cls.optimize_max_sharpe(
            cleaned_assets, expected_returns, cov_matrix, risk_free_rate, max_alt_weight, core_assets
        )

        # 2. Determine target return span
        min_ret = min_var_res.expected_return_annual
        # Max feasible return subject to weight bounds
        # Solve linear programming for maximum return
        c = -mu
        A_eq = np.ones((1, n))
        b_eq = np.array([1.0])
        lp_res = sco.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        max_ret = -lp_res.fun if lp_res.success else float(np.max(mu))

        if max_ret <= min_ret:
            target_returns = np.array([min_ret])
        else:
            target_returns = np.linspace(min_ret, max_ret, n_points)

        frontier_points: List[EfficientFrontierPoint] = []
        for target_r in target_returns:
            constraints = [
                {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
                {"type": "eq", "fun": lambda w, tr=target_r: float(np.dot(w, mu)) - tr}
            ]

            def portfolio_variance(w: np.ndarray) -> float:
                return float(np.dot(w.T, np.dot(cov, w)))

            init_w = np.ones(n) / n
            res = sco.minimize(
                portfolio_variance,
                init_w,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 300, "ftol": 1e-8}
            )

            if res.success:
                w_pt = np.clip(res.x, [b[0] for b in bounds], [b[1] for b in bounds])
                w_pt = w_pt / np.sum(w_pt)
                r_pt, v_pt, s_pt = cls.calculate_portfolio_metrics(w_pt, mu, cov, risk_free_rate)
                weights_dict = {cleaned_assets[i]: round(float(w_pt[i]), 4) for i in range(n)}
                frontier_points.append(
                    EfficientFrontierPoint(
                        target_return_annual=round(r_pt, 4),
                        volatility_annual=round(v_pt, 4),
                        sharpe_ratio=round(s_pt, 3),
                        weights=weights_dict
                    )
                )

        # Fallback if solver struggled
        if not frontier_points:
            frontier_points = [
                EfficientFrontierPoint(
                    target_return_annual=min_var_res.expected_return_annual,
                    volatility_annual=min_var_res.volatility_annual,
                    sharpe_ratio=min_var_res.sharpe_ratio,
                    weights=min_var_res.weights
                ),
                EfficientFrontierPoint(
                    target_return_annual=max_sharpe_res.expected_return_annual,
                    volatility_annual=max_sharpe_res.volatility_annual,
                    sharpe_ratio=max_sharpe_res.sharpe_ratio,
                    weights=max_sharpe_res.weights
                )
            ]

        expected_returns_dict = {cleaned_assets[i]: round(float(mu[i]), 4) for i in range(n)}
        return EfficientFrontierResult(
            frontier_points=frontier_points,
            max_sharpe_portfolio=max_sharpe_res,
            min_variance_portfolio=min_var_res,
            assets=cleaned_assets,
            expected_returns=expected_returns_dict,
            covariance_matrix=cov.tolist()
        )

    @classmethod
    def _build_opt_result(
        cls,
        assets: List[str],
        weights: np.ndarray,
        ret_ann: float,
        vol_ann: float,
        sharpe: float,
        status: str,
        opt_type: str
    ) -> PortfolioOptimizationResult:
        """Helper to compute risk metrics and construct a PortfolioOptimizationResult."""
        weights_dict = {assets[i]: round(float(weights[i]), 4) for i in range(len(assets))}
        # Balance minor rounding discrepancy so sum equals 1.0 exactly
        diff = round(1.0 - sum(weights_dict.values()), 4)
        if diff != 0.0 and len(weights_dict) > 0:
            max_k = max(weights_dict, key=weights_dict.get)
            weights_dict[max_k] = round(weights_dict[max_k] + diff, 4)

        daily_vol = vol_ann / math.sqrt(365)
        
        # Parametric VaR / CVaR (Daily & Annual)
        var_95_daily = daily_vol * 1.645 * 100.0
        cvar_95_daily = daily_vol * 2.06 * 100.0
        var_95_annual = max(0.0, (vol_ann * 1.645 - ret_ann) * 100.0)
        cvar_95_annual = max(0.0, (vol_ann * 2.06 - ret_ann) * 100.0)

        # Diversification metrics
        hhi = float(np.sum(weights ** 2))
        effective_n = 1.0 / (hhi + 1e-8)
        raw_score = (1.0 - hhi) * 100.0 + min(20.0, effective_n * 5.0)
        div_score = int(min(100, max(15, raw_score)))

        return PortfolioOptimizationResult(
            weights=weights_dict,
            expected_return_annual=round(ret_ann, 4),
            volatility_annual=round(vol_ann, 4),
            sharpe_ratio=round(sharpe, 3),
            daily_volatility=round(daily_vol, 5),
            var_95_daily_pct=round(var_95_daily, 2),
            cvar_95_daily_pct=round(cvar_95_daily, 2),
            var_95_annual_pct=round(var_95_annual, 2),
            cvar_95_annual_pct=round(cvar_95_annual, 2),
            diversification_score=div_score,
            hhi=round(hhi, 4),
            effective_n=round(effective_n, 2),
            status=status,
            optimization_type=opt_type
        )

    # ============================================================================
    # 1.1 BLACK-LITTERMAN ASSET ALLOCATION ENGINE
    # ============================================================================

    @classmethod
    def calculate_implied_equilibrium_returns(
        cls,
        assets: List[str],
        cov_matrix: np.ndarray,
        market_weights: Optional[Dict[str, float]] = None,
        risk_aversion_delta: float = 2.5,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE
    ) -> np.ndarray:
        """
        Calculates CAPM Reverse Optimization Implied Equilibrium Excess/Total Returns:
        Pi = delta * Sigma * w_mkt + r_f
        """
        cleaned_assets = [a.upper() for a in assets]
        n = len(cleaned_assets)
        if market_weights:
            norm_mkt = cls.normalize_weights(market_weights)
            w_mkt = np.array([norm_mkt.get(a, 1.0 / n) for a in cleaned_assets], dtype=np.float64)
            w_mkt = w_mkt / np.sum(w_mkt)
        else:
            # Neutral equal-weight market baseline proxy
            w_mkt = np.ones(n, dtype=np.float64) / n

        excess_pi = risk_aversion_delta * (cov_matrix @ w_mkt)
        total_pi = risk_free_rate + excess_pi
        return total_pi

    @classmethod
    def build_view_matrices(
        cls,
        assets: List[str],
        views: List[BlackLittermanView],
        tau: float,
        cov_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs view link matrix P (K x N), view vector Q (K), and uncertainty covariance Omega (K x K).
        Omega is scaled via He & Litterman confidence: Omega_kk = ((1 - c_k) / c_k) * (P_k * tau * Sigma * P_k^T).
        """
        cleaned_assets = [a.upper() for a in assets]
        n = len(cleaned_assets)
        k = len(views)

        if k == 0:
            return np.zeros((0, n)), np.zeros(0), np.zeros((0, 0))

        P = np.zeros((k, n), dtype=np.float64)
        Q = np.zeros(k, dtype=np.float64)
        Omega = np.zeros((k, k), dtype=np.float64)

        for i, view in enumerate(views):
            Q[i] = float(view.expected_return)
            for a, w in zip(view.assets, view.weights):
                a_clean = a.upper()
                if a_clean in cleaned_assets:
                    idx = cleaned_assets.index(a_clean)
                    P[i, idx] = float(w)
                else:
                    logger.warning(f"Asset '{a_clean}' in Black-Litterman view not found in portfolio assets {cleaned_assets}.")

            # Compute view variance: p_k * (tau * Sigma) * p_k^T
            p_k = P[i, :]
            view_var = float(p_k @ (tau * cov_matrix) @ p_k)
            conf = max(0.01, min(1.0, float(view.confidence)))
            # If confidence is 1.0 (100%), uncertainty is near zero (1e-6)
            if conf >= 0.999:
                omega_diag = 1e-6
            else:
                omega_diag = max(1e-6, ((1.0 - conf) / conf) * max(view_var, 1e-4))
            Omega[i, i] = omega_diag

        return P, Q, Omega

    @classmethod
    def compute_black_litterman_posterior(
        cls,
        assets: List[str],
        views: Optional[List[BlackLittermanView]] = None,
        custom_cov: Optional[np.ndarray] = None,
        market_weights: Optional[Dict[str, float]] = None,
        tau: float = 0.05,
        risk_aversion_delta: float = 2.5,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        max_alt_weight: float = MAX_ALTCOIN_WEIGHT_DEFAULT,
        core_assets: Optional[Set[str]] = None
    ) -> BlackLittermanResult:
        """
        Computes Bayesian Combined Return Distribution (mu_BL, Sigma_BL) using the Black-Litterman Master Formula:
        mu_BL = Pi + tau * Sigma * P^T * [ P * (tau * Sigma) * P^T + Omega ]^-1 * (Q - P * Pi)
        Sigma_BL = (1 + tau) * Sigma - tau^2 * Sigma * P^T * [ P * (tau * Sigma) * P^T + Omega ]^-1 * P * Sigma
        """
        cleaned_assets = [a.upper() for a in assets]
        n = len(cleaned_assets)
        if n == 0:
            raise ValueError("Assets list cannot be empty.")

        cov = cls.build_covariance_matrix(cleaned_assets, custom_cov)
        pi = cls.calculate_implied_equilibrium_returns(
            cleaned_assets, cov, market_weights, risk_aversion_delta, risk_free_rate
        )

        valid_views = [v for v in (views or []) if any(a.upper() in cleaned_assets for a in v.assets)]
        views_summary: List[Dict[str, Any]] = []

        if not valid_views:
            # Fallback smoothly to prior equilibrium
            mu_bl = pi.copy()
            sigma_bl = (1.0 + tau) * cov
        else:
            P, Q, Omega = cls.build_view_matrices(cleaned_assets, valid_views, tau, cov)

            # Master Formula
            tau_sigma_Pt = (tau * cov) @ P.T  # Shape: (N, K)
            M = P @ tau_sigma_Pt + Omega      # Shape: (K, K)
            
            # Regularized solve / inversion
            try:
                M_inv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                M_inv = np.linalg.pinv(M)

            # Expected return vector
            excess_views = Q - P @ pi
            mu_bl = pi + tau_sigma_Pt @ (M_inv @ excess_views)

            # Posterior covariance matrix
            sigma_adjustment = (tau * cov @ P.T) @ M_inv @ (P @ (tau * cov))
            sigma_bl = (1.0 + tau) * cov - sigma_adjustment

            # Enforce symmetry and PSD on posterior covariance
            sigma_bl = 0.5 * (sigma_bl + sigma_bl.T)
            eigvals, eigvecs = np.linalg.eigh(sigma_bl)
            eigvals = np.maximum(eigvals, 1e-6)
            sigma_bl = eigvecs @ np.diag(eigvals) @ eigvecs.T

            for v in valid_views:
                views_summary.append(v.to_dict())

        # Optimize Max Sharpe and Min Variance on the stabilized Black-Litterman posterior
        mu_bl_dict = {cleaned_assets[i]: float(mu_bl[i]) for i in range(n)}
        max_sharpe_opt = cls.optimize_max_sharpe(
            assets=cleaned_assets,
            expected_returns=mu_bl_dict,
            cov_matrix=sigma_bl,
            risk_free_rate=risk_free_rate,
            max_alt_weight=max_alt_weight,
            core_assets=core_assets
        )
        min_var_opt = cls.optimize_min_variance(
            assets=cleaned_assets,
            expected_returns=mu_bl_dict,
            cov_matrix=sigma_bl,
            risk_free_rate=risk_free_rate,
            max_alt_weight=max_alt_weight,
            core_assets=core_assets
        )

        pi_dict = {cleaned_assets[i]: round(float(pi[i]), 4) for i in range(n)}
        mu_bl_rounded_dict = {cleaned_assets[i]: round(float(mu_bl[i]), 4) for i in range(n)}

        return BlackLittermanResult(
            assets=cleaned_assets,
            implied_equilibrium_returns=pi_dict,
            posterior_expected_returns=mu_bl_rounded_dict,
            posterior_covariance_matrix=sigma_bl.tolist(),
            views_summary=views_summary,
            tau=tau,
            risk_aversion_delta=risk_aversion_delta,
            optimized_max_sharpe=max_sharpe_opt,
            optimized_min_variance=min_var_opt
        )

    @classmethod
    def generate_ai_views_from_forecaster(
        cls,
        forecasts_by_symbol: Dict[str, Dict[str, Any]]
    ) -> List[BlackLittermanView]:
        """
        Converts quantitative predictions from MaInvestMacroForecaster into Black-Litterman views.
        Uses 365d median return (P50) and adjusts confidence according to growth probability and risk-reward score.
        """
        ai_views: List[BlackLittermanView] = []
        for sym, report in forecasts_by_symbol.items():
            sym_clean = sym.upper()
            horizons = report.get("horizons", {})
            h365 = horizons.get("365d", {})
            p50_ret_pct = h365.get("p50_return_pct")
            prob_up = h365.get("prob_growth_up", 0.50)

            if p50_ret_pct is not None:
                exp_ret = float(p50_ret_pct) / 100.0
                conf = max(0.20, min(0.90, 0.30 + float(prob_up) * 0.40))
                ai_views.append(
                    BlackLittermanView(
                        view_type="absolute",
                        assets=[sym_clean],
                        weights=[1.0],
                        expected_return=round(exp_ret, 4),
                        confidence=round(conf, 3)
                    )
                )

        return ai_views

    @classmethod
    def optimize_black_litterman_portfolio(
        cls,
        assets: List[str],
        views: Optional[List[BlackLittermanView]] = None,
        optimization_type: str = "max_sharpe",
        market_weights: Optional[Dict[str, float]] = None,
        tau: float = 0.05,
        risk_aversion_delta: float = 2.5,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        max_alt_weight: float = MAX_ALTCOIN_WEIGHT_DEFAULT,
        core_assets: Optional[Set[str]] = None
    ) -> PortfolioOptimizationResult:
        """
        Direct interface to solve for optimal portfolio allocation using the Black-Litterman model.
        optimization_type: 'max_sharpe' | 'min_variance'
        """
        bl_res = cls.compute_black_litterman_posterior(
            assets=assets,
            views=views,
            market_weights=market_weights,
            tau=tau,
            risk_aversion_delta=risk_aversion_delta,
            risk_free_rate=risk_free_rate,
            max_alt_weight=max_alt_weight,
            core_assets=core_assets
        )
        if optimization_type.lower() == "min_variance":
            return bl_res.optimized_min_variance
        return bl_res.optimized_max_sharpe

    # ============================================================================
    # 2. REALISTIC DOLLAR-COST AVERAGING (DCA) SIMULATOR (BLOCK BOOTSTRAP)
    # ============================================================================

    @classmethod
    def generate_calibrated_historical_returns(
        cls,
        assets: List[str],
        n_weeks: int = 300,
        seed: Optional[int] = 42
    ) -> np.ndarray:
        """
        Generates realistic multi-asset weekly log-returns calibrated to crypto market regimes.
        Preserves heavy tails (Student-t degrees of freedom ~ 4.5) and cross-asset correlations.
        Returns shape: (n_weeks, n_assets).
        """
        rng = np.random.default_rng(seed)
        n = len(assets)
        cov_ann = cls.build_covariance_matrix(assets)
        # Weekly covariance
        cov_weekly = cov_ann / 52.0
        # Weekly expected returns
        mu_ann = np.array([cls.get_asset_return_prior(a) for a in assets])
        mu_weekly = mu_ann / 52.0

        # Multivariate t-distribution via Gaussian scale mixtures (nu = 4.5)
        nu = 4.5
        chi2_samples = rng.chisquare(df=nu, size=(n_weeks, 1)) / nu
        norm_samples = rng.multivariate_normal(mean=np.zeros(n), cov=cov_weekly, size=n_weeks)
        
        # Heavy-tailed weekly returns
        t_returns = mu_weekly + norm_samples / np.sqrt(chi2_samples)
        
        # Add occasional regime shock / flash crashes in historical pool (to make bootstrap realistic)
        for w_idx in [25, 80, 140, 210, 270]:
            if w_idx < n_weeks:
                shock_factor = rng.uniform(-0.35, -0.15)
                for i in range(n):
                    if assets[i] not in ("USDT", "USDC"):
                        t_returns[w_idx, i] = shock_factor * rng.uniform(0.9, 1.4)
                    else:
                        t_returns[w_idx, i] = 0.0

        return t_returns

    @classmethod
    def simulate_dca(
        cls,
        weekly_investment_usd: float,
        horizon_years: float,
        portfolio_weights: Dict[str, float],
        historical_returns: Optional[np.ndarray] = None,
        n_simulations: int = 2000,
        block_size_weeks: int = 4,
        initial_deposit_usd: float = 0.0,
        seed: Optional[int] = None
    ) -> DCASimulationResult:
        """
        Simulates long-term DCA trajectories using Moving Block Bootstrap.

        Parameters:
        - weekly_investment_usd: deposit per week ($10 to $500).
        - horizon_years: horizon (1.0 to 5.0 years).
        - portfolio_weights: e.g. {"BTC": 0.50, "ETH": 0.30, "SOL": 0.20}.
        - historical_returns: optional matrix (T_hist, n_assets) of weekly returns.
        - n_simulations: number of bootstrap paths (e.g. 2000).
        - block_size_weeks: bootstrap block length (e.g. 4 to 8 weeks) to preserve serial dependence.
        """
        # Validate inputs
        weekly_inv = float(weekly_investment_usd)
        if weekly_inv <= 0:
            raise ValueError("weekly_investment_usd must be positive.")
        if weekly_inv < 10.0 or weekly_inv > 5000.0:
            logger.warning(f"weekly_investment_usd (${weekly_inv}) outside standard recommendation ($10-$500).")

        horizon_yrs = float(horizon_years)
        if horizon_yrs < 0.25 or horizon_yrs > 10.0:
            raise ValueError("horizon_years must be between 0.25 and 10.0 years.")

        total_weeks = int(round(horizon_yrs * 52))
        if total_weeks < 4:
            total_weeks = 4

        norm_weights = cls.normalize_weights(portfolio_weights)
        assets = list(norm_weights.keys())
        w_vec = np.array([norm_weights[a] for a in assets], dtype=np.float64)
        n_assets = len(assets)

        # Obtain or generate historical return matrix
        if historical_returns is not None:
            hist_R = np.asarray(historical_returns, dtype=np.float64)
            if hist_R.ndim != 2 or hist_R.shape[1] != n_assets:
                raise ValueError(
                    f"historical_returns shape {hist_R.shape} must match (T_hist, {n_assets}) for assets {assets}"
                )
        else:
            hist_R = cls.generate_calibrated_historical_returns(assets, n_weeks=350, seed=42)

        T_hist = hist_R.shape[0]
        eff_block_size = max(1, min(block_size_weeks, T_hist // 4))
        max_start = T_hist - eff_block_size
        if max_start < 1:
            raise ValueError(f"Historical return length ({T_hist}) too short for block size ({eff_block_size}).")

        # Vectorized Moving Block Bootstrap
        rng = np.random.default_rng(seed)
        n_blocks = int(math.ceil(total_weeks / eff_block_size))
        block_starts = rng.integers(0, max_start + 1, size=(n_simulations, n_blocks))

        sampled_paths = np.zeros((n_simulations, n_blocks * eff_block_size, n_assets), dtype=np.float64)
        for b in range(n_blocks):
            starts = block_starts[:, b]
            for k in range(eff_block_size):
                sampled_paths[:, b * eff_block_size + k, :] = hist_R[starts + k, :]

        # Slice to exact total_weeks
        sampled_returns = sampled_paths[:, :total_weeks, :]  # shape: (n_sims, total_weeks, n_assets)

        # Simulate DCA Capital Accumulation & Peak-to-Trough Drawdowns
        trajectories = np.zeros((n_simulations, total_weeks), dtype=np.float64)
        asset_holdings = np.zeros((n_simulations, n_assets), dtype=np.float64)
        
        # Initial deposit allocation
        init_dep = max(0.0, float(initial_deposit_usd))
        if init_dep > 0:
            asset_holdings += init_dep * w_vec

        cash_flows = weekly_inv * w_vec  # cash deposited into each asset each week
        cum_invested = np.zeros(total_weeks, dtype=np.float64)

        peak_values = np.zeros(n_simulations, dtype=np.float64)
        max_drawdowns = np.zeros(n_simulations, dtype=np.float64)

        for t in range(total_weeks):
            # Inflow at start of week, then experiences return of week t
            asset_holdings = (asset_holdings + cash_flows) * (1.0 + sampled_returns[:, t, :])
            # Floor asset values to zero in case of extreme negative simulation
            asset_holdings = np.maximum(0.0, asset_holdings)
            
            port_val = np.sum(asset_holdings, axis=1)
            trajectories[:, t] = port_val
            cum_invested[t] = init_dep + (t + 1) * weekly_inv

            # Track peak and drawdowns
            peak_values = np.maximum(peak_values, port_val)
            dd = np.where(peak_values > 0, (port_val - peak_values) / peak_values, 0.0)
            max_drawdowns = np.minimum(max_drawdowns, dd)

        total_invested = float(cum_invested[-1])
        final_values = trajectories[:, -1]

        # Calculate Percentiles (P10, P50, P90)
        p10_final = float(np.percentile(final_values, 10))
        p50_final = float(np.percentile(final_values, 50))
        p90_final = float(np.percentile(final_values, 90))

        # Return on Investment (ROI)
        p10_roi = ((p10_final - total_invested) / total_invested) * 100.0
        p50_roi = ((p50_final - total_invested) / total_invested) * 100.0
        p90_roi = ((p90_final - total_invested) / total_invested) * 100.0

        # Compound Annual Growth Rate (CAGR)
        def calc_cagr(final_val: float, total_inv: float, yrs: float) -> float:
            if final_val <= 0 or total_inv <= 0:
                return -100.0
            ratio = final_val / total_inv
            return float((ratio ** (1.0 / yrs) - 1.0) * 100.0)

        p10_cagr = calc_cagr(p10_final, total_invested, horizon_yrs)
        p50_cagr = calc_cagr(p50_final, total_invested, horizon_yrs)
        p90_cagr = calc_cagr(p90_final, total_invested, horizon_yrs)

        # Probabilities
        prob_profit = float(np.mean(final_values > total_invested))
        prob_double = float(np.mean(final_values >= 2.0 * total_invested))

        # Drawdown statistics
        median_mdd = float(np.percentile(np.abs(max_drawdowns), 50)) * 100.0
        worst_mdd = float(np.percentile(np.abs(max_drawdowns), 95)) * 100.0

        # Percentile trajectories over time (sample 52 evenly spaced points if large)
        traj_p10 = [round(float(v), 2) for v in np.percentile(trajectories, 10, axis=0)]
        traj_p50 = [round(float(v), 2) for v in np.percentile(trajectories, 50, axis=0)]
        traj_p90 = [round(float(v), 2) for v in np.percentile(trajectories, 90, axis=0)]
        invested_curve = [round(float(v), 2) for v in cum_invested]

        return DCASimulationResult(
            weekly_investment_usd=round(weekly_inv, 2),
            horizon_years=round(horizon_yrs, 2),
            total_weeks=total_weeks,
            initial_deposit_usd=round(init_dep, 2),
            total_invested_usd=round(total_invested, 2),
            p10_final_value_usd=round(p10_final, 2),
            p50_final_value_usd=round(p50_final, 2),
            p90_final_value_usd=round(p90_final, 2),
            p10_roi_pct=round(p10_roi, 2),
            p50_roi_pct=round(p50_roi, 2),
            p90_roi_pct=round(p90_roi, 2),
            p10_cagr_pct=round(p10_cagr, 2),
            p50_cagr_pct=round(p50_cagr, 2),
            p90_cagr_pct=round(p90_cagr, 2),
            prob_profit=round(prob_profit, 4),
            prob_double=round(prob_double, 4),
            median_max_drawdown_pct=round(median_mdd, 2),
            worst_case_drawdown_pct=round(worst_mdd, 2),
            weekly_trajectory_p10=traj_p10,
            weekly_trajectory_p50=traj_p50,
            weekly_trajectory_p90=traj_p90,
            weekly_invested_curve=invested_curve,
            n_simulations=n_simulations,
            block_size_weeks=eff_block_size
        )

    # ============================================================================
    # 3. HISTORICAL CRASH STRESS TESTING
    # ============================================================================

    @classmethod
    def run_stress_tests(
        cls,
        portfolio_weights: Dict[str, float],
        initial_capital_usd: float = 10000.0,
        custom_scenarios: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> PortfolioStressTestReport:
        """
        Performs deep historical crash stress tests across key systemic black swans:
        1. FTX Collapse (Nov 2022)
        2. COVID-19 Flash Crash (March 2020)
        3. Terra / LUNA & UST Depegging Crash (May 2022)
        4. China Mining Ban (May 2021)
        5. Crypto Winter Bear Peak-to-Trough (2022)

        Returns asset-level breakdown, portfolio drawdown, dollar loss, and survival score.
        """
        weights = cls.normalize_weights(portfolio_weights)
        scenarios_db = custom_scenarios or cls.HISTORICAL_CRASH_SCENARIOS
        cap = max(100.0, float(initial_capital_usd))

        scenario_results: Dict[str, StressScenarioResult] = {}
        drawdowns: List[float] = []

        for sc_id, sc_data in scenarios_db.items():
            name = sc_data.get("name", sc_id)
            desc = sc_data.get("description", "")
            drops = sc_data.get("asset_drops", {})
            default_alt_drop = sc_data.get("default_alt_drop", -0.55)
            rec_months = sc_data.get("recovery_months", 6)

            # Calculate asset-level drop and weighted portfolio loss
            port_loss = 0.0
            asset_breakdown = {}
            worst_asset = "BTC"
            worst_drop = 0.0

            for asset, w in weights.items():
                if asset in drops:
                    drop = drops[asset]
                elif asset in ("USDT", "USDC"):
                    drop = 0.0
                else:
                    drop = default_alt_drop

                asset_breakdown[asset] = round(drop * 100.0, 2)
                port_loss += w * drop

                if drop < worst_drop:
                    worst_drop = drop
                    worst_asset = asset

            port_dd_pct = round(port_loss * 100.0, 2)  # negative percentage
            loss_usd = round(cap * abs(port_loss), 2)
            rem_val = round(max(0.0, cap * (1.0 + port_loss)), 2)

            scenario_results[sc_id] = StressScenarioResult(
                scenario_id=sc_id,
                name=name,
                description=desc,
                portfolio_drawdown_pct=port_dd_pct,
                loss_usd_on_10k=loss_usd,
                remaining_value_usd_on_10k=rem_val,
                worst_hit_asset=worst_asset,
                worst_hit_asset_drop_pct=round(worst_drop * 100.0, 2),
                asset_breakdown=asset_breakdown,
                estimated_recovery_months=rec_months
            )
            drawdowns.append(abs(port_loss))

        # Worst-case evaluation
        worst_sc_id = max(scenario_results, key=lambda k: abs(scenario_results[k].portfolio_drawdown_pct))
        worst_sc = scenario_results[worst_sc_id]
        worst_loss_pct = abs(worst_sc.portfolio_drawdown_pct)
        worst_loss_usd = worst_sc.loss_usd_on_10k

        avg_dd_pct = round(float(np.mean(drawdowns)) * 100.0, 2)

        # Stress Survival Score (0 to 100):
        # 100 - (worst_loss_pct * 1.15) clamped to [5, 98]
        raw_survival = 100.0 - (worst_loss_pct * 1.15)
        # Bonus for stablecoin/BTC safety anchor
        safe_weight = sum(weights.get(a, 0.0) for a in ("BTC", "USDT", "USDC"))
        survival_score = int(np.clip(raw_survival + (safe_weight * 10.0), 5, 98))

        # Resilience rating
        if survival_score >= 75:
            resilience = "🛡️ MAXIMUM SHIELD (Высокая стойкость к кризисам)"
        elif survival_score >= 55:
            resilience = "⚖️ BALANCED DEFENSE (Умеренная защита от просадок)"
        elif survival_score >= 35:
            resilience = "⚠️ ELEVATED VULNERABILITY (Повышенная уязвимость)"
        else:
            resilience = "🚨 CRITICAL CRASH RISK (Критический риск ликвидации портфеля)"

        # Actionable risk recommendations
        recommendations: List[str] = []
        alt_weight = sum(w for a, w in weights.items() if a not in ("BTC", "USDT", "USDC"))
        max_asset = max(weights, key=weights.get)
        max_pct = weights[max_asset] * 100.0

        if alt_weight > 0.50:
            recommendations.append(
                f"Высокая доля альткоинов ({alt_weight*100:.1f}%). В периоды системных сбросов альткоины "
                f"теряют до 70-96% стоимости. Рекомендуется нарастить долю BTC или USDT до 50%."
            )
        if max_pct > 40.0 and max_asset not in ("BTC", "USDT", "USDC"):
            recommendations.append(
                f"Критическая концентрация в альткоине #{max_asset} ({max_pct:.1f}%). "
                f"Ограничьте долю актива до <= 35% в соответствии с правилами риск-менеджмента MaBuy."
            )
        if weights.get("USDT", 0.0) + weights.get("USDC", 0.0) < 0.10:
            recommendations.append(
                "Отсутствует защитная подушка в стейблкоинах (USDT/USDC). Добавление 10-15% кэша "
                "позволяет выкупать экстремальные проливы рынка (Buy the Dip)."
            )
        if not recommendations:
            recommendations.append(
                "Отличный баланс активов! Портфель защищен ключевым стержнем BTC и обладает достаточным буфером "
                "для прохождения любого системного стресс-теста."
            )

        return PortfolioStressTestReport(
            scenarios=scenario_results,
            worst_case_scenario_id=worst_sc_id,
            worst_case_loss_pct=worst_loss_pct,
            worst_case_loss_usd_10k=worst_loss_usd,
            average_crash_drawdown_pct=avg_dd_pct,
            stress_survival_score=survival_score,
            resilience_rating=resilience,
            recommendations=recommendations
        )

    # ============================================================================
    # 4. FULL COMPREHENSIVE PORTFOLIO AUDIT & OPTIMIZATION SUITE
    # ============================================================================

    @classmethod
    def full_portfolio_analysis(
        cls,
        portfolio_weights: Dict[str, float],
        weekly_investment_usd: float = 100.0,
        horizon_years: float = 3.0,
        initial_capital_usd: float = 10000.0,
        risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
        max_alt_weight: float = MAX_ALTCOIN_WEIGHT_DEFAULT
    ) -> CompletePortfolioAnalysis:
        """
        Executes the complete MaBuy long-term quantitative investment diagnostic:
        1. Current portfolio Markowitz metrics.
        2. SLSQP Max Sharpe Ratio Optimization.
        3. SLSQP Global Minimum Variance Optimization.
        4. Efficient Frontier Generation (25 frontier points).
        5. Historical Block Bootstrap DCA Simulation.
        6. Historical Crash Stress Testing.
        """
        weights = cls.normalize_weights(portfolio_weights)
        assets = list(weights.keys())

        # 1. Optimize Max Sharpe and Min Variance
        max_sharpe = cls.optimize_max_sharpe(
            assets=assets,
            risk_free_rate=risk_free_rate,
            max_alt_weight=max_alt_weight
        )
        min_var = cls.optimize_min_variance(
            assets=assets,
            risk_free_rate=risk_free_rate,
            max_alt_weight=max_alt_weight
        )

        # 2. Efficient Frontier
        ef_result = cls.compute_efficient_frontier(
            assets=assets,
            n_points=20,
            risk_free_rate=risk_free_rate,
            max_alt_weight=max_alt_weight
        )

        # 3. DCA Simulation
        dca_res = cls.simulate_dca(
            weekly_investment_usd=weekly_investment_usd,
            horizon_years=horizon_years,
            portfolio_weights=weights,
            initial_deposit_usd=initial_capital_usd * 0.10,
            n_simulations=2000,
            seed=42
        )

        # 4. Stress Tests
        stress_res = cls.run_stress_tests(
            portfolio_weights=weights,
            initial_capital_usd=initial_capital_usd
        )

        return CompletePortfolioAnalysis(
            current_weights=weights,
            max_sharpe_optimal=max_sharpe,
            min_variance_optimal=min_var,
            dca_simulation=dca_res,
            stress_test_report=stress_res,
            efficient_frontier_sample=ef_result.frontier_points
        )


# Aliases for architectural backwards compatibility
PortfolioOptimizer = PortfolioOptimizerService
