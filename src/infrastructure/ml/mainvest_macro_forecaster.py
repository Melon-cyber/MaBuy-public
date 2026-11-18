r"""
MaInvest Macro Investment Forecaster Engine.
R&D Quantitative Laboratory — MaBuy / MaInvest Project.

Mathematical Foundations:
1. Merton Jump Diffusion (MJD) SDE with Poisson jump arrivals and normal jump log-size distribution:
   dS_t / S_t = (\mu - \lambda k) dt + \sigma dW_t + (e^Y - 1) dN_t
   where k = E[e^Y - 1] = exp(\mu_J + 0.5 \sigma_J^2) - 1.
2. Realized Volatility Estimators:
   - Garman-Klass OHLC Volatility (intraday high/low and open/close information)
   - Parkinson High/Low Extreme Volatility
   - Multi-window Volatility Cones (14d, 30d, 60d, 90d, 180d, 365d) and percentile ranking
3. Bitcoin Halving Macro Cycle Relative Stage Classifier:
   - 4-year cycle progress quantification
   - Asset-specific beta scaling for top crypto assets (BTC, ETH, SOL, BNB, XRP, ADA, DOGE, AVAX)
4. Multi-Horizon Trajectory Quantification (30, 90, 365 days):
   - Price Corridors: P10 (Bear Case), P50 (Median Target), P90 (Bull Case)
   - Expected CAGR (Compound Annual Growth Rate)
   - Probability of Growth P(UP) = P(S_H > S_0)
   - Value at Risk (VaR 95%) and Expected Shortfall (CVaR 95%)
"""

from __future__ import annotations

import datetime
import logging
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from src.domain.services.symbol_normalizer import SymbolNormalizer

logger = logging.getLogger("MaInvestMacroForecaster")


# =====================================================================
# Constants & Asset Universe Specifications
# =====================================================================

SUPPORTED_TOP_ASSETS: Dict[str, Dict[str, Any]] = {
    # Cryptocurrencies (Continuous 365 days / 24-7)
    "BTC": {
        "name": "Bitcoin",
        "asset_class": "crypto",
        "trading_days_per_year": 365,
        "beta": 1.00,
        "cycle_lag_days": 0,
        "historical_drift_prior": 0.35,
        "jump_sensitivity": 1.00,
    },
    "ETH": {
        "name": "Ethereum",
        "asset_class": "crypto",
        "trading_days_per_year": 365,
        "beta": 1.18,
        "cycle_lag_days": 35,
        "historical_drift_prior": 0.40,
        "jump_sensitivity": 1.15,
    },
    "SOL": {
        "name": "Solana",
        "asset_class": "crypto",
        "trading_days_per_year": 365,
        "beta": 1.45,
        "cycle_lag_days": 45,
        "historical_drift_prior": 0.55,
        "jump_sensitivity": 1.35,
    },
    "BNB": {
        "name": "BNB",
        "asset_class": "crypto",
        "trading_days_per_year": 365,
        "beta": 0.92,
        "cycle_lag_days": 20,
        "historical_drift_prior": 0.30,
        "jump_sensitivity": 0.90,
    },
    "TON": {
        "name": "Toncoin",
        "asset_class": "crypto",
        "trading_days_per_year": 365,
        "beta": 1.35,
        "cycle_lag_days": 40,
        "historical_drift_prior": 0.45,
        "jump_sensitivity": 1.30,
    },
    "XRP": {
        "name": "Ripple (XRP)",
        "asset_class": "crypto",
        "trading_days_per_year": 365,
        "beta": 1.25,
        "cycle_lag_days": 60,
        "historical_drift_prior": 0.35,
        "jump_sensitivity": 1.50,
    },
    "ADA": {
        "name": "Cardano",
        "asset_class": "crypto",
        "trading_days_per_year": 365,
        "beta": 1.30,
        "cycle_lag_days": 50,
        "historical_drift_prior": 0.35,
        "jump_sensitivity": 1.25,
    },
    "DOGE": {
        "name": "Dogecoin",
        "asset_class": "crypto",
        "trading_days_per_year": 365,
        "beta": 1.65,
        "cycle_lag_days": 75,
        "historical_drift_prior": 0.45,
        "jump_sensitivity": 1.80,
    },
    "AVAX": {
        "name": "Avalanche",
        "asset_class": "crypto",
        "trading_days_per_year": 365,
        "beta": 1.40,
        "cycle_lag_days": 45,
        "historical_drift_prior": 0.50,
        "jump_sensitivity": 1.30,
    },
    # US Equities (NYSE / NASDAQ Calendar — 252 trading days)
    "NVDA": {
        "name": "NVIDIA Corporation",
        "asset_class": "equity",
        "trading_days_per_year": 252,
        "beta": 1.75,
        "cycle_lag_days": 0,
        "historical_drift_prior": 0.35,
        "jump_sensitivity": 1.25,
    },
    "AAPL": {
        "name": "Apple Inc.",
        "asset_class": "equity",
        "trading_days_per_year": 252,
        "beta": 0.95,
        "cycle_lag_days": 0,
        "historical_drift_prior": 0.18,
        "jump_sensitivity": 0.85,
    },
    "TSLA": {
        "name": "Tesla Inc.",
        "asset_class": "equity",
        "trading_days_per_year": 252,
        "beta": 1.95,
        "cycle_lag_days": 0,
        "historical_drift_prior": 0.30,
        "jump_sensitivity": 1.50,
    },
    "MSFT": {
        "name": "Microsoft Corporation",
        "asset_class": "equity",
        "trading_days_per_year": 252,
        "beta": 1.05,
        "cycle_lag_days": 0,
        "historical_drift_prior": 0.20,
        "jump_sensitivity": 0.85,
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "asset_class": "equity",
        "trading_days_per_year": 252,
        "beta": 1.10,
        "cycle_lag_days": 0,
        "historical_drift_prior": 0.20,
        "jump_sensitivity": 0.90,
    },
    # Macro ETFs (NYSE / NASDAQ Calendar — 252 trading days)
    "SPY": {
        "name": "SPDR S&P 500 ETF Trust",
        "asset_class": "macro_etf",
        "trading_days_per_year": 252,
        "beta": 1.00,
        "cycle_lag_days": 0,
        "historical_drift_prior": 0.11,
        "jump_sensitivity": 0.60,
    },
    "QQQ": {
        "name": "Invesco QQQ Trust (Nasdaq 100)",
        "asset_class": "macro_etf",
        "trading_days_per_year": 252,
        "beta": 1.20,
        "cycle_lag_days": 0,
        "historical_drift_prior": 0.16,
        "jump_sensitivity": 0.75,
    },
    "GLD": {
        "name": "SPDR Gold Shares ETF",
        "asset_class": "commodity_etf",
        "trading_days_per_year": 252,
        "beta": 0.10,
        "cycle_lag_days": 0,
        "historical_drift_prior": 0.08,
        "jump_sensitivity": 0.40,
    },
}

GENERIC_ASSET_PARAMS: Dict[str, Any] = {
    "name": "Multi-Asset Security",
    "asset_class": "crypto",
    "trading_days_per_year": 365,
    "beta": 1.25,
    "cycle_lag_days": 45,
    "historical_drift_prior": 0.35,
    "jump_sensitivity": 1.20,
}

# Historical and scheduled Bitcoin Halving Epochs (UTC timestamps)
BITCOIN_HALVINGS: List[datetime.date] = [
    datetime.date(2012, 11, 28),  # Halving 1 (Block 210,000)
    datetime.date(2016, 7, 9),    # Halving 2 (Block 420,000)
    datetime.date(2020, 5, 11),   # Halving 3 (Block 630,000)
    datetime.date(2024, 4, 19),   # Halving 4 (Block 840,000)
    datetime.date(2028, 4, 17),   # Halving 5 (Block 1,050,000 - Estimated)
    datetime.date(2032, 4, 15),   # Halving 6 (Block 1,260,000 - Estimated)
]


# =====================================================================
# Data Structures & Result Objects
# =====================================================================

@dataclass(frozen=True)
class VolatilityConePoint:
    """Volatility metrics across a specific rolling window."""
    window_days: int
    min_vol: float
    p25_vol: float
    median_vol: float
    p75_vol: float
    max_vol: float
    current_vol: float
    percentile_rank: float
    regime: str  # "COMPRESSED", "NORMAL", "EXPANDED"


@dataclass(frozen=True)
class VolatilityAnalysisResult:
    """Comprehensive realized volatility analysis."""
    garman_klass_annualized: float
    parkinson_annualized: float
    close_to_close_annualized: float
    composite_annualized: float
    volatility_cones: List[VolatilityConePoint]
    dominant_vol_regime: str


@dataclass(frozen=True)
class MertonJumpParameters:
    """Calibrated Merton Jump Diffusion (MJD) model parameters."""
    drift_annualized: float
    continuous_vol_annualized: float
    jump_intensity_lambda: float
    jump_mean_mu: float
    jump_std_sigma: float
    jump_compensator_k: float
    jump_count_detected: int
    sample_days: int
    trading_days_per_year: int = 365


@dataclass(frozen=True)
class HorizonForecast:
    """Quantitative forecast metrics for a specific investment horizon."""
    horizon_days: int
    horizon_code: str  # e.g., "30d", "90d", "365d"
    current_price: float
    target_p10: float  # Bear Case
    target_p25: float
    target_p50: float  # Median Target
    target_p75: float
    target_p90: float  # Bull Case
    p10_return_pct: float
    p50_return_pct: float
    p90_return_pct: float
    expected_cagr: float
    prob_growth_up: float  # P(S_H > S_0) in [0.0, 1.0]
    prob_gain_25pct: float
    prob_gain_100pct: float
    prob_drawdown_30pct: float
    var_95_pct: float  # Value at Risk 95%
    cvar_95_pct: float  # Expected Shortfall / CVaR 95%


@dataclass(frozen=True)
class HalvingMacroCycleState:
    """Bitcoin Halving Macro-Cycle relative stage metrics."""
    reference_date: str
    cycle_index: int
    prev_halving_date: str
    next_halving_date: str
    days_since_halving: int
    days_to_next_halving: int
    cycle_progress_pct: float
    stage_code: str
    stage_name_en: str
    stage_name_ru: str
    regime_bias: str  # "STRONG_BULL", "BULL_EXPANSION", "DISTRIBUTION", "BEAR", "ACCUMULATION"
    cycle_momentum_multiplier: float


@dataclass
class MacroInvestmentForecastReport:
    """Complete quantitative investment forecast package."""
    symbol: str
    normalized_symbol: str
    asset_name: str
    current_price: float
    macro_cycle: HalvingMacroCycleState
    volatility: VolatilityAnalysisResult
    merton_params: MertonJumpParameters
    horizons: Dict[str, HorizonForecast]
    investment_recommendation: Dict[str, Any]
    simulated_trajectories_sample: Optional[List[List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts complete report to structured dictionary format."""
        return asdict(self)


# =====================================================================
# Realized Volatility Estimator Engine
# =====================================================================

class RealizedVolatilityEngine:
    """
    Computes High/Low and Open/Close realized volatility estimators
    and generates multi-horizon Volatility Cones.
    """

    @staticmethod
    def parkinson_volatility(highs: np.ndarray, lows: np.ndarray, annualized_factor: float = 365.0) -> float:
        """
        Calculates Parkinson (1980) High/Low Extreme Volatility:
        sigma_P = sqrt( (1 / (4 * ln(2) * N)) * sum( ln(H_i / L_i)^2 ) ) * sqrt(365)
        """
        if len(highs) == 0 or len(lows) == 0:
            return 0.0
        
        # Guard against zero or negative prices
        valid_mask = (highs > 0) & (lows > 0) & (highs >= lows)
        if not np.any(valid_mask):
            return 0.0
        
        h = highs[valid_mask]
        l = lows[valid_mask]
        
        log_hl = np.log(h / l)
        variance = np.sum(log_hl ** 2) / (4.0 * np.log(2.0) * len(log_hl))
        return float(np.sqrt(max(0.0, variance * annualized_factor)))

    @staticmethod
    def garman_klass_volatility(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        annualized_factor: float = 365.0
    ) -> float:
        """
        Calculates Garman-Klass (1980) OHLC Volatility:
        sigma_GK = sqrt( (1 / N) * sum( 0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2 ) ) * sqrt(365)
        """
        if len(opens) == 0 or len(highs) == 0 or len(lows) == 0 or len(closes) == 0:
            return 0.0
        
        valid_mask = (opens > 0) & (highs > 0) & (lows > 0) & (closes > 0) & (highs >= lows)
        if not np.any(valid_mask):
            return 0.0

        o = opens[valid_mask]
        h = highs[valid_mask]
        l = lows[valid_mask]
        c = closes[valid_mask]

        log_hl = np.log(h / l)
        log_co = np.log(c / o)
        
        c1 = 0.5
        c2 = 2.0 * np.log(2.0) - 1.0  # ~0.386294
        
        var_terms = c1 * (log_hl ** 2) - c2 * (log_co ** 2)
        variance = np.mean(var_terms)
        return float(np.sqrt(max(0.0, variance * annualized_factor)))

    @staticmethod
    def close_to_close_volatility(closes: np.ndarray, annualized_factor: float = 365.0) -> float:
        """Calculates standard Close-to-Close annualized log return volatility."""
        if len(closes) < 2:
            return 0.0
        
        valid_closes = closes[closes > 0]
        if len(valid_closes) < 2:
            return 0.0

        log_returns = np.diff(np.log(valid_closes))
        if len(log_returns) < 2:
            return 0.0
        
        std = np.std(log_returns, ddof=1)
        return float(std * np.sqrt(annualized_factor))

    @classmethod
    def compute_volatility_cones(
        cls,
        df: pd.DataFrame,
        windows: Optional[List[int]] = None,
        annualized_factor: float = 365.0
    ) -> List[VolatilityConePoint]:
        """
        Constructs multi-window Volatility Cones from historical OHLC data.
        Evaluates percentile rank of current realized volatility within historical distribution.
        """
        if windows is None:
            windows = [14, 30, 60, 90, 180, 365]

        n_rows = len(df)
        if n_rows < 15:
            # Insufficient data fallback
            return []

        opens = df["open"].to_numpy(dtype=np.float64)
        highs = df["high"].to_numpy(dtype=np.float64)
        lows = df["low"].to_numpy(dtype=np.float64)
        closes = df["close"].to_numpy(dtype=np.float64)

        cones: List[VolatilityConePoint] = []

        for window in windows:
            if n_rows < window:
                continue

            rolling_vols: List[float] = []
            # Calculate rolling Garman-Klass volatility over history
            for i in range(window, n_rows + 1):
                w_open = opens[i - window : i]
                w_high = highs[i - window : i]
                w_low = lows[i - window : i]
                w_close = closes[i - window : i]
                
                vol = cls.garman_klass_volatility(w_open, w_high, w_low, w_close, annualized_factor=annualized_factor)
                if vol > 0.0 and not np.isnan(vol):
                    rolling_vols.append(vol)

            if len(rolling_vols) == 0:
                continue

            arr_vols = np.array(rolling_vols, dtype=np.float64)
            current_vol = float(arr_vols[-1])
            min_v = float(np.min(arr_vols))
            p25_v = float(np.percentile(arr_vols, 25))
            median_v = float(np.median(arr_vols))
            p75_v = float(np.percentile(arr_vols, 75))
            max_v = float(np.max(arr_vols))

            # Percentile rank of current volatility in historical distribution
            pct_rank = float(stats.percentileofscore(arr_vols, current_vol, kind="weak"))

            if pct_rank < 25.0:
                regime = "COMPRESSED"
            elif pct_rank > 75.0:
                regime = "EXPANDED"
            else:
                regime = "NORMAL"

            cones.append(
                VolatilityConePoint(
                    window_days=window,
                    min_vol=round(min_v, 4),
                    p25_vol=round(p25_v, 4),
                    median_vol=round(median_v, 4),
                    p75_vol=round(p75_v, 4),
                    max_vol=round(max_v, 4),
                    current_vol=round(current_vol, 4),
                    percentile_rank=round(pct_rank, 2),
                    regime=regime,
                )
            )

        return cones

    @classmethod
    def analyze_realized_volatility(
        cls,
        df: pd.DataFrame,
        annualized_factor: float = 365.0
    ) -> VolatilityAnalysisResult:
        """Performs full spectrum realized volatility analysis."""
        opens = df["open"].to_numpy(dtype=np.float64)
        highs = df["high"].to_numpy(dtype=np.float64)
        lows = df["low"].to_numpy(dtype=np.float64)
        closes = df["close"].to_numpy(dtype=np.float64)

        # Lookback up to 90 days for current point estimates (or full available df)
        lookback = min(len(df), 90)
        sub_o = opens[-lookback:]
        sub_h = highs[-lookback:]
        sub_l = lows[-lookback:]
        sub_c = closes[-lookback:]

        gk_vol = cls.garman_klass_volatility(sub_o, sub_h, sub_l, sub_c, annualized_factor=annualized_factor)
        pk_vol = cls.parkinson_volatility(sub_h, sub_l, annualized_factor=annualized_factor)
        c2c_vol = cls.close_to_close_volatility(sub_c, annualized_factor=annualized_factor)

        # Composite blended estimator (40% GK, 35% PK, 25% C2C)
        valid_vols = [v for v in [gk_vol, pk_vol, c2c_vol] if v > 0.001]
        if valid_vols:
            composite = 0.40 * gk_vol + 0.35 * pk_vol + 0.25 * c2c_vol
        else:
            composite = 0.25 if annualized_factor == 252.0 else 0.60  # Default baseline volatility

        cones = cls.compute_volatility_cones(df, annualized_factor=annualized_factor)
        
        # Determine dominant regime across shortest and medium windows
        if cones:
            recent_cone = cones[0]
            dominant_regime = recent_cone.regime
        else:
            dominant_regime = "NORMAL"

        return VolatilityAnalysisResult(
            garman_klass_annualized=round(gk_vol, 4),
            parkinson_annualized=round(pk_vol, 4),
            close_to_close_annualized=round(c2c_vol, 4),
            composite_annualized=round(composite, 4),
            volatility_cones=cones,
            dominant_vol_regime=dominant_regime,
        )


# =====================================================================
# Bitcoin Halving Macro Cycle Classifier
# =====================================================================

class HalvingMacroCycleClassifier:
    """
    Quantifies the relative stage of the 4-year Bitcoin Halving cycle for crypto,
    and applies secular macro regime dynamics for US Equities and Macro ETFs.
    """

    @staticmethod
    def get_macro_cycle_state(
        current_date: Optional[datetime.date] = None,
        asset_class: str = "crypto"
    ) -> HalvingMacroCycleState:
        """
        Determines current position in the Bitcoin 4-year Halving cycle or Equity Macro Regime.
        """
        if current_date is None:
            current_date = datetime.date.today()

        # Non-crypto securities (US Equities, ETFs) operate under continuous macro secular trend
        if asset_class != "crypto":
            return HalvingMacroCycleState(
                reference_date=current_date.isoformat(),
                cycle_index=0,
                prev_halving_date="N/A",
                next_halving_date="N/A",
                days_since_halving=0,
                days_to_next_halving=0,
                cycle_progress_pct=50.0,
                stage_code="EQUITY_SECULAR_REGIME",
                stage_name_en="US Equity Secular Regime (NYSE/NASDAQ Calendar)",
                stage_name_ru="Режим фондового рынка США (Торговый календарь NYSE/NASDAQ)",
                regime_bias="MACRO_EQUITY",
                cycle_momentum_multiplier=1.00,
            )

        # Find surrounding halving dates
        prev_halving = BITCOIN_HALVINGS[0]
        next_halving = BITCOIN_HALVINGS[-1]
        cycle_idx = 1

        for i in range(len(BITCOIN_HALVINGS) - 1):
            if BITCOIN_HALVINGS[i] <= current_date < BITCOIN_HALVINGS[i + 1]:
                prev_halving = BITCOIN_HALVINGS[i]
                next_halving = BITCOIN_HALVINGS[i + 1]
                cycle_idx = i + 1
                break
        else:
            if current_date >= BITCOIN_HALVINGS[-1]:
                prev_halving = BITCOIN_HALVINGS[-1]
                # Extrapolate 1460 days (~4 years)
                next_halving = prev_halving + datetime.timedelta(days=1460)
                cycle_idx = len(BITCOIN_HALVINGS)

        days_since = (current_date - prev_halving).days
        total_cycle_days = max(1, (next_halving - prev_halving).days)
        days_to_next = max(0, (next_halving - current_date).days)
        progress_ratio = max(0.0, min(1.0, days_since / total_cycle_days))
        progress_pct = progress_ratio * 100.0

        # Cycle Stage Categorization:
        # Phase 1: 0% - 15% (0 ~ 220d post halving) -> Early Accumulation / Post-Halving Squeeze
        # Phase 2: 15% - 42% (220 ~ 610d post halving) -> Parabolic Bull Expansion
        # Phase 3: 42% - 60% (610 ~ 875d post halving) -> Distribution / Cycle Top Peak
        # Phase 4: 60% - 82% (875 ~ 1200d post halving) -> Bear Market Capitulation
        # Phase 5: 82% - 100% (1200 ~ 1460d post halving) -> Pre-Halving Rebound / Smart Money Accumulation

        if progress_ratio < 0.15:
            stage_code = "ACCUMULATION_EARLY"
            stage_en = "Early Post-Halving Accumulation"
            stage_ru = "Раннее накопление (Пост-халвинг консолидация)"
            regime_bias = "ACCUMULATION"
            momentum_mult = 1.15
        elif progress_ratio < 0.42:
            stage_code = "BULL_PARABOLIC_EXPANSION"
            stage_en = "Bull Market Parabolic Expansion"
            stage_ru = "Бычья параболическая экспансия (Главная фаза роста)"
            regime_bias = "STRONG_BULL"
            momentum_mult = 1.45
        elif progress_ratio < 0.60:
            stage_code = "DISTRIBUTION_CYCLE_TOP"
            stage_en = "Cycle Top Distribution & Peak Volatility"
            stage_ru = "Дистрибуция / Формирование вершины макро-цикла"
            regime_bias = "DISTRIBUTION"
            momentum_mult = 0.90
        elif progress_ratio < 0.82:
            stage_code = "BEAR_CAPITULATION"
            stage_en = "Bear Market Correction & Capitulation"
            stage_ru = "Медвежья капитуляция / Глубокая переоценка"
            regime_bias = "BEAR"
            momentum_mult = 0.65
        else:
            stage_code = "PRE_HALVING_RECOVERY"
            stage_en = "Pre-Halving Recovery & Bottom Building"
            stage_ru = "Предхалвинговое восстановление (Накопление дна)"
            regime_bias = "BULL_EXPANSION"
            momentum_mult = 1.25

        return HalvingMacroCycleState(
            reference_date=current_date.isoformat(),
            cycle_index=cycle_idx,
            prev_halving_date=prev_halving.isoformat(),
            next_halving_date=next_halving.isoformat(),
            days_since_halving=days_since,
            days_to_next_halving=days_to_next,
            cycle_progress_pct=round(progress_pct, 2),
            stage_code=stage_code,
            stage_name_en=stage_en,
            stage_name_ru=stage_ru,
            regime_bias=regime_bias,
            cycle_momentum_multiplier=round(momentum_mult, 3),
        )


# =====================================================================
# Merton Jump Diffusion (MJD) Calibration & Monte Carlo Engine
# =====================================================================

class MertonJumpDiffusionEngine:
    """
    Calibrates Merton (1976) Jump Diffusion parameters from historical time series
    and performs high-performance Monte Carlo path simulations.
    """

    @classmethod
    def calibrate_mjd_parameters(
        cls,
        df: pd.DataFrame,
        asset_info: Dict[str, Any],
        macro_cycle: HalvingMacroCycleState,
        vol_analysis: VolatilityAnalysisResult,
    ) -> MertonJumpParameters:
        """
        Calibrates continuous volatility, jump intensity (lambda), jump mean (mu_J),
        jump variance (sigma_J), and long-term compensated drift.
        Uses Median Absolute Deviation (MAD) robust jump filtering.
        """
        trading_days = float(asset_info.get("trading_days_per_year", 365.0))
        asset_class = asset_info.get("asset_class", "crypto")

        closes = df["close"].to_numpy(dtype=np.float64)
        if len(closes) < 10:
            # Fallback for ultra-short datasets
            fallback_drift = 0.12 if asset_class != "crypto" else 0.25
            fallback_vol = 0.25 if asset_class != "crypto" else 0.60
            return MertonJumpParameters(
                drift_annualized=fallback_drift,
                continuous_vol_annualized=fallback_vol,
                jump_intensity_lambda=1.5,
                jump_mean_mu=-0.01,
                jump_std_sigma=0.05,
                jump_compensator_k=-0.0087,
                jump_count_detected=0,
                sample_days=len(closes),
                trading_days_per_year=int(trading_days),
            )

        log_returns = np.diff(np.log(closes))
        n_days = len(log_returns)

        # 1. Robust jump detection using Median Absolute Deviation (MAD)
        med_ret = np.median(log_returns)
        mad = np.median(np.abs(log_returns - med_ret))
        sigma_mad = 1.4826 * mad if mad > 1e-8 else np.std(log_returns)
        sigma_mad = max(sigma_mad, 0.003)

        # Jump threshold at 2.58 standard deviations (~99% normal coverage)
        jump_threshold = 2.58 * sigma_mad
        jump_mask = np.abs(log_returns - med_ret) > jump_threshold
        jump_count = int(np.sum(jump_mask))
        diff_returns = log_returns[~jump_mask]

        # 2. Jump intensity: lambda = N_jumps / T_years
        time_span_years = max(n_days / trading_days, 0.1)
        jump_intensity = jump_count / time_span_years
        
        # Scale jump intensity by asset-specific sensitivity
        jump_sensitivity = asset_info.get("jump_sensitivity", 1.0)
        jump_intensity = max(0.2, min(24.0, jump_intensity * jump_sensitivity))

        # 3. Jump size distribution parameters Y ~ N(mu_J, sigma_J^2)
        if jump_count >= 2:
            jump_returns = log_returns[jump_mask]
            jump_mu = float(np.mean(jump_returns))
            jump_sigma = float(np.std(jump_returns, ddof=1))
            jump_sigma = max(jump_sigma, 0.015)
        else:
            jump_mu = -0.010 if asset_class != "crypto" else -0.015
            jump_sigma = 0.040 if asset_class != "crypto" else 0.075

        # Merton Jump Compensator: k = E[e^Y - 1] = exp(mu_J + 0.5 * sigma_J^2) - 1
        k_compensator = float(np.exp(jump_mu + 0.5 * (jump_sigma ** 2)) - 1.0)

        # 4. Continuous volatility estimation
        if len(diff_returns) >= 5:
            diff_vol = float(np.std(diff_returns, ddof=1) * np.sqrt(trading_days))
        else:
            diff_vol = vol_analysis.composite_annualized

        # Blend continuous volatility with composite realized volatility
        cont_vol = 0.50 * diff_vol + 0.50 * vol_analysis.composite_annualized
        min_vol = 0.08 if asset_class != "crypto" else 0.20
        max_vol = 1.20 if asset_class != "crypto" else 2.50
        cont_vol = max(min_vol, min(max_vol, cont_vol))

        # 5. Annualized Drift with Bayesian Shrinkage & Macro Regime Modulation
        # Raw historical mean log return annualized
        raw_historical_drift = float(np.mean(log_returns) * trading_days + 0.5 * (cont_vol ** 2))
        
        # Asset class prior and macro multiplier
        prior_drift = asset_info.get("historical_drift_prior", 0.15 if asset_class != "crypto" else 0.35)
        beta = asset_info.get("beta", 1.0)
        macro_mult = macro_cycle.cycle_momentum_multiplier

        # Modulate prior by macro regime
        regime_adjusted_prior = prior_drift * (1.0 + (macro_mult - 1.0) * beta)

        # Bayesian James-Stein shrinkage weight towards regime-adjusted prior
        shrinkage_weight = 60.0 / (n_days + 60.0)
        calibrated_drift = (1.0 - shrinkage_weight) * raw_historical_drift + shrinkage_weight * regime_adjusted_prior

        # Bounding drift to institutional macro bounds
        min_bound = -0.30 if asset_class != "crypto" else -0.40
        max_bound = 0.80 if asset_class != "crypto" else 1.50
        calibrated_drift = max(min_bound, min(max_bound, calibrated_drift))

        return MertonJumpParameters(
            drift_annualized=round(calibrated_drift, 4),
            continuous_vol_annualized=round(cont_vol, 4),
            jump_intensity_lambda=round(jump_intensity, 4),
            jump_mean_mu=round(jump_mu, 4),
            jump_std_sigma=round(jump_sigma, 4),
            jump_compensator_k=round(k_compensator, 4),
            jump_count_detected=jump_count,
            sample_days=n_days,
            trading_days_per_year=int(trading_days),
        )

    @classmethod
    def simulate_trajectories(
        cls,
        s0: float,
        params: MertonJumpParameters,
        horizons: List[int] = [30, 90, 365],
        num_simulations: int = 1000,
        random_seed: Optional[int] = None,
        trading_days_per_year: float = 365.0,
    ) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
        """
        Runs high-performance vectorized Monte Carlo simulation under Merton Jump Diffusion.
        Uses antithetic variates for variance reduction.

        Returns:
            - Dict mapping horizon_days -> array of simulated terminal prices (shape: [num_simulations])
            - Full price path matrix for a subset of trajectories (shape: [sample_paths, max_horizon + 1])
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        max_horizon = max(horizons)
        dt = 1.0 / trading_days_per_year
        sqrt_dt = np.sqrt(dt)

        mu = params.drift_annualized
        sigma = params.continuous_vol_annualized
        lam = params.jump_intensity_lambda
        mu_j = params.jump_mean_mu
        sigma_j = params.jump_std_sigma
        k = params.jump_compensator_k

        # Continuous drift per step with martingale jump compensation
        drift_dt = (mu - 0.5 * (sigma ** 2) - lam * k) * dt
        diff_vol_dt = sigma * sqrt_dt

        # Antithetic variates setup
        n_half = (num_simulations + 1) // 2
        total_sims = n_half * 2

        # 1. Brownian motion increments
        z_half = np.random.standard_normal((n_half, max_horizon))
        z = np.vstack([z_half, -z_half])[:num_simulations]

        # 2. Poisson jump counts per step
        # P(N = 0) = exp(-lambda * dt), P(N = 1) = lambda * dt * exp(-lambda * dt)
        poisson_mean = lam * dt
        jump_counts = np.random.poisson(poisson_mean, size=(num_simulations, max_horizon))

        # 3. Aggregate jump sizes
        total_jumps = int(np.sum(jump_counts))
        if total_jumps > 0:
            random_jump_draws = np.random.normal(mu_j, sigma_j, size=total_jumps)
            jump_matrix = np.zeros((num_simulations, max_horizon), dtype=np.float64)
            
            # Place jumps accurately into simulation grid
            sim_indices, time_indices = np.nonzero(jump_counts)
            jump_idx = 0
            for s_idx, t_idx in zip(sim_indices, time_indices):
                count = jump_counts[s_idx, t_idx]
                jump_matrix[s_idx, t_idx] = np.sum(random_jump_draws[jump_idx : jump_idx + count])
                jump_idx += count
        else:
            jump_matrix = np.zeros((num_simulations, max_horizon), dtype=np.float64)

        # 4. Total log-return increments per step
        log_increments = drift_dt + diff_vol_dt * z + jump_matrix

        # 5. Cumulative log-prices and price paths
        cum_log_paths = np.cumsum(log_increments, axis=1)
        full_price_paths = s0 * np.exp(cum_log_paths)  # shape: [num_simulations, max_horizon]

        # Extract terminal prices at designated horizons
        terminal_prices_by_horizon: Dict[int, np.ndarray] = {}
        for h in horizons:
            h_idx = min(h - 1, max_horizon - 1)
            terminal_prices_by_horizon[h] = full_price_paths[:, h_idx]

        # Sample trajectories for UI / visualization (prepend starting price S0)
        n_sample_trajectories = min(15, num_simulations)
        sample_paths = np.hstack([
            np.full((n_sample_trajectories, 1), s0),
            full_price_paths[:n_sample_trajectories, :]
        ])

        return terminal_prices_by_horizon, sample_paths


# =====================================================================
# Flagship MaInvest Macro Forecaster Engine
# =====================================================================

class MaInvestMacroForecaster:
    """
    Flagship Quantitative Investment Forecasting Engine for MaBuy / MaInvest.
    
    Provides long-term investment horizon price corridors (30d, 90d, 365d),
    expected CAGR, P(UP) probability of capital appreciation, and Halving cycle stages.
    """

    DEFAULT_HORIZONS: List[int] = [30, 90, 365]

    def __init__(self, default_simulations: int = 1000):
        self.default_simulations = max(500, default_simulations)
        self.vol_engine = RealizedVolatilityEngine()
        self.cycle_classifier = HalvingMacroCycleClassifier()
        self.mjd_engine = MertonJumpDiffusionEngine()

    @staticmethod
    def _validate_and_prepare_df(df: pd.DataFrame) -> pd.DataFrame:
        """Validates daily candle dataframe and standardizes column names."""
        if df is None or len(df) == 0:
            raise ValueError("Input daily_candles_df cannot be empty.")

        clean_df = df.copy()
        clean_df.columns = [str(col).strip().lower() for col in clean_df.columns]

        required_cols = {"open", "high", "low", "close"}
        missing = required_cols - set(clean_df.columns)
        if missing:
            raise ValueError(f"daily_candles_df missing required columns: {missing}")

        for col in ["open", "high", "low", "close"]:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        clean_df = clean_df.dropna(subset=["open", "high", "low", "close"])
        clean_df = clean_df[(clean_df["open"] > 0) & (clean_df["high"] > 0) & (clean_df["low"] > 0) & (clean_df["close"] > 0)]

        if len(clean_df) < 5:
            raise ValueError(f"Insufficient valid daily candles: found {len(clean_df)}, minimum 5 required.")

        return clean_df.sort_index()

    @classmethod
    def _get_asset_spec(cls, symbol: str) -> Tuple[str, Dict[str, Any]]:
        """Retrieves asset quantitative metadata and beta specifications."""
        clean_sym = symbol.upper().replace("USDT", "").replace("USD", "").replace("BUSD", "").replace("/", "")
        normalized = SymbolNormalizer.normalize(symbol)
        spec = SUPPORTED_TOP_ASSETS.get(normalized, SUPPORTED_TOP_ASSETS.get(clean_sym, GENERIC_ASSET_PARAMS.copy()))
        return normalized, spec

    def predict_investment_horizon(
        self,
        symbol: str,
        daily_candles_df: pd.DataFrame,
        num_simulations: Optional[int] = None,
        random_seed: Optional[int] = None,
        include_trajectory_samples: bool = True,
    ) -> Dict[str, Any]:
        """
        Main public interface for quantitative macro investment forecasting.
        
        Args:
            symbol: Ticker symbol (e.g. 'BTC', 'ETH', 'SOL', 'NVDA', 'SPY').
            daily_candles_df: Pandas DataFrame containing daily OHLCV candles.
            num_simulations: Number of Monte Carlo trajectories (default 1000, >= 500).
            random_seed: Optional random seed for reproducible quant backtesting.
            include_trajectory_samples: Whether to include sample paths for UI charts.

        Returns:
            Dict containing 30d, 90d, 365d price corridors (P10, P50, P90),
            expected CAGR, probability of growth P(UP), volatility cone metrics,
            and relative Halving macro-cycle stage.
        """
        clean_df = self._validate_and_prepare_df(daily_candles_df)
        sim_count = num_simulations if num_simulations is not None else self.default_simulations
        sim_count = max(500, sim_count)

        normalized_symbol, asset_info = self._get_asset_spec(symbol)
        current_price = float(clean_df["close"].iloc[-1])
        trading_days = float(asset_info.get("trading_days_per_year", 365.0))
        asset_class = asset_info.get("asset_class", "crypto")

        # 1. Realized Volatility & Cones
        vol_analysis = self.vol_engine.analyze_realized_volatility(clean_df, annualized_factor=trading_days)

        # 2. Macro Halving Cycle Stage / Equity Secular Regime
        ref_date = None
        for date_col in ["date", "timestamp", "time", "datetime"]:
            if date_col in clean_df.columns:
                try:
                    last_val = clean_df[date_col].iloc[-1]
                    if isinstance(last_val, (pd.Timestamp, datetime.datetime)):
                        ref_date = last_val.date()
                    elif isinstance(last_val, str):
                        ref_date = pd.to_datetime(last_val).date()
                except Exception:
                    pass
                break

        macro_cycle = self.cycle_classifier.get_macro_cycle_state(current_date=ref_date, asset_class=asset_class)

        # 3. Merton Jump Diffusion Calibration
        mjd_params = self.mjd_engine.calibrate_mjd_parameters(
            df=clean_df,
            asset_info=asset_info,
            macro_cycle=macro_cycle,
            vol_analysis=vol_analysis,
        )

        # 4. Monte Carlo Trajectory Simulation
        terminal_prices_by_horizon, sample_paths = self.mjd_engine.simulate_trajectories(
            s0=current_price,
            params=mjd_params,
            horizons=self.DEFAULT_HORIZONS,
            num_simulations=sim_count,
            random_seed=random_seed,
            trading_days_per_year=trading_days,
        )

        # 5. Multi-Horizon Percentiles & Quantitative Metrics
        horizons_report: Dict[str, HorizonForecast] = {}

        for h in self.DEFAULT_HORIZONS:
            term_prices = terminal_prices_by_horizon[h]
            h_code = f"{h}d"

            p10 = float(np.percentile(term_prices, 10))
            p25 = float(np.percentile(term_prices, 25))
            p50 = float(np.percentile(term_prices, 50))
            p75 = float(np.percentile(term_prices, 75))
            p90 = float(np.percentile(term_prices, 90))

            p10_ret = float(((p10 - current_price) / current_price) * 100.0)
            p50_ret = float(((p50 - current_price) / current_price) * 100.0)
            p90_ret = float(((p90 - current_price) / current_price) * 100.0)

            # Annualized Compound Annual Growth Rate (CAGR) based on median forecast
            ratio = max(0.001, p50 / current_price)
            cagr = float((ratio ** (trading_days / h) - 1.0) * 100.0)

            # Probabilities of various investment outcomes
            prob_up = float(np.mean(term_prices > current_price))
            prob_25 = float(np.mean(term_prices >= current_price * 1.25))
            prob_100 = float(np.mean(term_prices >= current_price * 2.00))
            prob_dd30 = float(np.mean(term_prices <= current_price * 0.70))

            # Value at Risk & Expected Shortfall (CVaR) on horizon returns
            horizon_returns = (term_prices - current_price) / current_price
            var_5th = float(np.percentile(horizon_returns, 5))
            var_95_pct = float(max(0.0, -var_5th * 100.0))
            
            tail_losses = horizon_returns[horizon_returns <= var_5th]
            if len(tail_losses) > 0:
                cvar_95_pct = float(max(0.0, -np.mean(tail_losses) * 100.0))
            else:
                cvar_95_pct = var_95_pct

            horizons_report[h_code] = HorizonForecast(
                horizon_days=h,
                horizon_code=h_code,
                current_price=round(current_price, 4),
                target_p10=round(p10, 4),
                target_p25=round(p25, 4),
                target_p50=round(p50, 4),
                target_p75=round(p75, 4),
                target_p90=round(p90, 4),
                p10_return_pct=round(p10_ret, 2),
                p50_return_pct=round(p50_ret, 2),
                p90_return_pct=round(p90_ret, 2),
                expected_cagr=round(cagr, 2),
                prob_growth_up=round(prob_up, 4),
                prob_gain_25pct=round(prob_25, 4),
                prob_gain_100pct=round(prob_100, 4),
                prob_drawdown_30pct=round(prob_dd30, 4),
                var_95_pct=round(var_95_pct, 2),
                cvar_95_pct=round(cvar_95_pct, 2),
            )

        # 6. Investment Recommendation & DCA Suitability Scoring
        p50_365 = horizons_report["365d"].p50_return_pct
        prob_up_365 = horizons_report["365d"].prob_growth_up
        cvar_365 = horizons_report["365d"].cvar_95_pct

        risk_reward_score = round(max(0.0, (p50_365 / max(cvar_365, 10.0)) * prob_up_365), 2)
        
        if prob_up_365 >= 0.65 and risk_reward_score >= 1.20:
            dca_suitability = "HIGH"
            rec_en = "High DCA Accumulation suitability. Favorable risk-to-tail-loss asymmetry."
            rec_ru = "Высокая целесообразность DCA-накопления. Благоприятная асимметрия доходности к хвостовому риску."
        elif prob_up_365 >= 0.50:
            dca_suitability = "MODERATE"
            rec_en = "Moderate suitability. Consider staggered accumulation with strict risk limits."
            rec_ru = "Умеренная целесообразность. Рекомендуются ступенчатые покупки с контролем размера позиции."
        else:
            dca_suitability = "LOW"
            rec_en = "Elevated macro drawdown risk. Caution advised for long-term spot entries."
            rec_ru = "Повышенный риск макро-просадки. Рекомендуется осторожность при долгосрочных входах."

        recommendation_dict = {
            "risk_reward_score": risk_reward_score,
            "dca_suitability": dca_suitability,
            "recommendation_en": rec_en,
            "recommendation_ru": rec_ru,
            "optimal_horizon": "365d" if p50_365 > 30.0 else "90d",
        }

        # 7. Sample trajectories for UI rendering
        sim_samples = None
        if include_trajectory_samples and sample_paths is not None:
            # Downsample steps to ~25 points along 365 days for compact payload
            sample_step = max(1, sample_paths.shape[1] // 25)
            downsampled = sample_paths[:, ::sample_step]
            sim_samples = np.round(downsampled, 2).tolist()

        full_report = MacroInvestmentForecastReport(
            symbol=symbol.upper(),
            normalized_symbol=normalized_symbol,
            asset_name=asset_info.get("name", symbol),
            current_price=round(current_price, 4),
            macro_cycle=macro_cycle,
            volatility=vol_analysis,
            merton_params=mjd_params,
            horizons=horizons_report,
            investment_recommendation=recommendation_dict,
            simulated_trajectories_sample=sim_samples,
        )

        return full_report.to_dict()


# Aliases for architectural backwards compatibility
MainvestMacroForecaster = MaInvestMacroForecaster

