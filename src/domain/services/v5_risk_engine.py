"""
MaBuy V5 — Risk & Compliance Department
V5DynamicRiskEngine: Strict Dynamic Risk Management & Adaptive Sizing System.

Core Pillars:
1. Dynamic Leverage Matrix (1x - 5x) based on Confidence (P_win), Garman-Klass Volatility, and Asset Tier (BTC/ETH vs SOL/ALTS).
2. Adaptive Fractional Kelly Sizing (0.15*K in Flat to 0.40*K in Clean Trend).
3. Portfolio Margin Cap (Total Margin <= 20% - 25% of account balance) & Micro-Deposit ($15-$100) capital defense.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class AssetTier(str, Enum):
    TIER_1_CORE = "TIER_1_CORE"      # BTC, ETH (High liquidity, tight spreads)
    TIER_2_MAJOR = "TIER_2_MAJOR"    # SOL, BNB (High momentum, moderate liquidity)
    TIER_3_ALTS = "TIER_3_ALTS"      # XRP, ADA, AVAX, DOGE, NEAR, TON (High beta, fat tails)


class MarketRegimeType(str, Enum):
    FLAT_RANGE = "FLAT_RANGE"        # ADX < 20 or CHOP >= 50.0
    TRANSITION = "TRANSITION"        # 20 <= ADX < 30 and CHOP < 50.0
    CLEAN_TREND = "CLEAN_TREND"      # ADX >= 30 and CHOP <= 38.2


@dataclass(frozen=True)
class LeverageMatrixResult:
    safe_leverage: int
    raw_matrix_leverage: int
    asset_tier: AssetTier
    gk_volatility_pct: float
    volatility_regime: str
    liquidation_distance_pct: float
    safety_buffer_ratio: float
    veto: bool
    veto_reason: Optional[str] = None


@dataclass(frozen=True)
class TradeRiskPlan:
    allowed: bool
    symbol: str
    asset_tier: AssetTier
    account_balance: float
    current_price: float
    stop_loss_price: float
    take_profit_price: float
    stop_loss_pct: float
    take_profit_pct: float
    reward_risk_ratio: float
    model_probability: float
    market_regime: MarketRegimeType
    kelly_fraction: float
    full_kelly: float
    final_risk_pct: float
    risk_amount_usd: float
    safe_leverage: int
    position_notional_usd: float
    position_coins: float
    required_margin_usd: float
    potential_profit_usd: float
    is_micro_cap: bool
    scaled_for_margin_cap: bool
    reason: str


@dataclass(frozen=True)
class PortfolioMarginStatus:
    total_balance: float
    max_portfolio_margin_cap_pct: float
    max_allowed_margin_usd: float
    currently_used_margin_usd: float
    available_margin_usd: float
    utilization_pct: float
    active_position_count: int
    max_allowed_positions: int
    can_open_new_trade: bool
    reason: str


class V5DynamicRiskEngine:
    """
    Production-grade Quantitative Risk Engine for MaBuy V5 Demon.
    Strictly implements:
    - Garman-Klass OHLC Volatility Estimator
    - 1x - 5x Dynamic Safe Leverage Matrix
    - 0.15*K to 0.40*K Regime-Adaptive Fractional Kelly Sizing
    - 20% - 25% Portfolio Margin Cap & Multi-Position Safeguards
    - Micro-Deposit ($15 - $100) Constraints and Binance Min-Notional Guards
    """

    # Asset tier mappings
    ASSET_TIERS: Dict[str, AssetTier] = {
        "BTC": AssetTier.TIER_1_CORE,
        "ETH": AssetTier.TIER_1_CORE,
        "SOL": AssetTier.TIER_2_MAJOR,
        "BNB": AssetTier.TIER_2_MAJOR,
        "XRP": AssetTier.TIER_3_ALTS,
        "ADA": AssetTier.TIER_3_ALTS,
        "AVAX": AssetTier.TIER_3_ALTS,
        "DOGE": AssetTier.TIER_3_ALTS,
        "NEAR": AssetTier.TIER_3_ALTS,
        "TON": AssetTier.TIER_3_ALTS,
    }

    # Minimum Model Win Probabilities (Demon Gatekeeper)
    MIN_PROBABILITY_TREND: float = 0.52
    MIN_PROBABILITY_FLAT: float = 0.58
    HIGH_CONFLUENCE_THRESHOLD: float = 0.68

    # Strict Risk-to-Reward minimum
    MIN_REWARD_TO_RISK: float = 1.80

    # Risk limits per trade
    MIN_TRADE_RISK_PCT: float = 1.00          # 1.0% Hard Floor
    MAX_TRADE_RISK_STD_PCT: float = 4.50      # 4.5% Standard Ceiling (Deposit > $100)
    MAX_TRADE_RISK_MICRO_FLAT_PCT: float = 2.00  # 2.0% Micro Ceiling in Flat ($15-$100)
    MAX_TRADE_RISK_MICRO_TREND_PCT: float = 2.50 # 2.5% Micro Ceiling in Trend ($15-$100)

    # Portfolio Constraints
    MICRO_DEPOSIT_THRESHOLD_USD: float = 100.0
    MICRO_DEPOSIT_FLOOR_USD: float = 15.0
    PORTFOLIO_MARGIN_CAP_PCT: float = 25.0    # 25% Max Aggregate Margin
    PORTFOLIO_MARGIN_CAP_MICRO_PCT: float = 20.0 # 20% Max Aggregate Margin for Micro accounts
    BINANCE_MIN_NOTIONAL_USD: float = 5.00    # Binance Futures min order size $5.00

    # Drawdown Emergency Limits
    MAX_DAILY_DRAWDOWN_PCT: float = -7.50     # Emergency Trading Halt
    MAX_CONSECUTIVE_LOSSES: int = 2           # Freeze threshold

    @classmethod
    def get_asset_tier(cls, symbol: str) -> AssetTier:
        clean = symbol.upper().replace("USDT", "").replace("USDC", "").replace("/", "").replace(":USDT", "")
        return cls.ASSET_TIERS.get(clean, AssetTier.TIER_3_ALTS)

    @classmethod
    def calculate_garman_klass_volatility(
        cls,
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        annualized: bool = False,
        periods_per_day: float = 24.0
    ) -> float:
        r"""
        Calculates the Garman-Klass (1980) OHLC Historical Volatility Estimator:
        \sigma_{GK}^2 = \frac{1}{2n} \sum_{i=1}^n \left( \ln\frac{H_i}{L_i} \right)^2 - \frac{2\ln 2 - 1}{n} \sum_{i=1}^n \left( \ln\frac{C_i}{O_i} \right)^2
        Garman-Klass is ~7.4x more statistically efficient than standard Close-to-Close volatility.
        Returns volatility as a percentage (e.g. 2.45 for 2.45%).
        """
        n = len(closes)
        if n < 5:
            return 2.50 # Fallback 2.5%

        # Clip arrays to avoid zero or negative division
        o = np.maximum(1e-8, opens.astype(np.float64))
        h = np.maximum(1e-8, highs.astype(np.float64))
        l = np.maximum(1e-8, lows.astype(np.float64))
        c = np.maximum(1e-8, closes.astype(np.float64))

        hl_ratio = np.log(h / l)
        co_ratio = np.log(c / o)

        term1 = 0.5 * (hl_ratio ** 2)
        term2 = (2.0 * np.log(2.0) - 1.0) * (co_ratio ** 2)
        gk_variance = np.mean(term1 - term2)

        # Variance must be strictly non-negative
        gk_variance = max(1e-9, float(gk_variance))
        gk_sigma = math.sqrt(gk_variance)

        if annualized:
            # Scale to daily/annualized depending on sampling
            gk_sigma = gk_sigma * math.sqrt(periods_per_day * 365.0)

        return float(gk_sigma * 100.0)

    @classmethod
    def evaluate_market_regime(
        cls,
        adx: float,
        choppiness_index: float
    ) -> Tuple[MarketRegimeType, float]:
        """
        Determines market regime and maps it to adaptive Fractional Kelly multiplier f_kelly in [0.15, 0.40].
        - FLAT_RANGE: f_kelly = 0.15 (15% of Kelly)
        - TRANSITION: f_kelly = 0.25 (Quarter-Kelly)
        - CLEAN_TREND: f_kelly = 0.40 (40% of Kelly)
        """
        if choppiness_index >= 50.0 or adx < 20.0:
            regime = MarketRegimeType.FLAT_RANGE
            kelly_fraction = 0.15
        elif adx >= 30.0 and choppiness_index <= 38.2:
            regime = MarketRegimeType.CLEAN_TREND
            kelly_fraction = 0.25  # Quarter-Kelly (was 0.40 — too aggressive for micro-capital)
        else:
            regime = MarketRegimeType.TRANSITION
            # Linear interpolation between 0.15 and 0.25 based on trend strength
            adx_norm = max(0.0, min(1.0, (adx - 20.0) / 10.0))
            chop_norm = max(0.0, min(1.0, (50.0 - choppiness_index) / 11.8))
            blend = 0.5 * (adx_norm + chop_norm)
            kelly_fraction = round(0.15 + blend * (0.25 - 0.15), 3)

        return regime, kelly_fraction

    @classmethod
    def calculate_safe_leverage(
        cls,
        symbol: str,
        model_probability: float,
        gk_volatility_pct: float,
        stop_loss_pct: float,
        market_regime: MarketRegimeType
    ) -> LeverageMatrixResult:
        """
        Dynamic Safe Leverage Matrix (1x - 5x) taking into account:
        1. Confidence (P_win)
        2. Garman-Klass Volatility (Low < 1.8%, Medium 1.8%-3.5%, High > 3.5%)
        3. Asset Tier (BTC/ETH vs SOL/BNB vs ALTS)
        4. Liquidation distance constraint: D_liq >= 2.5 * SL_pct
        """
        tier = cls.get_asset_tier(symbol)
        p = model_probability

        # Determine Volatility Regime
        if gk_volatility_pct < 1.80:
            vol_regime = "LOW"
        elif gk_volatility_pct <= 3.50:
            vol_regime = "MEDIUM"
        else:
            vol_regime = "HIGH"

        # --- DYNAMIC LEVERAGE MATRIX (1x - 5x) ---
        if tier == AssetTier.TIER_1_CORE:
            if p >= 0.70 and vol_regime in ("LOW", "MEDIUM"):
                matrix_lev = 5
            elif p >= 0.62 and vol_regime != "HIGH":
                matrix_lev = 4
            elif p >= 0.55 and vol_regime != "HIGH":
                matrix_lev = 3
            else:
                matrix_lev = 2 if vol_regime != "HIGH" else 1
        elif tier == AssetTier.TIER_2_MAJOR:
            if p >= 0.72 and vol_regime == "LOW":
                matrix_lev = 5
            elif p >= 0.68 and vol_regime in ("LOW", "MEDIUM"):
                matrix_lev = 4
            elif p >= 0.60 and vol_regime != "HIGH":
                matrix_lev = 3
            elif p >= 0.55:
                matrix_lev = 2
            else:
                matrix_lev = 1
        else: # TIER_3_ALTS
            if p >= 0.75 and vol_regime == "LOW":
                matrix_lev = 4
            elif p >= 0.68 and vol_regime in ("LOW", "MEDIUM"):
                matrix_lev = 3
            elif p >= 0.60 and vol_regime != "HIGH":
                matrix_lev = 2
            else:
                matrix_lev = 1

        # Adjust for Market Regime (In choppy/flat markets, cap leverage at max 3x)
        if market_regime == MarketRegimeType.FLAT_RANGE:
            matrix_lev = min(3, matrix_lev)

        # Anti-Liquidation Safety Constraint:
        # Approximate isolated liquidation distance: D_liq = 100 / Leverage
        # Safety requirement: D_liq >= 2.5 * SL_pct  ==>  Leverage <= 50.0 / SL_pct
        effective_sl_pct = max(0.5, stop_loss_pct)
        max_sl_bounded_leverage = max(1, int(50.0 / effective_sl_pct))

        # Final Hard Ceiling clamp [1, 5]
        safe_leverage = max(1, min(5, min(matrix_lev, max_sl_bounded_leverage)))

        liq_dist_pct = round(100.0 / safe_leverage, 2)
        safety_buffer = round(liq_dist_pct / effective_sl_pct, 2)

        veto = False
        veto_reason = None
        if safety_buffer < 2.0:
            veto = True
            veto_reason = f"LEVERAGE VETO: Safety buffer ({safety_buffer}x) is below minimum 2.0x required distance to liquidation."

        return LeverageMatrixResult(
            safe_leverage=safe_leverage,
            raw_matrix_leverage=matrix_lev,
            asset_tier=tier,
            gk_volatility_pct=round(gk_volatility_pct, 2),
            volatility_regime=vol_regime,
            liquidation_distance_pct=liq_dist_pct,
            safety_buffer_ratio=safety_buffer,
            veto=veto,
            veto_reason=veto_reason
        )

    @classmethod
    def calculate_trade_risk_plan(
        cls,
        account_balance: float,
        symbol: str,
        current_price: float,
        stop_loss_price: float,
        take_profit_price: float,
        model_probability: float,
        adx: float = 25.0,
        choppiness_index: float = 45.0,
        gk_volatility_pct: float = 2.20,
        current_daily_drawdown_pct: float = 0.0,
        portfolio_status: Optional[PortfolioMarginStatus] = None
    ) -> TradeRiskPlan:
        """
        Comprehensive Risk & Position Sizing Calculator.
        """
        if account_balance <= 0:
            raise ValueError("Account balance must be positive.")
        if current_price <= 0:
            raise ValueError("Current price must be positive.")

        tier = cls.get_asset_tier(symbol)
        is_micro_cap = account_balance <= cls.MICRO_DEPOSIT_THRESHOLD_USD

        # 1. Emergency Daily Drawdown Circuit Breaker (-7.5%)
        if current_daily_drawdown_pct <= cls.MAX_DAILY_DRAWDOWN_PCT:
            return cls._create_rejected_plan(
                symbol, tier, account_balance, current_price, stop_loss_price, take_profit_price,
                model_probability, is_micro_cap,
                f"HALT: Daily drawdown ({current_daily_drawdown_pct:.2f}%) hit limit ({cls.MAX_DAILY_DRAWDOWN_PCT}%)."
            )

        # 2. Market Regime & Adaptive Kelly Multiplier
        regime, kelly_fraction = cls.evaluate_market_regime(adx, choppiness_index)

        # 3. Model Probability Audit Gatekeeper
        min_p = cls.MIN_PROBABILITY_TREND if regime == MarketRegimeType.CLEAN_TREND else cls.MIN_PROBABILITY_FLAT
        if model_probability < min_p:
            return cls._create_rejected_plan(
                symbol, tier, account_balance, current_price, stop_loss_price, take_profit_price,
                model_probability, is_micro_cap,
                f"VETO: Model probability ({model_probability:.1%}) below regime minimum ({min_p:.1%})."
            )

        # 4. Stop-Loss, Take-Profit, and Payout Ratio (b = TP / SL)
        sl_pct = abs((current_price - stop_loss_price) / current_price) * 100.0
        tp_pct = abs((take_profit_price - current_price) / current_price) * 100.0

        if sl_pct <= 0.05 or tp_pct <= 0.05:
            return cls._create_rejected_plan(
                symbol, tier, account_balance, current_price, stop_loss_price, take_profit_price,
                model_probability, is_micro_cap,
                "VETO: Invalid Stop-Loss or Take-Profit distance (too tight/zero)."
            )

        rr_ratio = round(tp_pct / sl_pct, 2)
        if rr_ratio < cls.MIN_REWARD_TO_RISK:
            return cls._create_rejected_plan(
                symbol, tier, account_balance, current_price, stop_loss_price, take_profit_price,
                model_probability, is_micro_cap,
                f"VETO: Risk-to-Reward ratio (1:{rr_ratio:.2f}) is below strict minimum 1:{cls.MIN_REWARD_TO_RISK:.2f}."
            )

        # 5. Pure Kelly Formula: K = (p*(b+1) - 1) / b
        b = rr_ratio
        p = max(0.0, min(1.0, model_probability))
        kelly_full = (p * (b + 1.0) - 1.0) / b

        if kelly_full <= 0.0:
            return cls._create_rejected_plan(
                symbol, tier, account_balance, current_price, stop_loss_price, take_profit_price,
                model_probability, is_micro_cap,
                f"VETO: Mathematical Kelly edge is non-positive (K = {kelly_full:.4f})."
            )

        # 6. Regime-Adaptive Fractional Kelly Sizing
        raw_kelly_risk_pct = kelly_fraction * kelly_full * 100.0

        # Hard Risk Ceiling based on Deposit Size & Regime
        if is_micro_cap:
            max_risk_cap = cls.MAX_TRADE_RISK_MICRO_TREND_PCT if regime == MarketRegimeType.CLEAN_TREND else cls.MAX_TRADE_RISK_MICRO_FLAT_PCT
        else:
            max_risk_cap = cls.MAX_TRADE_RISK_STD_PCT

        final_risk_pct = max(cls.MIN_TRADE_RISK_PCT, min(max_risk_cap, raw_kelly_risk_pct))
        risk_amount_usd = account_balance * (final_risk_pct / 100.0)

        # 7. Safe Leverage Matrix Evaluation (1x - 5x)
        lev_result = cls.calculate_safe_leverage(
            symbol=symbol,
            model_probability=p,
            gk_volatility_pct=gk_volatility_pct,
            stop_loss_pct=sl_pct,
            market_regime=regime
        )

        if lev_result.veto:
            return cls._create_rejected_plan(
                symbol, tier, account_balance, current_price, stop_loss_price, take_profit_price,
                model_probability, is_micro_cap,
                lev_result.veto_reason or "Leverage safety check failed."
            )

        safe_lev = lev_result.safe_leverage

        # 8. Unconstrained Position Notional & Required Margin
        sl_decimal = sl_pct / 100.0
        position_notional_usd = risk_amount_usd / sl_decimal
        required_margin_usd = position_notional_usd / safe_lev

        scaled_for_margin = False

        # 9. Portfolio Margin Cap Integration
        cap_pct = cls.PORTFOLIO_MARGIN_CAP_MICRO_PCT if is_micro_cap else cls.PORTFOLIO_MARGIN_CAP_PCT
        max_portfolio_margin_usd = account_balance * (cap_pct / 100.0)

        if portfolio_status is not None:
            if not portfolio_status.can_open_new_trade:
                return cls._create_rejected_plan(
                    symbol, tier, account_balance, current_price, stop_loss_price, take_profit_price,
                    model_probability, is_micro_cap,
                    f"PORTFOLIO CAP VETO: {portfolio_status.reason}"
                )
            available_margin = portfolio_status.available_margin_usd
        else:
            available_margin = max_portfolio_margin_usd

        # If required margin exceeds available margin, scale down gracefully
        if required_margin_usd > available_margin:
            if available_margin <= 0.5:
                return cls._create_rejected_plan(
                    symbol, tier, account_balance, current_price, stop_loss_price, take_profit_price,
                    model_probability, is_micro_cap,
                    f"PORTFOLIO CAP VETO: Remaining margin capacity (${available_margin:.2f}) is exhausted."
                )

            # Downscale margin to fit available capacity
            required_margin_usd = available_margin
            position_notional_usd = required_margin_usd * safe_lev
            risk_amount_usd = position_notional_usd * sl_decimal
            final_risk_pct = (risk_amount_usd / account_balance) * 100.0
            scaled_for_margin = True

        # 10. Binance Futures Minimum Order Notional Check ($5.00)
        if position_notional_usd < cls.BINANCE_MIN_NOTIONAL_USD:
            min_notional_risk_usd = cls.BINANCE_MIN_NOTIONAL_USD * sl_decimal
            min_notional_risk_pct = (min_notional_risk_usd / account_balance) * 100.0

            if min_notional_risk_pct <= (max_risk_cap * 1.25) and (cls.BINANCE_MIN_NOTIONAL_USD / safe_lev) <= available_margin:
                position_notional_usd = cls.BINANCE_MIN_NOTIONAL_USD
                required_margin_usd = round(position_notional_usd / safe_lev, 2)
                risk_amount_usd = round(position_notional_usd * sl_decimal, 2)
                final_risk_pct = round((risk_amount_usd / account_balance) * 100.0, 2)
            else:
                return cls._create_rejected_plan(
                    symbol, tier, account_balance, current_price, stop_loss_price, take_profit_price,
                    model_probability, is_micro_cap,
                    f"MICRO-CAP VETO: Position notional (${position_notional_usd:.2f}) is below Binance minimum $5.00 notional."
                )

        position_coins = round(position_notional_usd / current_price, 6)
        potential_profit_usd = round(position_notional_usd * (tp_pct / 100.0), 2)

        reason = (
            f"APPROVED: {regime.value} sizing ({kelly_fraction:.2f}*K). "
            f"Risk: {final_risk_pct:.2f}% (${risk_amount_usd:.2f}), "
            f"Leverage: {safe_lev}x ({lev_result.volatility_regime} GK-Vol {lev_result.gk_volatility_pct}%), "
            f"Margin: ${required_margin_usd:.2f}."
        )

        return TradeRiskPlan(
            allowed=True,
            symbol=symbol,
            asset_tier=tier,
            account_balance=round(account_balance, 2),
            current_price=round(current_price, 4),
            stop_loss_price=round(stop_loss_price, 4),
            take_profit_price=round(take_profit_price, 4),
            stop_loss_pct=round(sl_pct, 2),
            take_profit_pct=round(tp_pct, 2),
            reward_risk_ratio=rr_ratio,
            model_probability=round(p, 3),
            market_regime=regime,
            kelly_fraction=round(kelly_fraction, 2),
            full_kelly=round(kelly_full, 4),
            final_risk_pct=round(final_risk_pct, 2),
            risk_amount_usd=round(risk_amount_usd, 2),
            safe_leverage=safe_lev,
            position_notional_usd=round(position_notional_usd, 2),
            position_coins=position_coins,
            required_margin_usd=round(required_margin_usd, 2),
            potential_profit_usd=potential_profit_usd,
            is_micro_cap=is_micro_cap,
            scaled_for_margin_cap=scaled_for_margin,
            reason=reason
        )

    @classmethod
    def evaluate_portfolio_margin_cap(
        cls,
        account_balance: float,
        active_positions_margins: List[float]
    ) -> PortfolioMarginStatus:
        """
        Evaluates portfolio margin utilization and enforces 20% - 25% max margin cap.
        Micro accounts ($15-$100): Max 20% cap, max 1-2 concurrent positions.
        Standard accounts (>$100): Max 25% cap, max 3-4 concurrent positions.
        """
        is_micro = account_balance <= cls.MICRO_DEPOSIT_THRESHOLD_USD
        cap_pct = cls.PORTFOLIO_MARGIN_CAP_MICRO_PCT if is_micro else cls.PORTFOLIO_MARGIN_CAP_PCT
        max_margin_usd = round(account_balance * (cap_pct / 100.0), 2)
        used_margin_usd = round(sum(active_positions_margins), 2)
        avail_margin_usd = max(0.0, round(max_margin_usd - used_margin_usd, 2))
        utilization_pct = round((used_margin_usd / max_margin_usd * 100.0) if max_margin_usd > 0 else 100.0, 1)

        pos_count = len(active_positions_margins)
        if is_micro:
            max_positions = 1 if account_balance <= 50.0 else 2
        else:
            max_positions = 4

        can_open = True
        reason = "OK: Margin capacity available."

        if pos_count >= max_positions:
            can_open = False
            reason = f"Concurrency limit reached ({pos_count}/{max_positions} active positions)."
        elif avail_margin_usd < 1.0:
            can_open = False
            reason = f"Margin capacity exhausted (Used ${used_margin_usd:.2f} / ${max_margin_usd:.2f}, {utilization_pct}% utilized)."

        return PortfolioMarginStatus(
            total_balance=account_balance,
            max_portfolio_margin_cap_pct=cap_pct,
            max_allowed_margin_usd=max_margin_usd,
            currently_used_margin_usd=used_margin_usd,
            available_margin_usd=avail_margin_usd,
            utilization_pct=utilization_pct,
            active_position_count=pos_count,
            max_allowed_positions=max_positions,
            can_open_new_trade=can_open,
            reason=reason
        )

    @classmethod
    def _create_rejected_plan(
        cls,
        symbol: str,
        tier: AssetTier,
        balance: float,
        price: float,
        sl: float,
        tp: float,
        prob: float,
        is_micro: bool,
        reason: str
    ) -> TradeRiskPlan:
        sl_pct = abs((price - sl) / price) * 100.0 if price > 0 else 0.0
        tp_pct = abs((tp - price) / price) * 100.0 if price > 0 else 0.0
        rr = round(tp_pct / max(0.01, sl_pct), 2)

        return TradeRiskPlan(
            allowed=False,
            symbol=symbol,
            asset_tier=tier,
            account_balance=balance,
            current_price=price,
            stop_loss_price=sl,
            take_profit_price=tp,
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            reward_risk_ratio=rr,
            model_probability=prob,
            market_regime=MarketRegimeType.FLAT_RANGE,
            kelly_fraction=0.15,
            full_kelly=0.0,
            final_risk_pct=0.0,
            risk_amount_usd=0.0,
            safe_leverage=1,
            position_notional_usd=0.0,
            position_coins=0.0,
            required_margin_usd=0.0,
            potential_profit_usd=0.0,
            is_micro_cap=is_micro,
            scaled_for_margin_cap=False,
            reason=reason
        )
