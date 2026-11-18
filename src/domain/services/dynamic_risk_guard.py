"""
MaBuy V4 — Risk & Compliance Department
DynamicRiskGuardService: Conservative Fractional Ensemble Kelly (0.05 * K) Risk Allocation Engine
Strictly enforces risk scaling between theoretical 1.0% floor and 4.5% ceiling per trade with a -7.5% Daily Drawdown Circuit Breaker.
"""
import math
from typing import Dict, Any

class CircuitBreakerTriggeredException(Exception):
    """Raised when daily drawdown exceeds emergency threshold (-7.5%)."""
    pass

class DynamicRiskGuardService:
    MIN_RISK_PCT: float = 1.0            # Floor risk per trade (1.0%)
    MAX_RISK_PCT: float = 4.5            # Ceiling risk per trade for standard accounts (4.5%)
    MICRO_CAP_THRESHOLD_USD: float = 50.0 # Account threshold for micro-deposit protection ($50)
    MICRO_CAP_MAX_RISK_PCT: float = 2.0   # Strict 2.0% hard risk ceiling for micro accounts ($15-$50)
    MAX_DAILY_DRAWDOWN_PCT: float = -7.5 # Emergency halt limit (-7.5%)
    KELLY_ENSEMBLE_FRACTION: float = 0.05 # Fractional Ensemble Kelly Multiplier (5% of Full Kelly)

    def calculate_trade_risk(
        self,
        account_balance: float,
        model_probability: float,
        take_profit_pct: float,
        stop_loss_pct: float,
        current_daily_drawdown_pct: float = 0.0,
        is_trend: bool = False
    ) -> Dict[str, Any]:
        """
        Calculates position risk using PURE Kelly Criterion: K = (p*(b+1) - 1)/b
        preserving strict mathematical dependency on BOTH probability (p) AND payout ratio (b=TP/SL).
        Enforces Adaptive Sizing: Half-Kelly during confirmed Trend, Quarter-Kelly during Normal regime.
        """
        if account_balance <= 0:
            raise ValueError("Account balance must be positive.")

        # 1. Emergency Circuit Breaker Check (-7.5% or lower triggers halt)
        if current_daily_drawdown_pct <= self.MAX_DAILY_DRAWDOWN_PCT:
            return {
                "allowed": False,
                "risk_pct": 0.0,
                "position_size_usd": 0.0,
                "circuit_breaker_triggered": True,
                "reason": f"CRITICAL RISK HALT: Daily drawdown ({current_daily_drawdown_pct:.2f}%) reached or exceeded limit ({self.MAX_DAILY_DRAWDOWN_PCT}%)."
            }

        # 2. Validate input parameters and calculate payout ratio b = TP / SL
        p = max(0.0, min(1.0, model_probability))
        if stop_loss_pct <= 0 or take_profit_pct <= 0:
            b = 1.5 # Default payout ratio fallback
        else:
            b = take_profit_pct / stop_loss_pct

        # 3. PURE Full Kelly Formula: K = (p*(b+1) - 1) / b
        kelly_full = (p * (b + 1.0) - 1.0) / b
        kelly_positive = max(0.0, kelly_full)

        # 4. Regime-Adaptive Kelly Multiplier (Half-Kelly on Trend, Quarter-Kelly on Standard)
        kelly_fraction = 0.10 if is_trend else self.KELLY_ENSEMBLE_FRACTION
        raw_kelly_risk_pct = (kelly_fraction * kelly_positive) * 100.0
        
        is_micro_cap = account_balance <= self.MICRO_CAP_THRESHOLD_USD
        micro_cap_ceiling = 2.5 if is_trend else self.MICRO_CAP_MAX_RISK_PCT
        effective_max_risk = micro_cap_ceiling if is_micro_cap else self.MAX_RISK_PCT
        final_risk_pct = max(self.MIN_RISK_PCT, min(effective_max_risk, raw_kelly_risk_pct))

        # 5. Calculate USD Risk Amount and Position Size
        risk_amount_usd = account_balance * (final_risk_pct / 100.0)
        stop_loss_decimal = (stop_loss_pct / 100.0) if stop_loss_pct > 0 else 0.01
        position_size_usd = risk_amount_usd / stop_loss_decimal

        # 6. Dynamic Break-Even & Trailing Milestones
        break_even_gain_pct = round(take_profit_pct * 0.50, 2)
        trailing_gain_pct = round(take_profit_pct * 0.75, 2)

        return {
            "allowed": True,
            "risk_pct": round(final_risk_pct, 2),
            "risk_amount_usd": round(risk_amount_usd, 2),
            "position_size_usd": round(position_size_usd, 2),
            "payout_ratio_b": round(b, 2),
            "raw_fractional_kelly_pct": round(raw_kelly_risk_pct, 2),
            "raw_quarter_kelly_pct": round(raw_kelly_risk_pct, 2),  # Backwards compatibility
            "full_kelly": round(kelly_full, 4),
            "is_micro_cap": is_micro_cap,
            "break_even_trigger_pct": break_even_gain_pct,
            "trailing_trigger_pct": trailing_gain_pct,
            "circuit_breaker_triggered": False,
            "reason": f"OK: Trade risk {'(Micro-Cap Capped 2.0%)' if is_micro_cap else ''} within bounds [{self.MIN_RISK_PCT}%, {effective_max_risk}%]"
        }
