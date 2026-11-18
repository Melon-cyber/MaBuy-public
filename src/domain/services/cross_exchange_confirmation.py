"""Cross-Exchange Confirmation Layer (CECL).

Institutional multi-venue signal confirmation service according to the Astra (gpt-6-astra)
Master Quant Blueprint. Evaluates cross-exchange consensus across Binance, Bybit, and OKX
perpetual futures to filter out false breakouts, exchange-specific liquidity wicks,
and toxic adverse price dislocations before risking live capital.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.domain.services.cross_exchange_state import CrossExchangeState, canonical_symbol

DecisionStatus = Literal["PASS", "VETO", "NEUTRAL", "DEGRADED"]


@dataclass(slots=True)
class CrossExchangeDecision:
    """Strictly typed outcome of the cross-exchange confirmation evaluation."""

    status: DecisionStatus
    score: float
    reason: str
    fresh_venues: List[str]
    external_return_bps: float
    dislocation_bps: float
    lead_lag_correlation: Optional[float]
    metrics: Dict[str, Any] = field(default_factory=dict)


class CrossExchangeConfirmation:
    """
    Evaluates multi-venue market consistency before order dispatch.
    
    Zero network calls during evaluate() — queries pre-warmed in-memory CrossExchangeState.
    """

    DEFAULT_DISLOCATION_THRESHOLDS_BPS = {
        "BTC": 8.0,
        "ETH": 12.0,
        "SOL": 20.0,
    }

    def __init__(
        self,
        state: CrossExchangeState,
        max_quote_age_sec: float = 5.0,
        return_window_sec: float = 3.0,
        return_confirmation_threshold_bps: float = 1.5,
        dislocation_thresholds_bps: Optional[Dict[str, float]] = None,
        weight_direction: float = 0.45,
        weight_lead_lag: float = 0.25,
        weight_dislocation: float = 0.20,
        weight_data_quality: float = 0.10,
    ) -> None:
        self.state = state
        self.max_quote_age_sec = max_quote_age_sec
        self.return_window_sec = return_window_sec
        self.return_confirmation_threshold_bps = return_confirmation_threshold_bps
        self.dislocation_thresholds = (
            dislocation_thresholds_bps
            if dislocation_thresholds_bps is not None
            else dict(self.DEFAULT_DISLOCATION_THRESHOLDS_BPS)
        )
        self.weight_direction = weight_direction
        self.weight_lead_lag = weight_lead_lag
        self.weight_dislocation = weight_dislocation
        self.weight_data_quality = weight_data_quality

    def evaluate(
        self,
        symbol: str,
        side: Literal["LONG", "SHORT", "BULLISH", "BEARISH"],
        current_price: Optional[float] = None,
        lead_lag_score: Optional[float] = None,
        now_ts: Optional[float] = None,
    ) -> CrossExchangeDecision:
        """
        Calculates cross-exchange confirmation score without executing any side-effects.

        Parameters
        ----------
        symbol : str
            Target asset symbol (e.g. 'BTC', 'ETHUSDT').
        side : str
            Proposed trade direction ('LONG' / 'BULLISH' or 'SHORT' / 'BEARISH').
        current_price : float, optional
            Current reference price (if available, used to refresh state).
        lead_lag_score : float, optional
            Normalized lead-lag score in [-1.0, 1.0] from cross_exchange_lead_lag.py.
        now_ts : float, optional
            Evaluation timestamp (defaults to time.time()).
        """
        sym = canonical_symbol(symbol)
        now = time.time() if now_ts is None else float(now_ts)
        is_long = side.upper() in ("LONG", "BULLISH", "BUY")

        fresh_venues = self.state.get_fresh_venues(sym, max_age_sec=self.max_quote_age_sec, now_ts=now)
        external_fresh = [v for v in fresh_venues if v in ("bybit", "okx")]

        # If neither Bybit nor OKX are fresh -> DEGRADED mode
        if not external_fresh:
            return CrossExchangeDecision(
                status="DEGRADED",
                score=0.0,
                reason="Нет свежих котировок Bybit/OKX (данные старше 5с)",
                fresh_venues=fresh_venues,
                external_return_bps=0.0,
                dislocation_bps=0.0,
                lead_lag_correlation=lead_lag_score,
                metrics={"external_venues_count": 0, "fresh_venues": fresh_venues},
            )

        # 1. External Return Component
        ext_ret = self.state.get_median_external_return(
            sym, window_sec=self.return_window_sec, max_age_sec=self.max_quote_age_sec, now_ts=now
        )
        ext_ret_val = ext_ret if ext_ret is not None else 0.0
        ext_ret_bps = ext_ret_val * 10000.0

        # Compare with threshold
        thresh_bps = self.return_confirmation_threshold_bps
        if is_long:
            if ext_ret_bps >= thresh_bps:
                c_direction = 1.0
            elif ext_ret_bps <= -thresh_bps:
                c_direction = -1.0  # External market dumping while signal is LONG!
            else:
                c_direction = 0.0
        else:  # SHORT
            if ext_ret_bps <= -thresh_bps:
                c_direction = 1.0
            elif ext_ret_bps >= thresh_bps:
                c_direction = -1.0  # External market pumping while signal is SHORT!
            else:
                c_direction = 0.0

        # 2. Price Dislocation Component
        dislocation_bps = self.state.get_dislocation_bps(
            sym, max_age_sec=self.max_quote_age_sec, now_ts=now
        )
        disloc_val = dislocation_bps if dislocation_bps is not None else 0.0
        max_disloc = self.dislocation_thresholds.get(sym, 15.0)

        d_dislocation = 0.0
        disloc_veto = False
        disloc_reason = ""

        if is_long:
            # If Binance is much more expensive than external venues, risk of buying the top
            if disloc_val > max_disloc:
                d_dislocation = min(1.0, (disloc_val - max_disloc) / max_disloc + 0.5)
                disloc_veto = True
                disloc_reason = f"Премия Binance +{disloc_val:.1f} bps превышает лимит {max_disloc:.1f} bps для LONG"
            elif disloc_val < -max_disloc:
                # Binance is at deep discount - favorable for LONG
                d_dislocation = 0.0
        else:
            # If Binance is much cheaper than external venues, risk of shorting the bottom
            if disloc_val < -max_disloc:
                d_dislocation = min(1.0, (abs(disloc_val) - max_disloc) / max_disloc + 0.5)
                disloc_veto = True
                disloc_reason = f"Дисконт Binance {disloc_val:.1f} bps превышает лимит {max_disloc:.1f} bps для SHORT"

        # 3. Lead-Lag Component
        if lead_lag_score is not None and math.isfinite(lead_lag_score):
            c_leadlag = max(-1.0, min(1.0, lead_lag_score if is_long else -lead_lag_score))
        else:
            c_leadlag = 0.0

        # 4. Data Quality
        # 3 venues fresh = 0 penalty; 2 fresh = 0.1 penalty; 1 fresh = 0.5 penalty
        venues_count = len(fresh_venues)
        q_data_risk = 0.0 if venues_count >= 3 else (0.2 if venues_count == 2 else 0.6)

        # 5. Composite Score Calculation
        composite_score = (
            self.weight_direction * c_direction
            + self.weight_lead_lag * c_leadlag
            - self.weight_dislocation * d_dislocation
            - self.weight_data_quality * q_data_risk
        )
        composite_score = max(-1.0, min(1.0, composite_score))

        # 6. Status Determination
        metrics = {
            "c_direction": c_direction,
            "c_leadlag": c_leadlag,
            "d_dislocation": d_dislocation,
            "q_data_risk": q_data_risk,
            "external_return_bps": ext_ret_bps,
            "dislocation_bps": disloc_val,
            "fresh_venues_count": len(fresh_venues),
        }

        if disloc_veto:
            return CrossExchangeDecision(
                status="VETO",
                score=composite_score,
                reason=disloc_reason,
                fresh_venues=fresh_venues,
                external_return_bps=ext_ret_bps,
                dislocation_bps=disloc_val,
                lead_lag_correlation=lead_lag_score,
                metrics=metrics,
            )

        if composite_score <= -0.35:
            reason = (
                f"Внешний рынок движется против сделки (возврат: {ext_ret_bps:+.1f} bps, скор: {composite_score:+.2f})"
            )
            return CrossExchangeDecision(
                status="VETO",
                score=composite_score,
                reason=reason,
                fresh_venues=fresh_venues,
                external_return_bps=ext_ret_bps,
                dislocation_bps=disloc_val,
                lead_lag_correlation=lead_lag_score,
                metrics=metrics,
            )
        elif composite_score >= 0.20:
            reason = (
                f"Подтверждено Bybit/OKX (возврат: {ext_ret_bps:+.1f} bps, скор: {composite_score:+.2f})"
            )
            return CrossExchangeDecision(
                status="PASS",
                score=composite_score,
                reason=reason,
                fresh_venues=fresh_venues,
                external_return_bps=ext_ret_bps,
                dislocation_bps=disloc_val,
                lead_lag_correlation=lead_lag_score,
                metrics=metrics,
            )
        else:
            reason = (
                f"Нейтральная динамика площадок (возврат: {ext_ret_bps:+.1f} bps, скор: {composite_score:+.2f})"
            )
            return CrossExchangeDecision(
                status="NEUTRAL",
                score=composite_score,
                reason=reason,
                fresh_venues=fresh_venues,
                external_return_bps=ext_ret_bps,
                dislocation_bps=disloc_val,
                lead_lag_correlation=lead_lag_score,
                metrics=metrics,
            )
