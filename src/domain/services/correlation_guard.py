"""
MaBuy Cross-Asset Correlation Guard.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger("CrossAssetCorrelationGuard")


@dataclass(frozen=True)
class CorrelationGuardEvaluation:
    allowed: bool
    candidate_symbol: str
    candidate_direction: str
    is_third_correlated_trade: bool
    confidence_required: float
    candidate_confidence: float
    margin_haircut_multiplier: float
    veto: bool
    rejection_reason: Optional[str]
    summary: str
    total_gross_notional_usd: float = 0.0
    aligned_notional_usd: float = 0.0


class CrossAssetCorrelationGuard:
    MONITORED_WHITELIST = ("BTC", "ETH", "SOL")
    ELEVATED_CONFIDENCE_THRESHOLD: float = 0.80  # 80% for 3rd correlated trade
    
    # Progressive haircuts for same-direction (aligned Beta) trades:
    FIRST_TRADE_HAIRCUT: float = 1.00      # 100% sizing
    SECOND_TRADE_HAIRCUT: float = 0.75     # 75% sizing (-25% haircut)
    THIRD_TRADE_HAIRCUT: float = 0.50      # 50% sizing (-50% haircut)

    # Hard portfolio bounds recommended by Astra Lead Quant Audit:
    MAX_GROSS_NOTIONAL_RATIO: float = 3.0  # Max total gross notional: 3.0x equity
    MAX_ALIGNED_NOTIONAL_RATIO: float = 2.2 # Max aligned (Beta=1) notional: 2.2x equity

    @staticmethod
    def normalize_direction(raw_direction: str) -> str:
        d = raw_direction.upper().strip()
        return "LONG" if d in ("BUY", "LONG", "BULLISH") else "SHORT"

    @classmethod
    def clean_symbol(cls, raw_sym: str) -> str:
        return raw_sym.upper().replace("/USDT:USDT", "").replace(":USDT", "").replace("/USDT", "").replace("USDT", "").strip()

    def evaluate_candidate_trade(
        self,
        candidate_symbol: str,
        candidate_direction: str,
        candidate_confidence: float,
        active_positions: List[Dict[str, Any]],
        portfolio_equity: float = 40.0,
        estimated_trade_notional: float = 21.0
    ) -> CorrelationGuardEvaluation:
        norm_sym = self.clean_symbol(candidate_symbol)
        norm_dir = self.normalize_direction(candidate_direction)
        norm_conf = candidate_confidence / 100.0 if candidate_confidence > 1.0 else candidate_confidence

        active_same_symbols = []
        existing_gross_notional = 0.0
        existing_aligned_notional = 0.0

        for pos in active_positions:
            pos_sym = self.clean_symbol(pos.get("symbol", ""))
            pos_amt = abs(float(pos.get("info", {}).get("positionAmt", 0) or pos.get("contracts", 0) or pos.get("amount", 0) or 0.0))
            pos_price = float(pos.get("markPrice", 0) or pos.get("entryPrice", 0) or pos.get("price", 0) or 0.0)
            pos_notional = float(pos.get("notional", 0) or (pos_amt * pos_price) or 0.0)
            existing_gross_notional += pos_notional

            if pos_sym in self.MONITORED_WHITELIST and pos_sym != norm_sym:
                pos_dir = self.normalize_direction(pos.get("direction") or pos.get("side") or "")
                if pos_dir == norm_dir:
                    active_same_symbols.append(pos_sym)
                    existing_aligned_notional += pos_notional

        num_aligned = len(active_same_symbols)

        # 1. Check Aggregate Gross Portfolio Notional Cap (Astra Critical Finding 3)
        projected_gross = existing_gross_notional + estimated_trade_notional
        max_allowed_gross = max(30.0, portfolio_equity * self.MAX_GROSS_NOTIONAL_RATIO)
        if projected_gross > max_allowed_gross and existing_gross_notional > 0:
            return CorrelationGuardEvaluation(
                allowed=False,
                candidate_symbol=norm_sym,
                candidate_direction=norm_dir,
                is_third_correlated_trade=(num_aligned >= 2),
                confidence_required=0.75,
                candidate_confidence=norm_conf,
                margin_haircut_multiplier=0.0,
                veto=True,
                rejection_reason=(
                    f"GROSS_NOTIONAL_VETO: Projected portfolio notional (${projected_gross:.2f}) "
                    f"exceeds {self.MAX_GROSS_NOTIONAL_RATIO:.1f}x equity cap (${max_allowed_gross:.2f})."
                ),
                summary="Vetoed by Portfolio Gross Exposure Cap.",
                total_gross_notional_usd=projected_gross,
                aligned_notional_usd=existing_aligned_notional
            )

        # 2. Check Aligned Beta Notional Cap
        projected_aligned = existing_aligned_notional + estimated_trade_notional
        max_allowed_aligned = max(25.0, portfolio_equity * self.MAX_ALIGNED_NOTIONAL_RATIO)
        if projected_aligned > max_allowed_aligned and num_aligned >= 1:
            return CorrelationGuardEvaluation(
                allowed=False,
                candidate_symbol=norm_sym,
                candidate_direction=norm_dir,
                is_third_correlated_trade=(num_aligned >= 2),
                confidence_required=0.75,
                candidate_confidence=norm_conf,
                margin_haircut_multiplier=0.0,
                veto=True,
                rejection_reason=(
                    f"ALIGNED_BETA_VETO: Aligned {norm_dir} notional (${projected_aligned:.2f}) "
                    f"exceeds {self.MAX_ALIGNED_NOTIONAL_RATIO:.1f}x equity cap (${max_allowed_aligned:.2f})."
                ),
                summary="Vetoed by Systemic Aligned Beta Cap.",
                total_gross_notional_usd=projected_gross,
                aligned_notional_usd=projected_aligned
            )

        # 3. Progressive Haircut & Confidence Gate for Aligned Positions
        if num_aligned >= 2:
            # 3rd correlated position
            if norm_conf < self.ELEVATED_CONFIDENCE_THRESHOLD:
                return CorrelationGuardEvaluation(
                    allowed=False,
                    candidate_symbol=norm_sym,
                    candidate_direction=norm_dir,
                    is_third_correlated_trade=True,
                    confidence_required=self.ELEVATED_CONFIDENCE_THRESHOLD,
                    candidate_confidence=norm_conf,
                    margin_haircut_multiplier=1.0,
                    veto=True,
                    rejection_reason=(
                        f"CORRELATION_VETO: 2 positions ({', '.join(active_same_symbols)}) already active in {norm_dir}. "
                        f"3rd correlated trade requires >= 80% confidence (got {norm_conf*100:.1f}%)."
                    ),
                    summary="Vetoed: 3rd correlated trade failed 80% confidence barrier.",
                    total_gross_notional_usd=projected_gross,
                    aligned_notional_usd=projected_aligned
                )
            else:
                return CorrelationGuardEvaluation(
                    allowed=True,
                    candidate_symbol=norm_sym,
                    candidate_direction=norm_dir,
                    is_third_correlated_trade=True,
                    confidence_required=self.ELEVATED_CONFIDENCE_THRESHOLD,
                    candidate_confidence=norm_conf,
                    margin_haircut_multiplier=self.THIRD_TRADE_HAIRCUT,
                    veto=False,
                    rejection_reason=None,
                    summary=(
                        f"Approved: 3rd correlated trade passed 80% barrier ({norm_conf*100:.1f}%). "
                        f"Margin scaled to {self.THIRD_TRADE_HAIRCUT*100:.0f}% (-50% haircut)."
                    ),
                    total_gross_notional_usd=projected_gross,
                    aligned_notional_usd=projected_aligned
                )
        elif num_aligned == 1:
            # 2nd correlated position: apply 25% progressive haircut
            return CorrelationGuardEvaluation(
                allowed=True,
                candidate_symbol=norm_sym,
                candidate_direction=norm_dir,
                is_third_correlated_trade=False,
                confidence_required=0.75,
                candidate_confidence=norm_conf,
                margin_haircut_multiplier=self.SECOND_TRADE_HAIRCUT,
                veto=False,
                rejection_reason=None,
                summary=(
                    f"Approved: 2nd correlated trade in {norm_dir} (with {active_same_symbols[0]}). "
                    f"Margin scaled to {self.SECOND_TRADE_HAIRCUT*100:.0f}% (-25% haircut)."
                ),
                total_gross_notional_usd=projected_gross,
                aligned_notional_usd=projected_aligned
            )

        # Standalone or diversified position
        return CorrelationGuardEvaluation(
            allowed=True,
            candidate_symbol=norm_sym,
            candidate_direction=norm_dir,
            is_third_correlated_trade=False,
            confidence_required=0.75,
            candidate_confidence=norm_conf,
            margin_haircut_multiplier=self.FIRST_TRADE_HAIRCUT,
            veto=False,
            rejection_reason=None,
            summary="Normal portfolio diversification. Full sizing permitted.",
            total_gross_notional_usd=projected_gross,
            aligned_notional_usd=projected_aligned
        )
