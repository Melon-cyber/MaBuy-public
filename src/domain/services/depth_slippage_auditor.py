import math
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class DepthExecutionAuditResult:
    is_liquid_enough: bool
    position_size_usd: float
    position_size_coins: float
    mid_price: float
    vwap_execution_price: float
    slippage_pct: float
    book_depth_usd_05pct: float
    liquidity_absorption_ratio: float # (Position Size / Depth within 0.5%)
    order_type_recommended: str # "LIMIT_POST_ONLY" or "MARKET_TAKER"
    is_hollow_book_veto: bool
    veto_reason: Optional[str]


class DepthIntegratedSlippageAuditor:
    """
    Simulates real-world execution by walking the order book (Depth Integration).
    Eliminates the 'Hollow Book' illusion during liquidation cascades (Z > 3.0).
    
    Guarantees:
    - Real VWAP execution price calculation across top-20 order book levels.
    - Automatic veto or switch to LIMIT_POST_ONLY if slippage > 0.25% or position consumes > 15% of depth.
    """

    def __init__(self, max_allowed_slippage_pct: float = 0.25, max_depth_absorption_pct: float = 15.0):
        self.max_allowed_slippage_pct = max_allowed_slippage_pct
        self.max_depth_absorption_pct = max_depth_absorption_pct

    def audit_execution_cost(
        self,
        bids: List[List[float]], # [[price, qty], ...]
        asks: List[List[float]], # [[price, qty], ...]
        position_size_usd: float,
        direction: str = "BULLISH"
    ) -> DepthExecutionAuditResult:
        if not bids or not asks or position_size_usd <= 0:
            return DepthExecutionAuditResult(
                is_liquid_enough=False,
                position_size_usd=position_size_usd,
                position_size_coins=0.0,
                mid_price=0.0,
                vwap_execution_price=0.0,
                slippage_pct=0.0,
                book_depth_usd_05pct=0.0,
                liquidity_absorption_ratio=1.0,
                order_type_recommended="LIMIT_POST_ONLY",
                is_hollow_book_veto=True,
                veto_reason="Empty order book data: cannot audit market depth"
            )

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid_price = (best_bid + best_ask) / 2.0
        target_coins = position_size_usd / mid_price

        # Select book side: Buying walks Asks, Selling walks Bids
        book_levels = asks if direction.upper() == "BULLISH" else bids
        
        # 1. Walk the book to calculate realistic VWAP execution price
        accumulated_coins = 0.0
        accumulated_cost_usd = 0.0
        depth_within_05pct_usd = 0.0

        for price_val, qty_val in book_levels:
            p = float(price_val)
            q = float(qty_val)
            level_cost = p * q

            # Check if within 0.5% of mid price
            dist_pct = abs(p - mid_price) / mid_price * 100.0
            if dist_pct <= 0.50:
                depth_within_05pct_usd += level_cost

            # Fill position
            needed = target_coins - accumulated_coins
            if needed > 0:
                fill_qty = min(q, needed)
                accumulated_coins += fill_qty
                accumulated_cost_usd += fill_qty * p

        # If book couldn't completely fill target coins
        if accumulated_coins < target_coins * 0.99:
            return DepthExecutionAuditResult(
                is_liquid_enough=False,
                position_size_usd=position_size_usd,
                position_size_coins=target_coins,
                mid_price=mid_price,
                vwap_execution_price=mid_price * 1.05 if direction.upper() == "BULLISH" else mid_price * 0.95,
                slippage_pct=5.0,
                book_depth_usd_05pct=depth_within_05pct_usd,
                liquidity_absorption_ratio=1.0,
                order_type_recommended="LIMIT_POST_ONLY",
                is_hollow_book_veto=True,
                veto_reason=f"HOLLOW BOOK VETO: Book only has {accumulated_coins:.4f} coins, needed {target_coins:.4f}"
            )

        vwap_price = accumulated_cost_usd / max(1e-6, accumulated_coins)
        slippage_pct = (abs(vwap_price - mid_price) / mid_price) * 100.0
        
        # Liquidity absorption check
        absorption_ratio = (position_size_usd / max(1.0, depth_within_05pct_usd)) * 100.0

        # Veto & Order Type Recommendation Logic
        is_hollow = False
        veto_reason = None
        order_type = "MARKET_TAKER"

        if slippage_pct > self.max_allowed_slippage_pct:
            is_hollow = True
            veto_reason = (
                f"HOLLOW BOOK VETO: Realistic slippage ({slippage_pct:.3f}%) exceeds max threshold "
                f"({self.max_allowed_slippage_pct}%). Market order would annihilate R:R."
            )
            order_type = "LIMIT_POST_ONLY"
        elif absorption_ratio > self.max_depth_absorption_pct:
            order_type = "LIMIT_POST_ONLY"

        is_liquid = bool(not is_hollow)

        return DepthExecutionAuditResult(
            is_liquid_enough=is_liquid,
            position_size_usd=round(position_size_usd, 2),
            position_size_coins=round(target_coins, 6),
            mid_price=round(mid_price, 4),
            vwap_execution_price=round(vwap_price, 4),
            slippage_pct=round(slippage_pct, 4),
            book_depth_usd_05pct=round(depth_within_05pct_usd, 2),
            liquidity_absorption_ratio=round(absorption_ratio, 2),
            order_type_recommended=order_type,
            is_hollow_book_veto=is_hollow,
            veto_reason=veto_reason
        )
