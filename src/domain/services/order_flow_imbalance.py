import math
from typing import Dict, List, Tuple, Any, Optional

class OrderFlowImbalanceService:
    """
    Microstructure Order Flow Imbalance (OFI) and Order Book Imbalance (OBI) Service.
    Analyzes live 20-level orderbook depth from Binance to detect:
    1. Order Book Imbalance (OBI): ratio of bid liquidity vs ask liquidity.
    2. Liquidity Walls: disproportionate clusters of limit orders acting as resistance/support.
    3. Microstructure Veto: rejects false breakout signals if orderbook flow contradicts technicals.
    """

    def __init__(self, veto_threshold: float = 0.25, confirmation_threshold: float = 0.20, min_wall_usd: float = 150_000.0):
        self.veto_threshold = veto_threshold
        self.confirmation_threshold = confirmation_threshold
        self.min_wall_usd = min_wall_usd

    @staticmethod
    def calculate_microprice(bids: List[List[Any]], asks: List[List[Any]]) -> float:
        """
        Calculates instantaneous top-of-book microprice:
        microprice = (P_ask * Q_bid + P_bid * Q_ask) / (Q_bid + Q_ask)
        If book is empty, returns 0.0.
        """
        if not bids or not asks:
            return 0.0
        try:
            bid_p = float(bids[0][0])
            bid_q = float(bids[0][1])
            ask_p = float(asks[0][0])
            ask_q = float(asks[0][1])
            denom = bid_q + ask_q
            if denom <= 0:
                return (bid_p + ask_p) / 2.0
            return (ask_p * bid_q + bid_p * ask_q) / denom
        except (IndexError, ValueError, TypeError):
            return 0.0

    @staticmethod
    def calculate_depth_microprice(bids: List[List[Any]], asks: List[List[Any]], levels: int = 5) -> float:
        """
        Calculates multi-level liquidity-weighted microprice across top N levels:
        microprice_N = sum(P_ask_i * Q_bid_i + P_bid_i * Q_ask_i) / sum(Q_bid_i + Q_ask_i)
        """
        if not bids or not asks:
            return 0.0
        try:
            num_bids = min(len(bids), levels)
            num_asks = min(len(asks), levels)
            n_levels = min(num_bids, num_asks)
            if n_levels == 0:
                return 0.0

            numerator = 0.0
            denominator = 0.0
            for i in range(n_levels):
                bid_p = float(bids[i][0])
                bid_q = float(bids[i][1])
                ask_p = float(asks[i][0])
                ask_q = float(asks[i][1])
                numerator += (ask_p * bid_q + bid_p * ask_q)
                denominator += (bid_q + ask_q)

            if denominator <= 0:
                mid = (float(bids[0][0]) + float(asks[0][0])) / 2.0
                return mid
            return numerator / denominator
        except (IndexError, ValueError, TypeError):
            return 0.0

    def calculate_obi(self, bids: List[List[Any]], asks: List[List[Any]], min_wall_usd: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculates Order Book Imbalance (OBI) on N depth levels.
        bids: list of [price, quantity]
        asks: list of [price, quantity]
        Returns dict with OBI score in [-1.0, +1.0], total bid/ask volume, microprice, and wall analysis.
        """
        wall_thresh = min_wall_usd if min_wall_usd is not None else self.min_wall_usd
        if not bids or not asks:
            return {
                "obi_score": 0.0,
                "microprice": 0.0,
                "depth_microprice": 0.0,
                "total_bid_vol": 0.0,
                "total_ask_vol": 0.0,
                "bid_ask_ratio": 1.0,
                "has_bid_wall": False,
                "has_ask_wall": False,
                "wall_price": None,
                "summary": "Нейтральный стакан (нет данных)"
            }

        total_bid_vol = 0.0
        total_ask_vol = 0.0
        bid_vols = []
        ask_vols = []

        for b in bids[:20]:
            qty = float(b[1])
            total_bid_vol += qty
            bid_vols.append((float(b[0]), qty))

        for a in asks[:20]:
            qty = float(a[1])
            total_ask_vol += qty
            ask_vols.append((float(a[0]), qty))

        denom = total_bid_vol + total_ask_vol
        if denom <= 0:
            obi_score = 0.0
        else:
            obi_score = (total_bid_vol - total_ask_vol) / denom

        # Detect liquidity walls (any level with >= 35% of total side volume and USD notional >= $150,000)
        has_bid_wall = False
        bid_wall_price = None
        if len(bid_vols) >= 4 and total_bid_vol > 0:
            for p, q in bid_vols:
                if (q / total_bid_vol) >= 0.35 and (p * q) >= wall_thresh:
                    has_bid_wall = True
                    bid_wall_price = p
                    break

        has_ask_wall = False
        ask_wall_price = None
        if len(ask_vols) >= 4 and total_ask_vol > 0:
            for p, q in ask_vols:
                if (q / total_ask_vol) >= 0.35 and (p * q) >= wall_thresh:
                    has_ask_wall = True
                    ask_wall_price = p
                    break

        # Summary text
        if obi_score >= self.confirmation_threshold:
            summary = f"🟢 Доминирование покупателей (+{round(obi_score * 100, 1)}%)"
            if has_bid_wall:
                summary += f" | Плотная стенка поддержки на ${bid_wall_price:,.2f}"
        elif obi_score <= -self.confirmation_threshold:
            summary = f"🔴 Давление продавцов ({round(obi_score * 100, 1)}%)"
            if has_ask_wall:
                summary += f" | Крупная стенка сопротивления на ${ask_wall_price:,.2f}"
        else:
            summary = f"⚖️ Сбалансированный стакан ({round(obi_score * 100, 1)}%)"

        top_microprice = self.calculate_microprice(bids, asks)
        depth_microprice = self.calculate_depth_microprice(bids, asks, levels=5)

        return {
            "obi_score": round(obi_score, 4),
            "microprice": round(top_microprice, 4),
            "depth_microprice": round(depth_microprice, 4),
            "total_bid_vol": round(total_bid_vol, 4),
            "total_ask_vol": round(total_ask_vol, 4),
            "bid_ask_ratio": round(total_bid_vol / max(0.0001, total_ask_vol), 2),
            "has_bid_wall": has_bid_wall,
            "has_ask_wall": has_ask_wall,
            "bid_wall_price": bid_wall_price,
            "ask_wall_price": ask_wall_price,
            "summary": summary
        }

    def evaluate_microstructure_veto(
        self, 
        technical_direction: str, 
        obi_data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluates whether orderbook microflow vetoes the technical signal.
        Returns: (is_vetoed: bool, veto_reason: Optional[str])
        """
        obi = obi_data.get("obi_score", 0.0)
        has_ask_wall = obi_data.get("has_ask_wall", False)
        ask_wall_price = obi_data.get("ask_wall_price")
        has_bid_wall = obi_data.get("has_bid_wall", False)
        bid_wall_price = obi_data.get("bid_wall_price")

        # 1. Bullish signal vs severe sell pressure
        if technical_direction == "BULLISH":
            if obi <= -self.veto_threshold:
                return True, f"Вето стакана: жесткий перевес продавцов ({round(obi * 100, 1)}%)"
            if has_ask_wall and ask_wall_price:
                return True, f"Вето стакана: крупная стенка продавцов на ${ask_wall_price:,.2f}"

        # 2. Bearish signal vs severe buy pressure
        if technical_direction == "BEARISH":
            if obi >= self.veto_threshold:
                return True, f"Вето стакана: жесткий перевес покупателей (+{round(obi * 100, 1)}%)"
            if has_bid_wall and bid_wall_price:
                return True, f"Вето стакана: плотная стенка поддержки на ${bid_wall_price:,.2f}"

        return False, None
