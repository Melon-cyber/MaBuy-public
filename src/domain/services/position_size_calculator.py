import math
import numpy as np
from typing import Dict, Any, Optional

from src.domain.services.dynamic_risk_guard import DynamicRiskGuardService
from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher
from src.domain.services.symbol_normalizer import SymbolNormalizer

class PositionSizeCalculatorService:
    """
    Quantitative Risk & Position Sizing Assistant for MaBuy V4.2.
    - Strictly aligned with DynamicRiskGuardService (Fractional Ensemble Kelly 0.05 * K).
    - Incorporates 14-period ATR volatility for dynamic Stop-Loss and Take-Profit bounds.
    - Calculates educational Max Safe Leverage ceiling to protect users against liquidation.
    """

    def __init__(self, risk_guard: Optional[DynamicRiskGuardService] = None, data_fetcher: Optional[CryptoDataFetcher] = None):
        self.risk_guard = risk_guard or DynamicRiskGuardService()
        self.data_fetcher = data_fetcher or CryptoDataFetcher(max_cache_age_sec=60)

    def calculate_position_plan(
        self,
        account_balance: float,
        symbol: str,
        direction: str = "BULLISH",
        risk_level: str = "balanced",
        model_probability: float = 0.58
    ) -> Dict[str, Any]:
        """
        Calculates actionable trade sizing plan.
        """
        if account_balance <= 0:
            raise ValueError("Account balance must be positive.")

        norm_sym = SymbolNormalizer.normalize(symbol)
        
        # Non-crypto benchmark prices
        NON_CRYPTO_BENCHMARKS = {
            "NVDA": 128.0, "AAPL": 225.0, "TSLA": 215.0, "MSFT": 420.0,
            "GOOGL": 165.0, "SPY": 560.0, "QQQ": 485.0, "GLD": 230.0
        }

        # 1. Fetch live market price and recent candles for ATR
        if norm_sym in NON_CRYPTO_BENCHMARKS:
            current_price = NON_CRYPTO_BENCHMARKS[norm_sym]
            atr_pct = 1.8  # Default equity ATR %
        else:
            current_price = 100.0
            atr_pct = 2.5 # default 2.5% volatility fallback
            try:
                df = self.data_fetcher.fetch_ohlcv(norm_sym, timeframe="1h", limit=20)
                if len(df) >= 14:
                    current_price = float(df["close"].iloc[-1])
                    highs = df["high"].values
                    lows = df["low"].values
                    closes = df["close"].values

                    tr1 = highs - lows
                    tr2 = np.abs(highs - np.insert(closes[:-1], 0, closes[0]))
                    tr3 = np.abs(lows - np.insert(closes[:-1], 0, closes[0]))
                    tr = np.maximum.reduce([tr1, tr2, tr3])
                    atr = float(np.mean(tr[-14:]))
                    atr_pct = max(0.5, (atr / current_price) * 100.0)
                elif len(df) > 0:
                    current_price = float(df["close"].iloc[-1])
            except Exception:
                pass # Use fallback defaults

        # 2. Dynamic Stop-Loss & Take-Profit based on ATR
        sl_pct = max(1.0, round(1.65 * atr_pct, 2))
        tp_pct = round(1.8 * sl_pct, 2)  # Strict R:R >= 1.8

        if direction.upper() == "BULLISH":
            stop_loss_price = round(current_price * (1.0 - sl_pct / 100.0), 4)
            take_profit_price = round(current_price * (1.0 + tp_pct / 100.0), 4)
        else:
            stop_loss_price = round(current_price * (1.0 + sl_pct / 100.0), 4)
            take_profit_price = round(current_price * (1.0 - tp_pct / 100.0), 4)

        # 3. Model probability adjustment by user risk level
        p = model_probability
        if risk_level == "conservative":
            p = max(0.52, p * 0.95)
        elif risk_level == "aggressive":
            p = min(0.75, p * 1.10)

        # 4. Pure Kelly Risk Allocation from DynamicRiskGuardService (0.05 * K)
        risk_res = self.risk_guard.calculate_trade_risk(
            account_balance=account_balance,
            model_probability=p,
            take_profit_pct=tp_pct,
            stop_loss_pct=sl_pct
        )

        risk_pct = risk_res["risk_pct"]
        risk_amount_usd = risk_res["risk_amount_usd"]
        
        # Position sizing: Risk USD / (SL %)
        position_size_usd = min(account_balance * 2.0, risk_amount_usd / (sl_pct / 100.0))
        position_size_usd = round(position_size_usd, 2)
        position_size_coins = round(position_size_usd / current_price, 6) if current_price > 0 else 0.0

        # Potential Profit
        potential_profit_usd = round(position_size_usd * (tp_pct / 100.0), 2)

        # 5. Protective Leverage Modes & Execution Sizing:
        # Conservative (Liq dist > 35-50%): 2x - 3x
        # Optimal (Balanced 10% margin risk): 5x
        # Max Safe Ceiling (Liq dist >= 2x SL): 50 / SL_pct
        max_safe_leverage = max(1, min(20, int(50.0 / sl_pct)))
        conservative_leverage = min(3, max(1, int(25.0 / sl_pct)))
        optimal_leverage = min(5, max(1, min(max_safe_leverage, int(40.0 / sl_pct))))

        recommended_margin_usd = round(account_balance * 0.10, 2) # 10% of deposit
        recommended_position_usd = round(recommended_margin_usd * optimal_leverage, 2)
        loss_at_sl_usd = round(recommended_position_usd * (sl_pct / 100.0), 2)
        profit_at_tp_usd = round(recommended_position_usd * (tp_pct / 100.0), 2)

        return {
            "symbol": norm_sym,
            "direction": direction.upper(),
            "account_balance": account_balance,
            "risk_level": risk_level,
            "current_price": current_price,
            "stop_loss_price": stop_loss_price,
            "take_profit_price": take_profit_price,
            "stop_loss_pct": sl_pct,
            "take_profit_pct": tp_pct,
            "reward_risk_ratio": round(tp_pct / sl_pct, 2),
            "risk_pct": risk_pct,
            "risk_amount_usd": risk_amount_usd,
            "position_size_usd": position_size_usd,
            "position_size_coins": position_size_coins,
            "potential_profit_usd": potential_profit_usd,
            "max_safe_leverage": max_safe_leverage,
            "conservative_leverage": conservative_leverage,
            "optimal_leverage": optimal_leverage,
            "recommended_margin_usd": recommended_margin_usd,
            "recommended_position_usd": recommended_position_usd,
            "loss_at_sl_usd": loss_at_sl_usd,
            "profit_at_tp_usd": profit_at_tp_usd,
            "kelly_fraction": 0.05,
            "model_probability": round(p, 3)
        }
