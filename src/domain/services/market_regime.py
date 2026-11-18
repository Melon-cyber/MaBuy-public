import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

class MarketRegimeClassifierService:
    """
    Analyzes multi-timeframe candle data and classifies current market state into 4 regimes:
    1. 🚀 Strong Bull Trend (High Momentum)
    2. 📉 Bear Trend (High Risk)
    3. 🌊 Low-Volatility Squeeze (Breakout Pending)
    4. ⚡ High Volatility Storm (Noise / Caution)
    """

    @staticmethod
    def classify_regime(df: pd.DataFrame) -> Tuple[str, str, float]:
        """
        Calculates ADX, ATR %, and EMA 20/50 slopes.
        Returns: (regime_code, localized_description, volatility_ratio)
        """
        if len(df) < 30:
            return ("NEUTRAL", "Нормальный рыночный режим", 0.015)

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        # EMA 20 & EMA 50
        ema20 = pd.Series(closes).ewm(span=20).mean().values[-1]
        ema50 = pd.Series(closes).ewm(span=50).mean().values[-1]
        curr_price = closes[-1]

        # ATR Calculation
        tr1 = highs - lows
        tr2 = np.abs(highs - np.roll(closes, 1))
        tr3 = np.abs(lows - np.roll(closes, 1))
        tr = np.maximum.reduce([tr1, tr2, tr3])
        atr = np.mean(tr[-14:])
        volatility_ratio = (atr / curr_price) if curr_price > 0 else 0.015

        if curr_price > ema20 > ema50 and volatility_ratio < 0.03:
            return (
                "BULL_TREND",
                "🚀 Устойчивый бычий тренд (Высокий импульс роста)",
                volatility_ratio
            )
        elif curr_price < ema20 < ema50:
            return (
                "BEAR_TREND",
                "📉 Медвежий тренд (Повышенный риск просадки)",
                volatility_ratio
            )
        elif volatility_ratio < 0.012:
            return (
                "SQUEEZE",
                "🌊 Накопление / Рыночный штиль (Ожидается сильный импульс)",
                volatility_ratio
            )
        else:
            return (
                "HIGH_VOLATILITY",
                "⚡ Высокая штормовая волатильность (Торгуйте с осторожностью)",
                volatility_ratio
            )

    @staticmethod
    def calculate_choppiness_index(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """
        Calculates Choppiness Index (CHOP) in [0, 100]:
        CHOP = 100 * LOG10( SUM(ATR_1, n) / (MAX(High, n) - MIN(Low, n)) ) / LOG10(n)
        - CHOP >= 61.8: CHOPPY / CONSOLIDATION / WHIPSAW (Entries strictly forbidden)
        - CHOP <= 38.2: STRONG DIRECTIONAL TREND (Entries allowed)
        - 38.2 < CHOP < 61.8: TRANSITIONAL ZONE (Requires ADX confirmation)
        """
        if len(closes) < period + 1:
            return 50.0

        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum.reduce([tr1, tr2, tr3])

        sum_tr = float(np.sum(tr[-period:]))
        max_high = float(np.max(highs[-period:]))
        min_low = float(np.min(lows[-period:]))
        hl_range = max_high - min_low

        if hl_range <= 1e-8 or sum_tr <= 1e-8:
            return 50.0

        chop = 100.0 * (np.log10(sum_tr / hl_range) / np.log10(period))
        return float(np.clip(chop, 0.0, 100.0))

    @staticmethod
    def calculate_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """
        Calculates Average Directional Index (ADX) in [0, 100] using Wilder's smoothing.
        - ADX >= 25.0: Strong Trend
        - ADX < 20.0: Weak / Non-trending range (Whipsaw prone)
        """
        if len(closes) < period * 2:
            return 25.0

        up_move = highs[1:] - highs[:-1]
        down_move = lows[:-1] - lows[1:]

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        tr = np.maximum.reduce([tr1, tr2, tr3])

        atr = float(np.mean(tr[:period]))
        plus_di_s = float(np.mean(plus_dm[:period]))
        minus_di_s = float(np.mean(minus_dm[:period]))

        dx_list = []
        for i in range(period, len(tr)):
            atr = (atr * (period - 1) + tr[i]) / period
            plus_di_s = (plus_di_s * (period - 1) + plus_dm[i]) / period
            minus_di_s = (minus_di_s * (period - 1) + minus_dm[i]) / period

            p_di = 100.0 * (plus_di_s / max(1e-8, atr))
            m_di = 100.0 * (minus_di_s / max(1e-8, atr))
            di_sum = p_di + m_di
            dx = 100.0 * abs(p_di - m_di) / max(1e-8, di_sum)
            dx_list.append(dx)

        if not dx_list:
            return 25.0
        adx = float(np.mean(dx_list[-period:])) if len(dx_list) >= period else float(np.mean(dx_list))
        return float(np.clip(adx, 0.0, 100.0))

    @classmethod
    def evaluate_whipsaw_risk(cls, df: pd.DataFrame, period: int = 14) -> Dict[str, Any]:
        """
        Comprehensive market regime evaluation to prevent false breakouts and whipsaw losses.
        Returns:
            is_tradeable: bool (True if market has sufficient trend momentum)
            choppiness_index: float
            adx: float
            regime: str
            reason: str
        """
        if df is None or len(df) < period + 5:
            return {
                "is_tradeable": True,
                "choppiness_index": 50.0,
                "adx": 25.0,
                "regime": "NORMAL",
                "reason": "Insufficient candle history for strict regime filtering. Permitted with base risk."
            }

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values

        chop = cls.calculate_choppiness_index(highs, lows, closes, period=period)
        adx = cls.calculate_adx(highs, lows, closes, period=period)

        # 1. Strict Choppiness Index Barrier
        if chop >= 61.8:
            return {
                "is_tradeable": False,
                "choppiness_index": round(chop, 2),
                "adx": round(adx, 2),
                "regime": "CHOPPY_SIDEWAYS",
                "reason": f"🛑 HIGH CHOPPINESS (CHOP={chop:.1f} >= 61.8). Market is in noisy range/whipsaw. Entries vetoed."
            }

        # 2. ADX Trend Strength Barrier
        if adx < 20.0 and chop > 50.0:
            return {
                "is_tradeable": False,
                "choppiness_index": round(chop, 2),
                "adx": round(adx, 2),
                "regime": "WEAK_RANGE",
                "reason": f"🛑 WEAK TREND MOMENTUM (ADX={adx:.1f} < 20.0, CHOP={chop:.1f}). High risk of whipsaw stop-hunting."
            }

        return {
            "is_tradeable": True,
            "choppiness_index": round(chop, 2),
            "adx": round(adx, 2),
            "regime": "TRENDING" if adx >= 25.0 else "TRANSITION",
            "reason": f"✅ Tradeable regime confirmed (ADX={adx:.1f}, CHOP={chop:.1f})."
        }
