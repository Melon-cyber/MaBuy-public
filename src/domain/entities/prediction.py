from dataclasses import dataclass
from typing import Optional, Dict, Any
from src.domain.entities.tier import SubscriptionTier


@dataclass
class PredictionRequest:
    """
    Encapsulates a user query for cryptocurrency price analysis and prediction.
    Supports multi-timeframe horizons: '1h', '4h', '1d' and bilingual localization ('ru', 'en').
    """
    user_id: str
    symbol: str
    tier: SubscriptionTier
    timeframe: str = "1h"
    lang: str = "ru"
    user_portfolio_context: Optional[Dict[str, Any]] = None


@dataclass
class PredictionResult:
    """
    Represents the output of the MaBuy Prediction Engine, tailored according to the user's tier.
    """
    symbol: str
    current_price: float
    predicted_target_price: float
    predicted_log_return: float
    direction: str                             # "BULLISH", "BEARISH", "NEUTRAL"
    confidence_score: float                   # Range [0.0, 1.0]
    volatility_estimate: float                # Rolling standard deviation or tree variance
    timeframe: str = "1h"                     # "1h", "4h", "1d"
    personal_recommendation: Optional[str] = None # Pro & Titan Tiers
    sentiment_score: Optional[float] = None       # Titan Tier Only (-1.0 to +1.0)
    news_summary: Optional[str] = None           # Titan Tier Only
    orderbook_imbalance: Optional[float] = None  # Range [-1.0, +1.0] (OFI Microstructure)
    orderbook_summary: Optional[str] = None      # Wall / Orderbook summary
    market_regime: Optional[str] = None          # "TREND", "MEAN_REVERTING", "CHOPPY_NOISE"
    hurst_exponent: Optional[float] = None       # Hurst Exponent H in [0.05, 0.95]
    model_version: str = "MaBuy-GBDT-Baseline-v1"
    tier_applied: SubscriptionTier = SubscriptionTier.FREE
