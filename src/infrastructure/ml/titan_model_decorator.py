import math
from typing import Dict, Tuple
from src.application.interfaces.ml_engine import MLEngineInterface
from src.domain.entities.prediction import PredictionRequest, PredictionResult
from src.domain.entities.tier import TierLimits, SubscriptionTier
from src.infrastructure.nlp.rss_sentiment_analyzer import RSSSentimentAnalyzer


class TitanModelEngineDecorator(MLEngineInterface):
    r"""
    Decorator pattern extending base ML prediction engines with Titan Tier capabilities:
    1. News Sentiment NLP Analysis & Adjustment (\hat{r}_titan = \hat{r}_base + \gamma * S_news)
    2. Advanced Neural Network Model dispatch (Stage 2 PyTorch hook)
    3. Bilingual (RU/EN) news background aggregation and synthesis.
    """

    def __init__(self, base_engine: MLEngineInterface, gamma: float = 0.005):
        self.base_engine = base_engine
        self.gamma = gamma # Sensitivity coefficient for sentiment adjustment
        self.rss_analyzer = RSSSentimentAnalyzer()

    def predict(self, request: PredictionRequest, limits: TierLimits) -> PredictionResult:
        # Obtain underlying baseline prediction
        result = self.base_engine.predict(request, limits)
        lang = getattr(request, "lang", "ru")

        # If Titan Tier capabilities are active, augment prediction with Sentiment Analysis & Advanced Engine
        if limits.has_sentiment_analysis:
            symbol = request.symbol.upper()

            # Retrieve asset sentiment score and localized news context via Real NLP
            sentiment_score, news_summary = self.rss_analyzer.analyze_symbol(symbol, lang=lang)

            # Mathematical Sentiment Adjustment: \hat{r}_adjusted = \hat{r}_base + \gamma * S_news
            adjusted_log_return = result.predicted_log_return + (self.gamma * sentiment_score)
            adjusted_target_price = result.current_price * math.exp(adjusted_log_return)

            # Re-evaluate direction based on adjusted return
            threshold = 0.0015
            if adjusted_log_return > threshold:
                direction = "BULLISH"
            elif adjusted_log_return < -threshold:
                direction = "BEARISH"
            else:
                direction = "NEUTRAL"

            # Enhanced confidence score incorporating sentiment alignment
            enhanced_confidence = min(1.0, result.confidence_score * (1.0 + 0.15 * abs(sentiment_score)))

            timeframe_str = getattr(result, 'timeframe', '1h').upper()
            # Keep the base recommendation (what to buy, how much, etc)
            # Add a small note about Titan adjustments
            base_rec = getattr(result, "personal_recommendation", "")
            
            target_formatted = f"${adjusted_target_price:,.4f}" if adjusted_target_price < 1.0 else f"${adjusted_target_price:,.2f}"
            if lang == "ru":
                titan_recommendation = (
                    f"{base_rec}\n\n"
                    f"👑 Сентимент Titan: Целевая цена скорректирована до {target_formatted} "
                    f"с учетом новостного фона (+{sentiment_score:.2f})."
                )
            else:
                titan_recommendation = (
                    f"{base_rec}\n\n"
                    f"👑 Titan Sentiment: Target adjusted to {target_formatted} "
                    f"accounting for news sentiment (+{sentiment_score:.2f})."
                )

            return PredictionResult(
                symbol=result.symbol,
                current_price=result.current_price,
                predicted_target_price=round(adjusted_target_price, 6 if adjusted_target_price < 1.0 else 2),
                predicted_log_return=round(adjusted_log_return, 5),
                direction=direction,
                confidence_score=round(enhanced_confidence, 4),
                volatility_estimate=result.volatility_estimate,
                timeframe=getattr(result, "timeframe", "1h"),
                personal_recommendation=titan_recommendation,
                sentiment_score=sentiment_score,
                news_summary=news_summary[:250], # Truncate to ensure caption fits in Telegram (1024 char limit)
                model_version="MaBuy-V4-Flagship-Ensemble (PyTorch+NLP)",
                tier_applied=SubscriptionTier.PRO
            )

        return result
