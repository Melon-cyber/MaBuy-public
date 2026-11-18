import os
from typing import List, Optional, Set
from src.domain.entities.tier import SubscriptionTier, TierLimits, TIER_CONFIGURATIONS
from src.domain.entities.prediction import PredictionRequest, PredictionResult
from src.domain.services.symbol_normalizer import SymbolNormalizer


class AccessPolicyException(Exception):
    """Base exception for Access Policy Guard violations."""
    pass


class DailyLimitExceededException(AccessPolicyException):
    """Raised when a user exceeds their tier's daily prediction request quota."""
    pass


class AssetNotAllowedException(AccessPolicyException):
    """Raised when a user requests an asset restricted under their subscription tier."""
    pass


class AccessPolicyService:
    """
    Domain Service enforcing subscription tier business policies.
    Guarantees asset access boundaries, daily request rate limits,
    payload sanitization, and Admin / VIP privilege overrides.
    """

    ADMIN_USER_IDS: Set[str] = {"100000001", "owner_admin", "admin", "alert_engine_scanner", "system_scanner", "channel_broadcast_bot", "broadcast_bot", "system"}

    def __init__(self):
        self.tier_config = TIER_CONFIGURATIONS
        # Load additional admin IDs from environment if set
        extra_admins = os.environ.get("ADMIN_TELEGRAM_IDS", "")
        if extra_admins:
            for admin_id in extra_admins.split(","):
                clean_id = admin_id.strip()
                if clean_id:
                    self.ADMIN_USER_IDS.add(clean_id)

    def is_admin(self, user_id: str) -> bool:
        """Checks if a user ID possesses permanent Admin privileges."""
        return str(user_id).strip() in self.ADMIN_USER_IDS

    def get_tier_limits(self, tier: SubscriptionTier, user_id: Optional[str] = None) -> TierLimits:
        """Returns limits; if user is Admin, returns Titan limits with unlimited daily quota."""
        if user_id and self.is_admin(user_id):
            pro_limits = self.tier_config[SubscriptionTier.PRO]
            return TierLimits(
                tier=SubscriptionTier.PRO,
                daily_prediction_limit=999999,  # Infinite quota for Admin
                daily_ai_limit=999999,          # Infinite AI quota for Admin
                allowed_assets=None,            # All assets
                has_personal_recommendations=True,
                has_sentiment_analysis=True,
                use_advanced_model=True,
                priority_support=True
            )

        return self.tier_config.get(tier, self.tier_config[SubscriptionTier.FREE])

    def validate_request(self, request: PredictionRequest, current_daily_usage: int) -> TierLimits:
        """Validates prediction request against subscription limits, asset guards, and quotas."""
        # 0. Admin Privilege Override
        if self.is_admin(request.user_id):
            return self.get_tier_limits(SubscriptionTier.PRO, user_id=request.user_id)

        limits = self.get_tier_limits(request.tier)

        # 1. Quota Check
        if current_daily_usage >= limits.daily_prediction_limit:
            raise DailyLimitExceededException(
                f"User [{request.user_id}] has exceeded daily prediction quota for '{request.tier.value}' tier "
                f"({current_daily_usage}/{limits.daily_prediction_limit}). Upgrade tier to get more requests."
            )

        # 2. Asset Access Guard
        normalized_symbol = SymbolNormalizer.normalize(request.symbol)
        if not limits.is_asset_allowed(normalized_symbol):
            allowed_str = ", ".join(limits.allowed_assets) if limits.allowed_assets else "All Cryptos"
            raise AssetNotAllowedException(
                f"Asset '{request.symbol}' is not available for '{request.tier.value}' tier. "
                f"Allowed assets: [{allowed_str}]. Upgrade to Amateur, Pro, or Titan."
            )

        return limits

    def filter_response_payload(self, result: PredictionResult, limits: TierLimits) -> PredictionResult:
        """
        Filters and sanitizes the prediction result according to the user's tier permissions.
        Redacts personal AI advice and news sentiment for tiers where features are locked.
        """
        rec = result.personal_recommendation if limits.has_personal_recommendations else None
        sent = result.sentiment_score if limits.has_sentiment_analysis else None
        summary = result.news_summary if limits.has_sentiment_analysis else None

        return PredictionResult(
            symbol=result.symbol,
            current_price=result.current_price,
            predicted_target_price=result.predicted_target_price,
            predicted_log_return=result.predicted_log_return,
            direction=result.direction,
            confidence_score=result.confidence_score,
            volatility_estimate=result.volatility_estimate,
            timeframe=getattr(result, "timeframe", "1h"),
            personal_recommendation=rec,
            sentiment_score=sent,
            news_summary=summary,
            model_version=result.model_version,
            tier_applied=limits.tier
        )
