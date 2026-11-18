from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional


class SubscriptionTier(str, Enum):
    FREE = "free"
    PRO = "pro"                  # MaPredict PRO (Day/Swing trading, 1h/4h signals, OFI, Kelly)
    MAINVEST = "mainvest"        # MaInvest WEALTH (Long-term Macro, 30d/90d/365d, Equities, Gold, DCA)
    ALL_ACCESS = "all_access"    # MaBuy All-Access BUNDLE (Both products discounted)


@dataclass(frozen=True)
class TierLimits:
    """
    Encapsulates subscription tier limits, permissions, and feature availability.
    Enforces distinct products: Free, MaPredict PRO ($14.99), MaInvest WEALTH ($14.99), and All-Access Bundle ($24.99).
    """
    tier: SubscriptionTier
    daily_prediction_limit: int                      # Max allowed predictions per day
    allowed_assets: Optional[Tuple[str, ...]] = None # None means all assets allowed
    has_personal_recommendations: bool = False        # Whether personal AI recommendations are included
    has_sentiment_analysis: bool = False              # Whether news/social sentiment NLP analysis is enabled
    use_advanced_model: bool = False                  # Whether to use advanced V4 Titan Ensemble vs baseline
    priority_support: bool = False                    # 24/7 priority support flag
    daily_ai_limit: int = 5                          # Max allowed MaIA AI copilot queries per day

    def is_asset_allowed(self, symbol: str) -> bool:
        """Checks if a given crypto asset symbol is permitted under this tier."""
        if self.allowed_assets is None:
            return True
        return symbol.upper() in [asset.upper() for asset in self.allowed_assets]


# Default Tier Configurations for Simplified SaaS Model
TOP_3_CRYPTOS: Tuple[str, ...] = ("BTC", "ETH", "BNB")

TIER_CONFIGURATIONS = {
    SubscriptionTier.FREE: TierLimits(
        tier=SubscriptionTier.FREE,
        daily_prediction_limit=2,
        daily_ai_limit=5,             # 5 AI queries/day for Free users
        allowed_assets=TOP_3_CRYPTOS,
        has_personal_recommendations=False,
        has_sentiment_analysis=False,
        use_advanced_model=False,
        priority_support=False
    ),
    SubscriptionTier.PRO: TierLimits(
        tier=SubscriptionTier.PRO,
        daily_prediction_limit=50,
        daily_ai_limit=10,            # 10 AI queries/day for MaPredict PRO ($14.99)
        allowed_assets=None,          # ALL assets allowed in MaPredict PRO
        has_personal_recommendations=True,
        has_sentiment_analysis=True,  # NLP Sentiment Alpha in PRO
        use_advanced_model=True,      # V4 Titan Ensemble Engine in PRO
        priority_support=True
    ),
    SubscriptionTier.MAINVEST: TierLimits(
        tier=SubscriptionTier.MAINVEST,
        daily_prediction_limit=50,
        daily_ai_limit=10,            # 10 AI queries/day for MaInvest WEALTH ($14.99)
        allowed_assets=None,          # Crypto + Equities + Gold/SPY in MaInvest WEALTH
        has_personal_recommendations=True,
        has_sentiment_analysis=True,
        use_advanced_model=True,      # Merton Jump Diffusion & Monte Carlo SDE
        priority_support=True
    ),
    SubscriptionTier.ALL_ACCESS: TierLimits(
        tier=SubscriptionTier.ALL_ACCESS,
        daily_prediction_limit=999999,  # Unlimited in All-Access Bundle ($24.99)
        daily_ai_limit=999999,          # Unlimited ∞ AI queries for MaBuy All-Access
        allowed_assets=None,
        has_personal_recommendations=True,
        has_sentiment_analysis=True,
        use_advanced_model=True,
        priority_support=True
    )
}
