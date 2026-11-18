import unittest
from src.domain.entities.tier import SubscriptionTier
from src.domain.entities.prediction import PredictionRequest
from src.domain.services.access_policy import (
    AccessPolicyService,
    AssetNotAllowedException,
    DailyLimitExceededException,
)


class TestTierPolicy(unittest.TestCase):

    def setUp(self):
        self.policy = AccessPolicyService()

    def test_free_tier_asset_restriction(self):
        req_btc = PredictionRequest(user_id="user_1", symbol="BTC", tier=SubscriptionTier.FREE)
        limits = self.policy.validate_request(req_btc, current_daily_usage=0)
        self.assertEqual(limits.tier, SubscriptionTier.FREE)

        req_sol = PredictionRequest(user_id="user_1", symbol="SOL", tier=SubscriptionTier.FREE)
        with self.assertRaises(AssetNotAllowedException):
            self.policy.validate_request(req_sol, current_daily_usage=0)

    def test_free_tier_daily_limit(self):
        req = PredictionRequest(user_id="user_2", symbol="BTC", tier=SubscriptionTier.FREE)
        limits = self.policy.validate_request(req, current_daily_usage=1)
        self.assertEqual(limits.daily_prediction_limit, 2)

        with self.assertRaises(DailyLimitExceededException):
            self.policy.validate_request(req, current_daily_usage=2)

    def test_pro_tier_full_capabilities(self):
        limits = self.policy.get_tier_limits(SubscriptionTier.PRO)
        self.assertEqual(limits.daily_prediction_limit, 50)
        self.assertEqual(limits.daily_ai_limit, 10)
        self.assertTrue(limits.has_personal_recommendations)
        self.assertTrue(limits.has_sentiment_analysis)
        self.assertTrue(limits.use_advanced_model)

    def test_ai_query_limits(self):
        free_limits = self.policy.get_tier_limits(SubscriptionTier.FREE)
        self.assertEqual(free_limits.daily_ai_limit, 5)

        pro_limits = self.policy.get_tier_limits(SubscriptionTier.PRO)
        self.assertEqual(pro_limits.daily_ai_limit, 10)

        invest_limits = self.policy.get_tier_limits(SubscriptionTier.MAINVEST)
        self.assertEqual(invest_limits.daily_ai_limit, 10)

        all_access_limits = self.policy.get_tier_limits(SubscriptionTier.ALL_ACCESS)
        self.assertTrue(all_access_limits.daily_ai_limit > 1000)

    def test_admin_vip_privilege_override(self):
        admin_req = PredictionRequest(user_id="100000001", symbol="SOL", tier=SubscriptionTier.FREE, timeframe="1d")
        limits = self.policy.validate_request(admin_req, current_daily_usage=999)
        self.assertEqual(limits.tier, SubscriptionTier.PRO)
        self.assertTrue(limits.daily_prediction_limit > 1000)
        self.assertTrue(limits.has_personal_recommendations)
        self.assertTrue(limits.has_sentiment_analysis)


if __name__ == "__main__":
    unittest.main()
