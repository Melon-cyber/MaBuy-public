import unittest
from src.domain.entities.prediction import PredictionRequest
from src.domain.entities.tier import SubscriptionTier
from src.domain.services.access_policy import AccessPolicyService
from src.infrastructure.persistence.in_memory_tracker import InMemoryUsageTracker
from src.infrastructure.ml.baseline_model import BaselineGBDTModelEngine
from src.infrastructure.ml.titan_model_decorator import TitanModelEngineDecorator
from src.application.use_cases.get_prediction_use_case import GetPredictionUseCase


class TestMLEnginesAndUseCases(unittest.TestCase):

    def setUp(self):
        self.policy = AccessPolicyService()
        self.tracker = InMemoryUsageTracker()
        self.base_engine = BaselineGBDTModelEngine()
        self.engine = TitanModelEngineDecorator(self.base_engine)
        self.use_case = GetPredictionUseCase(
            access_policy_service=self.policy,
            usage_tracker=self.tracker,
            ml_engine=self.engine
        )

    def test_use_case_execution_pro_tier(self):
        request = PredictionRequest(
            user_id="pro_user_1",
            symbol="BTC",
            tier=SubscriptionTier.PRO
        )

        result = self.use_case.execute(request)

        self.assertEqual(result.symbol, "BTC")
        self.assertGreater(result.current_price, 0)
        self.assertIn(result.direction, ["BULLISH", "BEARISH", "NEUTRAL"])
        self.assertIsNotNone(result.personal_recommendation)
        self.assertIsNotNone(result.sentiment_score) # PRO includes NLP Sentiment Analysis
        self.assertEqual(self.tracker.get_daily_usage("pro_user_1"), 1)

    def test_use_case_execution_free_tier(self):
        request = PredictionRequest(
            user_id="free_user_1",
            symbol="BTC",
            tier=SubscriptionTier.FREE
        )

        result = self.use_case.execute(request)

        self.assertEqual(result.symbol, "BTC")
        self.assertGreater(result.current_price, 0)
        self.assertIn(result.direction, ["BULLISH", "BEARISH", "NEUTRAL"])
        self.assertEqual(self.tracker.get_daily_usage("free_user_1"), 1)


if __name__ == "__main__":
    unittest.main()
