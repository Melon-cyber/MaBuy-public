from src.domain.entities.prediction import PredictionRequest, PredictionResult
from src.domain.services.access_policy import AccessPolicyService
from src.application.interfaces.usage_tracker import DailyUsageTrackerInterface
from src.application.interfaces.ml_engine import MLEngineInterface


class GetPredictionUseCase:
    """
    Main Application Use Case: Orchestrates tier validation, usage rate limiting,
    ML model execution, payload filtering, and usage counter incrementing.
    """

    def __init__(
        self,
        access_policy_service: AccessPolicyService,
        usage_tracker: DailyUsageTrackerInterface,
        ml_engine: MLEngineInterface,
    ):
        self.policy_service = access_policy_service
        self.usage_tracker = usage_tracker
        self.ml_engine = ml_engine

    def execute(self, request: PredictionRequest) -> PredictionResult:
        """
        Executes user prediction flow:
        1. Query current daily request count.
        2. Validate tier access limits and asset permissions.
        3. Invoke ML Engine for price prediction and analysis.
        4. Apply tier payload filtering (strip non-unlocked features).
        5. Increment user daily usage counter.
        """
        current_usage = self.usage_tracker.get_daily_usage(request.user_id)
        limits = self.policy_service.validate_request(request, current_usage)

        # Run prediction engine
        raw_prediction = self.ml_engine.predict(request, limits)

        # Tailor payload according to subscription tier
        final_prediction = self.policy_service.filter_response_payload(raw_prediction, limits)

        # Increment usage counter only upon successful prediction
        self.usage_tracker.increment_usage(request.user_id)

        return final_prediction
