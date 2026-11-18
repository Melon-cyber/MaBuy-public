from abc import ABC, abstractmethod
from src.domain.entities.prediction import PredictionRequest, PredictionResult
from src.domain.entities.tier import TierLimits


class MLEngineInterface(ABC):
    """
    Port (Interface) for MaBuy Machine Learning Prediction Engines.
    Allows seamless swapping of Stage 1 Baseline (Scikit-Learn GBDT), Stage 2 (PyTorch), or Titan NLP engines.
    """

    @abstractmethod
    def predict(self, request: PredictionRequest, limits: TierLimits) -> PredictionResult:
        """
        Executes prediction pipeline and returns structured PredictionResult.
        """
        pass
