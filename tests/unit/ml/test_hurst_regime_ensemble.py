import unittest
import numpy as np

from src.domain.entities.prediction import PredictionRequest
from src.domain.entities.tier import SubscriptionTier
from src.infrastructure.ml.v4_titan_ensemble import HurstRegimeDetector, MaBuyV4EnsembleEngine


class TestHurstRegimeEnsemble(unittest.TestCase):
    def test_hurst_trending_series(self):
        """Persistent autocorrelated momentum returns should produce Hurst H > 0.55."""
        np.random.seed(42)
        n = 200
        # AR(1) momentum returns with persistence phi = 0.70
        returns = [0.005]
        for _ in range(n - 1):
            returns.append(0.70 * returns[-1] + np.random.normal(0, 0.002))
        prices = 100.0 * np.exp(np.cumsum(returns))
        
        h = HurstRegimeDetector.calculate_hurst_from_prices(prices)
        self.assertGreater(h, 0.55)
        
        base_w = {"gbdt": 0.45, "pytorch": 0.45, "sentiment": 0.05, "drift": 0.05, "threshold": 0.006}
        cfg = HurstRegimeDetector.get_adaptive_weights(h, base_w)
        
        self.assertEqual(cfg["regime"], "TREND")
        # In trend, PyTorch Attention LSTM weight should be boosted over GBDT
        self.assertGreater(cfg["weights"]["pytorch"], cfg["weights"]["gbdt"])
        self.assertGreaterEqual(cfg["entropy_gate"], 0.40)

    def test_hurst_mean_reverting_series(self):
        """Anti-persistent mean-reverting returns should produce Hurst H < 0.45."""
        # Mean-reverting oscillating series: prices ping-ponging tightly
        prices = [100.0 if i % 2 == 0 else 101.5 for i in range(100)]
        
        h = HurstRegimeDetector.calculate_hurst_from_prices(prices)
        self.assertLess(h, 0.45)
        
        base_w = {"gbdt": 0.45, "pytorch": 0.45, "sentiment": 0.05, "drift": 0.05, "threshold": 0.006}
        cfg = HurstRegimeDetector.get_adaptive_weights(h, base_w)
        
        self.assertEqual(cfg["regime"], "MEAN_REVERTING")
        # In mean-reversion, GBDT weight should be boosted over PyTorch
        self.assertGreater(cfg["weights"]["gbdt"], cfg["weights"]["pytorch"])

    def test_hurst_random_walk_gating(self):
        """Random walk should have H near 0.50 and trigger entropy capital gating."""
        h = 0.50
        base_w = {"gbdt": 0.45, "pytorch": 0.45, "sentiment": 0.05, "drift": 0.05, "threshold": 0.006}
        cfg = HurstRegimeDetector.get_adaptive_weights(h, base_w)
        
        self.assertEqual(cfg["regime"], "CHOPPY_NOISE")
        self.assertAlmostEqual(cfg["weights"]["pytorch"], cfg["weights"]["gbdt"], places=2)
        # Risk must be gated down to 0.25
        self.assertAlmostEqual(cfg["entropy_gate"], 0.25, places=2)

    def test_v4_ensemble_prediction_includes_regime_and_hurst(self):
        """V4 Ensemble prediction should populate market_regime and hurst_exponent."""
        engine = MaBuyV4EnsembleEngine()
        req = PredictionRequest(
            user_id="hurst_tester",
            symbol="BTC",
            tier=SubscriptionTier.PRO,
            timeframe="1h",
            lang="ru"
        )
        res = engine.predict(req)
        self.assertIsNotNone(res.market_regime)
        self.assertIn(res.market_regime, ["TREND", "MEAN_REVERTING", "CHOPPY_NOISE"])
        self.assertIsNotNone(res.hurst_exponent)
        self.assertGreaterEqual(res.hurst_exponent, 0.05)
        self.assertLessEqual(res.hurst_exponent, 0.95)
        self.assertIsNotNone(res.personal_recommendation)
        self.assertGreater(len(res.personal_recommendation), 0)


if __name__ == "__main__":
    unittest.main()
