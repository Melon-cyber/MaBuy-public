import unittest
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from src.infrastructure.ml.training.validation import WalkForwardValidator


class TestWalkForwardValidator(unittest.TestCase):

    def test_evaluator(self):
        np.random.seed(42)
        X = np.random.randn(500, 10)
        y = 0.05 * X[:, 0] - 0.02 * X[:, 1] + np.random.normal(0, 0.01, size=500)

        validator = WalkForwardValidator(n_splits=3)
        model = HistGradientBoostingRegressor(random_state=42)

        summary, df_metrics = validator.evaluate(model, X, y)

        self.assertEqual(len(df_metrics), 3)
        self.assertIn("mean_directional_accuracy", summary)
        self.assertGreater(summary["mean_directional_accuracy"], 0.0)
        self.assertLessEqual(summary["mean_directional_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
