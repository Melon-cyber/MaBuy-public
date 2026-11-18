import unittest
import numpy as np

from src.presentation.telegram_bot.correlation_heatmap_generator import CorrelationHeatmapGenerator
from src.domain.services.correlation_matrix import MarketCorrelationMatrixService

class TestCorrelationHeatmap(unittest.TestCase):
    def test_generate_heatmap_bytes(self):
        symbols = ["BTC", "ETH", "SOL", "BNB"]
        corr_matrix = np.array([
            [1.0, 0.85, 0.72, 0.65],
            [0.85, 1.0, 0.78, 0.70],
            [0.72, 0.78, 1.0, 0.55],
            [0.65, 0.70, 0.55, 1.0]
        ])
        
        img_bytes = CorrelationHeatmapGenerator.generate_heatmap_bytes(
            symbols=symbols,
            corr_matrix=corr_matrix,
            title="TEST MATRIX"
        )
        
        self.assertIsInstance(img_bytes, bytes)
        self.assertGreater(len(img_bytes), 5000) # Valid PNG is usually > 50KB
        self.assertTrue(img_bytes.startswith(b'\x89PNG\r\n\x1a\n'))

    def test_correlation_matrix_service_output(self):
        data = MarketCorrelationMatrixService.get_correlation_matrix(lang="ru", include_image=True)
        self.assertIn("symbols", data)
        self.assertEqual(len(data["symbols"]), 8)
        self.assertIn("matrix", data)
        self.assertIn("pairs_summary", data)
        self.assertIn("stress_disclosure", data)
        self.assertIn("Black Swan", data["stress_disclosure"])
        self.assertIsNotNone(data.get("heatmap_bytes"))
        self.assertTrue(data["heatmap_bytes"].startswith(b'\x89PNG\r\n\x1a\n'))

if __name__ == "__main__":
    unittest.main()
