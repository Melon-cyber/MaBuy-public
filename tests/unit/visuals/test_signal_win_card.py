import os
import unittest
from src.presentation.visuals.signal_win_card import AnalyticalWinCardGenerator

class TestAnalyticalWinCardGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = AnalyticalWinCardGenerator()

    def test_generate_stories_card(self):
        path = self.generator.generate_stories_card(
            symbol="BTC",
            direction="BULLISH",
            entry_price=64000.0,
            target_price=67200.0,
            pnl_pct=5.0,
            confidence_pct=91,
            rr_ratio=2.2,
            is_square=False
        )
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        self.assertGreater(os.path.getsize(path), 1000)
        
        # Cleanup
        if os.path.exists(path):
            os.remove(path)

    def test_generate_square_card(self):
        path = self.generator.generate_stories_card(
            symbol="ETH",
            direction="BEARISH",
            entry_price=3400.0,
            target_price=3200.0,
            pnl_pct=5.88,
            confidence_pct=86,
            rr_ratio=2.2,
            is_square=True
        )
        self.assertIsNotNone(path)
        self.assertTrue(os.path.exists(path))
        
        # Cleanup
        if os.path.exists(path):
            os.remove(path)

if __name__ == "__main__":
    unittest.main()
