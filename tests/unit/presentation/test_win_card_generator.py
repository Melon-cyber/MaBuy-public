import unittest
import os
from src.presentation.telegram_bot.win_card_generator import ShareableWinCardGenerator

class TestWinCardGenerator(unittest.TestCase):
    def test_generate_win_card_bytes_paper_trading(self):
        png_bytes = ShareableWinCardGenerator.generate_win_card_bytes(
            symbol="SOL",
            pnl_pct=5.42,
            entry_price=145.0,
            exit_price=152.85,
            is_paper_trade=True,
            winrate_pct=75.0,
            total_trades=16,
            user_id="100000001"
        )
        self.assertIsInstance(png_bytes, bytes)
        self.assertGreater(len(png_bytes), 1000)
        # Check PNG header signature \x89PNG\r\n\x1a\n
        self.assertEqual(png_bytes[:4], b'\x89PNG')

    def test_generate_win_card_bytes_live_signal(self):
        png_bytes = ShareableWinCardGenerator.generate_win_card_bytes(
            symbol="BTC",
            pnl_pct=3.10,
            entry_price=64000.0,
            exit_price=65984.0,
            is_paper_trade=False,
            winrate_pct=71.4,
            total_trades=28
        )
        self.assertIsInstance(png_bytes, bytes)
        self.assertGreater(len(png_bytes), 1000)
        self.assertEqual(png_bytes[:4], b'\x89PNG')

    def test_generate_win_card_file(self):
        out_path = "temp_test_win_card.png"
        try:
            res_path = ShareableWinCardGenerator.generate_win_card_file(
                output_path=out_path,
                symbol="ETH",
                pnl_pct=4.15,
                entry_price=3100.0,
                exit_price=3228.65,
                is_paper_trade=True
            )
            self.assertTrue(os.path.exists(res_path))
            self.assertGreater(os.path.getsize(res_path), 1000)
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)

if __name__ == "__main__":
    unittest.main()
