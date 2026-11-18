import unittest
import os
import shutil
from src.domain.services.derivatives_flow_radar import DerivativesFlowRadarService
from src.presentation.telegram_bot.formatter import TelegramMessageFormatter

class TestDerivativesFlowRadarService(unittest.TestCase):
    def setUp(self):
        self.test_cache_dir = "data/cache/derivatives"

    def test_derivatives_radar_payload_structure(self):
        res = DerivativesFlowRadarService.get_derivatives_radar("BTC", current_spot_price=64000.0, lang="ru")
        self.assertEqual(res["symbol"], "BTC")
        self.assertIn("taker_buy_sell_ratio", res)
        self.assertIn("top_traders_long_pct", res)
        self.assertIn("top_traders_short_pct", res)
        self.assertIn("open_interest_usd_millions", res)
        self.assertIn("funding_rate_pct", res)
        self.assertIn("summary", res)
        self.assertIn("bias_label", res)
        self.assertGreaterEqual(res["confidence_pct"], 50.0)

    def test_derivatives_radar_localization_en(self):
        res = DerivativesFlowRadarService.get_derivatives_radar("ETH", current_spot_price=3200.0, lang="en")
        self.assertEqual(res["symbol"], "ETH")
        self.assertIn("Open Interest", res["summary"])

    def test_derivatives_formatter_and_keyboard(self):
        data = DerivativesFlowRadarService.get_derivatives_radar("SOL", current_spot_price=160.0, lang="ru")
        card = TelegramMessageFormatter.format_derivatives_radar_card(data, lang="ru")
        self.assertIn("РАДАР ПОЗИЦИЙ КИТОВ И ДЕРИВАТИВОВ", card)
        self.assertIn("ТОП-20% КРУПНЕЙШИХ ТРЕЙДЕРОВ", card)
        self.assertIn("Mark Price", card)

        kb = TelegramMessageFormatter.get_derivatives_keyboard("SOL", lang="ru")
        self.assertIn("inline_keyboard", kb)
        self.assertGreaterEqual(len(kb["inline_keyboard"]), 3)

if __name__ == "__main__":
    unittest.main()
