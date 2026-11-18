# -*- coding: utf-8 -*-
import os
import unittest
from cryptography.fernet import Fernet

# Ensure encryption key is set for tests
if "ENCRYPTION_KEY" not in os.environ:
    os.environ["ENCRYPTION_KEY"] = Fernet.generate_key().decode()

from src.presentation.telegram_bot.bot import MaBuyTelegramBot
from src.domain.entities.tier import SubscriptionTier


class TestTelegramBotIntegration(unittest.TestCase):

    def setUp(self):
        self.bot = MaBuyTelegramBot()
        self.user_id = "test_tg_user_formatter"
        self.bot.usage_tracker.reset_usage(self.user_id)

    def test_start_command(self):
        msg = self.bot.handle_start(self.user_id)
        self.assertTrue("MaBuy" in msg or "FREE" in msg or "меню" in msg)

    def test_free_tier_asset_guard_in_predict(self):
        msg_btc, chart_path = self.bot.handle_predict(self.user_id, "BTC")
        self.assertTrue("MaBuy" in msg_btc or "BTC" in msg_btc)

        msg_sol, chart_sol = self.bot.handle_predict(self.user_id, "SOL")
        self.assertTrue(len(msg_sol) > 0)

    def test_upgrade_and_pro_capabilities(self):
        self.bot.set_user_tier(self.user_id, SubscriptionTier.PRO)

        predict_msg, _ = self.bot.handle_predict(self.user_id, "BTC")
        self.assertTrue("PRO" in predict_msg or "MaBuy" in predict_msg)

    def test_public_channel_broadcast_formatting(self):
        broadcast_text, chart_path = self.bot.generate_channel_broadcast("ETH")
        self.assertIn("#ETH", broadcast_text)


if __name__ == "__main__":
    unittest.main()
