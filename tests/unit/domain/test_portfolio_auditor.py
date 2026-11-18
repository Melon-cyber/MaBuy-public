import os
import unittest
from cryptography.fernet import Fernet

# Ensure encryption key is set for tests
os.environ["ENCRYPTION_KEY"] = Fernet.generate_key().decode()

from src.domain.services.portfolio_auditor import PortfolioRiskAuditorService
from src.presentation.telegram_bot.formatter import TelegramMessageFormatter
from src.presentation.telegram_bot.bot import MaBuyTelegramBot


class TestPortfolioAuditor(unittest.TestCase):
    def setUp(self):
        self.auditor = PortfolioRiskAuditorService
        self.formatter = TelegramMessageFormatter()
        self.bot = MaBuyTelegramBot()

    def test_portfolio_auditor_covariance_and_var(self):
        """Verifies correlation-aware risk audit, VaR (95%) and diversification score."""
        classic = self.auditor.PRESETS["classic"]
        degen = self.auditor.PRESETS["degen"]
        balanced = self.auditor.PRESETS["balanced"]

        res_classic = self.auditor.audit_portfolio(classic, lang="ru")
        res_degen = self.auditor.audit_portfolio(degen, lang="ru")
        res_balanced = self.auditor.audit_portfolio(balanced, lang="ru")

        # Basic validity checks
        for res in (res_classic, res_degen, res_balanced):
            self.assertIn("portfolio_volatility_pct", res)
            self.assertIn("annual_volatility_pct", res)
            self.assertIn("diversification_score", res)
            self.assertIn("var_95_usd_per_100", res)
            self.assertIn("advice", res)
            self.assertGreater(res["diversification_score"], 0)
            self.assertLessEqual(res["diversification_score"], 100)

        # Degen (SOL, NEAR, TON) must have higher volatility than Balanced (BTC, ETH, SOL, USDT)
        self.assertGreater(res_degen["portfolio_volatility_pct"], res_balanced["portfolio_volatility_pct"])
        self.assertGreater(res_degen["var_95_usd_per_100"], res_balanced["var_95_usd_per_100"])

    def test_weight_normalization_and_edge_cases(self):
        """Verifies arbitrary weight sums (e.g. 150%) normalize to 100% strictly."""
        # Non-normalized input: 50 + 50 + 50 = 150
        raw_alloc = {"BTC": 50.0, "ETH": 50.0, "SOL": 50.0}
        res = self.auditor.audit_portfolio(raw_alloc, lang="ru")
        
        sum_weights = sum(res["weights"].values())
        self.assertAlmostEqual(sum_weights, 1.0, places=5)
        self.assertAlmostEqual(res["weights"]["BTC"], 1.0 / 3.0, places=4)

        # Single coin 100% BTC
        single_alloc = {"BTC": 100.0}
        res_single = self.auditor.audit_portfolio(single_alloc, lang="ru")
        self.assertEqual(res_single["weights"]["BTC"], 1.0)
        self.assertIn("Критическая концентрация", res_single["advice"])

        # Empty dictionary fallback
        res_empty = self.auditor.audit_portfolio({}, lang="ru")
        self.assertIn("BTC", res_empty["weights"])

    def test_telegram_formatting_and_keyboards(self):
        """Verifies visual diagnostic card formatting and <64 byte callback_data."""
        audit_res = self.auditor.audit_portfolio(self.auditor.PRESETS["balanced"], lang="ru")
        card_text = self.formatter.format_portfolio_audit_card(audit_res, lang="ru")

        self.assertIn("ИИ-АУДИТ РИСКОВ ПОРТФЕЛЯ", card_text)
        self.assertIn("1-Day VaR (95%):", card_text)
        self.assertIn("на каждые $100", card_text)
        self.assertIn("Диверсификация:", card_text)

        kb = self.formatter.get_portfolio_presets_keyboard(lang="ru")
        for row in kb["inline_keyboard"]:
            for btn in row:
                if "callback_data" in btn:
                    # Telegram Callback Data limit: 64 bytes
                    self.assertLessEqual(len(btn["callback_data"].encode("utf-8")), 64)

    def test_bot_service_portfolio_handlers(self):
        """Verifies MaBuyTelegramBot handles portfolio commands and presets."""
        user_id = "test_portfolio_user_1"
        
        # Default /portfolio
        default_reply = self.bot.handle_portfolio_audit(user_id, "")
        self.assertIn("ИИ-АУДИТ РИСКОВ ПОРТФЕЛЯ", default_reply)

        # Custom /portfolio BTC 50, SOL 50
        custom_reply = self.bot.handle_portfolio_audit(user_id, "BTC 50, SOL 50")
        self.assertIn("BTC", custom_reply)
        self.assertIn("SOL", custom_reply)

        # Preset
        preset_reply = self.bot.handle_portfolio_preset(user_id, "degen")
        self.assertIn("ИИ-АУДИТ РИСКОВ ПОРТФЕЛЯ", preset_reply)
        self.assertIn("NEAR", preset_reply)


if __name__ == "__main__":
    unittest.main()
