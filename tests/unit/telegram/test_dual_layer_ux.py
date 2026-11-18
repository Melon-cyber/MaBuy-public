import os
import unittest
from cryptography.fernet import Fernet

# Ensure encryption key is set for tests
os.environ["ENCRYPTION_KEY"] = Fernet.generate_key().decode()

from src.domain.entities.prediction import PredictionResult
from src.domain.entities.tier import SubscriptionTier
from src.presentation.telegram_bot.formatter import TelegramMessageFormatter
from src.presentation.telegram_bot.bot import MaBuyTelegramBot
from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker


class TestDualLayerUX(unittest.TestCase):
    def setUp(self):
        self.formatter = TelegramMessageFormatter()
        self.dummy_result = PredictionResult(
            symbol="BTC",
            current_price=64000.0,
            predicted_target_price=66500.0,
            predicted_log_return=0.038,
            direction="BULLISH",
            confidence_score=0.78,
            volatility_estimate=0.025,
            tier_applied=SubscriptionTier.PRO,
            model_version="MaBuy-V4.1-Titan-Ensemble",
            personal_recommendation="Крупные игроки выкупают пролив на споте.",
            sentiment_score=0.74
        )

    def test_beginner_view_formatting(self):
        """Verifies beginner card has traffic light, gauge bar, human reason, and $100 calculator."""
        card_ru = self.formatter.format_prediction_result(self.dummy_result, lang="ru", experience="beginner")
        
        self.assertIn("ПОКУПАТЬ (LONG)", card_ru)
        self.assertIn("AI: <b>78%</b>", card_ru)
        self.assertIn("Крупные игроки выкупают пролив", card_ru)
        self.assertIn("Сайзинг на $100:", card_ru)
        self.assertIn("(5x)", card_ru)
        self.assertIn("Вход:", card_ru)

    def test_pro_quant_matrix_formatting(self):
        """Verifies PRO view contains Order Book Imbalance, Kelly Math, and Neural Ensemble telemetry."""
        pro_ru = self.formatter.format_pro_quant_matrix(self.dummy_result, lang="ru")
        
        self.assertIn("🔬 <b>MaBuy Quant Matrix • #BTCUSDT [PRO]</b>", pro_ru)
        self.assertIn("Оптимальный риск (Quarter-Kelly):", pro_ru)
        self.assertIn("Order Book Imbalance (L2):", pro_ru)
        self.assertIn("Funding Rate (8h):", pro_ru)
        self.assertIn("АНСАМБЛЬ НЕЙРОСЕТЕЙ (V4.1)", pro_ru)

    def test_dual_layer_inline_keyboards(self):
        """Verifies clean minimalist interactive buttons with WebApp terminal and MaIA access."""
        kb = self.formatter.get_prediction_dual_keyboard("BTC", "1h", is_pro_view=False)
        
        # Check presence of direct WebApp button and MaIA button
        has_webapp = any("web_app" in btn for row in kb["inline_keyboard"] for btn in row)
        self.assertTrue(has_webapp)
        
        callbacks = [btn["callback_data"] for row in kb["inline_keyboard"] for btn in row if "callback_data" in btn]
        self.assertTrue(any("/ask" in cb for cb in callbacks))
        # verified keyboard

    def test_bot_service_dual_layer_interactions(self):
        """Verifies bot service handles pro view for free vs pro users and order copying."""
        test_db = ".test_dual_layer_bot.db"
        if os.path.exists(test_db):
            os.remove(test_db)
            
        tracker = SQLiteTracker(db_path=test_db)
        bot_service = MaBuyTelegramBot()
        bot_service.usage_tracker = tracker
        bot_service.get_prediction_use_case.usage_tracker = tracker
        
        import time
        # 1. Free user accessing PRO view gets promotional teaser
        free_user_id = "free_user_101"
        tracker.set_user_tier(free_user_id, SubscriptionTier.FREE, time.time() + 86400)
        free_pro_reply = bot_service.handle_predict_pro_view(free_user_id, "BTC", "1h")
        self.assertIn("КВАНТОВАЯ МАТРИЦА ДОСТУПНА В MABUY PRO", free_pro_reply)
        self.assertIn("PARI17", free_pro_reply)
        
        # 2. PRO user gets full quant matrix
        pro_user_id = "pro_user_202"
        tracker.set_user_tier(pro_user_id, SubscriptionTier.PRO, time.time() + 86400)
        pro_reply = bot_service.handle_predict_pro_view(pro_user_id, "BTC", "1h")
        self.assertIn("MaBuy Quant Matrix", pro_reply)
        
        # 3. Order copy snippet
        copy_reply = bot_service.handle_order_copy_snippet(pro_user_id, "BTC", "1h")
        self.assertIn("ПАРАМЕТРЫ ОРДЕРА ДЛЯ БИРЖИ", copy_reply)
        self.assertIn("BTC/USDT", copy_reply)
        
        # 4. Virtual trade open
        paper_reply = bot_service.handle_virtual_trade_open(pro_user_id, "BTC", "100")
        self.assertIn("ДЕМО-СДЕЛКА НА $10,000 УСПЕШНО ОТКРЫТА", paper_reply)
        
        # Clean up
        try:
            if os.path.exists(test_db):
                os.remove(test_db)
        except OSError:
            pass


if __name__ == "__main__":
    unittest.main()
