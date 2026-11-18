import unittest
from src.presentation.telegram_bot.formatter import TelegramMessageFormatter
from src.infrastructure.llm.persona_prompts import PersonaType

class TestMarketPulsePersona(unittest.TestCase):
    def test_format_market_pulse_card_with_yandere(self):
        signals = [
            {"symbol": "BTC", "direction": "BULLISH", "entry_price": 64000.0, "target_price": 67000.0, "stop_loss": 62500.0, "return_pct": 4.68, "confidence": 89},
            {"symbol": "ETH", "direction": "BULLISH", "entry_price": 3400.0, "target_price": 3600.0, "stop_loss": 3300.0, "return_pct": 5.88, "confidence": 87}
        ]
        card = TelegramMessageFormatter.format_market_pulse_card(
            signals=signals,
            user_deposit=1000.0,
            lang="ru",
            persona_narrative="(♡_♡) Господин! Весь рынок летит вверх, я выбрала лучшее для нас!",
            persona=PersonaType.YANDERE
        )
        self.assertIn("РЫНОЧНЫЙ ИМПУЛЬС", card)
        self.assertIn("Майя-Яндере", card)
        self.assertIn("Господин!", card)
        self.assertIn("#BTC", card)
        self.assertIn("#ETH", card)

    def test_format_market_pulse_card_with_tsundere(self):
        signals = [
            {"symbol": "SOL", "direction": "BEARISH", "entry_price": 145.0, "target_price": 135.0, "stop_loss": 150.0, "return_pct": -6.89, "confidence": 91}
        ]
        card = TelegramMessageFormatter.format_market_pulse_card(
            signals=signals,
            user_deposit=500.0,
            lang="ru",
            persona_narrative="(¬_¬) С-смотри сюда, дурачок! Не смей влезать во все сделки подряд!",
            persona=PersonaType.TSUNDERE
        )
        self.assertIn("Майя-Цундере", card)
        self.assertIn("дурачок", card)

if __name__ == "__main__":
    unittest.main()
