import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.domain.services.morning_briefing_service import MorningBriefingService
from src.infrastructure.llm.persona_prompts import PersonaType

class TestMorningBriefingService(unittest.TestCase):
    def setUp(self):
        self.service = MorningBriefingService()

    def test_generate_morning_briefing(self):
        mock_df = pd.DataFrame({
            "open": [60000.0] * 24,
            "high": [62000.0] * 24,
            "low": [59000.0] * 24,
            "close": [61500.0] * 24
        })
        with patch.object(self.service.fetcher, "fetch_ohlcv", return_value=mock_df):
            res_ru = self.service.generate_morning_briefing(lang="ru", persona=PersonaType.YANDERE)
            self.assertIn("УТРЕННИЙ БРИФИНГ", res_ru["html"])
            self.assertIn("сердце и депозит", res_ru["html"])
            self.assertEqual(len(res_ru["assets"]), 3)
            
            res_en = self.service.generate_morning_briefing(lang="en", persona=PersonaType.QUANT)
            self.assertIn("MORNING ALPHA PULSE", res_en["html"])

if __name__ == "__main__":
    unittest.main()
