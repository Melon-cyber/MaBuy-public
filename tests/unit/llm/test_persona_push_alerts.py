import unittest
import os
import sys
import asyncio
import tempfile
from cryptography.fernet import Fernet

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

os.environ['ENCRYPTION_KEY'] = 'LotP13ofZ_3chTwtYlLkqekLGX_bEw8rwaLm74YOCIE='

from src.infrastructure.llm.persona_prompts import PersonaType
from src.infrastructure.llm.copilot_engine import MaIACopilotEngine
from src.presentation.telegram_bot.formatter import TelegramMessageFormatter
from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker


class TestPersonaPushAlerts(unittest.TestCase):

    def setUp(self):
        os.environ['ENCRYPTION_KEY'] = 'LotP13ofZ_3chTwtYlLkqekLGX_bEw8rwaLm74YOCIE='
        self.copilot = MaIACopilotEngine()
        self.tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.tmp_file.close()
        self.tracker = SQLiteTracker(self.tmp_file.name)
        self.alert_data = {
            'symbol': 'BTC',
            'direction': 'BULLISH',
            'current_price': 64250.0,
            'target_price': 66050.0,
            'stop_loss_price': 63280.0,
            'tp_pct': 2.8,
            'sl_pct': 1.5,
            'reward_risk_ratio': 1.9,
            'max_safe_leverage': 5,
            'confidence_pct': 68,
            'timeframe': '1h',
            'personal_recommendation': 'Квантовый импульс V5'
        }

    def tearDown(self):
        try:
            if os.path.exists(self.tmp_file.name):
                os.remove(self.tmp_file.name)
        except Exception:
            pass

    def test_sqlite_user_persona_persistence(self):
        self.assertEqual(self.tracker.get_user_persona('user_1'), 'quant')
        self.tracker.set_user_persona('user_1', 'yandere')
        self.assertEqual(self.tracker.get_user_persona('user_1'), 'yandere')
        self.tracker.set_user_persona('user_1', 'tsundere')
        self.assertEqual(self.tracker.get_user_persona('user_1'), 'tsundere')

    def test_generate_push_alert_narratives_across_personas(self):
        for persona in [PersonaType.QUANT, PersonaType.ZEN, PersonaType.TSUNDERE, PersonaType.YANDERE, PersonaType.TITAN]:
            narrative = asyncio.run(
                self.copilot.generate_push_alert_narrative(
                    alert_data=self.alert_data,
                    persona=persona,
                    user_deposit=1000.0,
                    lang='ru'
                )
            )
            self.assertIsInstance(narrative, str)
            self.assertTrue(len(narrative) > 20)
            self.assertTrue('BTC' in narrative or 'Биткоин' in narrative or '64' in narrative or '68' in narrative or 'стоп' in narrative.lower() or 'цель' in narrative.lower())

    def test_format_push_alert_message_with_narrative(self):
        narrative = '(♡_♡) Г-господин... Наш Биткоин начинает расти! Я на 68% уверена...'
        msg = TelegramMessageFormatter.format_push_alert_message(
            self.alert_data,
            user_deposit=1000.0,
            lang='ru',
            persona_narrative=narrative
        )
        self.assertIn('(♡_♡) Г-господин', msg)
        self.assertIn('#BTC', msg)
        self.assertIn('64,250', msg)
        self.assertIn('TP', msg)


if __name__ == '__main__':
    unittest.main()
