import sys
import os
import unittest

sys.stdout.reconfigure(encoding='utf-8')
os.environ['ENCRYPTION_KEY'] = 'UN_5bf8rBxMiQOudLLjBJz42sY6vWznie0oxcjcKLYQ='
sys.path.insert(0, r'C:\Users\super\OneDrive\Изображения\Снимки экрана\MaBuy')

from src.infrastructure.ml.v4_titan_ensemble import MaBuyV4EnsembleEngine
from src.domain.entities.prediction import PredictionRequest
from src.domain.entities.tier import SubscriptionTier
from src.presentation.telegram_bot.bot import MaBuyTelegramBot

class TestMaBuyV4FlagshipEngine(unittest.TestCase):
    def setUp(self):
        self.v4_engine = MaBuyV4EnsembleEngine(db_path='data/test_v4_db.db')
        self.bot = MaBuyTelegramBot()

    def test_v4_ensemble_prediction_btc(self):
        req = PredictionRequest(
            user_id='v4_test_user_1',
            symbol='BTC',
            tier=SubscriptionTier.PRO,
            timeframe='1h',
            lang='ru'
        )
        res = self.v4_engine.predict(req)
        self.assertEqual(res.symbol, 'BTC')
        self.assertIn('V4-Titan-Ensemble', res.model_version)
        self.assertTrue(res.confidence_score >= 0.50)
        self.assertIsNotNone(res.personal_recommendation)
        print('=== TEST V4 ENSEMBLE BTC SUCCESS ===')
        print(f'Symbol: {res.symbol}, Dir: {res.direction}, Conf: {res.confidence_score*100:.1f}%, Target: ${res.predicted_target_price:,.2f}')

    def test_bot_predict_with_v4_engine(self):
        import time
        uid = f'v4_test_user_{int(time.time())}'
        self.bot.set_user_tier(uid, SubscriptionTier.PRO, duration_days=30)
        reply_text, chart_path = self.bot.handle_predict(uid, 'ETH 1h')
        self.assertIn('MaBuy', reply_text)
        print('\n=== TEST BOT INTEGRATION WITH V4 SUCCESS ===')
        print(reply_text[:250])

if __name__ == '__main__':
    unittest.main()
