import os
import unittest
import tempfile
import sqlite3
from src.domain.services.track_record_auditor import TrackRecordAuditorService
from src.presentation.telegram_bot.bot import MaBuyTelegramBot
from src.presentation.telegram_bot.win_card_generator import ShareableWinCardGenerator
from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker

class TestTrackRecordAndWinCard(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, 'test_mabuy.db')
        self.tracker = SQLiteTracker(db_path=self.db_path)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signal_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_time TEXT,
                    symbol TEXT,
                    direction TEXT,
                    confidence REAL,
                    model_version TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    pnl_usd REAL,
                    close_reason TEXT,
                    signal_hash TEXT UNIQUE,
                    telegram_msg_id INTEGER,
                    live_order_id TEXT,
                    status TEXT DEFAULT 'OPEN',
                    created_at TEXT
                )
            ''')
            conn.execute('''
                INSERT INTO signal_log (signal_time, symbol, direction, entry_price, exit_price, pnl_usd, status)
                VALUES ('2026-09-07T10:00:00Z', 'BTCUSDT', 'BULLISH', 60000.0, 61000.0, 5.5, 'CLOSED'),
                       ('2026-09-07T11:00:00Z', 'ETHUSDT', 'BEARISH', 2500.0, 2450.0, 3.2, 'CLOSED'),
                       ('2026-09-07T12:00:00Z', 'SOLUSDT', 'BULLISH', 130.0, NULL, NULL, 'OPEN')
            ''')

    def tearDown(self):
        try:
            self.tmp_dir.cleanup()
        except Exception:
            pass

    def test_auditor_performance_summary(self):
        auditor = TrackRecordAuditorService(db_path=self.db_path)
        stats = auditor.get_performance_summary(lang='ru')
        self.assertEqual(stats['total_signals'], 2)
        self.assertEqual(stats['open_signals'], 1)
        self.assertEqual(stats['win_rate_pct'], 100.0)
        self.assertEqual(len(stats['recent_signals']), 3)

    def test_handle_trackrecord_formatting(self):
        bot = MaBuyTelegramBot()
        bot.usage_tracker = self.tracker
        text = bot.handle_trackrecord('test_user_123')
        self.assertIn('ВЕРИФИЦИРОВАННЫЙ ТРЕК-РЕКОРД', text)
        self.assertIn('BTCUSDT', text)

    def test_wincard_generator(self):
        card_bytes = ShareableWinCardGenerator.generate_win_card_bytes(
            symbol='BTCUSDT',
            pnl_pct=2.45,
            entry_price=60000.0,
            exit_price=61470.0,
            is_paper_trade=False,
            winrate_pct=55.6,
            total_trades=36
        )
        self.assertTrue(card_bytes.startswith(b'\x89PNG\r\n\x1a\n'))

if __name__ == '__main__':
    unittest.main()
