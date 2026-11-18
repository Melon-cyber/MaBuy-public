"""
MaBuy V4 — Product UX & Dynamic Signal Card Test Suite.
Tests:
  1. TelegramMessageFormatter.format_channel_outcome_card formatting (HIT_TP, HIT_SL, EXPIRED).
  2. Idempotent SQLite migration and storage of channel_message_id.
  3. Safe Telegram editMessageText error handling (MessageNotModified, MessageNotFound, Rate Limits).
"""
import os
import shutil
import unittest
import urllib.error
from unittest.mock import patch, MagicMock

from cryptography.fernet import Fernet
from src.presentation.telegram_bot.formatter import TelegramMessageFormatter
from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker
from scripts.auto_trader_daemon import _edit_telegram_channel_message


class TestDynamicSignalCard(unittest.TestCase):
    def setUp(self):
        os.environ["ENCRYPTION_KEY"] = Fernet.generate_key().decode()
        self.test_db_dir = ".test_card_db"
        if os.path.exists(self.test_db_dir):
            shutil.rmtree(self.test_db_dir, ignore_errors=True)
        os.makedirs(self.test_db_dir, exist_ok=True)
        self.test_db_path = os.path.join(self.test_db_dir, "test_card.db")
        self.tracker = SQLiteTracker(db_path=self.test_db_path)

    def tearDown(self):
        if os.path.exists(self.test_db_dir):
            shutil.rmtree(self.test_db_dir, ignore_errors=True)

    def test_format_channel_outcome_card_hit_tp(self):
        text = TelegramMessageFormatter.format_channel_outcome_card(
            symbol="BTC",
            direction="BULLISH",
            entry_price=64000.0,
            target_price=66560.0,
            stop_loss=62720.0,
            outcome="HIT_TP",
            exit_price=66560.0,
            confidence=68.5,
            duration_str="4ч 12м"
        )
        self.assertIn("ТАРГЕТ ДОСТИГНУТ (+4.00% ПРОФИТ)", text)
        self.assertIn("ВЕРИФИЦИРОВАННЫЙ ПРОФИТ", text)
        self.assertIn("#BTC", text)
        self.assertIn("4ч 12м", text)
        self.assertIn("@MaBuyCryptoBot", text)

    def test_format_channel_outcome_card_hit_sl(self):
        text = TelegramMessageFormatter.format_channel_outcome_card(
            symbol="ETH",
            direction="BULLISH",
            entry_price=3500.0,
            target_price=3650.0,
            stop_loss=3430.0,
            outcome="HIT_SL",
            exit_price=3430.0,
            confidence=62.0,
            duration_str="1ч 45м"
        )
        self.assertIn("СТОП-ЛОСС СРАБОТАЛ (-2.00%)", text)
        self.assertIn("ЗАЩИТА КАПИТАЛА", text)
        self.assertIn("#ETH", text)

    def test_idempotent_sqlite_migration_and_channel_msg_id(self):
        # Record signal with channel_message_id
        signal_id = self.tracker.record_verified_signal(
            symbol="SOL",
            direction="BULLISH",
            entry=150.0,
            target=160.0,
            sl=145.0,
            conf=65.0,
            channel_message_id=98765,
            channel_id="@MaBuyCrypto"
        )
        self.assertIsNotNone(signal_id)

        # Retrieve open signals
        open_signals = self.tracker.get_open_verified_signals()
        self.assertEqual(len(open_signals), 1)
        self.assertEqual(open_signals[0]["channel_message_id"], 98765)
        self.assertEqual(open_signals[0]["channel_id"], "@MaBuyCrypto")

        # Test idempotent second initialization (must NOT raise OperationalError)
        second_tracker = SQLiteTracker(db_path=self.test_db_path)
        signals_check = second_tracker.get_open_verified_signals()
        self.assertEqual(len(signals_check), 1)

    @patch("scripts.auto_trader_daemon.urllib.request.urlopen")
    def test_edit_telegram_channel_message_success(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        # Should execute cleanly
        _edit_telegram_channel_message(
            token="test_token",
            channel_id="@MaBuyCrypto",
            message_id=12345,
            new_text="<b>Updated Text</b>",
            reply_markup={}
        )
        mock_urlopen.assert_called_once()

    @patch("scripts.auto_trader_daemon.urllib.request.urlopen")
    def test_edit_telegram_channel_message_handles_not_modified_gracefully(self, mock_urlopen):
        # Simulate Telegram HTTP 400 Bad Request: message is not modified
        fp_mock = MagicMock()
        fp_mock.read.return_value = b'{"ok":false,"error_code":400,"description":"Bad Request: message is not modified"}'
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.telegram.org",
            code=400,
            msg="Bad Request: message is not modified",
            hdrs={},
            fp=fp_mock
        )

    def test_format_channel_outcome_card_expired(self):
        text = TelegramMessageFormatter.format_channel_outcome_card(
            symbol="SOL",
            direction="BULLISH",
            entry_price=150.0,
            target_price=165.0,
            stop_loss=142.5,
            outcome="EXPIRED",
            exit_price=151.2,
            confidence=60.0,
            duration_str="48ч 00м"
        )
        self.assertIn("СИГНАЛ ЗАКРЫТ ПО ВРЕМЕНИ (+0.80%)", text)
        self.assertIn("ТАЙМАУТ 48 ЧАСОВ", text)
        self.assertIn("#SOL", text)
        self.assertIn("48ч 00м", text)

    @patch("scripts.auto_trader_daemon.urllib.request.urlopen")
    def test_edit_telegram_channel_message_handles_message_not_found(self, mock_urlopen):
        # Simulate Telegram HTTP 400 Bad Request: message to edit not found (e.g. deleted by user/admin)
        fp_mock = MagicMock()
        fp_mock.read.return_value = b'{"ok":false,"error_code":400,"description":"Bad Request: message to edit not found"}'
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.telegram.org",
            code=400,
            msg="Bad Request: message to edit not found",
            hdrs={},
            fp=fp_mock
        )

        res = _edit_telegram_channel_message(
            token="test_token",
            channel_id="@MaBuyCrypto",
            message_id=99999,
            new_text="<b>Updated Text</b>",
            reply_markup={}
        )
        self.assertFalse(res)

    @patch("scripts.auto_trader_daemon.time.sleep")
    @patch("scripts.auto_trader_daemon.urllib.request.urlopen")
    def test_edit_telegram_channel_message_handles_rate_limit_429_adaptive_retry(self, mock_urlopen, mock_sleep):
        # 1st attempt: 429 Too Many Requests with retry_after = 2
        fp_mock = MagicMock()
        fp_mock.read.return_value = b'{"ok":false,"error_code":429,"description":"Too Many Requests: retry after 2","parameters":{"retry_after":2}}'
        err_429 = urllib.error.HTTPError(
            url="https://api.telegram.org",
            code=429,
            msg="Too Many Requests",
            hdrs={},
            fp=fp_mock
        )

        # 2nd attempt: Success 200
        mock_success = MagicMock()
        mock_success.status = 200
        mock_success.__enter__.return_value = mock_success

        mock_urlopen.side_effect = [err_429, mock_success]

        res = _edit_telegram_channel_message(
            token="test_token",
            channel_id="@MaBuyCrypto",
            message_id=12345,
            new_text="<b>Updated Text</b>",
            reply_markup={},
            max_retries=1
        )
        self.assertTrue(res)
        mock_sleep.assert_called_once_with(2)


if __name__ == "__main__":
    unittest.main()
