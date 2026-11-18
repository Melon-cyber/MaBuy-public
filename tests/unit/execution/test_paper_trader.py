import os
import sys
import uuid
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.infrastructure.security.encryption import EncryptionService
from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker
from src.infrastructure.execution.paper_trader import PaperTrader


class TestPaperTraderATRStops(unittest.TestCase):
    def setUp(self):
        os.makedirs("data", exist_ok=True)
        if not os.environ.get("ENCRYPTION_KEY"):
            os.environ["ENCRYPTION_KEY"] = EncryptionService.generate_key()
        self.test_db_path = f"data/test_paper_trader_{uuid.uuid4().hex}.db"
        self.tracker = SQLiteTracker(db_path=self.test_db_path)
        self.user_id = f"test_user_atr_{uuid.uuid4().hex[:6]}"
        self.trader = PaperTrader(user_id=self.user_id, tracker=self.tracker)

    def tearDown(self):
        if os.path.exists(self.test_db_path):
            try:
                os.remove(self.test_db_path)
            except Exception:
                pass

    def test_get_price_and_atr_calculation(self):
        """Verifies calculation of 14-period Average True Range."""
        mock_df = pd.DataFrame({
            "high": [102.0] * 20,
            "low": [98.0] * 20,
            "close": [100.0] * 20
        })
        with patch.object(self.trader.fetcher, "fetch_ohlcv", return_value=mock_df):
            price, atr_pct = self.trader._get_price_and_atr("BTC")
            self.assertEqual(price, 100.0)
            # High-Low is 4.0, close is 100 -> ATR % = 4.0 / 100 = 0.04 (4%)
            self.assertAlmostEqual(atr_pct, 0.04, places=3)

    def test_execute_trade_success(self):
        """Verifies virtual paper trade execution and balance tracking."""
        mock_df = pd.DataFrame({
            "high": [50100.0] * 20,
            "low": [49900.0] * 20,
            "close": [50000.0] * 20
        })
        with patch.object(self.trader.fetcher, "fetch_ohlcv", return_value=mock_df):
            res = self.trader.execute_trade("BTC", "BULLISH", leverage=10)
            self.assertIsNotNone(res)
            self.assertEqual(res["status"], "PAPER_SUCCESS")
            self.assertEqual(res["side"], "BUY")
            self.assertEqual(res["entry_price"], 50000.0)
            
            open_trades = self.tracker.get_open_paper_trades(self.user_id)
            self.assertEqual(len(open_trades), 1)
            self.assertEqual(open_trades[0]["symbol"], "BTC")

    def test_trailing_stop_loss_trigger(self):
        """
        Verifies trailing stop triggers only after >= 1.0x ATR gain and 1.0x ATR pullback.
        """
        # 1. Open trade @ 50,000 USD, ATR = 2% ($1,000)
        entry_df = pd.DataFrame({
            "high": [50500.0] * 20,
            "low": [49500.0] * 20,
            "close": [50000.0] * 20
        })
        with patch.object(self.trader.fetcher, "fetch_ohlcv", return_value=entry_df):
            self.trader.execute_trade("BTC", "BULLISH", leverage=10)

        # 2. Price surges to 51,100 (+2.2% move, activates Trailing Stop >= 1.0x ATR, below TP 2.1x ATR)
        surge_df = pd.DataFrame({
            "high": [51600.0] * 20,
            "low": [50600.0] * 20,
            "close": [51100.0] * 20
        })
        with patch.object(self.trader.fetcher, "fetch_ohlcv", return_value=surge_df):
            self.trader.update_open_trades()
            open_trades = self.tracker.get_open_paper_trades(self.user_id)
            self.assertEqual(len(open_trades), 1)
            self.assertGreater(open_trades[0]["max_pnl_usd"], 0.0)

        # 3. Price pulls back to 49,900 (pullback >= 1.0x ATR from peak 50,900)
        pullback_df = pd.DataFrame({
            "high": [50400.0] * 20,
            "low": [49400.0] * 20,
            "close": [49900.0] * 20
        })
        with patch.object(self.trader.fetcher, "fetch_ohlcv", return_value=pullback_df):
            self.trader.update_open_trades()
            open_trades = self.tracker.get_open_paper_trades(self.user_id)
            # Trade must be closed by Trailing SL
            self.assertEqual(len(open_trades), 0)

    def test_hard_stop_loss_trigger(self):
        """Verifies hard stop loss closes trade when price drops by 1.15x ATR."""
        entry_df = pd.DataFrame({
            "high": [50500.0] * 20,
            "low": [49500.0] * 20,
            "close": [50000.0] * 20
        })
        with patch.object(self.trader.fetcher, "fetch_ohlcv", return_value=entry_df):
            self.trader.execute_trade("BTC", "BULLISH", leverage=10)

        # Price drops significantly (to 47,000, which is -6.0% > 2.0 x ATR)
        sl_df = pd.DataFrame({
            "high": [47500.0] * 20,
            "low": [46500.0] * 20,
            "close": [47000.0] * 20
        })
        with patch.object(self.trader.fetcher, "fetch_ohlcv", return_value=sl_df):
            self.trader.update_open_trades()
            open_trades = self.tracker.get_open_paper_trades(self.user_id)
            self.assertEqual(len(open_trades), 0)


if __name__ == "__main__":
    unittest.main()
