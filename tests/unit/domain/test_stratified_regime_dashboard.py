import os
os.environ["ENCRYPTION_KEY"] = "UN_5bf8rBxMiQOudLLjBJz42sY6vWznie0oxcjcKLYQ="

import time
import pytest
from src.domain.services.stratified_regime_tracker import StratifiedRegimeQuotaTracker
from src.presentation.telegram_bot.bot import MaBuyTelegramBot


class TestStratifiedRegimeDashboard:
    def test_stratified_regime_quota_math(self):
        tracker = StratifiedRegimeQuotaTracker()

        # 1. Initially unfulfilled
        readiness_empty = tracker.evaluate_live_readiness()
        assert readiness_empty["is_qualified_for_live_capital"] is False
        assert readiness_empty["total_trades"] == 0

        # 2. Record trades in BULL_TREND (20 trades: 15 wins, 5 losses)
        for i in range(15):
            tracker.record_shadow_trade("BULL_TREND", is_win=True, pnl_usd=10.0)
        for i in range(5):
            tracker.record_shadow_trade("BULL_TREND", is_win=False, pnl_usd=-4.0)

        # Record trades in SQUEEZE (25 trades: 20 wins, 5 losses)
        for i in range(20):
            tracker.record_shadow_trade("SQUEEZE", is_win=True, pnl_usd=8.0)
        for i in range(5):
            tracker.record_shadow_trade("SQUEEZE", is_win=False, pnl_usd=-3.0)

        eval_mid = tracker.evaluate_live_readiness()
        assert eval_mid["regimes_breakdown"]["BULL_TREND"]["trades_count"] == 20
        assert eval_mid["regimes_breakdown"]["BULL_TREND"]["is_quota_fulfilled"] is True
        assert eval_mid["regimes_breakdown"]["BULL_TREND"]["win_rate_pct"] == 75.0
        assert eval_mid["regimes_breakdown"]["SQUEEZE"]["is_quota_fulfilled"] is True
        assert eval_mid["is_qualified_for_live_capital"] is False # BEAR and VOLATILITY still 0

    def test_bot_regimes_dashboard_handler_and_persistence(self):
        bot = MaBuyTelegramBot()
        test_uid = f"regime_dash_test_{int(time.time()*1000)}"

        # 1. Record trades via sqlite persistence
        bot.usage_tracker.record_regime_trade("BULL_TREND", "BTC", is_win=True, pnl_usd=12.5)
        bot.usage_tracker.record_regime_trade("BEAR_TREND", "ETH", is_win=True, pnl_usd=8.2)

        # 2. Render dashboard
        dash_text, markup = bot.handle_regime_quota_dashboard(test_uid)

        assert "ФОРВАРД-ТЕСТ: ПРОГРЕСС 4 РЕЖИМОВ РЫНКА" in dash_text
        assert "Бычий тренд" in dash_text
        assert "Медвежий тренд" in dash_text
        assert "Флэт / Сквиз" in dash_text
        assert "Шторм / Волатильность" in dash_text
        assert "inline_keyboard" in markup
