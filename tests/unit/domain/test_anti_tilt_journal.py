import os
os.environ["ENCRYPTION_KEY"] = "UN_5bf8rBxMiQOudLLjBJz42sY6vWznie0oxcjcKLYQ="

import time
import pytest
from src.domain.services.anti_tilt_journal import AntiTiltJournalService, JournalEntry, DisciplineSummary
from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker
from src.presentation.telegram_bot.bot import MaBuyTelegramBot


class TestAntiTiltJournal:
    def test_discipline_score_calculation_and_grades(self):
        service = AntiTiltJournalService()
        user_id = "trader_stoic_1"

        # 1. Clean sheet
        empty_summary = service.evaluate_discipline(user_id, [])
        assert empty_summary.discipline_score == 75
        assert "A-Rank" in empty_summary.grade

        # 2. Perfect discipline series (4 followed plan logs: 75 + 4*5 = 95)
        entries_good = [
            JournalEntry(1, user_id, "FOLLOWED_PLAN", 45.0, True, "Perfect SL execution", time.time() - 3600),
            JournalEntry(2, user_id, "FOLLOWED_PLAN", 120.0, True, "Waited for 1h confirmation", time.time() - 2400),
            JournalEntry(3, user_id, "FOLLOWED_PLAN", -30.0, True, "Took clean SL without moving it", time.time() - 1200),
            JournalEntry(4, user_id, "FOLLOWED_PLAN", 80.0, True, "Target hit", time.time() - 600)
        ]
        summary_good = service.evaluate_discipline(user_id, entries_good)
        assert summary_good.discipline_score == 95
        assert "S-Rank" in summary_good.grade
        assert summary_good.plan_compliance_pct == 100.0

        # 3. Severe tilt episode (FOMO -15, Revenge -25, Overleveraged -20: 75 - 60 = 15)
        entries_tilt = [
            JournalEntry(1, user_id, "FOMO_ENTRY", -50.0, False, "Jumped into green candle", time.time() - 1800),
            JournalEntry(2, user_id, "REVENGE_TRADE", -120.0, False, "Tried to double down", time.time() - 1200),
            JournalEntry(3, user_id, "OVERLEVERAGED", -80.0, False, "Used 20x instead of 3x", time.time() - 600)
        ]
        summary_tilt = service.evaluate_discipline(user_id, entries_tilt)
        assert summary_tilt.discipline_score == 15
        assert "D-Rank" in summary_tilt.grade
        assert summary_tilt.fomo_count == 1
        assert summary_tilt.revenge_count == 1

    def test_tilt_cooldown_calculation(self):
        service = AntiTiltJournalService()
        now = time.time()

        # Tilt triggered 10 minutes ago (35 min remaining = 2100s)
        last_tilt_ts = now - (10 * 60)
        entries = [JournalEntry(1, "u1", "REVENGE_TRADE", -50.0, False, "tilt", last_tilt_ts)]

        summary = service.evaluate_discipline("u1", entries, last_tilt_timestamp=last_tilt_ts)
        assert summary.is_in_tilt_cooldown is True
        assert summary.cooldown_remaining_sec > 2000 and summary.cooldown_remaining_sec <= 2100

    def test_bot_and_sqlite_anti_tilt_integration(self):
        bot = MaBuyTelegramBot()
        test_uid = f"anti_tilt_test_{int(time.time()*1000)}"

        # 1. Initial dashboard
        dash_text, markup = bot.handle_discipline_dashboard(test_uid)
        assert "ДНЕВНИК ДИСЦИПЛИНЫ" in dash_text
        assert "inline_keyboard" in markup

        # 2. Log a good trade (+5)
        dash_after_good, _ = bot.handle_log_tilt(test_uid, "FOLLOWED_PLAN", pnl_usd=55.0)
        assert "80 / 100" in dash_after_good

        # 3. Log FOMO (-15)
        dash_after_fomo, _ = bot.handle_log_tilt(test_uid, "FOMO_ENTRY", pnl_usd=-25.0)
        assert "65 / 100" in dash_after_fomo
        assert "FOMO входы без сигнала: <b>1</b>" in dash_after_fomo
        assert "РЕЖИМ АНТИ-ТИЛЬТ ТАЙМАУТА АКТИВЕН" in dash_after_fomo

        # 4. Manual cooldown
        dash_cooldown, _ = bot.handle_tilt_cooldown(test_uid)
        assert "РЕЖИМ АНТИ-ТИЛЬТ ТАЙМАУТА АКТИВЕН" in dash_cooldown
