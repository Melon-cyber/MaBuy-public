import os
os.environ["ENCRYPTION_KEY"] = "UN_5bf8rBxMiQOudLLjBJz42sY6vWznie0oxcjcKLYQ="

import shutil
import tempfile
import pytest
from fastapi.testclient import TestClient

from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker
from src.domain.services.habit_ai_coach import HabitAICoachService
from src.presentation.telegram_bot.formatter import TelegramMessageFormatter
from src.presentation.telegram_bot.bot import MaBuyTelegramBot
from src.presentation.api.main import app


@pytest.fixture
def temp_tracker():
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_habits_prod.db")
    tracker = SQLiteTracker(db_path=db_path)
    yield tracker
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_habit_ai_coach_service():
    """Tests AI coach synthesis for week, month, and year horizons."""
    sample_stats = {
        "days": 7,
        "total_active_days": 6,
        "total_completed_tasks": 30,
        "total_possible_tasks": 35,
        "completion_rate": 85.7,
        "consistency_score": 85.7,
        "total_minutes_logged": 420,
        "total_hours_logged": 7.0,
        "xp_earned": 900,
        "streak_current": 12,
        "streak_best": 15,
        "best_habit": {"title": "SAT Math", "completed": 7},
        "weakest_habit": {"title": "Здоровый сон", "completed": 5},
        "category_breakdown": {
            "academic": {"completed": 14, "minutes": 210},
            "code": {"completed": 7, "minutes": 210}
        }
    }

    # Week
    res_w = HabitAICoachService.generate_ai_coaching_insight(sample_stats, period="week", lang="ru")
    assert "A+ (Элитный Титан)" in res_w["grade"]
    assert "7.0" in str(res_w["total_hours"])
    assert res_w["readiness_index"] >= 90
    assert "SAT Math" in res_w["recommendation"]

    # Month
    res_m = HabitAICoachService.generate_ai_coaching_insight(sample_stats, period="month", lang="ru")
    assert "30 дней" in res_m["title"]
    assert "95%" in str(res_m["recommendation"])

    # Year
    res_y = HabitAICoachService.generate_ai_coaching_insight(sample_stats, period="year", lang="ru")
    assert "Годовая" in res_y["title"]
    assert "Назарбаев Университет" in res_y["recommendation"]


def test_sqlite_tracker_multi_audio_appending(temp_tracker):
    """Tests multi-audio recording and appending to daily journal entry without data loss."""
    user_id = "100000001"
    date_str = "2026-08-21"

    # Append 1st voice note (morning thoughts)
    entry1 = temp_tracker.append_journal_audio(
        user_id=user_id,
        date_str=date_str,
        audio_path="/vault/media/voice_morning.ogg",
        duration=45,
        label="Утренний настрой"
    )
    assert len(entry1["audio_tracks"]) == 1
    assert entry1["audio_tracks"][0]["label"] == "Утренний настрой"
    assert entry1["audio_tracks"][0]["duration"] == 45
    track1_id = entry1["audio_tracks"][0]["id"]

    # Append 2nd voice note (evening debrief)
    entry2 = temp_tracker.append_journal_audio(
        user_id=user_id,
        date_str=date_str,
        audio_path="/vault/media/voice_evening.ogg",
        duration=90,
        label="Итоги SAT и кодинга"
    )
    assert len(entry2["audio_tracks"]) == 2
    assert entry2["audio_tracks"][0]["id"] == track1_id
    assert entry2["audio_tracks"][1]["label"] == "Итоги SAT и кодинга"
    assert entry2["audio_tracks"][1]["duration"] == 90

    # Retrieve entry by date
    fetched = temp_tracker.get_journal_entry_by_date(user_id, date_str)
    assert fetched is not None
    assert len(fetched["audio_tracks"]) == 2

    # Delete 1st audio track
    updated = temp_tracker.delete_journal_audio_track(user_id, date_str, track1_id)
    assert len(updated["audio_tracks"]) == 1
    assert updated["audio_tracks"][0]["label"] == "Итоги SAT и кодинга"


def test_sqlite_tracker_habits_statistics(temp_tracker):
    """Tests habits statistical aggregation over 7, 30, and 365 days."""
    user_id = "100000001"
    temp_tracker.init_default_academic_habits(user_id, force_reset=True)

    import datetime
    today_str = datetime.date.today().isoformat()
    yesterday_str = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    # Log several days of habit completions
    temp_tracker.log_habit_action(user_id, "sat_math", today_str, value=30, is_completed=True)
    temp_tracker.log_habit_action(user_id, "ielts_prep", today_str, value=30, is_completed=True)
    temp_tracker.log_habit_action(user_id, "ml_code", today_str, value=45, is_completed=True)

    temp_tracker.log_habit_action(user_id, "sat_math", yesterday_str, value=30, is_completed=True)
    temp_tracker.log_habit_action(user_id, "ml_code", yesterday_str, value=45, is_completed=True)

    # Week stats
    stats_w = temp_tracker.get_habits_statistics(user_id, period="week", lang="ru")
    assert stats_w["total_completed_tasks"] == 5
    assert stats_w["total_minutes_logged"] == 180
    assert stats_w["total_hours_logged"] == 3.0
    assert "ai_coach" in stats_w
    assert "grade" in stats_w["ai_coach"]

    # Month stats
    stats_m = temp_tracker.get_habits_statistics(user_id, period="month", lang="ru")
    assert stats_m["days"] == 30
    assert stats_m["total_completed_tasks"] == 5

    # Year stats
    stats_y = temp_tracker.get_habits_statistics(user_id, period="year", lang="ru")
    assert stats_y["days"] == 365


def test_telegram_formatters_and_handlers(temp_tracker):
    """Tests habit statistics and voice feedback formatting and bot handlers."""
    user_id = "100000001"
    temp_tracker.init_default_academic_habits(user_id, force_reset=True)

    bot_handler = MaBuyTelegramBot()
    bot_handler.usage_tracker = temp_tracker

    # Test /stats_habits handler
    text, markup = bot_handler.handle_habit_statistics(user_id, period="week")
    assert "НЕДЕЛЬНЫЙ СПРИНТ" in text
    assert "ВЕРДИКТ ИИ-КОУЧА" in text
    assert len(markup["inline_keyboard"]) == 2
    assert markup["inline_keyboard"][0][0]["callback_data"] == "stats_period_week"

    # Test month switch
    text_m, markup_m = bot_handler.handle_habit_statistics(user_id, period="month")
    assert "МЕСЯЧНЫЙ ОБЗОР" in text_m

    # Test voice note save handler
    v_text, v_markup = bot_handler.handle_save_voice_note(user_id, "/vault/media/sample_voice.ogg", duration=42)
    assert "ГОЛОСОВАЯ ЗАМЕТКА ДОБАВЛЕНА" in v_text
    assert "42 сек" in v_text
    assert "+50 XP" in v_text


def test_fastapi_analytics_and_multi_audio_endpoints(temp_tracker):
    """Tests FastAPI endpoints for habit analytics and multi-audio append."""
    from src.presentation.api.main import usage_tracker
    client = TestClient(app)
    user_id = "100000001"

    # 1. Test GET /api/v1/habits/analytics
    res = client.get(f"/api/v1/habits/analytics?user_id={user_id}&period=week&lang=ru")
    assert res.status_code == 200
    data = res.json()
    assert "ai_coach" in data
    assert "completion_rate" in data
    assert "category_breakdown" in data

    # 2. Test POST /api/v1/journal/audio (Append audio track)
    append_res = client.post("/api/v1/journal/audio", json={
        "user_id": user_id,
        "date_str": "2026-08-21",
        "audio_path": "/vault/media/test_voice_api.ogg",
        "duration": 35,
        "label": "Тестовая мысль API"
    })
    assert append_res.status_code == 200
    entry = append_res.json()["entry"]
    assert len(entry["audio_tracks"]) >= 1
    track_id = entry["audio_tracks"][-1]["id"]

    # 3. Test DELETE /api/v1/journal/audio/{track_id}
    del_res = client.delete(f"/api/v1/journal/audio/{track_id}?user_id={user_id}&date=2026-08-21")
    assert del_res.status_code == 200
    updated_entry = del_res.json()["entry"]
    assert not any(t["id"] == track_id for t in updated_entry.get("audio_tracks", []))
