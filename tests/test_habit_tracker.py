import os
os.environ["ENCRYPTION_KEY"] = "UN_5bf8rBxMiQOudLLjBJz42sY6vWznie0oxcjcKLYQ="

import time
import tempfile
import pytest
from fastapi.testclient import TestClient

from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker
from src.presentation.api.main import app
from src.presentation.telegram_bot.bot import MaBuyTelegramBot


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    tracker = SQLiteTracker(db_path=db_path)
    yield tracker
    try:
        os.remove(db_path)
    except Exception:
        pass


def test_rank_calculations(temp_db):
    """Verifies that the 10-level custom ranks calculate correctly based on XP."""
    r1 = temp_db.calculate_rank(150)
    assert r1["title"] == "NPC"
    assert r1["lvl"] == 1
    assert r1["icon"] == "🥔"
    assert r1["next_title"] == "Сайтама"

    r2 = temp_db.calculate_rank(1500)
    assert r2["title"] == "Сайтама"
    assert r2["lvl"] == 2
    assert r2["icon"] == "🥋"

    r3 = temp_db.calculate_rank(5000)
    assert r3["title"] == "Сигма"
    assert r3["lvl"] == 3
    assert r3["icon"] == "🗿"

    r4 = temp_db.calculate_rank(10000)
    assert r4["title"] == "Гигачад"
    assert r4["lvl"] == 4
    assert r4["icon"] == "🗿"

    r5 = temp_db.calculate_rank(20000)
    assert r5["title"] == "Грандмастер"
    assert r5["lvl"] == 5
    assert r5["icon"] == "🧙‍♂️"

    r6 = temp_db.calculate_rank(35000)
    assert r6["title"] == "Титан"
    assert r6["lvl"] == 6
    assert r6["icon"] == "🦾"

    r7 = temp_db.calculate_rank(60000)
    assert r7["title"] == "Берсерк"
    assert r7["lvl"] == 7
    assert r7["icon"] == "🗡️"

    r8 = temp_db.calculate_rank(85000)
    assert r8["title"] == "Йохан"
    assert r8["lvl"] == 8
    assert r8["icon"] == "♟️"

    r9 = temp_db.calculate_rank(120000)
    assert r9["title"] == "Истинный Воин"
    assert r9["lvl"] == 9
    assert r9["icon"] == "🛡️"

    r10 = temp_db.calculate_rank(160000)
    assert r10["title"] == "Фаундер"
    assert r10["lvl"] == 10
    assert r10["icon"] == "💼"


def test_habit_crud_and_status(temp_db):
    """Tests habit initialization, custom additions, completion logging, and streak logic."""
    user_id = "100000001"

    # 1. Defaults initialization
    habits = temp_db.get_user_habits(user_id)
    assert len(habits) == 5
    assert any(h["habit_id"] == "sat_math" for h in habits)
    assert any(h["habit_id"] == "ielts_prep" for h in habits)
    assert any(h["habit_id"] == "ml_code" for h in habits)

    # 2. Add custom habit
    custom_habit = {
        "title": "Gym Workout",
        "category": "health",
        "target_type": "checkbox",
        "target_value": 1,
        "icon": "🏋️",
        "color": "#e11d48",
        "deadline_time": "19:00"
    }
    saved = temp_db.save_habit(user_id, custom_habit)
    assert saved["title"] == "Gym Workout"
    h_id = saved["habit_id"]

    all_habits = temp_db.get_user_habits(user_id)
    assert len(all_habits) == 6

    # 3. Log completion
    today_status = temp_db.get_today_habits_status(user_id)
    assert today_status["completed_count"] == 0

    log_res = temp_db.log_habit_action(user_id, "sat_math", value=30, is_completed=True)
    assert log_res["completed_count"] == 1
    assert log_res["gamification"]["xp"] >= 50
    assert log_res["gamification"]["streak_current"] == 1

    # 4. Activity heatmap
    heatmap = temp_db.get_activity_heatmap(user_id, days=7)
    assert len(heatmap) == 7
    assert heatmap[-1]["completed"] == 1

    # 5. Delete habit
    deleted = temp_db.delete_habit(user_id, h_id, soft_delete=True)
    assert deleted is True
    active_habits = temp_db.get_user_habits(user_id)
    assert len(active_habits) == 5


def test_multimedia_journal_crud(temp_db):
    """Tests saving text, voice notes, photos and retrieving memory timeline."""
    user_id = "100000001"
    date_str = "2026-08-19"

    entry = temp_db.save_journal_entry(
        user_id=user_id,
        date_str=date_str,
        mood="🔥",
        title="Epic ML Breakthrough",
        content_text="Finished Attention LSTM training. SAT score 1480 on mock!",
        audio_path="/vault/media/voice_test.webm",
        media_paths=["/vault/media/screenshot1.png", "/vault/media/photo2.jpg"]
    )
    assert entry["date_str"] == date_str
    assert entry["mood"] == "🔥"
    assert len(entry["media_paths"]) == 2

    # Query timeline
    entries = temp_db.get_journal_entries(user_id)
    assert len(entries) >= 1
    assert entries[0]["content_text"] == "Finished Attention LSTM training. SAT score 1480 on mock!"

    # Query single date
    single = temp_db.get_journal_entry_by_date(user_id, date_str)
    assert single is not None
    assert single["title"] == "Epic ML Breakthrough"


def test_fastapi_habit_and_journal_endpoints():
    """Tests REST endpoints with authentication and privacy check."""
    client = TestClient(app)

    # 1. Private Vault Security: Unauthorized user rejected
    res_unauth = client.get("/api/v1/habits?user_id=stranger_hacker_999")
    assert res_unauth.status_code == 403

    # 2. Authorized user: owner (100000001)
    res_auth = client.get("/api/v1/habits?user_id=100000001")
    assert res_auth.status_code == 200
    data = res_auth.json()
    assert "habits" in data
    assert "gamification" in data
    assert data["gamification"]["rank"]["title"] in ["NPC", "Сайтама", "Сигма", "Гигачад", "Грандмастер", "Фаундер"]

    # 3. Log habit completion
    sat_habit = data["habits"][0]["habit_id"]
    res_log = client.post(f"/api/v1/habits/{sat_habit}/log", json={
        "user_id": "100000001",
        "value": 30,
        "is_completed": True
    })
    assert res_log.status_code == 200
    assert res_log.json()["completed_count"] >= 1

    # 4. Save journal entry
    res_journal = client.post("/api/v1/journal", json={
        "user_id": "100000001",
        "date_str": "2026-08-19",
        "mood": "🚀",
        "title": "Day 1 Complete",
        "content_text": "Building habit tracker with antigravity."
    })
    assert res_journal.status_code == 200
    assert res_journal.json()["entry"]["mood"] == "🚀"

    # 5. Serve WebApp HTML
    res_page = client.get("/habits")
    assert res_page.status_code == 200
    assert "MaBuy Titan" in res_page.text


def test_telegram_bot_habit_handlers():
    """Tests Telegram bot /habit, toggle, and morning/evening push messages."""
    bot = MaBuyTelegramBot()
    user_id = "100000001"

    # 1. Handle /habit command
    text, markup = bot.handle_habits(user_id)
    assert "ТРЕКЕР ПРИВЫЧЕК" in text
    assert "inline_keyboard" in markup
    assert any("WebApp" in btn.get("text", "") for row in markup["inline_keyboard"] for btn in row)

    # 2. Handle /journal command
    j_text, j_markup = bot.handle_journal(user_id)
    assert "ДНЕВНИК ВОСПОМИНАНИЙ" in j_text

    # 3. Morning Kickoff Alert
    morning_text, morning_markup = bot.send_morning_focus_alert(user_id)
    assert "ДОБРОЕ УТРО" in morning_text
    assert "SAT Math" in morning_text

    # 4. Evening Recap Alert
    evening_text, evening_markup = bot.send_evening_habit_recap(user_id)
    assert "ВЕЧЕРНИЙ ЧЕКИН" in evening_text or "ИДЕАЛЬНЫЙ ДЕНЬ" in evening_text
