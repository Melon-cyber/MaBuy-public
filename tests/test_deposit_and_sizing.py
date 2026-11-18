import pytest
from src.infrastructure.persistence.sqlite_tracker import SQLiteTracker
from src.presentation.telegram_bot.formatter import TelegramMessageFormatter

@pytest.fixture
def temp_tracker(tmp_path):
    db_file = tmp_path / "test_tracker.db"
    return SQLiteTracker(db_path=str(db_file))

def test_user_deposit_persistence(temp_tracker):
    user_id = "test_trader_999"
    temp_tracker.set_user_deposit(user_id, 1000.0)
    assert temp_tracker.get_user_deposit(user_id) == 1000.0

def test_format_push_alert_message_and_deposit_menu():
    alert_mock = {
        "symbol": "SOL",
        "direction": "BULLISH",
        "current_price": 150.0,
        "target_price": 155.0,
        "stop_loss_price": 147.0,
        "tp_pct": 3.33,
        "sl_pct": 2.0,
        "reward_risk_ratio": 1.67,
        "max_safe_leverage": 15,
        "confidence_pct": 75,
        "timeframe": "1h",
        "personal_recommendation": "Выкуп пролива."
    }

    msg = TelegramMessageFormatter.format_push_alert_message(alert_mock, user_deposit=2500.0, lang="ru")
    assert "#SOL" in msg
    assert "LONG" in msg

def test_format_prediction_result_with_custom_deposit():
    from src.domain.entities.prediction import PredictionResult

    pred = PredictionResult(
        symbol="BTC",
        current_price=60000.0,
        predicted_target_price=62000.0,
        predicted_log_return=0.033,
        direction="BULLISH",
        confidence_score=0.74,
        volatility_estimate=0.02,
        timeframe="1h",
        personal_recommendation="Выкуп пролива.",
        model_version="LightGBM_v4.1"
    )

    msg_ru = TelegramMessageFormatter.format_prediction_result(pred, lang="ru", experience="beginner", user_deposit=3500.0)
    assert "#BTC" in msg_ru
    assert "LONG" in msg_ru
