import pytest
import time
from datetime import datetime, timezone
from src.domain.services.astra_news_alpha_engine import AstraNewsAlphaEngine, NewsEngineConfig

UTC = timezone.utc

def test_sentiment_decay_kernel(tmp_path):
    config = NewsEngineConfig(sqlite_path=str(tmp_path / "news_test1.db"))
    engine = AstraNewsAlphaEngine(config=config)
    
    # Simulate an injected shock at t0
    t0 = time.time()
    half_life_min = 1.0  # 1 minute = 60s
    dt0 = datetime.fromtimestamp(t0, tz=UTC)
    
    engine.record_sentiment_event(
        symbol="BTC",
        sentiment_score=1.0,
        shock_intensity=3,
        half_life_minutes=half_life_min,
        event_time=dt0
    )
    
    # Immediately at t0, decayed score should be close to ~1.0
    score_t0 = engine.get_decayed_sentiment("BTC", current_time=t0)
    assert 0.90 <= score_t0 <= 1.0, f"Immediate score should be ~1.0, got {score_t0}"
    
    # After exactly 1 half-life (60s), score should be ~0.5
    score_t1 = engine.get_decayed_sentiment("BTC", current_time=t0 + 60.0)
    assert 0.45 <= score_t1 <= 0.55, f"Score after 1 half-life should be ~0.50, got {score_t1}"
    
    # After 4 half-lives (240s), score should decay to ~0.0625
    score_t4 = engine.get_decayed_sentiment("BTC", current_time=t0 + 240.0)
    assert score_t4 < 0.10, f"Score after 4 half-lives should be < 0.10, got {score_t4}"
    engine.close()

def test_news_gate_veto_logic(tmp_path):
    config = NewsEngineConfig(sqlite_path=str(tmp_path / "news_test2.db"))
    engine = AstraNewsAlphaEngine(config=config)
    now = time.time()
    dt = datetime.fromtimestamp(now, tz=UTC)
    
    # Inject severe negative shock on ETH
    engine.record_sentiment_event(
        symbol="ETH",
        sentiment_score=-0.95,
        shock_intensity=5,
        half_life_minutes=60.0,
        event_time=dt
    )
    
    # BULLISH trade on ETH should be VETOED
    passed_bull, score_bull, reason_bull = engine.evaluate_news_gate("ETH", "BULLISH")
    assert not passed_bull, "Severe negative shock must veto BULLISH trade"
    assert "VETO" in reason_bull or "SHOCK" in reason_bull or "adverse" in reason_bull or "bearish" in reason_bull.lower()
    
    # BEARISH trade on ETH should pass (and get confidence boost)
    passed_bear, score_bear, reason_bear = engine.evaluate_news_gate("ETH", "BEARISH")
    assert passed_bear, "Severe negative shock should allow BEARISH trade"
    engine.close()

def test_unaffected_symbols_neutral(tmp_path):
    config = NewsEngineConfig(sqlite_path=str(tmp_path / "news_test3.db"))
    engine = AstraNewsAlphaEngine(config=config)
    now = time.time()
    dt = datetime.fromtimestamp(now, tz=UTC)
    
    # Record event affecting only SOL
    engine.record_sentiment_event(
        symbol="SOL",
        sentiment_score=-0.80,
        shock_intensity=4,
        half_life_minutes=30.0,
        event_time=dt
    )
    
    # BTC should remain neutral (0.0) and pass
    passed, score, _ = engine.evaluate_news_gate("BTC", "BULLISH")
    assert passed
    assert abs(score) < 0.05, f"BTC sentiment should be ~0.0, got {score}"
    engine.close()


