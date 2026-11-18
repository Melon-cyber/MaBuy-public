import time
import pytest
from src.infrastructure.data_fetchers.liquidation_streamer import LiquidationCascadeRadar, LiquidationEvent


class TestLiquidationCascadeRadar:
    def test_record_liquidation_and_bounded_memory(self):
        radar = LiquidationCascadeRadar(max_history_events=100)
        base_time = 1700000000000

        # Push 150 events to verify maxlen=100 bound
        for i in range(150):
            radar.record_liquidation(
                symbol="BTCUSDT",
                side="SELL",
                orig_qty=1.5,
                price=50000.0,
                event_time_ms=base_time + i * 1000
            )

        events = radar._events_by_symbol["BTCUSDT"]
        assert len(events) == 100
        # Earliest event should be item index 50
        assert events[0].event_time == base_time + 50 * 1000
        assert events[-1].event_time == base_time + 149 * 1000

    def test_zscore_capitulation_spike_detection(self):
        radar = LiquidationCascadeRadar(max_history_events=1000)
        base_time = 1700000000000

        # Simulate 55 minutes of normal low volume (10k per 5 min)
        # 11 buckets of 5 min = 55 min
        for b in range(11):
            t = base_time + b * (5 * 60 * 1000)
            radar.record_liquidation("BTCUSDT", "SELL", 0.2, 50000.0, t) # 10k USD

        # In the 12th bucket (current 5 min), simulate massive long liquidation spike (500k USD)
        spike_time = base_time + 11 * (5 * 60 * 1000) + 10000
        radar.record_liquidation("BTCUSDT", "SELL", 10.0, 50000.0, spike_time)

        z_score, vol_5m, dominant_side, metrics = radar.calculate_liquidation_zscore("BTCUSDT", reference_time_ms=spike_time)

        assert z_score >= 3.0, f"Z-Score {z_score} should exceed 3.0 on massive liquidation spike"
        assert dominant_side == "LONG_LIQ"
        assert metrics["is_capitulation_spike"] is True
        assert metrics["volume_5m_usd"] >= 500000.0

    def test_cvd_calculation_by_event_time(self):
        radar = LiquidationCascadeRadar(max_history_events=500)
        base_time = 1700000000000

        # Record taker buys (+50k, +30k) and taker sells (-20k)
        radar.record_trade_delta("ETHUSDT", is_buyer_maker=False, volume_usd=50000.0, event_time_ms=base_time)
        radar.record_trade_delta("ETHUSDT", is_buyer_maker=False, volume_usd=30000.0, event_time_ms=base_time + 60000)
        radar.record_trade_delta("ETHUSDT", is_buyer_maker=True, volume_usd=20000.0, event_time_ms=base_time + 120000)

        cvd = radar.calculate_cvd_5m("ETHUSDT", reference_time_ms=base_time + 120000)
        # Expected: 50000 + 30000 - 20000 = 60000
        assert cvd == pytest.approx(60000.0)
