import pytest
from src.domain.services.distance_decay_obi import DistanceDecayedOBIService, DistanceDecayedOBIResult


class TestDistanceDecayedOBI:
    def test_d_obi_symmetry_and_weighting(self):
        service = DistanceDecayedOBIService(decay_factor=0.5, veto_threshold=0.40)
        
        # Bids: [50000, 10.0], [49950, 20.0]
        # Asks: [50050, 10.0], [50100, 20.0]
        bids = [[50000.0, 10.0], [49950.0, 20.0]]
        asks = [[50050.0, 10.0], [50100.0, 20.0]]
        
        res = service.calculate_d_obi(bids, asks)
        assert res.d_obi_score == pytest.approx(0.0, abs=1e-3)
        assert res.dominant_side == "NEUTRAL"
        assert res.microstructure_veto is False

    def test_spoof_wall_far_from_spread_discounted(self):
        service = DistanceDecayedOBIService(decay_factor=0.5, veto_threshold=0.40)
        
        # Realistic spread at 50,000
        # Close bids: 50,000 (10 BTC)
        # Close asks: 50,050 (10 BTC)
        # Fake spoof ask wall placed far at 51,500 (100 BTC)
        bids = [[50000.0, 10.0]]
        asks = [[50050.0, 10.0], [51500.0, 100.0]]
        
        res = service.calculate_d_obi(bids, asks)
        # Raw OBI would be (10 - 110) / 120 = -0.833 (Extremely bearish false signal)
        assert res.raw_obi_score < -0.80
        
        # Distance-Decayed OBI heavily discounts the 51,500 wall
        # So D-OBI should be much closer to 0.0 than raw_obi
        assert abs(res.d_obi_score) < abs(res.raw_obi_score)
        assert res.d_obi_score > -0.50

    def test_microstructure_veto_bullish_into_heavy_ask_pressure(self):
        service = DistanceDecayedOBIService(decay_factor=0.5, veto_threshold=0.40)
        
        # Heavy real ask pressure directly at the spread
        bids = [[50000.0, 2.0]]
        asks = [[50010.0, 25.0]]
        
        # Bullish signal trying to buy into heavy ask resistance
        res = service.calculate_d_obi(bids, asks, signal_direction="BULLISH")
        assert res.microstructure_veto is True
        assert "VETO: Bullish signal rejected by heavy Ask" in res.veto_reason
