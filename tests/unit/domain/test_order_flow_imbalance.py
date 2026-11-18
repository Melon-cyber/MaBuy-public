import unittest
from src.domain.services.order_flow_imbalance import OrderFlowImbalanceService

class TestOrderFlowImbalanceService(unittest.TestCase):
    def setUp(self):
        self.service = OrderFlowImbalanceService(veto_threshold=0.40, confirmation_threshold=0.30)

    def test_balanced_orderbook(self):
        # Equal bids and asks across 4 levels
        bids = [[60000.0, 10.0], [59950.0, 10.0], [59900.0, 10.0], [59850.0, 10.0]]
        asks = [[60050.0, 10.0], [60100.0, 10.0], [60150.0, 10.0], [60200.0, 10.0]]
        res = self.service.calculate_obi(bids, asks)
        
        self.assertAlmostEqual(res["obi_score"], 0.0, places=3)
        self.assertFalse(res["has_bid_wall"])
        self.assertFalse(res["has_ask_wall"])
        self.assertIn("Сбалансированный", res["summary"])

    def test_bullish_orderbook_dominance(self):
        # Massive bid liquidity (e.g. 90 BTC vs 10 BTC)
        bids = [[60000.0, 50.0], [59950.0, 40.0]]
        asks = [[60050.0, 5.0], [60100.0, 5.0]]
        res = self.service.calculate_obi(bids, asks)
        
        self.assertGreater(res["obi_score"], 0.70)
        self.assertIn("Доминирование покупателей", res["summary"])

    def test_bearish_orderbook_veto_on_bullish_signal(self):
        # Heavy sell pressure (OBI = -0.70)
        bids = [[60000.0, 15.0]]
        asks = [[60050.0, 85.0]]
        res = self.service.calculate_obi(bids, asks)
        
        # Technical direction is BULLISH, but orderbook has severe ask pressure
        is_vetoed, reason = self.service.evaluate_microstructure_veto("BULLISH", res)
        self.assertTrue(is_vetoed)
        self.assertIn("перевес продавцов", reason)

    def test_bid_wall_veto_on_bearish_signal(self):
        # Balanced overall OBI (-0.05), but a single massive bid wall at support (50% of bids)
        bids = [[59000.0, 50.0], [58900.0, 15.0], [58800.0, 15.0], [58700.0, 20.0]] # total = 100
        asks = [[60100.0, 25.0], [60200.0, 25.0], [60300.0, 25.0], [60400.0, 25.0]] # total = 100
        res = self.service.calculate_obi(bids, asks)
        
        self.assertTrue(res["has_bid_wall"])
        self.assertEqual(res["bid_wall_price"], 59000.0)
        
        # Technical direction is BEARISH, but hitting major bid support wall
        is_vetoed, reason = self.service.evaluate_microstructure_veto("BEARISH", res)
        self.assertTrue(is_vetoed)
        self.assertIn("стенка поддержки", reason)

    def test_sol_usd_notional_wall_check(self):
        # 6.0 SOL @ $140 = $840 notional (< $150,000) -> MUST NOT be flagged as wall
        sol_bids_small = [[140.0, 6.0], [139.9, 2.0], [139.8, 1.0], [139.7, 1.0]] # total = 10 SOL, 60% concentration
        sol_asks_small = [[140.1, 2.5], [140.2, 2.5], [140.3, 2.5], [140.4, 2.5]]
        res_small = self.service.calculate_obi(sol_bids_small, sol_asks_small)
        self.assertFalse(res_small["has_bid_wall"], "Small $840 SOL order must not be flagged as institutional wall")

        # 1500.0 SOL @ $140 = $210,000 notional (>= $150,000) -> MUST be flagged as wall
        sol_bids_whale = [[140.0, 1500.0], [139.9, 500.0], [139.8, 500.0], [139.7, 500.0]] # total = 3000 SOL, 50% concentration
        res_whale = self.service.calculate_obi(sol_bids_whale, sol_asks_small)
        self.assertTrue(res_whale["has_bid_wall"], "Institutional $210,000 SOL wall must be flagged")
        self.assertEqual(res_whale["bid_wall_price"], 140.0)

    def test_empty_orderbook_fallback(self):
        res = self.service.calculate_obi([], [])
        self.assertEqual(res["obi_score"], 0.0)
        is_vetoed, reason = self.service.evaluate_microstructure_veto("BULLISH", res)
        self.assertFalse(is_vetoed)
        self.assertIsNone(reason)

if __name__ == "__main__":
    unittest.main()
