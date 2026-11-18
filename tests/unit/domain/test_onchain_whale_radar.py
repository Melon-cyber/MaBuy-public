import unittest
from src.domain.services.derivatives_flow_radar import DerivativesFlowRadarService

class TestOnChainBackwardsCompatibility(unittest.TestCase):
    def test_derivatives_service_fallback(self):
        res = DerivativesFlowRadarService.get_derivatives_radar("BTC", current_spot_price=64000.0, lang="ru")
        self.assertEqual(res["symbol"], "BTC")
        self.assertGreaterEqual(res["confidence_pct"], 50.0)

if __name__ == "__main__":
    unittest.main()
