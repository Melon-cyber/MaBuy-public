import unittest
import time
from src.domain.entities.tick import Tick, MLSignalRequest
from src.domain.services.quant_analyzer import QuantAnalyzer


class TestQuantAnalyzer(unittest.TestCase):

    def setUp(self):
        self.quant_analyzer = QuantAnalyzer()
        # Force fallback heuristic mode for unit tests to verify feature math deterministically
        self.quant_analyzer.model = None

    def test_quant_analyzer_empty_ticks(self):
        request = MLSignalRequest(symbol="BTCUSDT", ticks=[])
        response = self.quant_analyzer.analyze(request)
        
        self.assertEqual(response.direction, "NEUTRAL")
        self.assertEqual(response.confidence_score, 0.0)

    def test_quant_analyzer_bullish_signal(self):
        base_time = int(time.time() * 1000)
        ticks = [
            Tick(trade_id=1, timestamp=base_time, price=60000.0, size=0.1, side=1),
            Tick(trade_id=2, timestamp=base_time+100, price=60010.0, size=0.5, side=1),
            Tick(trade_id=3, timestamp=base_time+200, price=60005.0, size=0.05, side=-1),
            Tick(trade_id=4, timestamp=base_time+300, price=60020.0, size=1.2, side=1),
        ]
        request = MLSignalRequest(symbol="BTCUSDT", ticks=ticks)
        
        response = self.quant_analyzer.analyze(request)
        
        self.assertEqual(response.direction, "BULLISH")
        self.assertTrue(response.confidence_score > 0.0)

    def test_quant_analyzer_bearish_signal(self):
        base_time = int(time.time() * 1000)
        ticks = [
            Tick(trade_id=1, timestamp=base_time, price=60000.0, size=0.1, side=-1),
            Tick(trade_id=2, timestamp=base_time+100, price=59990.0, size=1.5, side=-1),
            Tick(trade_id=3, timestamp=base_time+200, price=59995.0, size=0.05, side=1),
            Tick(trade_id=4, timestamp=base_time+300, price=59980.0, size=2.0, side=-1),
        ]
        request = MLSignalRequest(symbol="BTCUSDT", ticks=ticks)
        
        response = self.quant_analyzer.analyze(request)
        
        self.assertEqual(response.direction, "BEARISH")
        self.assertTrue(response.confidence_score > 0.0)


if __name__ == "__main__":
    unittest.main()
