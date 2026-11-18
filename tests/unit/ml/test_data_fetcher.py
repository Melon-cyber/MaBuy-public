import os
import shutil
import unittest
from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher


class TestCryptoDataFetcher(unittest.TestCase):

    def setUp(self):
        self.test_cache_dir = ".test_data_cache"
        self.fetcher = CryptoDataFetcher(cache_dir=self.test_cache_dir)

    def tearDown(self):
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir, ignore_errors=True)

    def test_fetch_ohlcv_shape_and_columns(self):
        df = self.fetcher.fetch_ohlcv(symbol="BTCUSDT", limit=100)
        self.assertEqual(len(df), 100)
        expected_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for col in expected_cols:
            self.assertIn(col, df.columns)

    def test_cache_creation(self):
        _ = self.fetcher.fetch_ohlcv(symbol="ETHUSDT", limit=50)
        cache_path = self.fetcher._get_cache_path("ETHUSDT", "1h")
        self.assertTrue(os.path.exists(cache_path))


if __name__ == "__main__":
    unittest.main()
