import unittest
import numpy as np
import pandas as pd
from src.infrastructure.ml.feature_engineering.pipeline import CryptoFeaturePipeline
from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher


class TestCryptoFeaturePipeline(unittest.TestCase):

    def setUp(self):
        fetcher = CryptoDataFetcher(cache_dir=".test_data_cache")
        self.raw_df = fetcher.fetch_ohlcv(symbol="BTCUSDT", limit=200)
        self.pipeline = CryptoFeaturePipeline()

    def test_transform_matrix_shapes(self):
        X, y, feature_names = self.pipeline.transform(self.raw_df)

        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertEqual(len(X), len(y))
        self.assertGreater(len(feature_names), 5)
        self.assertFalse(np.isnan(X).any(), "Feature matrix X contains unexpected NaNs")
        self.assertFalse(np.isnan(y).any(), "Target vector y contains unexpected NaNs")


if __name__ == "__main__":
    unittest.main()
