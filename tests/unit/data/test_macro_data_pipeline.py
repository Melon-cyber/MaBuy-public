"""
MaBuy V4 — Data Engineering Department Unit Test Suite.
Tests MacroDataPipeline:
  1. Live & Cached OHLCV Daily Ingestion (Binance API & Disk Caching).
  2. Indicator Calculations: 200-day SMA, 50-day EMA, Golden/Death Cross.
  3. Mayer Multiple & Valuation Regime classification.
  4. Historical Realized Volatilities (30d, 90d, 365d annualized).
  5. Daily Return Covariance & Correlation Matrices.
  6. Async interfaces & JSON serializability.
"""
import os
import shutil
import tempfile
import unittest
import asyncio
import json
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.infrastructure.data_fetchers.macro_data_pipeline import MacroDataPipeline
from src.infrastructure.data_fetchers.data_circuit_breaker import StaleDataException


class TestMacroDataPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="macro_cache_test_")
        self.pipeline = MacroDataPipeline(
            cache_dir=self.temp_dir,
            cache_ttl_sec=3600,
            default_limit=400
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _generate_mock_daily_df(self, n_days: int = 400, base_price: float = 60000.0, trend: float = 0.001) -> pd.DataFrame:
        """Generates synthetic daily candles for deterministic mathematical tests."""
        dates = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n_days, freq="D")
        records = []
        price = base_price
        np.random.seed(42)
        for d in dates:
            daily_ret = trend + np.random.normal(0, 0.02)
            open_p = price
            close_p = price * (1.0 + daily_ret)
            high_p = max(open_p, close_p) * 1.01
            low_p = min(open_p, close_p) * 0.99
            records.append({
                "timestamp": d,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": 1000.0
            })
            price = close_p
        return pd.DataFrame(records)

    def test_pipeline_initialization(self):
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertEqual(self.pipeline.cache_ttl_sec, 3600)
        self.assertEqual(self.pipeline.default_limit, 400)
        self.assertIn("BTC", self.pipeline.DEFAULT_SYMBOLS)

    def test_calculate_macro_indicators_math(self):
        """Validates mathematical correctness of 200 SMA, 50 EMA, Mayer Multiple, RV."""
        df = self._generate_mock_daily_df(n_days=400, base_price=50000.0)
        metrics = self.pipeline.calculate_macro_indicators(df, symbol="BTC")

        # Basic fields
        self.assertEqual(metrics["symbol"], "BTC")
        self.assertEqual(metrics["candles_count"], 400)
        self.assertGreater(metrics["current_price"], 0)

        # 200 SMA and 50 EMA
        expected_sma_200 = float(df["close"].rolling(200).mean().iloc[-1])
        expected_ema_50 = float(df["close"].ewm(span=50, adjust=False).mean().iloc[-1])
        self.assertAlmostEqual(metrics["sma_200"], round(expected_sma_200, 2), places=1)
        self.assertAlmostEqual(metrics["ema_50"], round(expected_ema_50, 2), places=1)

        # Cross Status
        if metrics["ema_50"] >= metrics["sma_200"]:
            self.assertEqual(metrics["trend_cross"], "GOLDEN_CROSS")
        else:
            self.assertEqual(metrics["trend_cross"], "DEATH_CROSS")

        # Mayer Multiple (Price / 200 SMA)
        expected_mayer = metrics["current_price"] / metrics["sma_200"]
        self.assertAlmostEqual(metrics["mayer_multiple"], round(expected_mayer, 3), places=2)
        self.assertIn(metrics["mayer_regime"], ["DEEP_VALUE", "ACCUMULATION", "BULLISH_EXPANSION", "OVERHEATED", "EUPHORIC_BUBBLE"])

        # Realized Volatility
        returns = df["close"].pct_change().dropna()
        expected_rv_30d = float(returns.tail(30).std() * np.sqrt(365.0))
        expected_rv_365d = float(returns.tail(365).std() * np.sqrt(365.0))
        self.assertAlmostEqual(metrics["realized_volatility_30d"], round(expected_rv_30d, 4), places=3)
        self.assertAlmostEqual(metrics["realized_volatility_365d"], round(expected_rv_365d, 4), places=3)

        # Drawdowns and Highs
        expected_high = float(df["high"].tail(365).max())
        expected_low = float(df["low"].tail(365).min())
        self.assertAlmostEqual(metrics["high_365d"], round(expected_high, 2), places=1)
        self.assertAlmostEqual(metrics["low_365d"], round(expected_low, 2), places=1)
        self.assertLessEqual(metrics["drawdown_from_365d_high_pct"], 0.0)

    def test_mayer_regimes_classification(self):
        """Verifies all Mayer Multiple boundary regimes."""
        # 1. Deep Value (< 0.8)
        df_low = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=250, freq="D"),
            "open": [100.0] * 249 + [50.0],
            "high": [105.0] * 249 + [52.0],
            "low": [95.0] * 249 + [48.0],
            "close": [100.0] * 249 + [50.0],
            "volume": [100.0] * 250
        })
        m_low = self.pipeline.calculate_macro_indicators(df_low, "SOL")
        self.assertEqual(m_low["mayer_regime"], "DEEP_VALUE")
        self.assertEqual(m_low["mayer_signal"], "STRONG_BUY")

        # 2. Euphoric Bubble (> 2.4)
        df_bubble = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=250, freq="D"),
            "open": [100.0] * 249 + [300.0],
            "high": [105.0] * 249 + [310.0],
            "low": [95.0] * 249 + [290.0],
            "close": [100.0] * 249 + [300.0],
            "volume": [100.0] * 250
        })
        m_bubble = self.pipeline.calculate_macro_indicators(df_bubble, "SOL")
        self.assertEqual(m_bubble["mayer_regime"], "EUPHORIC_BUBBLE")
        self.assertEqual(m_bubble["mayer_signal"], "STRONG_SELL")

    def test_caching_and_atomic_lock(self):
        """Verifies disk caching, freshness check and reload."""
        mock_raw = [
            [1700000000000 + i * 86400000, "100.0", "105.0", "95.0", "102.0", "500.0"]
            for i in range(250)
        ]
        with patch.object(self.pipeline, "_fetch_from_binance_network", return_value=mock_raw) as mock_net:
            # 1. First fetch -> calls network and writes cache
            df1 = self.pipeline.fetch_daily_candles("ETH", limit=250)
            self.assertEqual(len(df1), 250)
            mock_net.assert_called_once()
            
            cache_file = os.path.join(self.temp_dir, "ETH_1d.csv")
            self.assertTrue(os.path.exists(cache_file))

            # 2. Second fetch -> fast path (cache hit, no network call)
            mock_net.reset_mock()
            df2 = self.pipeline.fetch_daily_candles("ETH", limit=250)
            self.assertEqual(len(df2), 250)
            mock_net.assert_not_called()

            # 3. Force refresh -> bypasses cache and calls network again
            df3 = self.pipeline.fetch_daily_candles("ETH", limit=250, force_refresh=True)
            self.assertEqual(len(df3), 250)
            mock_net.assert_called_once()

    def test_covariance_and_correlation_matrices(self):
        """Verifies mathematical consistency of covariance and correlation matrices."""
        symbols = ["BTC", "ETH", "SOL"]
        # Pre-seed cache files with synthetic data
        for sym, base, trend in [("BTC", 60000.0, 0.001), ("ETH", 3000.0, 0.0012), ("SOL", 150.0, 0.002)]:
            df = self._generate_mock_daily_df(n_days=120, base_price=base, trend=trend)
            df.to_csv(os.path.join(self.temp_dir, f"{sym}_1d.csv"), index=False)

        # 1. Covariance Matrix
        cov = self.pipeline.get_assets_covariance_matrix(symbols, history_days=90)
        self.assertEqual(cov.shape, (3, 3))
        self.assertListEqual(list(cov.columns), symbols)
        self.assertListEqual(list(cov.index), symbols)
        # Diagonal elements (variance) must be positive
        for sym in symbols:
            self.assertGreater(cov.loc[sym, sym], 0.0)
        # Symmetry: cov(A, B) == cov(B, A)
        self.assertAlmostEqual(cov.loc["BTC", "ETH"], cov.loc["ETH", "BTC"], places=6)

        # 2. Annualized Covariance
        cov_ann = self.pipeline.get_assets_covariance_matrix(symbols, history_days=90, annualized=True)
        self.assertAlmostEqual(cov_ann.loc["BTC", "BTC"], cov.loc["BTC", "BTC"] * 365.0, places=4)

        # 3. Correlation Matrix
        corr = self.pipeline.get_assets_correlation_matrix(symbols, history_days=90)
        self.assertEqual(corr.shape, (3, 3))
        for sym in symbols:
            # Diagonal is strictly 1.0
            self.assertAlmostEqual(corr.loc[sym, sym], 1.0, places=5)
            for other in symbols:
                val = corr.loc[sym, other]
                self.assertTrue(-1.0 <= val <= 1.0)
                self.assertAlmostEqual(val, corr.loc[other, sym], places=5)

    def test_json_serializability_of_snapshots(self):
        """Ensures all fields returned by snapshot are native Python types and JSON serializable."""
        df = self._generate_mock_daily_df(n_days=250, base_price=70000.0)
        df.to_csv(os.path.join(self.temp_dir, "BTC_1d.csv"), index=False)

        snapshot = self.pipeline.get_macro_snapshot("BTC")
        json_str = json.dumps(snapshot)
        self.assertIsInstance(json_str, str)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["symbol"], "BTC")
        self.assertIsInstance(parsed["realized_volatility_30d_pct"], (int, float))
        self.assertIsInstance(parsed["recent_cross_detected"], bool)

    def test_async_methods(self):
        """Tests asynchronous non-blocking interfaces."""
        df = self._generate_mock_daily_df(n_days=220, base_price=50000.0)
        df.to_csv(os.path.join(self.temp_dir, "BTC_1d.csv"), index=False)

        async def run_async():
            df_async = await self.pipeline.fetch_daily_candles_async("BTC")
            snap_async = await self.pipeline.get_macro_snapshot_async("BTC")
            cov_async = await self.pipeline.get_assets_covariance_matrix_async(["BTC"], history_days=30)
            return len(df_async), snap_async["symbol"], cov_async.shape

        n_rows, sym, shape = asyncio.run(run_async())
        self.assertEqual(n_rows, 220)
        self.assertEqual(sym, "BTC")
        self.assertEqual(shape, (1, 1))


if __name__ == "__main__":
    unittest.main()
