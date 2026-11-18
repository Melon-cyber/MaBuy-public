"""
MaBuy V4 — Data Engineering Department Unit & Multiprocessing Test Suite.
Tests:
  1. DataCircuitBreaker state machine (CLOSED -> OPEN -> HALF_OPEN -> CLOSED).
  2. AtomicCacheGuard single-flight locking and stale lock reclamation.
  3. Multiprocessing stress test verifying zero corruption under concurrent OS processes.
  4. Live-trading StaleDataException enforcement.
"""
import os
import time
import shutil
import unittest
import multiprocessing
import pandas as pd
from unittest.mock import patch, MagicMock

from src.infrastructure.data_fetchers.data_circuit_breaker import (
    DataCircuitBreaker,
    CircuitState,
    StaleDataException
)
from src.infrastructure.data_fetchers.atomic_cache_guard import (
    AtomicCacheGuard,
    LockAcquisitionTimeout
)
from src.infrastructure.data_fetchers.binance_fetcher import CryptoDataFetcher
from src.infrastructure.nlp.rss_sentiment_analyzer import RSSSentimentAnalyzer


def _multiprocess_worker_write(file_path: str, worker_id: int, write_count: int, results_queue):
    """Worker function executed in a separate OS process to test AtomicCacheGuard under true multiprocessing."""
    try:
        for i in range(write_count):
            with AtomicCacheGuard(file_path, timeout=5.0):
                # Read current count, increment, write back
                val = 0
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        content = f.read().strip()
                        if content:
                            val = int(content)
                val += 1
                time.sleep(0.005)  # Simulate file I/O latency
                with open(file_path, "w") as f:
                    f.write(str(val))
        results_queue.put((worker_id, True, None))
    except Exception as e:
        results_queue.put((worker_id, False, str(e)))


class TestDataCircuitBreaker(unittest.TestCase):
    def setUp(self):
        self.cb = DataCircuitBreaker("test_service", failure_threshold=3, recovery_timeout_sec=0.2)
        self.cb.reset()

    def test_circuit_breaker_transitions(self):
        self.assertEqual(self.cb.state, CircuitState.CLOSED)
        self.assertTrue(self.cb.allow_request())

        # 1st failure
        self.cb.record_failure(Exception("Network error 1"))
        self.assertEqual(self.cb.state, CircuitState.CLOSED)
        self.assertTrue(self.cb.allow_request())

        # 2nd failure
        self.cb.record_failure(Exception("Network error 2"))
        self.assertEqual(self.cb.state, CircuitState.CLOSED)
        self.assertTrue(self.cb.allow_request())

        # 3rd failure -> Trip to OPEN
        self.cb.record_failure(Exception("Network error 3"))
        self.assertEqual(self.cb.state, CircuitState.OPEN)
        self.assertTrue(self.cb.is_open())
        self.assertFalse(self.cb.allow_request())

        # Wait for cooldown
        time.sleep(0.25)
        # Should transition to HALF_OPEN probe
        self.assertTrue(self.cb.allow_request())
        self.assertEqual(self.cb.state, CircuitState.HALF_OPEN)

        # Successful probe resets to CLOSED
        self.cb.record_success()
        self.assertEqual(self.cb.state, CircuitState.CLOSED)
        self.assertEqual(self.cb.failure_count, 0)

    def test_half_open_failure_re_trips(self):
        # Trip to OPEN
        for _ in range(3):
            self.cb.record_failure()
        self.assertEqual(self.cb.state, CircuitState.OPEN)

        time.sleep(0.25)
        self.assertTrue(self.cb.allow_request())
        self.assertEqual(self.cb.state, CircuitState.HALF_OPEN)

        # Failure in HALF_OPEN trips immediately back to OPEN
        self.cb.record_failure(Exception("Probe failed"))
        self.assertEqual(self.cb.state, CircuitState.OPEN)
        self.assertFalse(self.cb.allow_request())


class TestAtomicCacheGuard(unittest.TestCase):
    def setUp(self):
        self.test_dir = ".test_cache_dir"
        os.makedirs(self.test_dir, exist_ok=True)
        self.test_file = os.path.join(self.test_dir, "test_data.txt")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_basic_acquire_and_release(self):
        guard = AtomicCacheGuard(self.test_file, timeout=1.0)
        with guard:
            self.assertTrue(os.path.exists(guard.lock_file_path))
        self.assertFalse(os.path.exists(guard.lock_file_path))

    def test_stale_lock_reclamation(self):
        lock_file = self.test_file + ".lock"
        # Create a simulated abandoned lockfile
        with open(lock_file, "w") as f:
            f.write("99999:dead_pid")

        # Artificially set mtime to 20 seconds ago
        old_time = time.time() - 20.0
        os.utime(lock_file, (old_time, old_time))

        # AtomicCacheGuard with stale_lock_max_age=5.0 should reclaim the lock
        guard = AtomicCacheGuard(self.test_file, timeout=2.0, stale_lock_max_age=5.0)
        with guard:
            self.assertTrue(guard._is_locked)
        self.assertFalse(os.path.exists(lock_file))

    def test_multiprocess_concurrency(self):
        """Spawns 4 independent OS processes incrementing a shared counter under AtomicCacheGuard."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

        num_processes = 4
        writes_per_process = 10
        queue = multiprocessing.Queue()
        processes = []

        for p_id in range(num_processes):
            p = multiprocessing.Process(
                target=_multiprocess_worker_write,
                args=(self.test_file, p_id, writes_per_process, queue)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout=10.0)

        # Check worker results
        results = []
        while not queue.empty():
            results.append(queue.get())

        self.assertEqual(len(results), num_processes)
        for worker_id, success, err in results:
            self.assertTrue(success, f"Worker {worker_id} failed with error: {err}")

        with open(self.test_file, "r") as f:
            final_count = int(f.read().strip())

        expected_count = num_processes * writes_per_process
        self.assertEqual(final_count, expected_count, "Multiprocess counter mismatch: atomic lock failure!")


class TestCryptoDataFetcherResilience(unittest.TestCase):
    def setUp(self):
        self.test_cache = ".test_crypto_cache"
        os.makedirs(self.test_cache, exist_ok=True)
        self.fetcher = CryptoDataFetcher(cache_dir=self.test_cache, max_cache_age_sec=2)
        self.fetcher.circuit_breaker.reset()

    def tearDown(self):
        if os.path.exists(self.test_cache):
            shutil.rmtree(self.test_cache, ignore_errors=True)
        self.fetcher.circuit_breaker.reset()

    @patch("src.infrastructure.data_fetchers.binance_fetcher.urllib.request.urlopen")
    def test_require_fresh_stale_data_exception(self, mock_urlopen):
        # Directly trip the circuit breaker with 3 failures
        for _ in range(3):
            self.fetcher.circuit_breaker.record_failure(Exception("Connection Refused"))

        self.assertTrue(self.fetcher.circuit_breaker.is_open())

        # When require_fresh=True and no fresh data exists, must raise StaleDataException
        with self.assertRaises(StaleDataException):
            self.fetcher.fetch_ohlcv("BTC", timeframe="1h", limit=10, require_fresh=True)

    @patch("src.infrastructure.data_fetchers.binance_fetcher.urllib.request.urlopen")
    def test_graceful_cached_fallback_for_prediction(self, mock_urlopen):
        # First save a valid cache
        cache_path = self.fetcher._get_cache_path("BTCUSDT", "1h")
        sample_df = pd.DataFrame([{
            "timestamp": pd.Timestamp.now(),
            "open": 64000.0,
            "high": 64500.0,
            "low": 63800.0,
            "close": 64200.0,
            "volume": 500.0
        }])
        sample_df.to_csv(cache_path, index=False)

        # Network down
        mock_urlopen.side_effect = Exception("Binance Outage 503")
        
        # Non-trading fetch should gracefully return cached dataframe
        df = self.fetcher.fetch_ohlcv("BTC", timeframe="1h", limit=1, require_fresh=False)
        self.assertEqual(len(df), 1)
class TestRSSSentimentAnalyzerResilience(unittest.TestCase):
    def setUp(self):
        self.test_cache = ".test_rss_cache"
        os.makedirs(self.test_cache, exist_ok=True)
        self.analyzer = RSSSentimentAnalyzer(cache_ttl_seconds=2, cache_dir=self.test_cache)
        self.analyzer.circuit_breaker.reset()

    def tearDown(self):
        if os.path.exists(self.test_cache):
            shutil.rmtree(self.test_cache, ignore_errors=True)
        self.analyzer.circuit_breaker.reset()

    def test_clean_text_strips_html_and_unescapes(self):
        raw = "<p>Bitcoin &amp; Ethereum are <b>surging</b> today! &#8217;Awesome&#8217; <a href='https://link'>Read more</a></p>"
        cleaned = RSSSentimentAnalyzer._clean_text(raw)
        self.assertEqual(cleaned, "Bitcoin & Ethereum are surging today! ’Awesome’ Read more")

    def test_word_boundary_symbol_matching(self):
        # Must not match "eth" inside words like "whether", "tether", "together", "method"
        mock_news = [
            ("Market update: Whether prices fall or rise", "Discussion on broader macro outlook."),
            ("Tether announces $1B supply expansion", "Stablecoin minting update."),
            ("Ethereum breaks key resistance level", "ETH transaction volume spikes to new highs."),
            ("Solana ecosystem sees major DEX inflows", "SOL token gains 12% in 24h.")
        ]
        with patch.object(self.analyzer, "_get_all_news_items", return_value=mock_news):
            score_eth, summary_eth = self.analyzer.analyze_symbol("ETH", lang="ru")
            self.assertIn("Ethereum breaks key resistance level", summary_eth)
            self.assertGreater(score_eth, 0.0)

            score_sol, summary_sol = self.analyzer.analyze_symbol("SOL", lang="ru")
            self.assertIn("Solana ecosystem", summary_sol)

            score_btc, summary_btc = self.analyzer.analyze_symbol("BTC", lang="ru")
            self.assertEqual(score_btc, 0.0)
            self.assertIn("Нейтральный фон", summary_btc)

    def test_cross_feed_deduplication(self):
        feed_data = [
            ("Bitcoin Surges Past $70k", "Description A"),
            ("Bitcoin Surges Past $70k", "Description B from other feed"),
            ("Ethereum 2.0 Update", "Description C")
        ]
        with patch.object(self.analyzer, "_fetch_and_parse", return_value=feed_data[:2]):
            with patch.object(self.analyzer, "_fetch_cryptopanic_api", return_value=[feed_data[2]]):
                items = self.analyzer._get_all_news_items()
                # Duplicate Bitcoin item must be removed
                titles = [it[0] for it in items]
                self.assertEqual(titles.count("Bitcoin Surges Past $70k"), 1)
                self.assertIn("Ethereum 2.0 Update", titles)


if __name__ == "__main__":
    unittest.main()
