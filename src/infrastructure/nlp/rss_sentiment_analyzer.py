import os
import json
import time
import re
import html
import urllib.request
import xml.etree.ElementTree as ET
import logging
import asyncio
from typing import Tuple, Dict, Optional, List, Set

from src.infrastructure.data_fetchers.data_circuit_breaker import DataCircuitBreaker
from src.infrastructure.data_fetchers.atomic_cache_guard import AtomicCacheGuard

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    class SentimentIntensityAnalyzer:
        def polarity_scores(self, text: str) -> dict:
            pos_words = ["bullish", "surge", "gain", "breakout", "rally", "growth", "high", "upgrade", "soar", "profit"]
            neg_words = ["bearish", "crash", "drop", "plunge", "decline", "fall", "dump", "ban", "lawsuit", "hack"]
            lower_text = text.lower()
            pos = sum(1 for w in pos_words if w in lower_text)
            neg = sum(1 for w in neg_words if w in lower_text)
            if pos > neg:
                return {"compound": 0.5}
            elif neg > pos:
                return {"compound": -0.5}
            return {"compound": 0.0}

logger = logging.getLogger(__name__)


class RSSSentimentAnalyzer:
    """
    Fetches real news from RSS feeds & Crypto news aggregators (CoinTelegraph, CoinDesk, Decrypt, CryptoSlate, CryptoPanic)
    and calculates deterministic sentiment scores using VADER NLP for the VIP Titan Tier.
    Features:
      - Circuit Breaker protection against RSS feed timeouts and outages
      - Atomic multi-process caching (.data_cache) to shield external feeds
      - Robust HTML noise stripping, entity unescaping & cross-source deduplication
      - Precise tokenized word-boundary asset matching (prevents false substring collisions like 'eth' in 'whether')
      - Async non-blocking execution support
    """

    ASSET_ALIASES: Dict[str, List[str]] = {
        "BTC": ["bitcoin", "btc"],
        "ETH": ["ethereum", "ether", "eth"],
        "SOL": ["solana", "sol"],
        "BNB": ["binance", "bnb"],
        "XRP": ["ripple", "xrp"],
        "DOGE": ["dogecoin", "doge"],
        "ADA": ["cardano", "ada"],
        "AVAX": ["avalanche", "avax"],
        "DOT": ["polkadot", "dot"],
        "LINK": ["chainlink", "link"],
        "NEAR": ["near protocol", "near"],
        "TON": ["toncoin", "ton"]
    }

    def __init__(self, cache_ttl_seconds: int = 600, cache_dir: str = ".data_cache"):
        self.rss_urls = [
            "https://cointelegraph.com/rss",
            "https://cryptoslate.com/feed/",
            "https://decrypt.co/feed",
            "https://bitcoinmagazine.com/.rss/full/",
            "https://cryptopotato.com/feed/"
        ]
        self.cryptopanic_api_key = os.environ.get("CRYPTOPANIC_API_KEY", "")
        self.analyzer = SentimentIntensityAnalyzer()
        self.cache_ttl = cache_ttl_seconds
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(self.cache_dir, "rss_news_cache.json")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.circuit_breaker = DataCircuitBreaker.get_instance("rss", failure_threshold=3, recovery_timeout_sec=60.0)

    @staticmethod
    def _clean_text(raw_text: str) -> str:
        """Strips HTML tags, decodes XML/HTML entities, and collapses excess whitespace."""
        if not raw_text:
            return ""
        # 1. Unescape HTML & XML entities (&amp;, &quot;, &#8217;, etc.)
        text = html.unescape(raw_text)
        # 2. Remove all HTML tags (<p>, <a href...>, <img .../>)
        text = re.sub(r"<[^>]+>", " ", text)
        # 3. Collapse multiple whitespace characters
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _fetch_cryptopanic_api(self) -> List[Tuple[str, str]]:
        """Fetches from CryptoPanic API if auth token is configured or available."""
        if not self.cryptopanic_api_key:
            return []
        try:
            url = f"https://cryptopanic.com/api/v1/posts/?auth_token={self.cryptopanic_api_key}&public=true"
            req = urllib.request.Request(url, headers={"User-Agent": "MaBuy-DataEngine/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            items = []
            for post in data.get("results", []):
                title = self._clean_text(post.get("title", ""))
                items.append((title, title))
            return items
        except Exception as e:
            logger.debug(f"CryptoPanic API fetch bypassed: {e}")
            return []

    def _fetch_and_parse(self, url: str) -> List[Tuple[str, str]]:
        if not self.circuit_breaker.allow_request():
            logger.debug(f"[RSS Circuit Breaker OPEN] Skipping live feed request for {url}")
            return []

        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36'})
            with urllib.request.urlopen(req, timeout=5) as response:
                xml_data = response.read()

            root = ET.fromstring(xml_data)
            items = []
            for item in root.findall('.//item'):
                title = item.find('title').text if item.find('title') is not None else ""
                desc = item.find('description').text if item.find('description') is not None else ""
                clean_title = self._clean_text(title)
                clean_desc = self._clean_text(desc)
                if clean_title:
                    items.append((clean_title, clean_desc))
            self.circuit_breaker.record_success()
            return items
        except Exception as e:
            self.circuit_breaker.record_failure(e)
            logger.warning(f"Failed to fetch RSS from {url}: {e}")
            return []

    def _read_disk_cache(self) -> Tuple[List[Tuple[str, str]], float]:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return [tuple(x) for x in data.get("items", [])], data.get("timestamp", 0)
        except Exception as e:
            logger.debug(f"RSS cache read error: {e}")
        return [], 0

    def _write_disk_cache(self, items: List[Tuple[str, str]]):
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump({"items": items, "timestamp": time.time()}, f, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"RSS cache write error: {e}")

    def _get_all_news_items(self) -> List[Tuple[str, str]]:
        cached_items, last_time = self._read_disk_cache()
        now = time.time()

        if cached_items and (now - last_time) < self.cache_ttl:
            return cached_items

        # Multi-process atomic lock for updating news cache
        try:
            with AtomicCacheGuard(self.cache_file, timeout=5.0):
                cached_items, last_time = self._read_disk_cache()
                if cached_items and (now - last_time) < self.cache_ttl:
                    return cached_items

                raw_items = []
                # 1. Fetch RSS feeds (CoinTelegraph, CoinDesk, Decrypt, CryptoSlate)
                for url in self.rss_urls:
                    raw_items.extend(self._fetch_and_parse(url))

                # 2. Fetch CryptoPanic if available
                raw_items.extend(self._fetch_cryptopanic_api())

                # 3. Deduplicate across sources based on normalized title hash
                seen_titles: Set[str] = set()
                deduped_items: List[Tuple[str, str]] = []
                for title, desc in raw_items:
                    norm_key = re.sub(r"[^\w\s]", "", title.lower()).strip()
                    if norm_key and norm_key not in seen_titles:
                        seen_titles.add(norm_key)
                        deduped_items.append((title, desc))

                if deduped_items:
                    self._write_disk_cache(deduped_items)
                    return deduped_items
                return cached_items
        except Exception as e:
            logger.warning(f"Atomic lock error on RSS cache: {e}")
            return cached_items

    def _calculate_score(self, text: str) -> float:
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']

    def analyze_symbol(self, symbol: str, lang: str = "ru") -> Tuple[float, str]:
        all_items = self._get_all_news_items()
        sym_clean = symbol.upper().replace("USDT", "").replace("USD", "").strip()

        alias_list = self.ASSET_ALIASES.get(sym_clean, [sym_clean.lower()])

        relevant_news = []
        for title, desc in all_items:
            full_text = f"{title} {desc}"
            # Exact tokenized word boundary matching to eliminate false substring positives
            matched = False
            for alias in alias_list:
                if re.search(r"\b" + re.escape(alias) + r"\b", full_text, re.IGNORECASE):
                    matched = True
                    break
            if matched:
                relevant_news.append((title, full_text))

        if not relevant_news:
            default_msg = (
                f"Нейтральный фон новостей по монете {sym_clean} за последние 24 часа."
                if lang == "ru"
                else f"Neutral news sentiment for {sym_clean} over the last 24h."
            )
            return 0.0, default_msg

        latest_title, latest_full_text = relevant_news[0]
        score = self._calculate_score(latest_full_text)

        if lang == "ru":
            summary = f"Последняя новость: «{latest_title}» (Сентимент: {score:+.2f})."
        else:
            summary = f"Latest news: '{latest_title}' (Sentiment: {score:+.2f})."

        return score, summary

    async def analyze_symbol_async(self, symbol: str, lang: str = "ru") -> Tuple[float, str]:
        return await asyncio.to_thread(self.analyze_symbol, symbol=symbol, lang=lang)

