from __future__ import annotations

import argparse
import asyncio
import hashlib
import html
import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence


LOGGER = logging.getLogger(__name__)

UTC = timezone.utc

_ALLOWED_CATEGORIES = {
    "MACRO",
    "REGULATORY",
    "EXCHANGE",
    "PROTOCOL",
    "MARKET_FLOW",
}

_ALLOWED_DIRECTIONAL_TAGS = {
    "BULLISH_CATALYST",
    "BEARISH_HEADWIND",
    "NEUTRAL_NOISE",
}


class NewsEngineError(RuntimeError):
    """Base exception raised by the Astra news alpha engine."""


class FeedFetchError(NewsEngineError):
    """Raised when a remote news feed cannot be fetched or parsed."""


class LLMAnalysisError(NewsEngineError):
    """Raised when the remote Astra/OpenAI compatible API response is invalid."""


class NewsCategory(str, Enum):
    """Supported institutional news categories."""

    MACRO = "MACRO"
    REGULATORY = "REGULATORY"
    EXCHANGE = "EXCHANGE"
    PROTOCOL = "PROTOCOL"
    MARKET_FLOW = "MARKET_FLOW"


class DirectionalTag(str, Enum):
    """Directional classification assigned to a news event."""

    BULLISH_CATALYST = "BULLISH_CATALYST"
    BEARISH_HEADWIND = "BEARISH_HEADWIND"
    NEUTRAL_NOISE = "NEUTRAL_NOISE"


@dataclass(frozen=True, slots=True)
class NewsArticle:
    """Canonical raw news article representation."""

    source: str
    title: str
    body: str = ""
    url: str = ""
    published_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def normalized_title(self) -> str:
        """Return a normalized title suitable for deterministic de-duplication."""
        return re.sub(r"\s+", " ", html.unescape(self.title)).strip()

    def title_hash(self) -> str:
        """Return SHA-256 hash of the normalized title."""
        normalized = self.normalized_title().lower().encode("utf-8", errors="replace")
        return hashlib.sha256(normalized).hexdigest()


@dataclass(frozen=True, slots=True)
class SentimentAnalysis:
    """Validated sentiment interpretation of a single news article."""

    sentiment_score: float
    shock_intensity: int
    half_life_minutes: float
    symbols: tuple[str, ...]
    category: NewsCategory
    directional_tag: DirectionalTag
    model_source: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.sentiment_score):
            raise ValueError("sentiment_score must be finite.")
        if not -1.0 <= self.sentiment_score <= 1.0:
            raise ValueError("sentiment_score must be within [-1.0, 1.0].")
        if not 1 <= self.shock_intensity <= 5:
            raise ValueError("shock_intensity must be within [1, 5].")
        if not math.isfinite(self.half_life_minutes) or self.half_life_minutes <= 0.0:
            raise ValueError("half_life_minutes must be finite and greater than zero.")
        if not self.symbols:
            raise ValueError("At least one affected symbol is required.")


@dataclass(frozen=True, slots=True)
class SentimentEvent:
    """Persisted and time-decayed alpha event."""

    article: NewsArticle
    analysis: SentimentAnalysis
    event_time: datetime
    processed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def title_hash(self) -> str:
        """Return canonical title hash for this event."""
        return self.article.title_hash()


@dataclass(frozen=True, slots=True)
class AstraClientConfig:
    """Configuration for Astra/OpenAI-compatible sentiment inference."""

    base_url: str = "https://ttmapi.site/v1"
    model: str = "gpt-6-astra"
    api_key: str | None = field(
        default_factory=lambda: os.getenv("ASTRA_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    timeout_seconds: float = 8.0
    enabled: bool = True

    def endpoint(self) -> str:
        """Return the normalized OpenAI-compatible chat completion endpoint."""
        return f"{self.base_url.rstrip('/')}/chat/completions"


@dataclass(frozen=True, slots=True)
class NewsEngineConfig:
    """Top-level configuration for ingestion, persistence, and decay behavior."""

    cryptopanic_api_key: str | None = field(
        default_factory=lambda: os.getenv("CRYPTOPANIC_API_KEY")
    )
    cryptopanic_api_url: str = "https://cryptopanic.com/api/v1/posts/"
    cryptopanic_rss_url: str = "https://cryptopanic.com/news/rss/"
    cointelegraph_rss_url: str = "https://cointelegraph.com/rss"
    binance_rss_url: str = "https://www.binance.com/en/support/announcement/rss"
    sqlite_path: str = "astra_news_alpha_events.sqlite3"
    request_timeout_seconds: float = 8.0
    max_feed_bytes: int = 4_000_000
    max_articles_per_source: int = 100
    max_workers: int = 4
    cache_capacity: int = 50_000
    event_retention_hours: float = 168.0
    local_news: tuple[str | Mapping[str, Any], ...] = ()
    enable_simulated_cryptopanic_fallback: bool = True
    astra_client: AstraClientConfig = field(default_factory=AstraClientConfig)

    def __post_init__(self) -> None:
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be greater than zero.")
        if self.max_feed_bytes <= 0:
            raise ValueError("max_feed_bytes must be greater than zero.")
        if self.max_articles_per_source <= 0:
            raise ValueError("max_articles_per_source must be greater than zero.")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be greater than zero.")
        if self.cache_capacity <= 0:
            raise ValueError("cache_capacity must be greater than zero.")
        if self.event_retention_hours <= 0:
            raise ValueError("event_retention_hours must be greater than zero.")


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Detailed institutional Gate 3.8 decision."""

    passed: bool
    sentiment_score: float
    reason: str
    confidence_adjustment: float


class AstraSentimentClient:
    """OpenAI-compatible Astra sentiment client with deterministic local fallback."""

    _POSITIVE_TERMS: tuple[tuple[str, float], ...] = (
        ("approval", 0.52),
        ("approved", 0.52),
        ("adoption", 0.34),
        ("all-time high", 0.42),
        ("buyback", 0.28),
        ("bullish", 0.34),
        ("breakout", 0.31),
        ("capital inflow", 0.39),
        ("etf inflow", 0.46),
        ("institutional inflow", 0.46),
        ("listing", 0.22),
        ("partnership", 0.22),
        ("rally", 0.28),
        ("record inflow", 0.48),
        ("reserve", 0.20),
        ("surge", 0.27),
        ("upgrade", 0.18),
        ("whale accumulation", 0.34),
    )

    _NEGATIVE_TERMS: tuple[tuple[str, float], ...] = (
        ("ban", 0.48),
        ("bankruptcy", 0.65),
        ("bearish", 0.34),
        ("breach", 0.42),
        ("collapse", 0.62),
        ("delist", 0.30),
        ("exploit", 0.56),
        ("hack", 0.55),
        ("investigation", 0.32),
        ("lawsuit", 0.33),
        ("liquidation", 0.40),
        ("outflow", 0.31),
        ("sanction", 0.45),
        ("selloff", 0.42),
        ("security incident", 0.47),
        ("shutdown", 0.46),
        ("stolen", 0.49),
        ("withdrawal halt", 0.54),
    )

    _SYMBOL_PATTERNS: Mapping[str, tuple[str, ...]] = {
        "BTC": ("bitcoin", "btc", "xbt"),
        "ETH": ("ethereum", "ether", "eth"),
        "SOL": ("solana", "sol"),
        "BNB": ("bnb", "binance coin"),
        "XRP": ("xrp", "ripple"),
        "DOGE": ("dogecoin", "doge"),
        "ADA": ("cardano", "ada"),
        "AVAX": ("avalanche", "avax"),
        "LINK": ("chainlink", "link"),
        "MATIC": ("polygon", "matic", "pol"),
    }

    def __init__(self, config: AstraClientConfig) -> None:
        """Initialize client configuration."""
        self._config = config

    def analyze(self, article: NewsArticle) -> SentimentAnalysis:
        """
        Analyze article sentiment through Astra when available.

        Any networking, API, schema, or parsing failure automatically falls back
        to the deterministic local heuristic.
        """
        if article.metadata.get("simulated", False):
            return SentimentAnalysis(
                sentiment_score=0.0,
                shock_intensity=1,
                half_life_minutes=15.0,
                symbols=("ALL",),
                category=NewsCategory.MARKET_FLOW,
                directional_tag=DirectionalTag.NEUTRAL_NOISE,
                model_source="simulated-neutral-guard",
            )

        if self._config.enabled and self._config.api_key:
            try:
                return self._analyze_remote(article)
            except (LLMAnalysisError, urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                LOGGER.warning(
                    "Astra remote analysis failed for title_hash=%s; using local fallback: %s",
                    article.title_hash(),
                    exc,
                )

        return self._analyze_heuristic(article)

    def _analyze_remote(self, article: NewsArticle) -> SentimentAnalysis:
        """Submit strict JSON request to the configured Astra-compatible endpoint."""
        prompt = self._build_prompt(article)
        payload: dict[str, Any] = {
            "model": self._config.model,
            "temperature": 0.0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Astra, an institutional crypto cross-asset news alpha "
                        "classification engine. You produce only valid JSON and never prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
            "User-Agent": "AstraNewsAlphaEngine/1.0",
        }

        request = urllib.request.Request(
            self._config.endpoint(),
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self._config.timeout_seconds,
            ) as response:
                raw_response = response.read(2_000_000)
        except urllib.error.HTTPError as exc:
            body = exc.read(10_000).decode("utf-8", errors="replace")
            raise LLMAnalysisError(
                f"Astra API HTTP {exc.code}: {body[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise LLMAnalysisError(f"Astra API transport failure: {exc.reason}") from exc

        try:
            response_object = json.loads(raw_response.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMAnalysisError("Astra API returned non-JSON response.") from exc

        content = self._extract_completion_content(response_object)
        result = self._extract_json_object(content)
        return self._validate_remote_result(result)

    @staticmethod
    def _build_prompt(article: NewsArticle) -> str:
        """Build a strict institutional quantitative classification prompt."""
        title = article.normalized_title()
        body = re.sub(r"\s+", " ", article.body).strip()[:8_000]
        source = article.source
        published = _ensure_utc(article.published_at).isoformat()

        return (
            "Assess the following crypto news item for short-horizon institutional "
            "alpha. Account for catalyst credibility, market-wide contagion risk, "
            "regulatory severity, exchange counterparty risk, protocol exploit risk, "
            "and likely information half-life.\n\n"
            f"SOURCE: {source}\n"
            f"PUBLISHED_AT_UTC: {published}\n"
            f"TITLE: {title}\n"
            f"BODY: {body}\n\n"
            "Return ONLY one JSON object with exactly these fields:\n"
            "{\n"
            '  "sentiment_score": float from -1.0 to 1.0,\n'
            '  "shock_intensity": integer from 1 to 5,\n'
            '  "half_life_minutes": positive float,\n'
            '  "symbols": array of uppercase symbols such as ["BTC","ETH","SOL"] '
            'or ["ALL"],\n'
            '  "category": one of "MACRO","REGULATORY","EXCHANGE","PROTOCOL","MARKET_FLOW",\n'
            '  "directional_tag": one of "BULLISH_CATALYST","BEARISH_HEADWIND","NEUTRAL_NOISE"\n'
            "}\n"
            "Do not include markdown, explanations, confidence fields, or extra keys."
        )

    @staticmethod
    def _extract_completion_content(response_object: Mapping[str, Any]) -> str:
        """Extract assistant content from common OpenAI-compatible response formats."""
        choices = response_object.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMAnalysisError("Astra response contains no completion choices.")

        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise LLMAnalysisError("Astra completion choice has invalid structure.")

        message = first_choice.get("message")
        if not isinstance(message, Mapping):
            raise LLMAnalysisError("Astra completion choice has no message object.")

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

        if isinstance(content, list):
            fragments: list[str] = []
            for item in content:
                if isinstance(item, Mapping):
                    text = item.get("text")
                    if isinstance(text, str):
                        fragments.append(text)
            joined = "".join(fragments).strip()
            if joined:
                return joined

        raise LLMAnalysisError("Astra completion response contained no textual content.")

    @staticmethod
    def _extract_json_object(content: str) -> Mapping[str, Any]:
        """Extract a JSON object even when a provider includes accidental wrapping text."""
        stripped = content.strip()
        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, Mapping):
                return decoded
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for index, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                decoded, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, Mapping):
                return decoded

        raise LLMAnalysisError("Unable to locate valid JSON object in Astra response.")

    @staticmethod
    def _validate_remote_result(result: Mapping[str, Any]) -> SentimentAnalysis:
        """Strictly validate and normalize the remote model output."""
        expected_keys = {
            "sentiment_score",
            "shock_intensity",
            "half_life_minutes",
            "symbols",
            "category",
            "directional_tag",
        }
        if set(result.keys()) != expected_keys:
            missing = expected_keys.difference(result.keys())
            extra = set(result.keys()).difference(expected_keys)
            raise LLMAnalysisError(
                f"Astra response schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}."
            )

        try:
            sentiment_score = float(result["sentiment_score"])
            shock_intensity = int(result["shock_intensity"])
            half_life_minutes = float(result["half_life_minutes"])
        except (TypeError, ValueError) as exc:
            raise LLMAnalysisError("Astra numeric fields are malformed.") from exc

        if not math.isfinite(sentiment_score) or not -1.0 <= sentiment_score <= 1.0:
            raise LLMAnalysisError("Astra sentiment_score is outside [-1.0, 1.0].")
        if shock_intensity < 1 or shock_intensity > 5:
            raise LLMAnalysisError("Astra shock_intensity is outside [1, 5].")
        if not math.isfinite(half_life_minutes) or half_life_minutes <= 0.0:
            raise LLMAnalysisError("Astra half_life_minutes is invalid.")

        raw_symbols = result["symbols"]
        if not isinstance(raw_symbols, list):
            raise LLMAnalysisError("Astra symbols must be a JSON array.")

        symbols = _normalize_symbols(raw_symbols)
        category_value = str(result["category"]).strip().upper()
        directional_value = str(result["directional_tag"]).strip().upper()

        if category_value not in _ALLOWED_CATEGORIES:
            raise LLMAnalysisError(f"Unsupported Astra category: {category_value}.")
        if directional_value not in _ALLOWED_DIRECTIONAL_TAGS:
            raise LLMAnalysisError(
                f"Unsupported Astra directional_tag: {directional_value}."
            )

        return SentimentAnalysis(
            sentiment_score=round(sentiment_score, 6),
            shock_intensity=shock_intensity,
            half_life_minutes=round(half_life_minutes, 4),
            symbols=symbols,
            category=NewsCategory(category_value),
            directional_tag=DirectionalTag(directional_value),
            model_source="astra-remote",
        )

    def _analyze_heuristic(self, article: NewsArticle) -> SentimentAnalysis:
        """Perform deterministic regex/VADER-enhanced local sentiment analysis."""
        text = f"{article.title} {article.body}".lower()
        lexical_score = 0.0

        for phrase, weight in self._POSITIVE_TERMS:
            lexical_score += text.count(phrase) * weight

        for phrase, weight in self._NEGATIVE_TERMS:
            lexical_score -= text.count(phrase) * weight

        lexical_score = max(-1.0, min(1.0, lexical_score))

        vader_score = self._try_vader_score(text)
        if vader_score is not None:
            sentiment_score = max(
                -1.0,
                min(1.0, (0.60 * vader_score) + (0.40 * lexical_score)),
            )
        else:
            sentiment_score = lexical_score

        category = self._infer_category(text)
        symbols = self._infer_symbols(text)

        shock_intensity = 1
        if abs(sentiment_score) >= 0.15:
            shock_intensity = 2
        if abs(sentiment_score) >= 0.35:
            shock_intensity = 3
        if abs(sentiment_score) >= 0.55:
            shock_intensity = 4
        if abs(sentiment_score) >= 0.80:
            shock_intensity = 5

        severe_terms = (
            "bankruptcy",
            "collapse",
            "systemic",
            "emergency",
            "hack",
            "exploit",
            "ban",
            "sanction",
            "approval",
            "etf inflow",
        )
        if any(term in text for term in severe_terms):
            shock_intensity = max(shock_intensity, 4)

        half_life_minutes = {
            NewsCategory.MACRO: 240.0,
            NewsCategory.REGULATORY: 360.0,
            NewsCategory.EXCHANGE: 180.0,
            NewsCategory.PROTOCOL: 120.0,
            NewsCategory.MARKET_FLOW: 60.0,
        }[category]

        if "rumor" in text or "unconfirmed" in text:
            half_life_minutes = min(half_life_minutes, 30.0)
        if shock_intensity >= 5:
            half_life_minutes *= 1.5

        if sentiment_score >= 0.10:
            directional_tag = DirectionalTag.BULLISH_CATALYST
        elif sentiment_score <= -0.10:
            directional_tag = DirectionalTag.BEARISH_HEADWIND
        else:
            directional_tag = DirectionalTag.NEUTRAL_NOISE

        return SentimentAnalysis(
            sentiment_score=round(sentiment_score, 6),
            shock_intensity=shock_intensity,
            half_life_minutes=round(half_life_minutes, 4),
            symbols=symbols,
            category=category,
            directional_tag=directional_tag,
            model_source="local-heuristic",
        )

    @staticmethod
    def _try_vader_score(text: str) -> float | None:
        """Use VADER opportunistically when installed without making it mandatory."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError:
            return None

        try:
            score = float(SentimentIntensityAnalyzer().polarity_scores(text)["compound"])
        except (KeyError, TypeError, ValueError):
            return None

        if not math.isfinite(score):
            return None
        return max(-1.0, min(1.0, score))

    @staticmethod
    def _infer_category(text: str) -> NewsCategory:
        """Infer institutional event category from deterministic keyword rules."""
        if any(
            term in text
            for term in (
                "sec",
                "regulator",
                "regulatory",
                "legislation",
                "lawmakers",
                "lawsuit",
                "sanction",
                "ban",
                "etf approval",
            )
        ):
            return NewsCategory.REGULATORY

        if any(
            term in text
            for term in (
                "fed",
                "federal reserve",
                "interest rate",
                "inflation",
                "cpi",
                "recession",
                "treasury",
                "central bank",
                "macroeconomic",
            )
        ):
            return NewsCategory.MACRO

        if any(
            term in text
            for term in (
                "exchange",
                "binance",
                "coinbase",
                "kraken",
                "withdrawal",
                "listing",
                "delist",
                "custody",
            )
        ):
            return NewsCategory.EXCHANGE

        if any(
            term in text
            for term in (
                "exploit",
                "hack",
                "smart contract",
                "bridge",
                "mainnet",
                "protocol",
                "vulnerability",
                "fork",
            )
        ):
            return NewsCategory.PROTOCOL

        return NewsCategory.MARKET_FLOW

    def _infer_symbols(self, text: str) -> tuple[str, ...]:
        """Extract affected spot symbols, falling back to the broad market."""
        symbols: list[str] = []
        for symbol, terms in self._SYMBOL_PATTERNS.items():
            for term in terms:
                if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text):
                    symbols.append(symbol)
                    break

        broad_market_terms = (
            "crypto market",
            "cryptocurrency market",
            "digital assets",
            "bitcoin market",
            "altcoin market",
            "market-wide",
        )

        if any(term in text for term in broad_market_terms) and not symbols:
            return ("ALL",)

        return tuple(symbols) if symbols else ("ALL",)


class AstraNewsAlphaEngine:
    """
    Thread-safe institutional crypto-news alpha engine.

    The engine provides concurrent source ingestion, title-hash de-duplication,
    Astra/OpenAI-compatible structured sentiment analysis, exponential alpha
    decay, persistent SQLite event logging, and Gate 3.8 trade veto logic.
    """

    def __init__(self, config: NewsEngineConfig | None = None) -> None:
        """Initialize persistence, caches, feed clients, and the sentiment client."""
        self._config = config or NewsEngineConfig()
        self._lock = threading.RLock()
        self._sentiment_client = AstraSentimentClient(self._config.astra_client)
        self._events: list[SentimentEvent] = []
        self._title_cache: OrderedDict[str, float] = OrderedDict()

        self._initialize_database()
        self._load_recent_events()

    def close(self) -> None:
        """Close the SQLite connection safely."""
        with self._lock:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                try:
                    connection.close()
                except sqlite3.Error as exc:
                    LOGGER.warning("Unable to close SQLite connection: %s", exc)

    def __enter__(self) -> AstraNewsAlphaEngine:
        """Support context manager usage."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Close persistent resources on context manager exit."""
        self.close()

    def fetch_all(self) -> list[NewsArticle]:
        """
        Concurrently fetch CryptoPanic, CoinTelegraph, Binance, and local feeds.

        Individual feed failures are logged to SQLite and do not prevent healthy
        sources from returning their available articles.
        """
        fetchers: dict[str, Any] = {
            "cryptopanic": self.fetch_cryptopanic,
            "cointelegraph": self.fetch_cointelegraph,
            "binance": self.fetch_binance_announcements,
            "local": self.fetch_local_news,
        }

        articles: list[NewsArticle] = []
        with ThreadPoolExecutor(
            max_workers=min(self._config.max_workers, len(fetchers)),
            thread_name_prefix="astra-news-fetch",
        ) as executor:
            futures = {
                executor.submit(fetcher): source for source, fetcher in fetchers.items()
            }

            for future in as_completed(futures):
                source = futures[future]
                try:
                    source_articles = future.result()
                    articles.extend(source_articles)
                except Exception as exc:
                    self._log_fetch_error(source, str(exc))
                    LOGGER.warning("News fetch failed for source=%s: %s", source, exc)

        articles.sort(key=lambda item: _ensure_utc(item.published_at), reverse=True)
        return articles

    async def fetch_all_async(self) -> list[NewsArticle]:
        """Asynchronously execute the concurrent feed ingestion operation."""
        return await asyncio.to_thread(self.fetch_all)

    def fetch_cryptopanic(self) -> list[NewsArticle]:
        """Fetch CryptoPanic API data when keyed, otherwise use its public RSS feed."""
        if self._config.cryptopanic_api_key:
            try:
                query = urllib.parse.urlencode(
                    {
                        "auth_token": self._config.cryptopanic_api_key,
                        "kind": "news",
                        "public": "true",
                    }
                )
                url = f"{self._config.cryptopanic_api_url}?{query}"
                payload = self._fetch_json(url)
                return self._parse_cryptopanic_api(payload)
            except (FeedFetchError, ValueError, TypeError) as exc:
                LOGGER.warning("CryptoPanic API unavailable; trying RSS fallback: %s", exc)

        try:
            return self._fetch_rss(self._config.cryptopanic_rss_url, "cryptopanic")
        except FeedFetchError as exc:
            if not self._config.enable_simulated_cryptopanic_fallback:
                raise

            now = datetime.now(UTC)
            return [
                NewsArticle(
                    source="cryptopanic-simulated-fallback",
                    title="CryptoPanic source unavailable; no actionable market event.",
                    body=(
                        "Simulated monitoring fallback generated because the public "
                        "CryptoPanic feed could not be reached. This item is explicitly "
                        "excluded from directional alpha generation."
                    ),
                    published_at=now,
                    metadata={"simulated": True, "fetch_error": str(exc)},
                )
            ]

    def fetch_cointelegraph(self) -> list[NewsArticle]:
        """Fetch and parse the CoinTelegraph public RSS feed."""
        return self._fetch_rss(self._config.cointelegraph_rss_url, "cointelegraph")

    def fetch_binance_announcements(self) -> list[NewsArticle]:
        """Fetch and parse the Binance announcements RSS feed."""
        return self._fetch_rss(self._config.binance_rss_url, "binance")

    def fetch_local_news(self) -> list[NewsArticle]:
        """Convert configured local text or mapping fixtures into canonical articles."""
        articles: list[NewsArticle] = []

        for index, entry in enumerate(self._config.local_news):
            try:
                if isinstance(entry, str):
                    text = entry.strip()
                    if not text:
                        continue
                    title = re.sub(r"\s+", " ", text)[:240]
                    articles.append(
                        NewsArticle(
                            source="local",
                            title=title,
                            body=text,
                            published_at=datetime.now(UTC),
                            metadata={"local_index": index},
                        )
                    )
                    continue

                if not isinstance(entry, Mapping):
                    raise ValueError("Local feed entry must be a string or mapping.")

                title = str(entry.get("title", "")).strip()
                body = str(entry.get("body", entry.get("text", ""))).strip()
                if not title:
                    title = re.sub(r"\s+", " ", body)[:240]
                if not title:
                    raise ValueError("Local feed entry has neither title nor body.")

                source = str(entry.get("source", "local")).strip() or "local"
                url = str(entry.get("url", "")).strip()
                published_at = _parse_datetime(entry.get("published_at"))

                metadata = entry.get("metadata", {})
                if not isinstance(metadata, Mapping):
                    metadata = {"raw_metadata": str(metadata)}

                articles.append(
                    NewsArticle(
                        source=source,
                        title=title,
                        body=body,
                        url=url,
                        published_at=published_at,
                        metadata=dict(metadata),
                    )
                )
            except (TypeError, ValueError) as exc:
                self._log_fetch_error("local", f"entry_index={index}: {exc}")
                LOGGER.warning("Skipping invalid local news item index=%d: %s", index, exc)

        return articles[: self._config.max_articles_per_source]

    def ingest_and_analyze(
        self,
        articles: Iterable[NewsArticle] | None = None,
    ) -> list[SentimentEvent]:
        """
        De-duplicate, analyze, persist, and retain valid sentiment events.

        If articles is omitted, all configured sources are concurrently fetched.
        """
        raw_articles = list(articles) if articles is not None else self.fetch_all()
        accepted_events: list[SentimentEvent] = []

        for article in raw_articles:
            try:
                event = self.ingest_article(article)
                if event is not None:
                    accepted_events.append(event)
            except Exception as exc:
                self._log_fetch_error(
                    article.source,
                    f"ingestion_failure title_hash={article.title_hash()}: {exc}",
                )
                LOGGER.exception(
                    "Unable to ingest article title_hash=%s",
                    article.title_hash(),
                )

        self.prune_expired_events()
        return accepted_events

    async def ingest_and_analyze_async(
        self,
        articles: Iterable[NewsArticle] | None = None,
    ) -> list[SentimentEvent]:
        """Asynchronously execute article ingestion and sentiment analysis."""
        materialized = list(articles) if articles is not None else None
        return await asyncio.to_thread(self.ingest_and_analyze, materialized)

    def ingest_article(self, article: NewsArticle) -> SentimentEvent | None:
        """
        Ingest one article and return its event, or None when title-hash duplicated.

        The title hash is reserved before remote inference to avoid concurrent
        duplicate analysis requests for identical headlines.
        """
        normalized_title = article.normalized_title()
        if not normalized_title:
            raise ValueError("News article title cannot be empty.")

        canonical_article = NewsArticle(
            source=article.source.strip() or "unknown",
            title=normalized_title,
            body=article.body.strip(),
            url=article.url.strip(),
            published_at=_ensure_utc(article.published_at),
            metadata=dict(article.metadata),
        )

        title_hash = canonical_article.title_hash()
        if not self._reserve_title_hash(title_hash):
            return None

        analysis = self._sentiment_client.analyze(canonical_article)
        event_time = min(_ensure_utc(canonical_article.published_at), datetime.now(UTC))
        event = SentimentEvent(
            article=canonical_article,
            analysis=analysis,
            event_time=event_time,
        )

        self._persist_event(event)
        with self._lock:
            self._events.append(event)

        return event

    def composite_sentiment(
        self,
        symbol: str,
        at_time: datetime | None = None,
    ) -> float:
        """
        Compute bounded exponentially decayed symbol sentiment.

        S_sym(t) = sum_i[
            s_i * (gamma_i / 3.0) *
            exp(-ln(2) / t_half_i * (t - t_i))
        ]

        Events tagged with ALL affect every requested symbol.
        """
        normalized_symbol = _normalize_symbol(symbol)
        evaluation_time = _ensure_utc(at_time or datetime.now(UTC))
        retention_cutoff = evaluation_time - timedelta(
            hours=self._config.event_retention_hours
        )

        with self._lock:
            events = tuple(self._events)

        composite = 0.0
        log_two = math.log(2.0)

        for event in events:
            if event.event_time < retention_cutoff:
                continue

            symbols = event.analysis.symbols
            if normalized_symbol not in symbols and "ALL" not in symbols:
                continue

            age_minutes = max(
                0.0,
                (evaluation_time - _ensure_utc(event.event_time)).total_seconds() / 60.0,
            )
            half_life = event.analysis.half_life_minutes
            decay = math.exp(-(log_two / half_life) * age_minutes)
            contribution = (
                event.analysis.sentiment_score
                * (event.analysis.shock_intensity / 3.0)
                * decay
            )
            composite += contribution

        return max(-1.0, min(1.0, composite))

    def record_sentiment_event(
        self,
        symbol: str,
        sentiment_score: float,
        shock_intensity: int = 3,
        half_life_minutes: float = 60.0,
        event_time: datetime | None = None,
        category: NewsCategory = NewsCategory.MACRO,
        directional_tag: DirectionalTag | None = None,
    ) -> SentimentEvent:
        """Programmatically record a sentiment event for testing, API webhook, or manual injection."""
        norm_sym = _normalize_symbol(symbol)
        ev_time = _ensure_utc(event_time or datetime.now(UTC))
        if directional_tag is None:
            if sentiment_score >= 0.10:
                directional_tag = DirectionalTag.BULLISH_CATALYST
            elif sentiment_score <= -0.10:
                directional_tag = DirectionalTag.BEARISH_HEADWIND
            else:
                directional_tag = DirectionalTag.NEUTRAL_NOISE

        analysis = SentimentAnalysis(
            sentiment_score=sentiment_score,
            shock_intensity=shock_intensity,
            half_life_minutes=half_life_minutes,
            symbols=(norm_sym,),
            category=category,
            directional_tag=directional_tag,
            model_source="programmatic",
        )
        article = NewsArticle(
            source="manual",
            title=f"Programmatic Event {norm_sym} {sentiment_score:+.2f} {time.time()}",
            body="",
            url="",
            published_at=ev_time,
        )
        event = SentimentEvent(
            article=article,
            analysis=analysis,
            event_time=ev_time,
        )
        self._persist_event(event)
        with self._lock:
            self._events.append(event)
        return event

    def get_decayed_sentiment(
        self,
        symbol: str,
        current_time: float | datetime | None = None,
    ) -> float:
        """Alias for composite_sentiment accepting timestamp float or datetime."""
        if isinstance(current_time, (int, float)):
            dt = datetime.fromtimestamp(current_time, tz=UTC)
        else:
            dt = current_time
        return self.composite_sentiment(symbol, at_time=dt)

    def evaluate_news_gate(
        self,
        symbol: str,
        proposed_direction: str,
    ) -> tuple[bool, float, str]:
        """
        Apply institutional Gate 3.8 news veto and confidence modulation rules.

        Returns:
            Tuple of (passed, bounded_sentiment_score, reason).

        Rules:
            * BULLISH proposals are vetoed below -0.50.
            * BEARISH proposals are vetoed above +0.50.
            * Directionally confirming sentiment in [0.20, 0.50] grants a
              recommended +0.03 ML confidence adjustment.
        """
        normalized_symbol = _normalize_symbol(symbol)
        direction = proposed_direction.strip().upper()

        if direction not in {"BULLISH", "BEARISH"}:
            decision = GateDecision(
                passed=False,
                sentiment_score=0.0,
                reason=(
                    "Gate 3.8 VETO: invalid proposed_direction. "
                    "Expected BULLISH or BEARISH."
                ),
                confidence_adjustment=0.0,
            )
            self._log_gate_decision(normalized_symbol, direction, decision)
            return decision.passed, decision.sentiment_score, decision.reason

        score = self.composite_sentiment(normalized_symbol)

        if direction == "BULLISH" and score < -0.50:
            decision = GateDecision(
                passed=False,
                sentiment_score=score,
                reason=(
                    "Gate 3.8 VETO: Bearish Macro Shock. "
                    f"Composite news sentiment={score:+.4f} is below -0.50."
                ),
                confidence_adjustment=0.0,
            )
        elif direction == "BEARISH" and score > 0.50:
            decision = GateDecision(
                passed=False,
                sentiment_score=score,
                reason=(
                    "Gate 3.8 VETO: Bullish Macro Shock. "
                    f"Composite news sentiment={score:+.4f} is above +0.50."
                ),
                confidence_adjustment=0.0,
            )
        elif direction == "BULLISH" and 0.20 <= score <= 0.50:
            decision = GateDecision(
                passed=True,
                sentiment_score=score,
                reason=(
                    "Gate 3.8 PASS: bullish news confirms proposed direction; "
                    "recommended ML confidence adjustment=+0.03."
                ),
                confidence_adjustment=0.03,
            )
        elif direction == "BEARISH" and -0.50 <= score <= -0.20:
            decision = GateDecision(
                passed=True,
                sentiment_score=score,
                reason=(
                    "Gate 3.8 PASS: bearish news confirms proposed direction; "
                    "recommended ML confidence adjustment=+0.03."
                ),
                confidence_adjustment=0.03,
            )
        else:
            decision = GateDecision(
                passed=True,
                sentiment_score=score,
                reason=(
                    "Gate 3.8 PASS: no conflicting macro news veto. "
                    "recommended ML confidence adjustment=+0.00."
                ),
                confidence_adjustment=0.0,
            )

        self._log_gate_decision(normalized_symbol, direction, decision)
        return decision.passed, decision.sentiment_score, decision.reason

    def prune_expired_events(self) -> int:
        """Prune expired in-memory and SQLite events according to retention policy."""
        cutoff = datetime.now(UTC) - timedelta(hours=self._config.event_retention_hours)
        cutoff_timestamp = cutoff.timestamp()

        with self._lock:
            original_count = len(self._events)
            self._events = [
                event for event in self._events if event.event_time >= cutoff
            ]
            removed_count = original_count - len(self._events)

            try:
                with self._connection:
                    self._connection.execute(
                        "DELETE FROM news_events WHERE event_time < ?",
                        (cutoff_timestamp,),
                    )
            except sqlite3.Error as exc:
                LOGGER.warning("Unable to prune expired SQLite events: %s", exc)

        return removed_count

    def recent_events(self) -> tuple[SentimentEvent, ...]:
        """Return immutable snapshot of current in-memory sentiment events."""
        with self._lock:
            return tuple(self._events)

    def _fetch_json(self, url: str) -> Mapping[str, Any]:
        """Fetch a bounded JSON payload with strict timeout and validation."""
        raw_payload = self._fetch_bytes(url)
        try:
            parsed = json.loads(raw_payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FeedFetchError(f"Invalid JSON feed response from {url}") from exc

        if not isinstance(parsed, Mapping):
            raise FeedFetchError(f"JSON feed response from {url} is not an object.")
        return parsed

    def _fetch_rss(self, url: str, source: str) -> list[NewsArticle]:
        """Fetch and parse RSS 2.0 or Atom XML feed."""
        raw_payload = self._fetch_bytes(url)
        try:
            root = ET.fromstring(raw_payload)
        except ET.ParseError as exc:
            raise FeedFetchError(f"Invalid XML RSS payload from {url}") from exc

        articles = self._parse_rss_or_atom(root, source)
        if not articles:
            raise FeedFetchError(f"No usable articles found in RSS feed {url}.")
        return articles[: self._config.max_articles_per_source]

    def _fetch_bytes(self, url: str) -> bytes:
        """Fetch bounded remote bytes using standard-library HTTP transport."""
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/rss+xml, application/xml, application/json, text/xml, */*",
                "User-Agent": "AstraNewsAlphaEngine/1.0",
            },
            method="GET",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self._config.request_timeout_seconds,
            ) as response:
                payload = response.read(self._config.max_feed_bytes + 1)
        except urllib.error.HTTPError as exc:
            raise FeedFetchError(f"HTTP {exc.code} while fetching {url}") from exc
        except urllib.error.URLError as exc:
            raise FeedFetchError(f"Transport failure while fetching {url}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise FeedFetchError(f"Timed out while fetching {url}") from exc

        if len(payload) > self._config.max_feed_bytes:
            raise FeedFetchError(
                f"Feed {url} exceeds max_feed_bytes={self._config.max_feed_bytes}."
            )
        if not payload:
            raise FeedFetchError(f"Feed {url} returned an empty response.")

        return payload

    def _parse_cryptopanic_api(self, payload: Mapping[str, Any]) -> list[NewsArticle]:
        """Parse CryptoPanic API results into canonical articles."""
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise FeedFetchError("CryptoPanic API response has no results array.")

        articles: list[NewsArticle] = []
        for item in raw_results[: self._config.max_articles_per_source]:
            if not isinstance(item, Mapping):
                continue

            title = str(item.get("title", "")).strip()
            if not title:
                continue

            domain = item.get("domain")
            source_name = "cryptopanic"
            if isinstance(domain, str) and domain.strip():
                source_name = f"cryptopanic:{domain.strip()}"

            published_at = _parse_datetime(item.get("published_at"))
            url = str(item.get("url", "")).strip()

            articles.append(
                NewsArticle(
                    source=source_name,
                    title=title,
                    body=str(item.get("description", "")).strip(),
                    url=url,
                    published_at=published_at,
                    metadata={
                        "currencies": item.get("currencies", []),
                        "kind": item.get("kind", ""),
                    },
                )
            )

        if not articles:
            raise FeedFetchError("CryptoPanic API response contained no usable articles.")
        return articles

    def _parse_rss_or_atom(self, root: ET.Element, source: str) -> list[NewsArticle]:
        """Parse RSS item and Atom entry records without external feed libraries."""
        articles: list[NewsArticle] = []

        for element in root.iter():
            local_name = _xml_local_name(element.tag)
            if local_name not in {"item", "entry"}:
                continue

            title = _xml_child_text(element, "title")
            if not title:
                continue

            body = (
                _xml_child_text(element, "description")
                or _xml_child_text(element, "content")
                or _xml_child_text(element, "summary")
                or ""
            )
            url = _xml_link(element)
            published_at = _parse_datetime(
                _xml_child_text(element, "pubDate")
                or _xml_child_text(element, "published")
                or _xml_child_text(element, "updated")
                or _xml_child_text(element, "date")
            )

            articles.append(
                NewsArticle(
                    source=source,
                    title=title,
                    body=_strip_html(body),
                    url=url,
                    published_at=published_at,
                )
            )

            if len(articles) >= self._config.max_articles_per_source:
                break

        return articles

    def _initialize_database(self) -> None:
        """Create SQLite event-log schema with safe WAL-mode persistence."""
        sqlite_path = self._config.sqlite_path
        if sqlite_path != ":memory:":
            Path(sqlite_path).expanduser().resolve().parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        try:
            self._connection = sqlite3.connect(
                sqlite_path,
                timeout=10.0,
                check_same_thread=False,
            )
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute("PRAGMA foreign_keys=ON")

            with self._connection:
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS news_events (
                        title_hash TEXT PRIMARY KEY,
                        source TEXT NOT NULL,
                        title TEXT NOT NULL,
                        body TEXT NOT NULL,
                        url TEXT NOT NULL,
                        event_time REAL NOT NULL,
                        processed_at REAL NOT NULL,
                        sentiment_score REAL NOT NULL,
                        shock_intensity INTEGER NOT NULL,
                        half_life_minutes REAL NOT NULL,
                        symbols_json TEXT NOT NULL,
                        category TEXT NOT NULL,
                        directional_tag TEXT NOT NULL,
                        model_source TEXT NOT NULL,
                        metadata_json TEXT NOT NULL
                    )
                    """
                )
                self._connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_news_events_event_time
                    ON news_events(event_time)
                    """
                )
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS fetch_errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at REAL NOT NULL,
                        source TEXT NOT NULL,
                        error_message TEXT NOT NULL
                    )
                    """
                )
                self._connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS gate_decisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at REAL NOT NULL,
                        symbol TEXT NOT NULL,
                        proposed_direction TEXT NOT NULL,
                        passed INTEGER NOT NULL,
                        sentiment_score REAL NOT NULL,
                        confidence_adjustment REAL NOT NULL,
                        reason TEXT NOT NULL
                    )
                    """
                )
        except sqlite3.Error as exc:
            raise NewsEngineError(
                f"Unable to initialize SQLite database '{sqlite_path}': {exc}"
            ) from exc

    def _load_recent_events(self) -> None:
        """Load retained events from SQLite into in-memory decay state."""
        cutoff = datetime.now(UTC) - timedelta(hours=self._config.event_retention_hours)

        with self._lock:
            try:
                rows = self._connection.execute(
                    """
                    SELECT
                        title_hash, source, title, body, url, event_time, processed_at,
                        sentiment_score, shock_intensity, half_life_minutes, symbols_json,
                        category, directional_tag, model_source, metadata_json
                    FROM news_events
                    WHERE event_time >= ?
                    ORDER BY event_time ASC
                    """,
                    (cutoff.timestamp(),),
                ).fetchall()
            except sqlite3.Error as exc:
                raise NewsEngineError(f"Unable to load persisted news events: {exc}") from exc

            for row in rows:
                try:
                    symbols_payload = json.loads(row[10])
                    metadata_payload = json.loads(row[14])
                    analysis = SentimentAnalysis(
                        sentiment_score=float(row[7]),
                        shock_intensity=int(row[8]),
                        half_life_minutes=float(row[9]),
                        symbols=_normalize_symbols(symbols_payload),
                        category=NewsCategory(str(row[11])),
                        directional_tag=DirectionalTag(str(row[12])),
                        model_source=str(row[13]),
                    )
                    article = NewsArticle(
                        source=str(row[1]),
                        title=str(row[2]),
                        body=str(row[3]),
                        url=str(row[4]),
                        published_at=datetime.fromtimestamp(float(row[5]), tz=UTC),
                        metadata=metadata_payload if isinstance(metadata_payload, Mapping) else {},
                    )
                    event = SentimentEvent(
                        article=article,
                        analysis=analysis,
                        event_time=datetime.fromtimestamp(float(row[5]), tz=UTC),
                        processed_at=datetime.fromtimestamp(float(row[6]), tz=UTC),
                    )
                    self._events.append(event)
                    self._title_cache[str(row[0])] = float(row[6])
                except (
                    TypeError,
                    ValueError,
                    KeyError,
                    json.JSONDecodeError,
                ) as exc:
                    LOGGER.warning("Skipping malformed persisted event row: %s", exc)

            self._trim_title_cache()

    def _reserve_title_hash(self, title_hash: str) -> bool:
        """Atomically reserve title hash in memory and against persisted history."""
        with self._lock:
            if title_hash in self._title_cache:
                self._title_cache.move_to_end(title_hash)
                return False

            try:
                existing = self._connection.execute(
                    "SELECT 1 FROM news_events WHERE title_hash = ? LIMIT 1",
                    (title_hash,),
                ).fetchone()
            except sqlite3.Error as exc:
                LOGGER.warning("Unable to query title de-duplication store: %s", exc)
                existing = None

            if existing is not None:
                self._title_cache[title_hash] = time.time()
                self._trim_title_cache()
                return False

            self._title_cache[title_hash] = time.time()
            self._trim_title_cache()
            return True

    def _trim_title_cache(self) -> None:
        """Bound in-memory title hash cache capacity."""
        while len(self._title_cache) > self._config.cache_capacity:
            self._title_cache.popitem(last=False)

    def _persist_event(self, event: SentimentEvent) -> None:
        """Persist an event to SQLite while retaining operational availability."""
        try:
            with self._lock:
                with self._connection:
                    self._connection.execute(
                        """
                        INSERT OR IGNORE INTO news_events (
                            title_hash, source, title, body, url, event_time, processed_at,
                            sentiment_score, shock_intensity, half_life_minutes, symbols_json,
                            category, directional_tag, model_source, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.title_hash,
                            event.article.source,
                            event.article.title,
                            event.article.body,
                            event.article.url,
                            event.event_time.timestamp(),
                            event.processed_at.timestamp(),
                            event.analysis.sentiment_score,
                            event.analysis.shock_intensity,
                            event.analysis.half_life_minutes,
                            json.dumps(event.analysis.symbols, separators=(",", ":")),
                            event.analysis.category.value,
                            event.analysis.directional_tag.value,
                            event.analysis.model_source,
                            json.dumps(
                                dict(event.article.metadata),
                                separators=(",", ":"),
                                default=str,
                            ),
                        ),
                    )
        except sqlite3.Error as exc:
            LOGGER.error(
                "Unable to persist sentiment event title_hash=%s: %s",
                event.title_hash,
                exc,
            )

    def _log_fetch_error(self, source: str, error_message: str) -> None:
        """Persist bounded fetch and ingestion errors without propagating logging failures."""
        try:
            with self._lock:
                with self._connection:
                    self._connection.execute(
                        """
                        INSERT INTO fetch_errors (created_at, source, error_message)
                        VALUES (?, ?, ?)
                        """,
                        (
                            datetime.now(UTC).timestamp(),
                            source[:128],
                            error_message[:4_000],
                        ),
                    )
        except sqlite3.Error as exc:
            LOGGER.warning("Unable to persist fetch error log: %s", exc)

    def _log_gate_decision(
        self,
        symbol: str,
        proposed_direction: str,
        decision: GateDecision,
    ) -> None:
        """Persist Gate 3.8 outcomes for auditability."""
        try:
            with self._lock:
                with self._connection:
                    self._connection.execute(
                        """
                        INSERT INTO gate_decisions (
                            created_at, symbol, proposed_direction, passed,
                            sentiment_score, confidence_adjustment, reason
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            datetime.now(UTC).timestamp(),
                            symbol,
                            proposed_direction,
                            int(decision.passed),
                            decision.sentiment_score,
                            decision.confidence_adjustment,
                            decision.reason,
                        ),
                    )
        except sqlite3.Error as exc:
            LOGGER.warning("Unable to persist Gate 3.8 decision: %s", exc)


def _normalize_symbol(symbol: str) -> str:
    """Validate and normalize a requested trading symbol."""
    normalized = symbol.strip().upper()
    if not normalized or not re.fullmatch(r"[A-Z0-9]{2,20}", normalized):
        raise ValueError(
            "symbol must contain 2-20 uppercase alphanumeric characters after normalization."
        )
    return normalized


def _normalize_symbols(raw_symbols: Sequence[Any]) -> tuple[str, ...]:
    """Normalize and de-duplicate API/model supplied affected symbols."""
    normalized_symbols: list[str] = []

    for raw_symbol in raw_symbols:
        if not isinstance(raw_symbol, str):
            raise LLMAnalysisError("Each affected symbol must be a string.")

        candidate = raw_symbol.strip().upper()
        if candidate == "ALL":
            return ("ALL",)

        if not re.fullmatch(r"[A-Z0-9]{2,20}", candidate):
            raise LLMAnalysisError(f"Invalid affected symbol: {raw_symbol!r}.")

        if candidate not in normalized_symbols:
            normalized_symbols.append(candidate)

    if not normalized_symbols:
        raise LLMAnalysisError("At least one affected symbol is required.")

    return tuple(normalized_symbols)


def _ensure_utc(value: datetime) -> datetime:
    """Return timezone-aware UTC datetime, treating naive input as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_datetime(value: Any) -> datetime:
    """Parse common RSS, ISO-8601, timestamp, and datetime inputs safely."""
    if value is None or value == "":
        return datetime.now(UTC)

    if isinstance(value, datetime):
        return _ensure_utc(value)

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (ValueError, OSError, OverflowError):
            return datetime.now(UTC)

    if not isinstance(value, str):
        return datetime.now(UTC)

    candidate = value.strip()
    if not candidate:
        return datetime.now(UTC)

    try:
        return _ensure_utc(parsedate_to_datetime(candidate))
    except (TypeError, ValueError, IndexError, OverflowError):
        pass

    try:
        return _ensure_utc(datetime.fromisoformat(candidate.replace("Z", "+00:00")))
    except ValueError:
        return datetime.now(UTC)


def _xml_local_name(tag: str) -> str:
    """Return XML local name independent of namespace."""
    return tag.rsplit("}", 1)[-1]


def _xml_child_text(element: ET.Element, name: str) -> str:
    """Return first matching direct child text by local XML name."""
    for child in element:
        if _xml_local_name(child.tag) == name:
            return "".join(child.itertext()).strip()
    return ""


def _xml_link(element: ET.Element) -> str:
    """Extract RSS or Atom link target from an XML feed entry."""
    direct_text = _xml_child_text(element, "link")
    if direct_text:
        return direct_text

    for child in element:
        if _xml_local_name(child.tag) != "link":
            continue
        href = child.attrib.get("href", "").strip()
        relation = child.attrib.get("rel", "alternate").strip().lower()
        if href and relation in {"", "alternate"}:
            return href

    return ""


def _strip_html(value: str) -> str:
    """Remove simple HTML markup and normalize whitespace from feed content."""
    without_tags = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return re.sub(r"\s+", " ", without_tags).strip()


def _event_to_dict(event: SentimentEvent) -> dict[str, Any]:
    """Serialize an event for CLI diagnostics."""
    return {
        "title_hash": event.title_hash,
        "source": event.article.source,
        "title": event.article.title,
        "url": event.article.url,
        "event_time": event.event_time.isoformat(),
        "processed_at": event.processed_at.isoformat(),
        "sentiment_score": event.analysis.sentiment_score,
        "shock_intensity": event.analysis.shock_intensity,
        "half_life_minutes": event.analysis.half_life_minutes,
        "symbols": list(event.analysis.symbols),
        "category": event.analysis.category.value,
        "directional_tag": event.analysis.directional_tag.value,
        "model_source": event.analysis.model_source,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Astra institutional crypto news alpha engine diagnostics."
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Local raw news text. May be supplied multiple times.",
    )
    parser.add_argument(
        "--symbol",
        default="BTC",
        help="Symbol to evaluate through Gate 3.8. Default: BTC.",
    )
    parser.add_argument(
        "--direction",
        choices=("BULLISH", "BEARISH"),
        default="BULLISH",
        help="Proposed trade direction. Default: BULLISH.",
    )
    parser.add_argument(
        "--database",
        default=":memory:",
        help="SQLite database path. Default: in-memory.",
    )
    parser.add_argument(
        "--fetch-remote",
        action="store_true",
        help="Fetch CryptoPanic, CoinTelegraph, and Binance feeds.",
    )
    parser.add_argument(
        "--enable-llm",
        action="store_true",
        help="Enable Astra remote API when ASTRA_API_KEY or OPENAI_API_KEY is available.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    local_items: tuple[str, ...]
    if args.text:
        local_items = tuple(args.text)
    else:
        local_items = (
            "Bitcoin ETF reports record institutional inflows as major allocators increase exposure.",
            "Exchange confirms temporary withdrawal suspension following a security incident.",
        )

    client_config = AstraClientConfig(enabled=bool(args.enable_llm))
    config = NewsEngineConfig(
        sqlite_path=args.database,
        local_news=local_items,
        astra_client=client_config,
    )

    with AstraNewsAlphaEngine(config) as engine:
        if args.fetch_remote:
            events = engine.ingest_and_analyze()
        else:
            events = engine.ingest_and_analyze(engine.fetch_local_news())

        passed, score, reason = engine.evaluate_news_gate(
            args.symbol,
            args.direction,
        )

        print(
            json.dumps(
                {
                    "ingested_events": [_event_to_dict(event) for event in events],
                    "gate_decision": {
                        "symbol": args.symbol.upper(),
                        "proposed_direction": args.direction,
                        "passed": passed,
                        "sentiment_score": score,
                        "reason": reason,
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )