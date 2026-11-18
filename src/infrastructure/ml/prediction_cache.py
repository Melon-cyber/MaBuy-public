"""
High-performance in-memory cache for ML prediction results.
Reduces latency to under 1ms for duplicate queries within a configurable TTL window (default 10s).
"""

import time
from threading import Lock
from typing import Optional, Dict, Tuple, Any


class PredictionCacheService:
    """
    High-performance in-memory cache for ML prediction results.
    Reduces latency to under 1ms for duplicate queries within a configurable TTL window (default 10s).
    """

    def __init__(self, ttl_seconds: float = 10.0):
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[Tuple[str, str, str], Tuple[float, Any]] = {}
        self._lock = Lock()

    def get(self, symbol: str, timeframe: str, tier_name: str) -> Optional[Any]:
        key = (symbol.upper(), timeframe.lower(), tier_name.lower())
        with self._lock:
            if key in self._cache:
                timestamp, result = self._cache[key]
                if time.time() - timestamp < self.ttl_seconds:
                    return result
                del self._cache[key]
        return None

    def set(self, symbol: str, timeframe: str, tier_name: str, result: Any) -> None:
        key = (symbol.upper(), timeframe.lower(), tier_name.lower())
        with self._lock:
            self._cache[key] = (time.time(), result)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
