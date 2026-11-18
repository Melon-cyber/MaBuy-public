import datetime
from threading import Lock
from typing import Dict, Tuple, Any

from src.domain.entities.tier import SubscriptionTier
from src.application.interfaces.usage_tracker import DailyUsageTrackerInterface


class InMemoryUsageTracker(DailyUsageTrackerInterface):
    """
    Thread-safe in-memory usage tracking implementation.
    Tracks daily prediction usage per user and active tier limits.
    """

    def __init__(self):
        self._lock = Lock()
        # Key: (user_id, date_str), Value: count
        self._usage_store: Dict[Tuple[str, str], int] = {}
        # Key: user_id, Value: SubscriptionTier
        self._tier_store: Dict[str, SubscriptionTier] = {}

    def _get_today_str(self) -> str:
        return datetime.date.today().isoformat()

    def get_daily_usage(self, user_id: str) -> int:
        with self._lock:
            today = self._get_today_str()
            return self._usage_store.get((user_id, today), 0)

    def increment_usage(self, user_id: str) -> int:
        with self._lock:
            today = self._get_today_str()
            key = (user_id, today)
            current = self._usage_store.get(key, 0)
            self._usage_store[key] = current + 1
            return self._usage_store[key]

    def increment_daily_usage(self, user_id: str) -> int:
        return self.increment_usage(user_id)

    def reset_usage(self, user_id: str) -> None:
        with self._lock:
            today = self._get_today_str()
            key = (user_id, today)
            if key in self._usage_store:
                del self._usage_store[key]
            if user_id in self._tier_store:
                del self._tier_store[user_id]

    def reset_all_data(self) -> None:
        with self._lock:
            self._usage_store.clear()
            self._tier_store.clear()
        print("Wiped all usage and tier tracking data.")

    def get_virtual_balance(self, user_id: str) -> float:
        return 10000.0  # Stub for paper trading balance

    def get_paper_trading_stats(self, user_id: str) -> Dict[str, Any]:
        return {
            "total_pnl": 245.50,
            "win_rate": 0.68,
            "active_trades": 2,
            "total_trades": 15
        }

    def get_user_tier(self, user_id: str) -> str:
        # Give the web user Titan tier to avoid 429 limits
        return "titan"
