from abc import ABC, abstractmethod


class DailyUsageTrackerInterface(ABC):
    """
    Port (Interface) for tracking and persisting daily user prediction requests.
    Supports Clean Architecture by decoupling business logic from storage backends (Redis/DB/In-Memory).
    """

    @abstractmethod
    def get_daily_usage(self, user_id: str) -> int:
        """Returns the number of predictions requested by the user today."""
        pass

    @abstractmethod
    def increment_usage(self, user_id: str) -> int:
        """Increments and returns the updated daily request count for the user."""
        pass

    @abstractmethod
    def reset_usage(self, user_id: str) -> None:
        """Resets the usage counter for the user (e.g. at midnight UTC)."""
        pass
