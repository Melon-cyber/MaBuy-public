import unittest
from src.infrastructure.persistence.in_memory_tracker import InMemoryUsageTracker


class TestUsageTracker(unittest.TestCase):

    def test_in_memory_usage_tracker(self):
        tracker = InMemoryUsageTracker()
        user_id = "test_user_123"

        self.assertEqual(tracker.get_daily_usage(user_id), 0)

        tracker.increment_usage(user_id)
        self.assertEqual(tracker.get_daily_usage(user_id), 1)

        tracker.increment_usage(user_id)
        tracker.increment_usage(user_id)
        self.assertEqual(tracker.get_daily_usage(user_id), 3)

        tracker.reset_usage(user_id)
        self.assertEqual(tracker.get_daily_usage(user_id), 0)


if __name__ == "__main__":
    unittest.main()
