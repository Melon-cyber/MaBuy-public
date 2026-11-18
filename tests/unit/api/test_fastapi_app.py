import os
os.environ["ENCRYPTION_KEY"] = "UN_5bf8rBxMiQOudLLjBJz42sY6vWznie0oxcjcKLYQ="
import unittest
from fastapi.testclient import TestClient
from src.presentation.api.main import app


class TestFastApiApp(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_check_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("active_models", data)

    def test_prediction_endpoint_free_tier_valid(self):
        import time
        payload = {
            "user_id": f"test_api_free_valid_{int(time.time() * 1000)}",
            "symbol": "BTC",
            "tier": "free",
            "timeframe": "1h"
        }
        response = self.client.post("/api/v1/predictions", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["symbol"], "BTC")
        self.assertIsNone(data["personal_recommendation"]) # Free tier stripped

    def test_prediction_endpoint_free_tier_restricted_asset(self):
        import time
        payload = {
            "user_id": f"test_api_free_restricted_{int(time.time() * 1000)}",
            "symbol": "SOL",
            "tier": "free",
            "timeframe": "1h"
        }
        response = self.client.post("/api/v1/predictions", json=payload)
        self.assertEqual(response.status_code, 403) # AssetNotAllowedException

    def test_prediction_endpoint_pro_tier(self):
        import time
        payload = {
            "user_id": f"test_api_pro_{int(time.time() * 1000)}",
            "symbol": "BTC",
            "tier": "pro",
            "timeframe": "1h"
        }
        response = self.client.post("/api/v1/predictions", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsNotNone(data["personal_recommendation"])
        self.assertIsNotNone(data["sentiment_score"])

    def test_mlops_drift_endpoint(self):
        response = self.client.get("/api/v1/mlops/drift?symbol=BTCUSDT&candle_limit=200")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["symbol"], "BTCUSDT")
        self.assertIn("summary", data)

    def test_calculator_endpoint(self):
        response = self.client.get("/calculator")
        self.assertEqual(response.status_code, 200)
        self.assertIn("MaBuy Safe-Leverage", response.text)
        self.assertIn("Калькулятор риска", response.text)


if __name__ == "__main__":
    unittest.main()
