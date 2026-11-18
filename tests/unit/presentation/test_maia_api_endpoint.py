import os
os.environ["ENCRYPTION_KEY"] = "UN_5bf8rBxMiQOudLLjBJz42sY6vWznie0oxcjcKLYQ="
import unittest
from fastapi.testclient import TestClient
from src.presentation.api.main import app

class TestMaIAApiEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_maia_ask_quant_persona(self):
        payload = {
            "user_id": "100000001",
            "query": "Что думаешь по SOL на споте?",
            "persona": "quant",
            "deposit": 1000.0
        }
        response = self.client.post("/api/v1/maia/ask", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertIn("oled_expression", data)
        self.assertEqual(data["persona"], "quant")
        self.assertIn("SOL", data["target_symbol"])
        self.assertIn("DYOR", data["text"])
        self.assertGreater(data["latency_ms"], 0)

    def test_maia_ask_zen_persona(self):
        payload = {
            "user_id": "100000001",
            "query": "Как распределить $1000 по стратегии DCA?",
            "persona": "zen",
            "deposit": 1000.0
        }
        response = self.client.post("/api/v1/maia/ask", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["persona"], "zen")
        self.assertIn("DYOR", data["text"])

    def test_maia_ask_tsundere_persona(self):
        payload = {
            "user_id": "100000001",
            "query": "Стоит ли брать плечо 20x по BTC?",
            "persona": "tsundere",
            "deposit": 500.0
        }
        response = self.client.post("/api/v1/maia/ask", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["persona"], "tsundere")
        self.assertIn("DYOR", data["text"])

    def test_maia_ask_titan_persona(self):
        payload = {
            "user_id": "100000001",
            "query": "Анализ макро-модели по NVDA",
            "persona": "titan",
            "deposit": 2500.0
        }
        response = self.client.post("/api/v1/maia/ask", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["persona"], "titan")
        self.assertIn("NVDA", data["target_symbol"])

if __name__ == "__main__":
    unittest.main()
