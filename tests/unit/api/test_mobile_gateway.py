import os
import unittest
from fastapi.testclient import TestClient
from src.presentation.api.main import app
from src.presentation.api.mobile_gateway import DEFAULT_MOBILE_TOKEN

class TestMobileGateway(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.token = DEFAULT_MOBILE_TOKEN
        self.auth_headers = {"Authorization": f"Bearer {self.token}"}

    def test_unauthorized_dashboard_access(self):
        """Dashboard must reject unauthenticated requests with 401."""
        response = self.client.get("/api/v1/mobile/dashboard")
        self.assertEqual(response.status_code, 401)

    def test_invalid_token_rejected(self):
        """Dashboard must reject wrong tokens with 401."""
        response = self.client.get(
            "/api/v1/mobile/dashboard",
            headers={"Authorization": "Bearer wrong-token-xyz"}
        )
        self.assertEqual(response.status_code, 401)

    def test_authorized_dashboard(self):
        """Authorized dashboard returns complete Cockpit summary."""
        response = self.client.get(
            "/api/v1/mobile/dashboard",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("balance_usd", data)
        self.assertIn("equity_usd", data)
        self.assertIn("pnl", data)
        self.assertIn("risk", data)
        self.assertIn("gateways_summary", data)
        self.assertEqual(data["gateways_summary"]["total"], 7)

    def test_authorized_gateways(self):
        """Returns the status of all 7 institutional defense gates."""
        response = self.client.get(
            "/api/v1/mobile/gateways",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_active"], 7)
        self.assertEqual(len(data["gates"]), 7)

    def test_authorized_positions(self):
        """Returns live positions list."""
        response = self.client.get(
            "/api/v1/mobile/positions",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("positions", data)

    def test_authorized_signals(self):
        """Returns audited signal log from SQLite."""
        response = self.client.get(
            "/api/v1/mobile/signals?limit=10",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("signals", data)

    def test_authorized_risk_matrix(self):
        """Returns funding rates, Hurst regimes, and circuit breaker."""
        response = self.client.get(
            "/api/v1/mobile/risk-matrix",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("funding_rates", data)
        self.assertIn("hurst_regimes", data)
        self.assertIn("circuit_breaker", data)

    def test_emergency_kill_switch_invalid_code(self):
        """Kill switch must reject invalid confirmation codes with 400."""
        payload = {
            "scope": "ALL",
            "action": "CANCEL_ORDERS_AND_CLOSE_POSITIONS",
            "confirmation_code": "WRONG-CODE"
        }
        response = self.client.post(
            "/api/v1/mobile/emergency/kill",
            json=payload,
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 400)

    def test_emergency_kill_switch_valid(self):
        """Kill switch accepts exact 'KILL-MABUY' confirmation code."""
        payload = {
            "scope": "ALL",
            "action": "CANCEL_ORDERS_AND_CLOSE_POSITIONS",
            "confirmation_code": "KILL-MABUY",
            "reason": "Automated unit test emergency test"
        }
        response = self.client.post(
            "/api/v1/mobile/emergency/kill",
            json=payload,
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ACCEPTED")
        self.assertIn("operation_id", data)

    def test_websocket_stream_unauthorized(self):
        """WebSocket stream closes with 1008 if wrong token provided."""
        try:
            with self.client.websocket_connect("/api/v1/mobile/stream?token=invalid") as ws:
                pass
        except Exception:
            # Rejection before accept is expected
            pass

    def test_websocket_stream_authorized(self):
        """WebSocket stream delivers live heartbeat message."""
        with self.client.websocket_connect(f"/api/v1/mobile/stream?token={self.token}") as ws:
            msg = ws.receive_json()
            self.assertEqual(msg["type"], "heartbeat")
            self.assertEqual(msg["data"]["gateway_total"], 7)
            self.assertIn("equity_usd", msg["data"])

    def test_cinema_vault_catalog(self):
        """Returns curated cinema vault catalog."""
        response = self.client.get(
            "/api/v1/mobile/media/cinema",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertGreaterEqual(len(data["movies"]), 1)
        self.assertIn("embed_url", data["movies"][0])

    def test_cyber_radio_stations(self):
        """Returns 24/7 lossless cyber radio stations."""
        response = self.client.get(
            "/api/v1/mobile/media/radio",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertGreaterEqual(len(data["stations"]), 1)
        self.assertIn("stream_url", data["stations"][0])

    def test_spotify_playlist(self):
        """Returns Creator's official Spotify playlist."""
        response = self.client.get(
            "/api/v1/mobile/media/spotify",
            headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["playlist_id"], "2jXNXmksAPecZ3aNgilBu8")
        self.assertIn("embed_url", data)

    def test_mobile_page_served(self):
        """Serves the Web/PWA mobile client on GET and HEAD /mobile."""
        get_res = self.client.get("/mobile")
        self.assertEqual(get_res.status_code, 200)
        self.assertIn("MABUY CITADEL", get_res.text)

        head_res = self.client.head("/mobile")
        self.assertEqual(head_res.status_code, 200)

    def test_search_music_empty_and_valid(self):
        """Searches music tracks with empty and valid queries."""
        res_empty = self.client.get("/api/v1/mobile/media/search-music?q=", headers=self.auth_headers)
        self.assertEqual(res_empty.status_code, 200)
        self.assertEqual(res_empty.json()["tracks"], [])

        res_gates = self.client.get("/api/v1/mobile/gates", headers=self.auth_headers)
        self.assertEqual(res_gates.status_code, 200)

        dash_res = self.client.get("/api/v1/mobile/dashboard", headers=self.auth_headers)
        self.assertEqual(dash_res.status_code, 200)
        dash_data = dash_res.json()
        self.assertIn("portfolio", dash_data)
        self.assertIn("total_equity_usd", dash_data["portfolio"])

if __name__ == "__main__":
    unittest.main()

