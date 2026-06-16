"""Tests for the FastAPI endpoints using TestClient."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture()
def client():
    # Seed app state with minimal mocks so lifespan is bypassed
    app.state.loader = MagicMock()
    app.state.inference = None
    app.state.explainer = None
    app.state.score_cache = {}
    app.state.alert_store = []
    app.state.asset_store = {}
    app.state.horizon_url = "https://horizon-testnet.stellar.org"
    app.state.contract_id = "CTEST"
    app.state.models_loaded = False

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "version" in data
        assert "horizon_url" in data
        assert "models_loaded" in data

    def test_health_status_value(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"


# ── Alerts ────────────────────────────────────────────────────────────────────

class TestAlerts:
    def _seed_alerts(self, client, n=5):
        now = datetime.now(timezone.utc)
        client.app.state.alert_store = [
            {
                "wallet": f"G{'A' * 55}{i}",
                "asset_pair": "XLM:native/USDC",
                "score": 80 + i,
                "flagged_at": now,
                "benford_flag": True,
                "ml_flag": True,
            }
            for i in range(n)
        ]

    def test_empty_store_returns_empty_list(self, client):
        resp = client.get("/alerts/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["alerts"] == []

    def test_alerts_schema(self, client):
        self._seed_alerts(client)
        data = client.get("/alerts/recent").json()
        assert "alerts" in data
        assert "total" in data
        assert "page" in data
        assert "limit" in data

    def test_alerts_filtered_by_min_score(self, client):
        self._seed_alerts(client, n=5)  # scores 80–84
        data = client.get("/alerts/recent?min_score=83").json()
        assert all(a["score"] >= 83 for a in data["alerts"])

    def test_alerts_pagination(self, client):
        self._seed_alerts(client, n=5)
        data = client.get("/alerts/recent?limit=2&page=1").json()
        assert len(data["alerts"]) <= 2

    def test_alerts_sorted_by_score_desc(self, client):
        self._seed_alerts(client, n=5)
        data = client.get("/alerts/recent").json()
        scores = [a["score"] for a in data["alerts"]]
        assert scores == sorted(scores, reverse=True)


# ── Assets ────────────────────────────────────────────────────────────────────

class TestAssets:
    def test_empty_store_returns_empty(self, client):
        resp = client.get("/assets/risk-ranking")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_asset_ranking_schema(self, client):
        data = client.get("/assets/risk-ranking").json()
        assert "assets" in data
        assert "total" in data
        assert "window" in data

    def test_window_param_echoed(self, client):
        data = client.get("/assets/risk-ranking?window=7d").json()
        assert data["window"] == "7d"

    def test_invalid_window_returns_422(self, client):
        resp = client.get("/assets/risk-ranking?window=99y")
        assert resp.status_code == 422

    def test_seeded_asset_appears_in_ranking(self, client):
        now = datetime.now(timezone.utc)
        client.app.state.asset_store["24h"] = [
            {"asset_code": "USDC", "asset_issuer": "GA5Z", "score": 85, "wallet": "GAA", "last_updated": now},
            {"asset_code": "USDC", "asset_issuer": "GA5Z", "score": 70, "wallet": "GAB", "last_updated": now},
        ]
        data = client.get("/assets/risk-ranking?window=24h").json()
        assert data["total"] == 1
        usdc = data["assets"][0]
        assert usdc["asset_code"] == "USDC"
        assert usdc["avg_score"] == pytest.approx(77.5)
        assert usdc["flagged_wallet_count"] == 1  # only score 85 ≥ 75
