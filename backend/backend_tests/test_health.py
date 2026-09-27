from __future__ import annotations


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_reports_model_loaded(self, client):
        response = client.get("/api/v1/health")
        body = response.json()
        assert body["model_loaded"] is True
        assert body["model_name"] == "fake_test_model"

    def test_health_reports_database_connected(self, client):
        response = client.get("/api/v1/health")
        body = response.json()
        assert body["database_connected"] is True
        assert body["status"] == "ok"
