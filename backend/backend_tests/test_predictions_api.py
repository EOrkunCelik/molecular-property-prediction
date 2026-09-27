from __future__ import annotations


class TestPredictEndpoint:
    def test_predict_valid_smiles_returns_201(self, client):
        response = client.post("/api/v1/predict", json={"smiles": "CCO"})
        assert response.status_code == 201
        body = response.json()
        assert body["canonical_smiles"]
        assert body["molecular_formula"] == "C2H6O"
        assert "predicted_log_solubility" in body
        assert "predicted_solubility_mol_per_l" in body
        assert body["model_info"]["model_name"] == "fake_test_model"

    def test_predict_invalid_smiles_returns_422(self, client):
        response = client.post("/api/v1/predict", json={"smiles": "not a smiles!!"})
        assert response.status_code == 422

    def test_predict_blank_smiles_returns_422(self, client):
        response = client.post("/api/v1/predict", json={"smiles": "   "})
        assert response.status_code == 422

    def test_predict_missing_field_returns_422(self, client):
        response = client.post("/api/v1/predict", json={})
        assert response.status_code == 422

    def test_predicted_solubility_conversion_is_consistent(self, client):
        response = client.post("/api/v1/predict", json={"smiles": "CCO"})
        body = response.json()
        assert abs(body["predicted_solubility_mol_per_l"] - 10 ** body["predicted_log_solubility"]) < 1e-9


class TestPredictionHistoryEndpoints:
    def test_list_predictions_empty_initially(self, client):
        response = client.get("/api/v1/predictions")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 0
        assert body["items"] == []

    def test_list_predictions_after_creating_some(self, client):
        for smiles in ["CCO", "c1ccccc1", "CCCO"]:
            client.post("/api/v1/predict", json={"smiles": smiles})

        response = client.get("/api/v1/predictions")
        body = response.json()
        assert body["total"] == 3
        assert len(body["items"]) == 3
        # newest first
        assert body["items"][0]["original_smiles"] == "CCCO"

    def test_pagination_limit(self, client):
        for smiles in ["CCO", "c1ccccc1", "CCCO"]:
            client.post("/api/v1/predict", json={"smiles": smiles})

        response = client.get("/api/v1/predictions", params={"limit": 2})
        body = response.json()
        assert len(body["items"]) == 2
        assert body["total"] == 3

    def test_get_prediction_by_id(self, client):
        create_resp = client.post("/api/v1/predict", json={"smiles": "CCO"})
        prediction_id = create_resp.json()["id"]

        get_resp = client.get(f"/api/v1/predictions/{prediction_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == prediction_id

    def test_get_nonexistent_prediction_returns_404(self, client):
        response = client.get("/api/v1/predictions/does-not-exist")
        assert response.status_code == 404


class TestModelUnavailable:
    """Verifies the `FileNotFoundError -> 503` handler in app.main (see its docstring):
    if the model hasn't been trained yet, /predict should fail cleanly with a 503 and
    an actionable message, not an opaque 500.
    """

    def test_predict_returns_503_when_model_not_trained(self, client, db_session_factory):
        from app.main import app
        from app.ml_runtime.model_loader import get_predictor

        def raise_not_found():
            raise FileNotFoundError(
                "No trained model found. Run `python scripts/train_model.py` first."
            )

        # Overrides the `client` fixture's FakePredictor override for this test only;
        # the fixture's own teardown (`app.dependency_overrides.clear()`) resets this
        # after the test regardless of how it exits, so no manual restoration is needed.
        app.dependency_overrides[get_predictor] = raise_not_found

        response = client.post("/api/v1/predict", json={"smiles": "CCO"})
        assert response.status_code == 503
        assert "train_model.py" in response.json()["detail"]


class TestVisualizationEndpoints:
    def test_visualize_valid_smiles_returns_svg(self, client):
        response = client.get("/api/v1/visualize", params={"smiles": "CCO"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/svg+xml")
        assert b"<svg" in response.content

    def test_visualize_invalid_smiles_returns_422(self, client):
        response = client.get("/api/v1/visualize", params={"smiles": "not a smiles!!"})
        assert response.status_code == 422

    def test_prediction_structure_endpoint(self, client):
        create_resp = client.post("/api/v1/predict", json={"smiles": "CCO"})
        prediction_id = create_resp.json()["id"]

        response = client.get(f"/api/v1/predictions/{prediction_id}/structure")
        assert response.status_code == 200
        assert b"<svg" in response.content
