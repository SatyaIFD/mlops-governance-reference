"""
API Integration Tests - Project 1: Credit Card Fraud Detection
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.inference.main import app, verify_api_key
from src.inference import main


@pytest.fixture(autouse=True)
def mock_mlflow_model():
    """Injects a mock model and overrides API key for baseline functional tests."""
    app.dependency_overrides[verify_api_key] = lambda: None
    with patch("src.inference.main.mlflow.xgboost.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.9, 0.1]]
        mock_model.get_booster.return_value.feature_names = main.EXPECTED_FEATURE_NAMES
        mock_load.return_value = mock_model

        main.model = mock_model
        yield mock_model

        main.model = None
        app.dependency_overrides.clear()


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200  # nosec B101
        assert response.json()["status"] == "HEALTHY"  # nosec B101
        assert response.json()["model_loaded"] is True  # nosec B101


def test_predict_success():
    mock_features = [0.0] + [-1.35] * 28 + [85.00]
    payload = {"features": mock_features}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        assert response.status_code == 200  # nosec B101
        assert "prediction" in response.json()  # nosec B101


def test_predict_contract_violation():
    with TestClient(app) as client:
        response = client.post("/predict", json={"features": [0.0, 1.0]})
        assert response.status_code == 422  # nosec B101


def test_predict_rejects_nan_or_infinity():
    """Verifies that NaN or Inf values are rejected at the schema boundary without tripping the circuit breaker."""
    with TestClient(app) as client:
        # allow_nan=True in json client serialization to test server rejection
        response = client.post(
            "/predict",
            content='{"features": [NaN, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}',
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422  # nosec B101


def test_auth_missing_server_config_returns_500(monkeypatch):
    """Verifies that missing EXPECTED_API_KEY fails closed with HTTP 500."""
    app.dependency_overrides.clear()
    monkeypatch.delenv("EXPECTED_API_KEY", raising=False)
    mock_features = [0.0] * 30
    with TestClient(app) as client:
        response = client.post("/predict", json={"features": mock_features})
        assert response.status_code == 500  # nosec B101


def test_auth_invalid_or_missing_client_key_returns_401(monkeypatch):
    """Verifies that missing or wrong X-API-Key returns HTTP 401 when server key is configured."""
    app.dependency_overrides.clear()
    monkeypatch.setenv("EXPECTED_API_KEY", "test-secret-key")
    mock_features = [0.0] * 30
    with TestClient(app) as client:
        res_missing = client.post("/predict", json={"features": mock_features})
        assert res_missing.status_code == 401  # nosec B101

        res_wrong = client.post(
            "/predict",
            json={"features": mock_features},
            headers={"X-API-Key": "wrong-key"},
        )
        assert res_wrong.status_code == 401  # nosec B101

        res_ok = client.post(
            "/predict",
            json={"features": mock_features},
            headers={"X-API-Key": "test-secret-key"},
        )
        assert res_ok.status_code == 200  # nosec B101
