"""
API Integration Tests
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from src.inference.main import app


@pytest.fixture(autouse=True)
def mock_mlflow_model():
    """Injects a mock model to bypass empty MLflow registries in CI runners."""
    with patch("src.inference.main.mlflow.pyfunc.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.9, 0.1]]
        mock_load.return_value = mock_model
        yield mock_model


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
