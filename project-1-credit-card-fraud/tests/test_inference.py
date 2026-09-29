"""
Automated Integration Tests for Inference API
Validates the FastAPI data contract, feature ordering, and model prediction logic.
"""

import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Dynamically inject Project 1 root into sys.path to resolve the local 'src' namespace
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

# Dynamically set tracking URI relative to project root
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{BASE_DIR}/mlflow.db"

from src.inference.app import app  # noqa: E402


def test_health_check():
    """Verifies the health probe endpoint is active."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200  # nosec B101
        assert response.json()["status"] == "HEALTHY"  # nosec B101
        assert response.json()["model_loaded"] is True  # nosec B101


def test_predict_success():
    """Verifies that the /predict endpoint correctly processes a valid feature payload."""
    mock_features = [0.0] + [-1.35] * 28 + [85.00]
    payload = {"features": mock_features}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

        # We append response.text so Pytest prints the exact 500 error payload
        assert response.status_code == 200, response.text  # nosec B101
        data = response.json()
        assert "prediction" in data  # nosec B101
        assert "risk_score" in data  # nosec B101
        assert isinstance(data["prediction"], int)  # nosec B101


def test_predict_contract_violation():
    """Verifies that the API correctly rejects malformed input (contract enforcement)."""
    bad_payload = {"features": [0.1, 0.2, 0.3, 0.4, 0.5]}
    with TestClient(app) as client:
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422  # nosec B101
