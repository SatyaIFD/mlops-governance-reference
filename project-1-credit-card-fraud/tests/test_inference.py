"""
Automated Integration Tests for Inference API
Validates the FastAPI data contract, feature ordering, and model prediction logic.
"""

import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Dynamically set tracking URI relative to project root
BASE_DIR = Path(__file__).resolve().parents[1]
os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{BASE_DIR}/mlflow.db"

from src.inference.app import app

def test_health_check():
    """Verifies the health probe endpoint is active."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "HEALTHY"
        assert response.json()["model_loaded"] is True

def test_predict_success():
    """Verifies that the /predict endpoint correctly processes a valid feature payload."""
    mock_features = [0.0] + [-1.35] * 28 + [85.00]
    payload = {
        "features": mock_features
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "risk_score" in data
        assert isinstance(data["prediction"], int)

def test_predict_contract_violation():
    """Verifies that the API correctly rejects malformed input (contract enforcement)."""
    bad_payload = {"features": [0.1, 0.2, 0.3, 0.4, 0.5]}
    with TestClient(app) as client:
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422