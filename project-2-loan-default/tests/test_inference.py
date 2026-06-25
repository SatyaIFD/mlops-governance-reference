"""
Unit Test Suite - Project 2 Inference Layer
Validates FastAPI endpoints and structural feature transformations.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Ensure project source root is on path for easy importing
tests_dir = Path(__file__).resolve().parent
project_root = tests_dir.parent
sys.path.append(str(project_root))

from src.inference.app import app

client = TestClient(app)

def test_health_endpoint():
    """Asserts that the monitoring health check endpoint responds successfully."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "HEALTHY"

def test_invalid_payload_schema():
    """Asserts that incoming malformed payloads are caught and rejected by Pydantic validation boundaries."""
    # Payload is missing required features (like CreditScore, Income)
    malformed_payload = {
        "Age": 25,
        "Education": "Bachelor's"
    }
    response = client.post("/predict", json=malformed_payload)
    assert response.status_code == 422  # Unprocessable Entity