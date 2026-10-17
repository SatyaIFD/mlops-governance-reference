import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Ensure project source root is on path for easy importing
tests_dir = Path(__file__).resolve().parent
project_root = tests_dir.parent
sys.path.append(str(project_root))

from src.inference.app import app  # noqa: E402


@pytest.fixture(autouse=True)
def mock_mlflow_model():
    """Injects a mock model to bypass empty MLflow registries in CI runners."""
    with patch("src.inference.app.mlflow.pyfunc.load_model") as mock_load:
        mock_model = MagicMock()
        # Simulate a standard risk prediction so the test assertion passes
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.8, 0.2]]
        mock_load.return_value = mock_model
        yield mock_model


@pytest.fixture
def client():
    """Context-managed test client that guarantees startup events execute perfectly."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_payload():
    """Returns a fully compliant underwriting payload."""
    return {
        "Age": 25,
        "Income": 55000,
        "CreditScore": 680,
        "LoanAmount": 15000,
        "DTIRatio": 0.28,
        "MonthsEmployed": 36,
        "InterestRate": 5.4,
        "NumCreditLines": 3,
        "LoanTerm": 48,
        "Education": "Bachelors",
        "EmploymentType": "Full-time",
        "MaritalStatus": "Single",
        "HasMortgage": "No",
        "HasDependents": "No",
        "LoanPurpose": "Auto",
        "HasCoSigner": "No",
    }


def test_health_endpoint(client):
    """Asserts that the monitoring health check endpoint responds successfully."""
    response = client.get("/health")
    assert response.status_code == 200  # nosec B101
    assert response.json()["status"] == "HEALTHY"  # nosec B101


def test_predict_success(client, valid_payload):
    """Asserts that a valid payload returns a 200 OK and correct risk telemetry."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200  # nosec B101

    data = response.json()
    assert "risk_metrics" in data  # nosec B101
    assert "governance" in data  # nosec B101
    assert data["risk_metrics"]["risk_score_tier"] == "STANDARD RISK"  # nosec B101
    assert data["governance"]["underwriting_decision"] == "APPROVED"  # nosec B101


def test_invalid_payload_schema(client):
    """Asserts that incoming malformed payloads are caught and rejected by Pydantic validation boundaries."""
    malformed_payload = {"Age": 25, "Education": "Bachelors"}
    response = client.post("/predict", json=malformed_payload)
    assert response.status_code == 422  # nosec B101
