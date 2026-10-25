import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

tests_dir = Path(__file__).resolve().parent
project_root = tests_dir.parent
sys.path.append(str(project_root))

from src.inference.main import app  # noqa: E402
from src.inference import main as p2_main  # noqa: E402


@pytest.fixture(autouse=True)
def mock_mlflow_model():
    """Injects a mock model to bypass empty MLflow registries in CI runners."""
    with patch("src.inference.main.mlflow.xgboost.load_model") as mock_load:
        mock_model = MagicMock()
        # Predict a low risk of default (0.2) to test standard routing
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.8, 0.2]]
        mock_model.feature_names_in_ = ["Age", "Income", "CreditScore"]
        mock_load.return_value = mock_model

        p2_main.model = mock_model
        yield mock_model

        p2_main.model = None


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_payload():
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
    response = client.get("/health")
    assert response.status_code == 200  # nosec B101
    assert response.json()["status"] == "HEALTHY"  # nosec B101


def test_predict_success_low_risk(client, valid_payload):
    """Asserts that a low-risk payload is scored normally by the EU AI Act router."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200  # nosec B101

    data = response.json()
    assert "risk_metrics" in data  # nosec B101
    assert "governance" in data  # nosec B101
    # The probability injected by the mock is 0.2 (< 0.50 threshold)
    assert data["governance"]["action"] == "SCORE_ONLY"  # nosec B101


def test_predict_success_high_risk(client, valid_payload):
    """Asserts that a high-risk payload forces a manual review per the EU AI Act."""
    # Dynamically alter the mock to simulate a high default risk (0.85)
    p2_main.model.predict_proba.return_value = [[0.15, 0.85]]

    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200  # nosec B101

    data = response.json()
    assert data["governance"]["action"] == "MANUAL_REVIEW_REQUIRED"  # nosec B101


def test_invalid_payload_schema(client):
    malformed_payload = {"Age": 25, "Education": "Bachelors"}
    response = client.post("/predict", json=malformed_payload)
    assert response.status_code == 422  # nosec B101
