import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

tests_dir = Path(__file__).resolve().parent
project_root = tests_dir.parent
sys.path.append(str(project_root))

from src.inference.main import app, verify_api_key  # noqa: E402
from src.inference import main as p2_main  # noqa: E402

EXPECTED_MOCK_FEATURES = [
    "Age",
    "Income",
    "CreditScore",
    "LoanAmount",
    "DTIRatio",
    "MonthsEmployed",
    "NumCreditLines",
    "InterestRate",
    "LoanTerm",
    "is_young",
    "Education_High School",
    "Education_Masters",
    "Education_PhD",
    "EmploymentType_Part-time",
    "EmploymentType_Self-employed",
    "EmploymentType_Unemployed",
    "MaritalStatus_Married",
    "MaritalStatus_Single",
    "HasMortgage_Yes",
    "HasDependents_Yes",
    "LoanPurpose_Business",
    "LoanPurpose_Home",
    "LoanPurpose_Personal",
    "HasCoSigner_Yes",
]


@pytest.fixture(autouse=True)
def mock_mlflow_model(tmp_path, monkeypatch):
    """Injects a mock model with realistic training one-hot columns and isolates telemetry logs."""
    monkeypatch.setenv("LOAN_TELEMETRY_LOG_PATH", str(tmp_path / "telemetry.jsonl"))
    app.dependency_overrides[verify_api_key] = lambda: None

    with patch("src.inference.main.mlflow.xgboost.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.8, 0.2]]
        mock_model.get_booster.return_value.feature_names = EXPECTED_MOCK_FEATURES
        mock_load.return_value = mock_model

        p2_main.model = mock_model
        p2_main.feature_columns = EXPECTED_MOCK_FEATURES
        yield mock_model

        p2_main.model = None
        p2_main.feature_columns = None
        app.dependency_overrides.clear()


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
        "Education": "PhD",
        "EmploymentType": "Self-employed",
        "MaritalStatus": "Single",
        "HasMortgage": "Yes",
        "HasDependents": "No",
        "LoanPurpose": "Business",
        "HasCoSigner": "Yes",
    }


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200  # nosec B101
    assert response.json()["status"] == "HEALTHY"  # nosec B101


def test_predict_success_low_risk_and_decision_semantics(client, valid_payload):
    """Verifies low-risk prediction, EU AI Act routing, and explicit decision semantics."""
    p2_main.model.predict_proba.return_value = [[0.8, 0.2]]

    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200  # nosec B101

    data = response.json()
    assert "risk_metrics" in data  # nosec B101
    assert "governance" in data  # nosec B101
    assert data["governance"]["action"] == "MANUAL_REVIEW_REQUIRED"  # nosec B101
    assert data["governance"]["model_recommendation"] == "APPROVE"  # nosec B101
    assert data["governance"]["review_status"] == "PENDING_HUMAN_REVIEW"  # nosec B101
    assert data["governance"]["final_decision"] is None  # nosec B101
    assert data["governance"]["age_group"] == "UNDER_30"  # nosec B101


def test_single_row_categorical_features_preserved(client, valid_payload):
    """Proves single-row categorical inputs are not dropped by get_dummies before model scoring."""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200  # nosec B101

    passed_df = p2_main.model.predict_proba.call_args[0][0]
    assert list(passed_df.columns) == EXPECTED_MOCK_FEATURES  # nosec B101
    assert int(passed_df["Education_PhD"].iloc[0]) == 1  # nosec B101
    assert int(passed_df["EmploymentType_Self-employed"].iloc[0]) == 1  # nosec B101
    assert int(passed_df["HasMortgage_Yes"].iloc[0]) == 1  # nosec B101
    assert int(passed_df["HasDependents_Yes"].iloc[0]) == 0  # nosec B101
    assert int(passed_df["LoanPurpose_Business"].iloc[0]) == 1  # nosec B101
    assert int(passed_df["HasCoSigner_Yes"].iloc[0]) == 1  # nosec B101


def test_predict_success_high_risk(client, valid_payload):
    """Asserts that a high-risk payload produces DECLINE recommendation and requires manual review."""
    p2_main.model.predict_proba.return_value = [[0.15, 0.85]]

    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200  # nosec B101

    data = response.json()
    assert data["governance"]["action"] == "MANUAL_REVIEW_REQUIRED"  # nosec B101
    assert data["governance"]["model_recommendation"] == "DECLINE"  # nosec B101


def test_invalid_payload_schema(client):
    malformed_payload = {"Age": 25, "Education": "Bachelors"}
    response = client.post("/predict", json=malformed_payload)
    assert response.status_code == 422  # nosec B101


def test_predict_rejects_nan_float(client, valid_payload):
    response = client.post(
        "/predict",
        content='{"Age": 25, "Income": 55000, "CreditScore": 680, "LoanAmount": 15000, "DTIRatio": NaN, "MonthsEmployed": 36, "InterestRate": 5.4, "NumCreditLines": 3, "LoanTerm": 48, "Education": "Bachelors", "EmploymentType": "Full-time", "MaritalStatus": "Single", "HasMortgage": "No", "HasDependents": "No", "LoanPurpose": "Auto", "HasCoSigner": "No"}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422  # nosec B101


def test_auth_missing_server_config_returns_500(client, valid_payload, monkeypatch):
    app.dependency_overrides.clear()
    monkeypatch.delenv("EXPECTED_API_KEY", raising=False)
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 500  # nosec B101


def test_auth_invalid_or_missing_client_key_returns_401(
    client, valid_payload, monkeypatch
):
    app.dependency_overrides.clear()
    monkeypatch.setenv("EXPECTED_API_KEY", "loan-secret-key")

    res_missing = client.post("/predict", json=valid_payload)
    assert res_missing.status_code == 401  # nosec B101

    res_wrong = client.post(
        "/predict", json=valid_payload, headers={"X-API-Key": "wrong"}
    )
    assert res_wrong.status_code == 401  # nosec B101

    res_ok = client.post(
        "/predict", json=valid_payload, headers={"X-API-Key": "loan-secret-key"}
    )
    assert res_ok.status_code == 200  # nosec B101
