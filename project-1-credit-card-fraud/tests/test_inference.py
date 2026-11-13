from pathlib import Path

"""
API Integration Tests - Project 1: Credit Card Fraud Detection
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from src.inference import main
from src.inference.main import app, verify_api_key


@pytest.fixture(autouse=True)
def mock_mlflow_model(tmp_path, monkeypatch):
    """Injects a mock model, isolated segment_thresholds.json, and overrides API key for baseline functional tests."""
    import json

    art_dir = tmp_path / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    default_policy = {
        "low_value_threshold": 0.50,
        "high_value_threshold": 0.70,
        "scaled_amount_q80": 1.0,
        "calibration_split": "validation_holdout_20pct",
        "mlflow_run_id": "run_test",
    }
    (art_dir / "segment_thresholds.json").write_text(
        json.dumps(default_policy), encoding="utf-8"
    )
    monkeypatch.setattr("src.inference.main.PROJECT_ROOT", tmp_path)
    app.dependency_overrides[verify_api_key] = lambda: None
    with (
        patch("src.inference.main.MlflowClient") as mock_client_cls,
        patch("src.inference.main.mlflow.xgboost.load_model") as mock_load,
    ):
        mock_mv = MagicMock()
        mock_mv.version = "1"
        mock_mv.current_stage = "Production"
        mock_mv.run_id = "run_test"
        mock_client_cls.return_value.search_model_versions.return_value = [mock_mv]
        mock_model = MagicMock()
        mock_model.predict.return_value = [0]
        mock_model.predict_proba.return_value = [[0.9, 0.1]]
        mock_model.get_booster.return_value.feature_names = main.EXPECTED_FEATURE_NAMES
        mock_load.return_value = mock_model

        main.model = mock_model
        main.segment_policy = default_policy.copy()
        yield mock_model

        main.model = None
        main.segment_policy = None
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


def test_non_ascii_api_key_returns_401(monkeypatch):
    """Verifies that non-ASCII X-API-Key values raise HTTP 401 instead of ValueError/HTTP 500."""
    from fastapi import HTTPException

    monkeypatch.setenv("EXPECTED_API_KEY", "test-secret-key")
    with pytest.raises(HTTPException) as exc_info:
        verify_api_key("키-non-ascii")
    assert exc_info.value.status_code == 401  # nosec B101


def test_portable_mlflow3_model_id_fallback(tmp_path, monkeypatch):
    """Verifies _load_xgboost_model_portable resolves MLflow 3 models:/m-<id> sources under /app/mlruns."""
    artifact_dir = tmp_path / "1" / "models" / "m-abc123" / "artifacts"
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "MLmodel").write_text("model_uuid: test", encoding="utf-8")

    mock_client = MagicMock()
    mock_mv = MagicMock()
    mock_mv.storage_location = "file:///host/mlruns/1/models/m-abc123/artifacts"
    mock_mv.source = "models:/m-abc123"
    mock_mv.run_id = "run_999"
    mock_client.get_model_version.return_value = mock_mv

    calls = []

    valid_mock = MagicMock()
    valid_mock.get_booster.return_value.feature_names = main.EXPECTED_FEATURE_NAMES

    def fake_load(uri):
        calls.append(uri)
        if uri.startswith("models:/"):
            raise OSError("Host path missing inside container")
        return valid_mock

    monkeypatch.setattr("src.inference.main.mlflow.xgboost.load_model", fake_load)
    monkeypatch.setattr(
        "src.inference.main.Path",
        lambda p: tmp_path if p == "/app/mlruns" else Path(p),
    )
    loaded_mod, resolved_src = main._load_xgboost_model_portable(
        mock_client, "fraud_detection_xgb", "models:/fraud_detection_xgb/1", 1
    )
    assert loaded_mod is valid_mock  # nosec B101
    assert "m-abc123" in resolved_src  # nosec B101


def test_segment_specific_cutoff_and_run_id_verification(tmp_path, monkeypatch):
    """Verifies that /predict applies low_value_threshold vs high_value_threshold based on scaled_amount_q80 and verifies mlflow_run_id."""
    import json

    # Probability 0.60 is >= low_value_threshold (0.50) but < high_value_threshold (0.70)
    main.model.predict_proba.return_value = [[0.40, 0.60]]
    low_amt_features = [0.0] * 28 + [0.2, 0.0]  # scaled_amount = 0.2 < 1.0
    high_amt_features = [0.0] * 28 + [2.5, 0.0]  # scaled_amount = 2.5 >= 1.0
    with TestClient(app) as client:
        res_low = client.post("/predict", json={"features": low_amt_features})
        assert res_low.status_code == 200  # nosec B101
        assert res_low.json()["prediction"] == 1  # nosec B101

        res_high = client.post("/predict", json={"features": high_amt_features})
        assert res_high.status_code == 200  # nosec B101
        assert res_high.json()["prediction"] == 0  # nosec B101

    monkeypatch.setattr("src.inference.main.PROJECT_ROOT", tmp_path)
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "segment_thresholds.json").write_text(
        json.dumps(
            {
                "low_value_threshold": 0.35,
                "high_value_threshold": 0.65,
                "scaled_amount_q80": 1.08,
                "mlflow_run_id": "run_good_123",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Segment threshold run_id mismatch"):
        main._load_startup_segment_policy(expected_run_id="run_wrong_999")
    loaded_pol = main._load_startup_segment_policy(expected_run_id="run_good_123")
    assert loaded_pol["high_value_threshold"] == 0.65  # nosec B101
