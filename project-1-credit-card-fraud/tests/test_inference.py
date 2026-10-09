import os
import pytest
from fastapi.testclient import TestClient
from src.inference.app import app


@pytest.fixture
def client():
    if os.environ.get("GITHUB_ACTIONS") == "true":
        # Override the app state to simulate a loaded model for CI tests
        app.state.model = "mock_model"
        app.state.model_version = "mock_version"
    return TestClient(app)


def test_health_check(client):
    """Verifies the health probe endpoint is active."""
    response = client.get("/health")
    assert response.status_code == 200  # nosec B101
    assert response.json()["status"] == "HEALTHY"  # nosec B101

    # Only assert model is loaded if we are not in CI mocking it
    if os.environ.get("GITHUB_ACTIONS") != "true":
        assert response.json()["model_loaded"] is True  # nosec B101


def test_predict_success(client):
    """Verifies that the /predict endpoint correctly processes a valid feature payload."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        pytest.skip("Skipping live prediction endpoint test in CI/CD runner")

    mock_features = [0.0] + [-1.35] * 28 + [85.00]
    payload = {"features": mock_features}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text  # nosec B101
