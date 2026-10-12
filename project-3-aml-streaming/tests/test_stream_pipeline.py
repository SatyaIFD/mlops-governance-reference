from unittest.mock import MagicMock, patch
from src.ingestion.stream_ingestor import StreamingIngestionEngine
from src.features.velocity import StreamingStateManager
from src.models.anomaly_detector import AMLAnomalyDetector


def test_streaming_ingestion_engine_validation(tmp_path):
    """Verifies schema validation and DLQ routing logic."""
    engine = StreamingIngestionEngine(dlq_output_dir=tmp_path)
    valid_payload = {
        "Timestamp": "2024-03-15T10:00:00Z",
        "Sender_account": "ACC_123",
        "Receiver_account": "ACC_456",
        "Amount": 1500.00,
        "Payment_currency": "USD",
    }
    is_valid, processed = engine.validate_and_route(valid_payload)
    assert is_valid is True  # nosec B101
    assert processed["Amount"] == 1500.0  # nosec B101


def test_velocity_state_manager_enrichment():
    """Verifies stateful feature engineering over lookback windows."""
    manager = StreamingStateManager()
    tx1 = {
        "Timestamp": "2024-03-15T10:00:00Z",
        "Sender_account": "A",
        "Receiver_account": "B",
        "Amount": 1000.0,
    }
    tx2 = {
        "Timestamp": "2024-03-15T10:05:00Z",
        "Sender_account": "A",
        "Receiver_account": "C",
        "Amount": 500.0,
    }

    manager.update_and_enrich(tx1)
    enriched = manager.update_and_enrich(tx2)

    assert "pass_through_ratio_1h" in enriched  # nosec B101
    assert "receiver_inflow_count_1h" in enriched  # nosec B101


@patch("src.models.anomaly_detector.mlflow.pyfunc.load_model")
@patch("src.models.anomaly_detector.MlflowClient")
@patch("src.models.anomaly_detector.mlflow.set_tracking_uri")
def test_anomaly_detector_mlflow_integration(
    mock_set_uri, mock_client, mock_load_model
):
    """Verifies the detector correctly loads from MLflow and processes features."""
    # Setup mock MLflow responses
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance

    mock_version = MagicMock()
    mock_version.version = "1"
    mock_client_instance.search_model_versions.return_value = [mock_version]

    # Setup mock sklearn model
    mock_sklearn = MagicMock()
    mock_sklearn.predict_proba.return_value = [
        [0.1, 0.85]
    ]  # Simulate 85% fraud probability

    # Setup mock MLflow wrapper
    mock_wrapper = MagicMock()
    mock_wrapper._model_impl.sklearn_model = mock_sklearn
    mock_load_model.return_value = mock_wrapper

    # Execute test
    detector = AMLAnomalyDetector()
    detector.load("mock_model_name")

    sample_enriched = {
        "Amount": 9000.00,
        "pass_through_ratio_1h": 0.95,
        "receiver_inflow_count_1h": 3,
        "receiver_inflow_amount_1h": 27000.00,
    }

    prob, is_anomaly = detector.score_transaction(sample_enriched)

    # Verify assertions
    assert prob == 0.85  # nosec B101
    assert is_anomaly == 1  # nosec B101
    mock_set_uri.assert_called_once()
    mock_load_model.assert_called_once_with("models:/mock_model_name/1")
