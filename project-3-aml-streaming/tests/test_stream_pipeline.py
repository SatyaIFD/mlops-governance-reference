from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.ingestion.stream_ingestor import StreamingIngestionEngine
from src.features.velocity import StreamingStateManager
from src.models.anomaly_detector import AMLAnomalyDetector


def test_streaming_ingestion_engine_validation(tmp_path):
    engine = StreamingIngestionEngine(dlq_output_dir=tmp_path)
    valid_payload = {
        "Timestamp": "2024-03-15 10:00:00",
        "Sender_account": "ACC_123",
        "Receiver_account": "ACC_456",
        "Amount": 1500.00,
        "Payment_currency": "USD",
    }
    is_valid, processed = engine.validate_and_route(valid_payload)
    assert is_valid is True


def test_velocity_state_manager_enrichment():
    manager = StreamingStateManager()
    tx1 = {
        "Timestamp": datetime(2024, 3, 15, 10, 0, 0),
        "Sender_account": "A",
        "Receiver_account": "B",
        "Amount": 1000.0,
    }
    tx2 = {
        "Timestamp": datetime(2024, 3, 15, 10, 5, 0),
        "Sender_account": "A",
        "Receiver_account": "C",
        "Amount": 500.0,
    }
    manager.update_and_enrich(tx1)
    enriched = manager.update_and_enrich(tx2)
    assert "pass_through_ratio_1h" in enriched


@patch("src.models.anomaly_detector.sio.load")
@patch("src.models.anomaly_detector.sio.get_untrusted_types")
def test_anomaly_detector_skops_integration(mock_get_untrusted, mock_sio_load):
    mock_sklearn = MagicMock()
    mock_sklearn.predict_proba.return_value = [[0.1, 0.85]]
    mock_sio_load.return_value = mock_sklearn
    mock_get_untrusted.return_value = [
        "sklearn.ensemble._forest.RandomForestClassifier"
    ]

    detector = AMLAnomalyDetector()
    detector.load(Path("mock_model.skops"))
    sample_enriched = {
        "Amount": 9000.00,
        "pass_through_ratio_1h": 0.95,
        "receiver_inflow_count_1h": 3,
        "receiver_inflow_amount_1h": 27000.00,
    }
    prob, is_anomaly = detector.score_transaction(sample_enriched)

    assert prob == 0.85
    mock_get_untrusted.assert_called_once_with(file=Path("mock_model.skops"))
    mock_sio_load.assert_called_once_with(
        Path("mock_model.skops"),
        trusted=["sklearn.ensemble._forest.RandomForestClassifier"],
    )
