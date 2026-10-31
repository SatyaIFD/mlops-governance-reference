import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.ingestion.stream_ingestor import (
    StreamingIngestionEngine,
    KeyVersionedPseudonymizer,
)
from src.features.velocity import StreamingStateManager, CANONICAL_AML_FEATURES
from src.models.anomaly_detector import AMLAnomalyDetector
from scripts.replay_dlq import replay_dlq_events


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
    assert is_valid is True  # nosec B101
    assert processed["event_id"] is not None  # nosec B101
    assert engine.get_data_quality_report()["data_quality_pct"] == 100.0  # nosec B101


def test_streaming_ingestion_rejects_nan_inf_and_bad_types(tmp_path):
    engine = StreamingIngestionEngine(dlq_output_dir=tmp_path)
    for bad_amount in (float("nan"), float("inf"), -50.0, ["invalid_list"]):
        ok, _ = engine.validate_and_route(
            {
                "Timestamp": "2024-03-15 10:00:00",
                "Sender_account": "ACC_1",
                "Receiver_account": "ACC_2",
                "Amount": bad_amount,
            }
        )
        assert ok is False  # nosec B101

    report = engine.get_data_quality_report()
    assert report["dlq_events"] == 4  # nosec B101
    assert report["data_quality_pct"] == 0.0  # nosec B101


def test_velocity_state_manager_enrichment_and_all_7_features():
    manager = StreamingStateManager()
    tx1 = {
        "Timestamp": datetime(2024, 3, 15, 10, 0, 0),
        "Sender_account": "B",
        "Receiver_account": "A",
        "Amount": 10000.0,
    }
    tx2 = {
        "Timestamp": datetime(2024, 3, 15, 10, 5, 0),
        "Sender_account": "A",
        "Receiver_account": "C",
        "Amount": 9000.0,
    }
    manager.update_and_enrich(tx1)
    enriched = manager.update_and_enrich(tx2)

    for col in CANONICAL_AML_FEATURES:
        assert col in enriched  # nosec B101
    assert enriched["is_structuring"] == 1  # nosec B101
    assert enriched["pass_through_ratio_1h"] == 0.9  # nosec B101
    assert enriched["fan_out_count_24h"] == 1  # nosec B101


def test_event_time_watermark_eviction_preserves_historical_window():
    """Proves historical 2024 events are evicted relative to event watermark, not wall-clock year."""
    manager = StreamingStateManager()
    t0 = datetime(2024, 1, 1, 12, 0, 0)
    manager.update_and_enrich(
        {
            "Timestamp": t0,
            "Sender_account": "OLD_ACC",
            "Receiver_account": "MID_ACC",
            "Amount": 500.0,
        }
    )
    t1 = t0 + timedelta(hours=2)
    manager.update_and_enrich(
        {
            "Timestamp": t1,
            "Sender_account": "RECENT_ACC",
            "Receiver_account": "RX_ACC",
            "Amount": 800.0,
        }
    )

    # Evict using event-time watermark (t1 - 24h): both accounts are within 2 hours of t1 and MUST survive
    manager._evict_dormant_accounts()
    assert "OLD_ACC" in manager.state  # nosec B101
    assert "RECENT_ACC" in manager.state  # nosec B101

    # Advance watermark by 25 hours: OLD_ACC is now >24h behind watermark and must be evicted
    t2 = t0 + timedelta(hours=25)
    manager.update_and_enrich(
        {
            "Timestamp": t2,
            "Sender_account": "NEW_ACC",
            "Receiver_account": "RX_ACC",
            "Amount": 300.0,
        }
    )
    manager._evict_dormant_accounts()
    assert "OLD_ACC" not in manager.state  # nosec B101
    assert "NEW_ACC" in manager.state  # nosec B101


def test_duplicate_event_idempotency():
    """Verifies that replaying an event with the same event_id does not double-count receiver inflow."""
    manager = StreamingStateManager()
    tx = {
        "event_id": "evt_001",
        "Timestamp": datetime(2024, 3, 15, 10, 0, 0),
        "Sender_account": "A",
        "Receiver_account": "B",
        "Amount": 1000.0,
    }
    first = manager.update_and_enrich(tx)
    second = manager.update_and_enrich(tx)
    assert first["receiver_inflow_count_1h"] == 1  # nosec B101
    assert second["receiver_inflow_count_1h"] == 1  # nosec B101
    assert second["receiver_inflow_amount_1h"] == 1000.0  # nosec B101


def test_dlq_jsonl_replay_and_hmac_continuity(tmp_path):
    pseudo = KeyVersionedPseudonymizer(key_version="v1", secret_key="unit-test-secret")  # nosec B106
    ingestor = StreamingIngestionEngine(
        dlq_output_dir=tmp_path, pseudonymize_accounts=True, pseudonymizer=pseudo
    )
    state = StreamingStateManager()

    # 1. Quarantine a record with a malformed timestamp, then repair the timestamp in JSONL
    ingestor.validate_and_route(
        {
            "Timestamp": "BAD_DATE",
            "Sender_account": "ACC_SENDER",
            "Receiver_account": "ACC_RECEIVER",
            "Amount": 2500.0,
        }
    )
    dlq_file = tmp_path / "dead_letter_queue.jsonl"
    assert dlq_file.exists()  # nosec B101

    record = json.loads(dlq_file.read_text(encoding="utf-8").strip())
    # Account IDs in DLQ are already HMAC-pseudonymized with v1 prefix
    assert record["raw_payload"]["Sender_account"].startswith("hmac_v1_")  # nosec B101
    record["raw_payload"]["Timestamp"] = "2024-03-15 11:00:00"
    dlq_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    # 2. Replay the repaired JSONL and verify it recovers cleanly without double-hashing
    summary = replay_dlq_events(tmp_path, ingestor, state_manager=state)
    assert summary["replayed_clean"] == 1  # nosec B101
    assert summary["replayed_failed"] == 0  # nosec B101
    expected_sender_token = pseudo.pseudonymize("ACC_SENDER")
    assert expected_sender_token in state.state  # nosec B101


@patch("src.models.anomaly_detector.sio.load")
def test_anomaly_detector_skops_integration(mock_sio_load):
    mock_sklearn = MagicMock()
    mock_sklearn.predict_proba.return_value = [[0.1, 0.85]]
    mock_sio_load.return_value = mock_sklearn

    detector = AMLAnomalyDetector()
    detector.load(Path("mock_model.skops"))
    sample_enriched = {
        "Amount": 9000.00,
        "pass_through_ratio_1h": 0.95,
        "receiver_inflow_count_1h": 3,
        "receiver_inflow_amount_1h": 27000.00,
    }
    prob, is_anomaly = detector.score_transaction(sample_enriched)

    assert prob == 0.85  # nosec B101
    assert is_anomaly == 1  # nosec B101
    mock_sio_load.assert_called_once_with(
        Path("mock_model.skops"),
        trusted=[
            "sklearn.ensemble._forest.RandomForestClassifier",
            "sklearn.tree._tree.Tree",
        ],
    )
