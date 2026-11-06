import json
from datetime import datetime, timedelta

import pandas as pd
from scripts.replay_dlq import replay_dlq_events
from src.features.velocity import CANONICAL_AML_FEATURES, StreamingStateManager
from src.ingestion.stream_ingestor import (
    KeyVersionedPseudonymizer,
    StreamingIngestionEngine,
)
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

    manager._evict_dormant_accounts()
    assert "OLD_ACC" in manager.state  # nosec B101
    assert "RECENT_ACC" in manager.state  # nosec B101

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
    assert record["raw_payload"]["Sender_account"].startswith("hmac_v1_")  # nosec B101
    record["raw_payload"]["Timestamp"] = "2024-03-15 11:00:00"
    dlq_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    summary = replay_dlq_events(tmp_path, ingestor, state_manager=state)
    assert summary["replayed_clean"] == 1  # nosec B101
    assert summary["replayed_failed"] == 0  # nosec B101
    expected_sender_token = pseudo.pseudonymize("ACC_SENDER")
    assert expected_sender_token in state.state  # nosec B101


def test_anomaly_detector_real_skops_roundtrip_and_threshold_persistence(tmp_path):
    """Trains a real RandomForestClassifier, saves via skops with custom threshold, reloads, and scores."""
    df = pd.DataFrame(
        {
            "Amount": [50.0, 9500.0, 10.0, 25000.0, 120.0, 8900.0],
            "pass_through_ratio_1h": [0.0, 0.95, 0.1, 1.0, 0.05, 0.92],
            "is_structuring": [0, 1, 0, 0, 0, 1],
            "velocity_acceleration": [1.0, 12.5, 0.8, 18.0, 1.1, 9.5],
            "fan_out_count_24h": [1, 8, 1, 14, 2, 6],
            "receiver_inflow_count_1h": [1, 5, 1, 15, 1, 7],
            "receiver_inflow_amount_1h": [
                50.0,
                45000.0,
                10.0,
                150000.0,
                120.0,
                62000.0,
            ],
            "Is_laundering": [0, 1, 0, 1, 0, 1],
        }
    )
    artifact_file = tmp_path / "aml_streaming_model"
    detector = AMLAnomalyDetector(threshold=0.65)
    detector.train(df)
    detector.save(artifact_file)

    reloaded = AMLAnomalyDetector(threshold=0.40)
    reloaded.load(artifact_file)
    assert reloaded.threshold == 0.65  # nosec B101

    prob, is_anomaly = reloaded.score_transaction(
        {
            "Amount": 9500.0,
            "pass_through_ratio_1h": 0.95,
            "is_structuring": 1,
            "velocity_acceleration": 12.5,
            "fan_out_count_24h": 8,
            "receiver_inflow_count_1h": 5,
            "receiver_inflow_amount_1h": 45000.0,
        }
    )
    assert prob >= 0.65  # nosec B101
    assert is_anomaly == 1  # nosec B101


def test_anomaly_detector_rejects_tampered_sha256(tmp_path):
    """Verifies that load() recomputes SHA-256 and raises ValueError if .meta.json hash mismatches."""
    import pytest

    df = pd.DataFrame(
        {
            "Amount": [50.0, 9500.0, 10.0, 25000.0],
            "pass_through_ratio_1h": [0.0, 0.95, 0.1, 1.0],
            "is_structuring": [0, 1, 0, 0],
            "velocity_acceleration": [1.0, 12.5, 0.8, 18.0],
            "fan_out_count_24h": [1, 8, 1, 14],
            "receiver_inflow_count_1h": [1, 5, 1, 15],
            "receiver_inflow_amount_1h": [50.0, 45000.0, 10.0, 150000.0],
            "Is_laundering": [0, 1, 0, 1],
        }
    )
    artifact_file = tmp_path / "aml_streaming_model"
    detector = AMLAnomalyDetector(threshold=0.55)
    detector.train(df)
    detector.save(artifact_file)

    meta_file = tmp_path / "aml_streaming_model.meta.json"
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    meta["sha256"] = "0" * 64
    meta_file.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        AMLAnomalyDetector().load(artifact_file)


def test_end_to_end_pipeline_on_synthetic_csv(tmp_path, monkeypatch):
    """Runs run_production_stream_pipeline end-to-end on a synthetic CSV and verifies alert/report parity."""
    from src.streaming.generator import _write_synthetic_saml_fallback
    from src.streaming.pipeline import run_production_stream_pipeline

    csv_path = tmp_path / "SAML-D.csv"
    _write_synthetic_saml_fallback(csv_path, num_rows=120)

    df = pd.DataFrame(
        {
            "Amount": [50.0, 9250.0, 10.0, 9250.0, 120.0, 9250.0],
            "pass_through_ratio_1h": [0.0, 0.95, 0.1, 1.0, 0.05, 0.92],
            "is_structuring": [0, 1, 0, 1, 0, 1],
            "velocity_acceleration": [1.0, 12.5, 0.8, 18.0, 1.1, 9.5],
            "fan_out_count_24h": [1, 8, 1, 14, 2, 6],
            "receiver_inflow_count_1h": [1, 5, 1, 15, 1, 7],
            "receiver_inflow_amount_1h": [
                50.0,
                45000.0,
                10.0,
                150000.0,
                120.0,
                62000.0,
            ],
            "Is_laundering": [0, 1, 0, 1, 0, 1],
        }
    )
    model_path = tmp_path / "aml_streaming_model"
    det = AMLAnomalyDetector(threshold=0.50)
    det.train(df)
    det.save(model_path, mlflow_run_id="test_run_123")

    monkeypatch.setenv("AML_DATA_PATH", str(csv_path))
    monkeypatch.setenv("AML_MODEL_PATH", str(model_path))
    monkeypatch.setattr(
        "src.streaming.pipeline.resolve_runtime_paths",
        lambda: (
            csv_path,
            tmp_path / "dlq",
            tmp_path / "pipeline_execution_report.json",
            model_path,
        ),
    )

    summary = run_production_stream_pipeline(
        max_events=60, delay_seconds=0.0, warmup_events=40
    )
    assert summary["streaming_performance"]["total_processed"] == 60  # nosec B101
    assert summary["data_governance"]["dlq_events"] == 0  # nosec B101
    alerts_file = tmp_path / "aml_alerts.jsonl"
    alert_lines = (
        len(alerts_file.read_text(encoding="utf-8").splitlines())
        if alerts_file.exists()
        else 0
    )
    assert alert_lines == summary["streaming_performance"]["anomalies_flagged"]  # nosec B101


def test_batch_scoring_fallback_does_not_double_count(tmp_path, monkeypatch):
    """Forces score_batch to raise and verifies per-item fallback scores each event exactly once."""
    from src.streaming.generator import _write_synthetic_saml_fallback
    from src.streaming.pipeline import run_production_stream_pipeline

    csv_path = tmp_path / "SAML-D.csv"
    _write_synthetic_saml_fallback(csv_path, num_rows=50)

    df = pd.DataFrame(
        {
            "Amount": [50.0, 9250.0, 10.0, 9250.0, 120.0, 9250.0],
            "pass_through_ratio_1h": [0.0, 0.95, 0.1, 1.0, 0.05, 0.92],
            "is_structuring": [0, 1, 0, 1, 0, 1],
            "velocity_acceleration": [1.0, 12.5, 0.8, 18.0, 1.1, 9.5],
            "fan_out_count_24h": [1, 8, 1, 14, 2, 6],
            "receiver_inflow_count_1h": [1, 5, 1, 15, 1, 7],
            "receiver_inflow_amount_1h": [
                50.0,
                45000.0,
                10.0,
                150000.0,
                120.0,
                62000.0,
            ],
            "Is_laundering": [0, 1, 0, 1, 0, 1],
        }
    )
    model_path = tmp_path / "aml_streaming_model"
    det = AMLAnomalyDetector(threshold=0.50)
    det.train(df)
    det.save(model_path, mlflow_run_id="test_fallback_run")

    monkeypatch.setattr(
        "src.streaming.pipeline.resolve_runtime_paths",
        lambda: (
            csv_path,
            tmp_path / "dlq",
            tmp_path / "pipeline_execution_report.json",
            model_path,
        ),
    )
    monkeypatch.setattr(
        AMLAnomalyDetector,
        "score_batch",
        lambda self, batch: (_ for _ in ()).throw(
            RuntimeError("Simulated batch failure")
        ),
    )

    summary = run_production_stream_pipeline(
        max_events=30, delay_seconds=0.0, warmup_events=10
    )
    assert summary["streaming_performance"]["total_processed"] == 30  # nosec B101
    assert summary["data_governance"]["clean_events"] == 40  # nosec B101
    assert summary["data_governance"]["dlq_events"] == 0  # nosec B101


def test_anomaly_detector_requires_meta_file_and_all_features(tmp_path):
    """Verifies load() fails if .meta.json is deleted and score_transaction() fails if a feature is missing."""
    import pytest

    df = pd.DataFrame(
        {
            "Amount": [50.0, 9500.0, 10.0, 25000.0],
            "pass_through_ratio_1h": [0.0, 0.95, 0.1, 1.0],
            "is_structuring": [0, 1, 0, 0],
            "velocity_acceleration": [1.0, 12.5, 0.8, 18.0],
            "fan_out_count_24h": [1, 8, 1, 14],
            "receiver_inflow_count_1h": [1, 5, 1, 15],
            "receiver_inflow_amount_1h": [50.0, 45000.0, 10.0, 150000.0],
            "Is_laundering": [0, 1, 0, 1],
        }
    )
    artifact_file = tmp_path / "aml_streaming_model"
    det = AMLAnomalyDetector(threshold=0.55)
    det.train(df)
    det.save(artifact_file)

    with pytest.raises(KeyError, match="Missing required AML features"):
        det.score_transaction({"Amount": 100.0})

    (tmp_path / "aml_streaming_model.meta.json").unlink()
    with pytest.raises(
        FileNotFoundError, match="Required model metadata sidecar missing"
    ):
        AMLAnomalyDetector().load(artifact_file)


def test_recording_failure_routes_to_dlq_without_incrementing_processed_count(
    tmp_path, monkeypatch
):
    """Verifies that a bad label during _record_scored_item routes to DLQ and is not counted in total_processed."""
    import csv
    from src.streaming.generator import _write_synthetic_saml_fallback
    from src.streaming.pipeline import run_production_stream_pipeline

    csv_path = tmp_path / "SAML-D.csv"
    _write_synthetic_saml_fallback(csv_path, num_rows=25)

    # Corrupt Is_laundering on the very last row so _record_scored_item raises ValueError
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    rows[-1]["Is_laundering"] = "CORRUPT_LABEL"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    df = pd.DataFrame(
        {
            "Amount": [50.0, 9250.0, 10.0, 9250.0, 120.0, 9250.0],
            "pass_through_ratio_1h": [0.0, 0.95, 0.1, 1.0, 0.05, 0.92],
            "is_structuring": [0, 1, 0, 1, 0, 1],
            "velocity_acceleration": [1.0, 12.5, 0.8, 18.0, 1.1, 9.5],
            "fan_out_count_24h": [1, 8, 1, 14, 2, 6],
            "receiver_inflow_count_1h": [1, 5, 1, 15, 1, 7],
            "receiver_inflow_amount_1h": [
                50.0,
                45000.0,
                10.0,
                150000.0,
                120.0,
                62000.0,
            ],
            "Is_laundering": [0, 1, 0, 1, 0, 1],
        }
    )
    model_path = tmp_path / "aml_streaming_model"
    det = AMLAnomalyDetector(threshold=0.50)
    det.train(df)
    det.save(model_path, mlflow_run_id="test_rec_fail")

    monkeypatch.setattr(
        "src.streaming.pipeline.resolve_runtime_paths",
        lambda: (
            csv_path,
            tmp_path / "dlq",
            tmp_path / "pipeline_execution_report.json",
            model_path,
        ),
    )

    summary = run_production_stream_pipeline(
        max_events=15, delay_seconds=0.0, warmup_events=10
    )
    assert summary["status"] == "COMPLETE"  # nosec B101
    assert summary["streaming_performance"]["total_processed"] == 14  # nosec B101
    assert summary["data_governance"]["dlq_events"] == 1  # nosec B101
    assert summary["data_governance"]["clean_events"] == 24  # nosec B101
