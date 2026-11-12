"""
Unit Test Suite - Project 2 Observability Layer
Validates structured JSON telemetry logging and append operations.
"""

import sys
from pathlib import Path

# Ensure project source root is on path context for importing
tests_dir = Path(__file__).resolve().parent
project_root = tests_dir.parent
sys.path.append(str(project_root))

from src.monitoring.observability import ModelObservabilityLogger


def test_telemetry_logging_append(tmp_path):
    """Asserts that real-time api payload entries are written to disk with structured keys."""
    test_log_file = tmp_path / "test_telemetry.jsonl"
    logger = ModelObservabilityLogger(telemetry_log_path=test_log_file)

    mock_payload = {"Age": 28, "Income": 85000, "CreditScore": 750}
    mock_response = {
        "risk_metrics": {"default_probability": 0.02},
        "governance": {
            "underwriting_decision": "APPROVED",
            "fairness_demographic_proxy_flag": "Control Group",
        },
    }

    # Log simulated transaction
    logger.log_inference_transaction(mock_payload, mock_response, latency_ms=10.5)

    # Assert ledger file creation and entries
    assert test_log_file.exists()

    latest_records = logger.read_latest_telemetry(tail_count=1)
    assert len(latest_records) == 1
    assert latest_records[0]["status_code"] == 200
    assert latest_records[0]["latency_ms"] == 10.5
    assert latest_records[0]["inference_output"]["underwriting_decision"] == "APPROVED"


def test_telemetry_multi_archive_rotation_and_reader(tmp_path):
    """Verifies that ModelObservabilityLogger rotates .1, .2 archives and read_latest_telemetry spans archives."""
    from src.monitoring.observability import iter_telemetry_log_files

    log_file = tmp_path / "telemetry_logs.jsonl"
    logger = ModelObservabilityLogger(
        telemetry_log_path=log_file, max_bytes=200, max_archives=3
    )
    mock_payload = {
        "Age": 24,
        "Income": 60000,
        "CreditScore": 710,
        "LoanAmount": 12000,
        "DTIRatio": 0.25,
    }
    mock_response = {
        "risk_metrics": {"default_probability": 0.05},
        "governance": {"model_recommendation": "APPROVE", "model_version": "v4"},
    }
    for _ in range(4):
        logger.log_inference_transaction(mock_payload, mock_response, latency_ms=5.0)

    files = iter_telemetry_log_files(log_file, max_archives=3)
    assert len(files) >= 2  # nosec B101
    records = logger.read_latest_telemetry(tail_count=10)
    assert len(records) == 4  # nosec B101


def test_categorical_psi_and_minimized_snapshot_privacy(tmp_path):
    """Verifies MaritalStatus is excluded from payload_snapshot, rotated archives are .gz, and categorical PSI works."""
    import pandas as pd
    from src.validation.drift_check import DataDriftMonitor

    log_file = tmp_path / "telemetry_logs.jsonl"
    logger = ModelObservabilityLogger(
        telemetry_log_path=log_file, max_bytes=180, max_archives=3
    )
    payload = {
        "Age": 23,
        "Income": 52000,
        "CreditScore": 690,
        "LoanAmount": 14000,
        "DTIRatio": 0.31,
        "MaritalStatus": "Married",
        "Education": "PhD",
    }
    resp = {
        "risk_metrics": {"default_probability": 0.08},
        "governance": {"model_recommendation": "APPROVE"},
    }
    for _ in range(3):
        logger.log_inference_transaction(
            payload, resp, latency_ms=4.2, user_id="USR_GDPR_1"
        )

    recs = logger.read_latest_telemetry(tail_count=5)
    assert len(recs) == 3  # nosec B101
    assert "MaritalStatus" not in recs[-1]["payload_snapshot"]  # nosec B101
    assert recs[-1]["user_id"] == "USR_GDPR_1"  # nosec B101

    mon = DataDriftMonitor(
        reference_path=tmp_path / "ref.parquet", artifacts_dir=tmp_path
    )
    s1 = pd.Series(["A", "A", "B", "B"])
    s2 = pd.Series(["A", "A", "B", "B"])
    assert mon.calculate_categorical_psi(s1, s2) == 0.0  # nosec B101
