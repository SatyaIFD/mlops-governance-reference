"""
Unit Test Suite - Project 2 Observability Layer
Validates structured JSON telemetry logging and append operations.
"""

import pytest
import json
from pathlib import Path
import sys

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
        "governance": {"underwriting_decision": "APPROVED", "fairness_demographic_proxy_flag": "Control Group"}
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