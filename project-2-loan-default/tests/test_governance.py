"""
Unit Test Suite - Project 2 Governance & Fairness Monitoring Layer
Validates automated compliance report generation, cohort size gating, and zero-baseline states.
"""

import json
import sys
from pathlib import Path

tests_dir = Path(__file__).resolve().parent
project_root = tests_dir.parent
sys.path.append(str(project_root))

from src.governance.audit import compile_compliance_audit
from src.monitoring.drift_monitor import run_production_audit


def test_audit_report_generation_execution(tmp_path, monkeypatch):
    """Asserts that the compliance auditing compilation completes and reports UNKNOWN when drift report is absent."""
    monkeypatch.setattr("src.governance.audit.PROJECT_ROOT", tmp_path)

    artifacts_dir = tmp_path / "artifacts"
    docs_dir = tmp_path / "docs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    compile_compliance_audit()
    model_card = (docs_dir / "model_card.md").read_text(encoding="utf-8")
    assert "UNKNOWN (NO_DRIFT_REPORT)" in model_card  # nosec B101


def test_drift_monitor_reports_inconclusive_on_zero_baseline(tmp_path):
    """Verifies both min_cohort_size=30 gating and zero-baseline approval gating."""
    log_file = tmp_path / "telemetry_logs.jsonl"
    report_file = tmp_path / "compliance_audit_report.json"

    report_file.write_text(
        json.dumps({"performance_metrics": {"test_roc_auc": 0.7227}}),
        encoding="utf-8",
    )

    entries = [
        {
            "schema_version": 2,
            "status_code": 200,
            "payload_snapshot": {"Age": 24},
            "inference_output": {"model_recommendation": "DECLINE"},
        },
        {
            "schema_version": 2,
            "status_code": 200,
            "payload_snapshot": {"Age": 42},
            "inference_output": {"model_recommendation": "DECLINE"},
        },
    ]
    log_file.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
    )

    # 1. Default min_cohort_size=30 triggers insufficient cohort size
    result_small = run_production_audit(log_path=log_file, report_path=report_file)
    assert (
        result_small["fairness_metrics"]["demographic_parity_status"]
        == "INCONCLUSIVE (INSUFFICIENT_DATA)"
    )  # nosec B101

    # 2. With min_cohort_size=1, directly exercises the zero-baseline approval rate branch
    result_zero = run_production_audit(
        log_path=log_file, report_path=report_file, min_cohort_size=1
    )
    assert (
        result_zero["fairness_metrics"]["demographic_parity_status"]
        == "INCONCLUSIVE (INSUFFICIENT_DATA)"
    )  # nosec B101
    assert result_zero["fairness_metrics"]["disparate_impact_ratio"] is None  # nosec B101
    assert result_zero["performance_metrics"]["test_roc_auc"] == 0.7227  # nosec B101
