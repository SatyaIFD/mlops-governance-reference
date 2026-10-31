"""
Unit Test Suite - Project 2 Governance & Fairness Monitoring Layer
Validates automated compliance report generation and inconclusive/breach states.
"""

import json
import sys
from pathlib import Path

tests_dir = Path(__file__).resolve().parent
project_root = tests_dir.parent
sys.path.append(str(project_root))

from src.governance.audit import compile_compliance_audit  # noqa: E402
from src.monitoring.drift_monitor import run_production_audit  # noqa: E402


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
    """Verifies that zero baseline approvals report INCONCLUSIVE rather than defaulting to 1.0 COMPLIANT."""
    log_file = tmp_path / "telemetry_logs.jsonl"
    report_file = tmp_path / "compliance_audit_report.json"

    # Seed existing performance metrics to verify drift_monitor does not overwrite them with hardcoded 91.4%
    report_file.write_text(
        json.dumps({"performance_metrics": {"test_roc_auc": 0.7227}}),
        encoding="utf-8",
    )

    # Write records where all applicants in both cohorts are declined
    entries = [
        {
            "status_code": 200,
            "payload_snapshot": {"Age": 24},
            "inference_output": {"model_recommendation": "DECLINE"},
        },
        {
            "status_code": 200,
            "payload_snapshot": {"Age": 42},
            "inference_output": {"model_recommendation": "DECLINE"},
        },
    ]
    log_file.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
    )

    result = run_production_audit(log_path=log_file, report_path=report_file)
    assert (
        result["fairness_metrics"]["demographic_parity_status"]
        == "INCONCLUSIVE (INSUFFICIENT_DATA)"
    )  # nosec B101
    assert result["fairness_metrics"]["disparate_impact_ratio"] is None  # nosec B101
    assert result["performance_metrics"]["test_roc_auc"] == 0.7227  # nosec B101
