"""
Unit Test Suite - Project 2 Governance & Fairness Monitoring Layer
Validates automated compliance report generation, cohort size gating, zero-baseline states,
Katz 95% confidence interval straddling, and strict holdout vs. live fairness separation.
"""

import json
import sys
from pathlib import Path

import pytest

tests_dir = Path(__file__).resolve().parent
project_root = tests_dir.parent
sys.path.append(str(project_root))

from src.governance.audit import compile_compliance_audit
from src.monitoring.drift_monitor import _compute_di_95_ci, run_production_audit


def test_audit_report_generation_execution(tmp_path, monkeypatch):
    """Asserts that the compliance auditing compilation reports UNKNOWN when holdout and drift reports are absent."""
    monkeypatch.setattr("src.governance.audit.PROJECT_ROOT", tmp_path)

    artifacts_dir = tmp_path / "artifacts"
    docs_dir = tmp_path / "docs"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    compile_compliance_audit()
    model_card = (docs_dir / "model_card.md").read_text(encoding="utf-8")
    assert "UNKNOWN (NO_HOLDOUT_EVALUATION)" in model_card  # nosec B101
    assert "UNKNOWN (NO_DRIFT_REPORT)" in model_card  # nosec B101


def test_drift_monitor_reports_inconclusive_on_zero_baseline_and_preserves_holdout(
    tmp_path,
):
    """Verifies min_cohort_size=30 gating, zero-baseline gating, and that live runs never populate fairness_metrics when absent."""
    log_file = tmp_path / "telemetry_logs.jsonl"
    report_file = tmp_path / "compliance_audit_report.json"

    report_file.write_text(
        json.dumps({"performance_metrics": {"test_roc_auc": 0.7227}}),
        encoding="utf-8",
    )

    entries = [
        {
            "schema_version": 2,
            "traffic_source": "synthetic_representative_simulation",
            "status_code": 200,
            "payload_snapshot": {"Age": 24},
            "inference_output": {"model_recommendation": "DECLINE"},
        },
        {
            "schema_version": 2,
            "traffic_source": "synthetic_representative_simulation",
            "status_code": 200,
            "payload_snapshot": {"Age": 42},
            "inference_output": {"model_recommendation": "DECLINE"},
        },
    ]
    log_file.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
    )

    # 1. Default min_cohort_size=30 triggers insufficient cohort size and does NOT create fairness_metrics
    result_small = run_production_audit(log_path=log_file, report_path=report_file)
    assert "fairness_metrics" not in result_small  # nosec B101
    assert (
        result_small["live_telemetry_fairness"]["demographic_parity_status"]
        == "INCONCLUSIVE (INSUFFICIENT_DATA)"
    )  # nosec B101
    assert (
        result_small["live_telemetry_fairness"]["data_source"]
        == "synthetic_representative_simulation"
    )  # nosec B101

    # 2. With min_cohort_size=1, directly exercises the zero-baseline approval rate branch
    result_zero = run_production_audit(
        log_path=log_file, report_path=report_file, min_cohort_size=1
    )
    assert (
        result_zero["live_telemetry_fairness"]["demographic_parity_status"]
        == "INCONCLUSIVE (INSUFFICIENT_DATA)"
    )  # nosec B101
    assert result_zero["live_telemetry_fairness"]["disparate_impact_ratio"] is None  # nosec B101
    assert result_zero["performance_metrics"]["test_roc_auc"] == 0.7227  # nosec B101


def test_katz_ci_straddling_and_holdout_preservation(tmp_path):
    """Verifies _compute_di_95_ci math, CI_STRADDLES_THRESHOLD status on 31/40 vs 35/40, and holdout preservation."""
    ci = _compute_di_95_ci(31, 40, 35, 40)
    assert ci == [pytest.approx(0.7223, abs=1e-3), pytest.approx(1.0861, abs=1e-3)]  # nosec B101

    log_file = tmp_path / "telemetry_logs.jsonl"
    report_file = tmp_path / "compliance_audit_report.json"
    report_file.write_text(
        json.dumps(
            {
                "fairness_metrics": {
                    "demographic_parity_status": "COMPLIANT (FAIR)",
                    "disparate_impact_ratio": 1.0001,
                }
            }
        ),
        encoding="utf-8",
    )

    rows = []
    for i in range(40):
        rows.append(
            {
                "schema_version": 2,
                "traffic_source": "live_production_traffic",
                "status_code": 200,
                "payload_snapshot": {"Age": 24},
                "inference_output": {
                    "model_recommendation": "APPROVE" if i < 31 else "DECLINE"
                },
            }
        )
        rows.append(
            {
                "schema_version": 2,
                "traffic_source": "live_production_traffic",
                "status_code": 200,
                "payload_snapshot": {"Age": 45},
                "inference_output": {
                    "model_recommendation": "APPROVE" if i < 35 else "DECLINE"
                },
            }
        )
    log_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    res = run_production_audit(log_path=log_file, report_path=report_file)
    assert res["fairness_metrics"]["disparate_impact_ratio"] == 1.0001  # nosec B101
    assert (
        res["live_telemetry_fairness"]["demographic_parity_status"]
        == "INCONCLUSIVE (CI_STRADDLES_THRESHOLD)"
    )  # nosec B101
    assert res["live_telemetry_fairness"]["data_source"] == "live_production_traffic"  # nosec B101


def test_zero_protected_approvals_flags_breach_and_mixed_sources_are_filtered(
    tmp_path,
):
    """Verifies that 0/40 protected approvals vs 35/40 baseline approvals triggers NON-COMPLIANT (BREACH) and mixed sources are not pooled."""
    log_file = tmp_path / "telemetry_logs.jsonl"
    report_file = tmp_path / "compliance_audit_report.json"

    rows = [
        {
            "schema_version": 2,
            "traffic_source": "synthetic_stress_simulation",
            "status_code": 200,
            "payload_snapshot": {"Age": 22},
            "inference_output": {"model_recommendation": "APPROVE"},
        }
    ]
    for i in range(40):
        rows.append(
            {
                "schema_version": 2,
                "traffic_source": "live_production_traffic",
                "status_code": 200,
                "payload_snapshot": {"Age": 24},
                "inference_output": {"model_recommendation": "DECLINE"},
            }
        )
        rows.append(
            {
                "schema_version": 2,
                "traffic_source": "live_production_traffic",
                "status_code": 200,
                "payload_snapshot": {"Age": 45},
                "inference_output": {
                    "model_recommendation": "APPROVE" if i < 35 else "DECLINE"
                },
            }
        )
    log_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    res = run_production_audit(log_path=log_file, report_path=report_file)
    live = res["live_telemetry_fairness"]
    assert live["data_source"] == "live_production_traffic"  # nosec B101
    assert live["protected_cohort_count"] == 40  # nosec B101
    assert live["disparate_impact_ratio"] == 0.0  # nosec B101
    assert live["demographic_parity_status"] == "NON-COMPLIANT (BREACH)"  # nosec B101
    assert live["di_95_ci"][1] < 0.80  # nosec B101
