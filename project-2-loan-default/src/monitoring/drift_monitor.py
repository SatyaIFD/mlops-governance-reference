"""
Real-Time Fairness & Recommendation Disparity Monitor - Project 2: Loan Default Prediction
Evaluates live telemetry logs across protected (<30) and baseline (>=30) age cohorts.
Never reports COMPLIANT when observations are missing, rejected, or have zero baseline rates.
"""

import json
from pathlib import Path
import pandas as pd


def run_production_audit(
    log_path: Path | None = None,
    report_path: Path | None = None,
    min_cohort_size: int = 1,
) -> dict:
    print("🕵️‍♂️ Initializing real-time governance & recommendation parity scanner...")

    src_dir = Path(__file__).resolve().parent
    base_dir = src_dir.parents[1]
    log_path = log_path or (base_dir / "artifacts" / "telemetry_logs.jsonl")
    report_path = report_path or (
        base_dir / "artifacts" / "compliance_audit_report.json"
    )

    existing_report = {}
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                existing_report = json.load(f)
        except Exception:
            existing_report = {}

    if not log_path.exists():
        print(f"⚠️ Telemetry log ledger not found at: {log_path}")
        status_block = {
            "demographic_parity_status": "UNKNOWN (NO_TELEMETRY)",
            "disparate_impact_ratio": None,
            "sample_size_valid": False,
        }
        existing_report["fairness_metrics"] = status_block
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(existing_report, f, indent=4)
        return existing_report

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line.strip())
                if rec.get("status_code") == 200:
                    records.append(rec)

    if not records:
        print("⚠️ No valid HTTP 200 inference records in telemetry log.")
        status_block = {
            "demographic_parity_status": "INCONCLUSIVE (INSUFFICIENT_DATA)",
            "disparate_impact_ratio": None,
            "sample_size_valid": False,
        }
        existing_report["fairness_metrics"] = status_block
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(existing_report, f, indent=4)
        return existing_report

    df = pd.DataFrame(records)
    payloads = pd.json_normalize(df["payload_snapshot"])
    outputs = pd.json_normalize(df["inference_output"])
    audit_df = pd.concat([payloads, outputs], axis=1)

    if "Age" not in audit_df.columns:
        status_block = {
            "demographic_parity_status": "INCONCLUSIVE (MISSING_COHORT_FIELD)",
            "disparate_impact_ratio": None,
            "sample_size_valid": False,
        }
        existing_report["fairness_metrics"] = status_block
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(existing_report, f, indent=4)
        return existing_report

    decision_col = (
        "model_recommendation"
        if "model_recommendation" in audit_df.columns
        else "underwriting_decision"
    )
    if decision_col not in audit_df.columns:
        status_block = {
            "demographic_parity_status": "INCONCLUSIVE (MISSING_DECISION_FIELD)",
            "disparate_impact_ratio": None,
            "sample_size_valid": False,
        }
        existing_report["fairness_metrics"] = status_block
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(existing_report, f, indent=4)
        return existing_report

    valid_decisions = {"APPROVE", "APPROVED", "DECLINE", "REJECTED"}
    audit_df = audit_df[audit_df[decision_col].isin(valid_decisions)].copy()

    young_df = audit_df[audit_df["Age"] < 30]
    mature_df = audit_df[audit_df["Age"] >= 30]

    if len(young_df) < min_cohort_size or len(mature_df) < min_cohort_size:
        fairness_status = "INCONCLUSIVE (INSUFFICIENT_DATA)"
        di_ratio = None
        approval_protected = (
            float(young_df[decision_col].isin({"APPROVE", "APPROVED"}).mean())
            if len(young_df) > 0
            else None
        )
        approval_baseline = (
            float(mature_df[decision_col].isin({"APPROVE", "APPROVED"}).mean())
            if len(mature_df) > 0
            else None
        )
    else:
        approval_protected = float(
            young_df[decision_col].isin({"APPROVE", "APPROVED"}).mean()
        )
        approval_baseline = float(
            mature_df[decision_col].isin({"APPROVE", "APPROVED"}).mean()
        )

        if approval_baseline <= 0.0:
            fairness_status = "INCONCLUSIVE (INSUFFICIENT_DATA)"
            di_ratio = None
        else:
            di_ratio = round(approval_protected / approval_baseline, 4)
            if 0.80 <= di_ratio <= 1.25:
                fairness_status = "COMPLIANT"
            else:
                fairness_status = "NON-COMPLIANT (BREACH)"

    existing_report["fairness_metrics"] = {
        "demographic_parity_status": fairness_status,
        "disparate_impact_ratio": di_ratio,
        "protected_cohort_count": int(len(young_df)),
        "baseline_cohort_count": int(len(mature_df)),
        "protected_approval_rate": round(approval_protected, 4)
        if approval_protected is not None
        else None,
        "baseline_approval_rate": round(approval_baseline, 4)
        if approval_baseline is not None
        else None,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(existing_report, f, indent=4)

    print(f"⚖️ Fairness Status: {fairness_status} | Disparate Impact Ratio: {di_ratio}")
    return existing_report


if __name__ == "__main__":
    run_production_audit()
