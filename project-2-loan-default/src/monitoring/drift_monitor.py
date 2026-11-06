"""
Real-Time Fairness & Recommendation Disparity Monitor - Project 2: Loan Default Prediction
Evaluates live telemetry logs across protected (<30) and baseline (>=30) age cohorts.
Preserves holdout evaluation metrics under fairness_metrics / holdout_fairness_metrics
while recording live telemetry fairness under live_telemetry_fairness with 95% CI.
"""

import json
import math
from pathlib import Path

import pandas as pd


def _compute_di_95_ci(
    k_prot: int, n_prot: int, k_base: int, n_base: int
) -> list[float] | None:
    """Computes the Katz logarithmic 95% confidence interval for the Disparate Impact ratio."""
    if n_prot <= 0 or n_base <= 0 or k_prot <= 0 or k_base <= 0:
        return None
    p1 = k_prot / n_prot
    p0 = k_base / n_base
    di = p1 / p0
    se_log = math.sqrt(max(0.0, (1.0 - p1) / k_prot + (1.0 - p0) / k_base))
    low = round(di * math.exp(-1.96 * se_log), 4)
    high = round(di * math.exp(1.96 * se_log), 4)
    return [low, high]


def run_production_audit(
    log_path: Path | None = None,
    report_path: Path | None = None,
    min_cohort_size: int = 30,
    data_source: str = "synthetic_traffic_simulation",
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

    # Restore or preserve holdout fairness_metrics so live telemetry never overwrites holdout metrics
    if "holdout_fairness_metrics" in existing_report:
        existing_report["fairness_metrics"] = existing_report[
            "holdout_fairness_metrics"
        ].copy()
    elif (
        "fairness_metrics" in existing_report
        and "protected_cohort_count" not in existing_report["fairness_metrics"]
    ):
        existing_report["holdout_fairness_metrics"] = existing_report[
            "fairness_metrics"
        ].copy()

    def _save_with_live_block(block: dict) -> dict:
        existing_report["live_telemetry_fairness"] = block
        if "fairness_metrics" not in existing_report:
            existing_report["fairness_metrics"] = block
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(existing_report, f, indent=4)
        return existing_report

    if not log_path.exists():
        print(f"⚠️ Telemetry log ledger not found at: {log_path}")
        return _save_with_live_block(
            {
                "data_source": data_source,
                "demographic_parity_status": "UNKNOWN (NO_TELEMETRY)",
                "disparate_impact_ratio": None,
                "di_95_ci": None,
                "sample_size_valid": False,
            }
        )

    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line.strip())
                if rec.get("status_code") == 200 and rec.get("schema_version") == 2:
                    records.append(rec)

    if not records:
        print("⚠️ No valid HTTP 200 v2 inference records in telemetry log.")
        return _save_with_live_block(
            {
                "data_source": data_source,
                "demographic_parity_status": "INCONCLUSIVE (INSUFFICIENT_DATA)",
                "disparate_impact_ratio": None,
                "di_95_ci": None,
                "sample_size_valid": False,
            }
        )

    df = pd.DataFrame(records)
    payloads = pd.json_normalize(df["payload_snapshot"])
    outputs = pd.json_normalize(df["inference_output"])
    audit_df = pd.concat([payloads, outputs], axis=1)

    if "Age" not in audit_df.columns:
        return _save_with_live_block(
            {
                "data_source": data_source,
                "demographic_parity_status": "INCONCLUSIVE (MISSING_COHORT_FIELD)",
                "disparate_impact_ratio": None,
                "di_95_ci": None,
                "sample_size_valid": False,
            }
        )

    decision_col = (
        "model_recommendation"
        if "model_recommendation" in audit_df.columns
        else "underwriting_decision"
    )
    if decision_col not in audit_df.columns:
        return _save_with_live_block(
            {
                "data_source": data_source,
                "demographic_parity_status": "INCONCLUSIVE (MISSING_DECISION_FIELD)",
                "disparate_impact_ratio": None,
                "di_95_ci": None,
                "sample_size_valid": False,
            }
        )

    valid_decisions = {"APPROVE", "APPROVED", "DECLINE", "REJECTED"}
    audit_df = audit_df[audit_df[decision_col].isin(valid_decisions)].copy()

    young_df = audit_df[audit_df["Age"] < 30]
    mature_df = audit_df[audit_df["Age"] >= 30]
    n_prot = len(young_df)
    n_base = len(mature_df)
    k_prot = (
        int(young_df[decision_col].isin({"APPROVE", "APPROVED"}).sum())
        if n_prot > 0
        else 0
    )
    k_base = (
        int(mature_df[decision_col].isin({"APPROVE", "APPROVED"}).sum())
        if n_base > 0
        else 0
    )

    approval_protected = float(k_prot / n_prot) if n_prot > 0 else None
    approval_baseline = float(k_base / n_base) if n_base > 0 else None
    ci_bounds = _compute_di_95_ci(k_prot, n_prot, k_base, n_base)

    if n_prot < min_cohort_size or n_base < min_cohort_size:
        fairness_status = "INCONCLUSIVE (INSUFFICIENT_DATA)"
        di_ratio = None
    elif approval_baseline is None or approval_baseline <= 0.0:
        fairness_status = "INCONCLUSIVE (INSUFFICIENT_DATA)"
        di_ratio = None
    else:
        di_ratio = round(approval_protected / approval_baseline, 4)
        if 0.80 <= di_ratio <= 1.25:
            fairness_status = "COMPLIANT"
        else:
            fairness_status = "NON-COMPLIANT (BREACH)"

    live_block = {
        "data_source": data_source,
        "demographic_parity_status": fairness_status,
        "disparate_impact_ratio": di_ratio,
        "di_95_ci": ci_bounds,
        "min_cohort_size_required": min_cohort_size,
        "protected_cohort_count": n_prot,
        "baseline_cohort_count": n_base,
        "protected_approval_rate": round(approval_protected, 4)
        if approval_protected is not None
        else None,
        "baseline_approval_rate": round(approval_baseline, 4)
        if approval_baseline is not None
        else None,
    }

    print(
        f"⚖️ Live Fairness ({data_source}): {fairness_status} | DI: {di_ratio} | 95% CI: {ci_bounds}"
    )
    return _save_with_live_block(live_block)


if __name__ == "__main__":
    run_production_audit()
