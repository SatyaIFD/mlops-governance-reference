"""
Real-Time Fairness & Recommendation Disparity Monitor - Project 2: Loan Default Prediction
Evaluates live telemetry logs across protected (<30) and baseline (>=30) age cohorts.
Filters to a single consistent traffic_source, applies Haldane-Anscombe continuity
correction for zero-approval cohorts, and classifies fairness using the 95% Katz CI.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


def _compute_di_95_ci(
    k_prot: int, n_prot: int, k_base: int, n_base: int
) -> list[float] | None:
    """
    Computes the Katz logarithmic 95% confidence interval for the Disparate Impact ratio.
    Applies a Haldane-Anscombe (+0.5) continuity correction when k_prot == 0 (with k_base > 0)
    so zero-approval protected cohorts produce a valid upper bound and trigger NON-COMPLIANT.
    """
    if n_prot <= 0 or n_base <= 0 or k_base <= 0:
        return None
    if k_prot == 0:
        kp = 0.5
        np_adj = n_prot + 1.0
        kb = float(k_base) + 0.5
        nb_adj = n_base + 1.0
    else:
        kp = float(k_prot)
        np_adj = float(n_prot)
        kb = float(k_base)
        nb_adj = float(n_base)

    p1 = kp / np_adj
    p0 = kb / nb_adj
    di = p1 / p0
    se_log = math.sqrt(max(0.0, (1.0 - p1) / kp + (1.0 - p0) / kb))
    low = 0.0 if k_prot == 0 else round(di * math.exp(-1.96 * se_log), 4)
    high = round(di * math.exp(1.96 * se_log), 4)
    return [low, high]


def _select_single_source_records(
    records: list[dict], target_source: str | None = None
) -> tuple[list[dict], str]:
    """Filters records to a single traffic_source (never pooling mixed sources)."""
    if not records:
        return [], "none"
    chosen = (
        target_source
        or str(records[-1].get("traffic_source") or "unlabeled_telemetry_logs").strip()
    )
    filtered = [
        r
        for r in records
        if str(r.get("traffic_source") or "unlabeled_telemetry_logs").strip() == chosen
    ]
    return filtered, chosen


def run_production_audit(
    log_path: Path | None = None,
    report_path: Path | None = None,
    min_cohort_size: int = 30,
    fail_on_breach: bool = False,
    traffic_source: str | None = None,
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

    if "holdout_fairness_metrics" in existing_report:
        existing_report["fairness_metrics"] = existing_report[
            "holdout_fairness_metrics"
        ].copy()
    elif (
        "fairness_metrics" in existing_report
        and "protected_cohort_count" not in existing_report["fairness_metrics"]
        and "data_source" not in existing_report["fairness_metrics"]
    ):
        existing_report["holdout_fairness_metrics"] = existing_report[
            "fairness_metrics"
        ].copy()
    else:
        existing_report.pop("fairness_metrics", None)

    def _save_with_live_block(block: dict) -> dict:
        existing_report["live_telemetry_fairness"] = block
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(existing_report, f, indent=4)
        return existing_report

    if not log_path.exists():
        print(f"⚠️ Telemetry log ledger not found at: {log_path}")
        return _save_with_live_block(
            {
                "data_source": "none",
                "demographic_parity_status": "UNKNOWN (NO_TELEMETRY)",
                "disparate_impact_ratio": None,
                "di_95_ci": None,
                "sample_size_valid": False,
            }
        )

    all_records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line.strip())
                if rec.get("status_code") == 200 and rec.get("schema_version") == 2:
                    all_records.append(rec)

    records, detected_source = _select_single_source_records(
        all_records, target_source=traffic_source
    )

    if not records:
        print("⚠️ No valid HTTP 200 v2 inference records in telemetry log.")
        return _save_with_live_block(
            {
                "data_source": detected_source,
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
                "data_source": detected_source,
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
                "data_source": detected_source,
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
        if ci_bounds is None:
            fairness_status = "INCONCLUSIVE (INSUFFICIENT_DATA)"
        elif ci_bounds[0] >= 0.80 and ci_bounds[1] <= 1.25:
            fairness_status = "COMPLIANT"
        elif ci_bounds[1] < 0.80 or ci_bounds[0] > 1.25:
            fairness_status = "NON-COMPLIANT (BREACH)"
        else:
            fairness_status = "INCONCLUSIVE (CI_STRADDLES_THRESHOLD)"

    live_block = {
        "data_source": detected_source,
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
        f"⚖️ Live Fairness ({detected_source}): {fairness_status} | DI: {di_ratio} | 95% CI: {ci_bounds}"
    )
    saved = _save_with_live_block(live_block)
    if fail_on_breach and fairness_status == "NON-COMPLIANT (BREACH)":
        print("🚨 Live demographic parity breach confirmed outside 95% CI bounds.")
        sys.exit(1)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fail-on-breach",
        action="store_true",
        help="Exit non-zero if the 95% CI confirms a demographic parity breach.",
    )
    parser.add_argument(
        "--traffic-source",
        type=str,
        default=None,
        help="Filter telemetry records to a specific traffic_source.",
    )
    args = parser.parse_args()
    run_production_audit(
        fail_on_breach=args.fail_on_breach, traffic_source=args.traffic_source
    )
