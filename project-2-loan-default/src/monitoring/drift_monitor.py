"""
Real-Time Fairness & Recommendation Disparity Monitor - Project 2: Loan Default Prediction
Evaluates telemetry logs across protected (<30 / UNDER_30) and baseline (>=30 / 30_AND_OVER) cohorts.
Groups records by (traffic_source, model_version, decision_threshold) so retraining or threshold
updates never pool mixed policies together.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

DEFAULT_TRAFFIC_SOURCE = "live_production_traffic"


def _compute_di_95_ci(
    k_prot: int, n_prot: int, k_base: int, n_base: int
) -> list[float] | None:
    """
    Computes the Katz logarithmic 95% confidence interval for the Disparate Impact ratio.
    Applies Haldane-Anscombe (+0.5) continuity correction whenever any cell in the 2x2
    approval/decline table is zero (k == 0 or k == n).
    """
    if n_prot <= 0 or n_base <= 0 or k_base <= 0:
        return None
    if k_prot == 0 or k_prot == n_prot or k_base == n_base:
        kp = float(k_prot) + 0.5
        np_adj = float(n_prot) + 1.0
        kb = float(k_base) + 0.5
        nb_adj = float(n_base) + 1.0
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


def _extract_policy_key(rec: dict) -> tuple[str, str]:
    out = rec.get("inference_output", {}) or {}
    mv = str(out.get("model_version") or "unversioned")
    dt = out.get("decision_threshold")
    dt_str = f"{float(dt):.4f}" if isinstance(dt, (int, float)) else "default"
    return mv, dt_str


def _evaluate_source_records(
    records: list[dict], source_name: str, min_cohort_size: int
) -> dict:
    if not records:
        return {
            "data_source": source_name,
            "demographic_parity_status": "INCONCLUSIVE (INSUFFICIENT_DATA)",
            "disparate_impact_ratio": None,
            "di_95_ci": None,
            "sample_size_valid": False,
        }

    # Filter to the most recent active (model_version, decision_threshold) policy within this source
    active_mv, active_dt = _extract_policy_key(records[-1])
    policy_records = [
        r for r in records if _extract_policy_key(r) == (active_mv, active_dt)
    ]

    df = pd.DataFrame(policy_records)
    payloads = pd.json_normalize(df["payload_snapshot"])
    outputs = pd.json_normalize(df["inference_output"])
    audit_df = pd.concat([payloads, outputs], axis=1)

    if "age_bucket" in audit_df.columns:
        is_young_mask = audit_df["age_bucket"] == "UNDER_30"
        is_mature_mask = audit_df["age_bucket"] == "30_AND_OVER"
    elif "Age" in audit_df.columns:
        is_young_mask = audit_df["Age"] < 30
        is_mature_mask = audit_df["Age"] >= 30
    elif "age_group" in audit_df.columns:
        is_young_mask = audit_df["age_group"] == "UNDER_30"
        is_mature_mask = audit_df["age_group"] == "30_AND_OVER"
    else:
        return {
            "data_source": source_name,
            "demographic_parity_status": "INCONCLUSIVE (MISSING_COHORT_FIELD)",
            "disparate_impact_ratio": None,
            "di_95_ci": None,
            "sample_size_valid": False,
        }

    decision_col = (
        "model_recommendation"
        if "model_recommendation" in audit_df.columns
        else "underwriting_decision"
    )
    if decision_col not in audit_df.columns:
        return {
            "data_source": source_name,
            "demographic_parity_status": "INCONCLUSIVE (MISSING_DECISION_FIELD)",
            "disparate_impact_ratio": None,
            "di_95_ci": None,
            "sample_size_valid": False,
        }

    valid_decisions = {"APPROVE", "APPROVED", "DECLINE", "REJECTED"}
    valid_mask = audit_df[decision_col].isin(valid_decisions)
    young_df = audit_df[valid_mask & is_young_mask]
    mature_df = audit_df[valid_mask & is_mature_mask]

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

    return {
        "data_source": source_name,
        "model_version": active_mv,
        "decision_threshold": float(active_dt) if active_dt != "default" else None,
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


def run_production_audit(
    log_path: Path | None = None,
    report_path: Path | None = None,
    min_cohort_size: int = 30,
    fail_on_breach: bool = False,
    traffic_source: str | None = None,
    include_synthetic_breach: bool = False,
    require_live_traffic: bool = False,
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

    def _save_blocks(
        live_block: dict | None,
        synthetic_block: dict | None,
        by_source: dict[str, dict],
        by_policy: dict[str, dict],
    ) -> dict:
        existing_report["live_telemetry_fairness"] = live_block
        existing_report["synthetic_telemetry_fairness"] = synthetic_block
        existing_report["live_telemetry_fairness_by_source"] = by_source
        existing_report["live_telemetry_fairness_by_policy"] = by_policy
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(existing_report, f, indent=4)
        return existing_report

    if not log_path.exists():
        print(f"⚠️ Telemetry log ledger not found at: {log_path}")
        saved = _save_blocks(None, None, {}, {})
        if require_live_traffic:
            print(
                "🚨 NOT_EVALUATED: --require-live-traffic set but no telemetry log exists."
            )
            sys.exit(2)
        return saved

    grouped: dict[str, list[dict]] = {}
    grouped_by_policy: dict[str, list[dict]] = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line.strip())
                if rec.get("status_code") == 200 and rec.get("schema_version") == 2:
                    src = str(
                        rec.get("traffic_source") or DEFAULT_TRAFFIC_SOURCE
                    ).strip()
                    grouped.setdefault(src, []).append(rec)
                    mv, dt_str = _extract_policy_key(rec)
                    pol_key = f"{src}|model={mv}|threshold={dt_str}"
                    grouped_by_policy.setdefault(pol_key, []).append(rec)

    if not grouped:
        print("⚠️ No valid HTTP 200 v2 inference records in telemetry log.")
        saved = _save_blocks(None, None, {}, {})
        if require_live_traffic:
            print(
                "🚨 NOT_EVALUATED: --require-live-traffic set but zero v2 records exist."
            )
            sys.exit(2)
        return saved

    by_source = {
        src: _evaluate_source_records(recs, src, min_cohort_size)
        for src, recs in grouped.items()
    }
    by_policy = {
        pkey: _evaluate_source_records(recs, pkey, min_cohort_size)
        for pkey, recs in grouped_by_policy.items()
    }

    if traffic_source:
        live_block = by_source.get(
            traffic_source,
            _evaluate_source_records([], traffic_source, min_cohort_size),
        )
    else:
        live_block = by_source.get(DEFAULT_TRAFFIC_SOURCE)

    synthetic_sources = [k for k in by_source if k != DEFAULT_TRAFFIC_SOURCE]
    synthetic_block = (
        by_source[max(synthetic_sources, key=lambda k: len(grouped[k]))]
        if synthetic_sources
        else None
    )

    if live_block is None:
        print(
            "⏭️ Live Fairness Gate: NOT_EVALUATED (NO_LIVE_TRAFFIC) — live_telemetry_fairness is null."
        )
    if live_block:
        print(
            f"⚖️ Live Telemetry Fairness ({live_block['data_source']}): "
            f"{live_block['demographic_parity_status']} | "
            f"DI: {live_block['disparate_impact_ratio']} | 95% CI: {live_block['di_95_ci']}"
        )
    if synthetic_block:
        print(
            f"🧪 Synthetic Telemetry Fairness ({synthetic_block['data_source']}): "
            f"{synthetic_block['demographic_parity_status']} | "
            f"DI: {synthetic_block['disparate_impact_ratio']} | 95% CI: {synthetic_block['di_95_ci']}"
        )

    saved = _save_blocks(live_block, synthetic_block, by_source, by_policy)
    if require_live_traffic and live_block is None:
        print(
            "🚨 NOT_EVALUATED: --require-live-traffic set and no live_production_traffic records found."
        )
        sys.exit(2)

    if fail_on_breach:
        gated_blocks = (
            list(by_source.values())
            if include_synthetic_breach
            else ([live_block] if live_block is not None else [])
        )
        if any(
            b.get("demographic_parity_status") == "NON-COMPLIANT (BREACH)"
            for b in gated_blocks
        ):
            print("🚨 Demographic parity breach confirmed outside 95% CI bounds.")
            sys.exit(1)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fail-on-breach",
        action="store_true",
        help="Exit non-zero if 95% CI confirms a demographic parity breach.",
    )
    parser.add_argument(
        "--include-synthetic",
        action="store_true",
        help="Also gate synthetic simulation traffic when --fail-on-breach is set.",
    )
    parser.add_argument(
        "--require-live-traffic",
        action="store_true",
        help="Exit with code 2 (NOT_EVALUATED) when no live_production_traffic rows exist.",
    )
    parser.add_argument(
        "--traffic-source",
        type=str,
        default=None,
        help="Filter telemetry records to a specific traffic_source.",
    )
    args = parser.parse_args()
    run_production_audit(
        fail_on_breach=args.fail_on_breach,
        traffic_source=args.traffic_source,
        include_synthetic_breach=args.include_synthetic,
        require_live_traffic=args.require_live_traffic,
    )
