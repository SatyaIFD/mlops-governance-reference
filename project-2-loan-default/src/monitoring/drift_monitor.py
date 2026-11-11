"""
Real-Time Fairness & Recommendation Disparity Monitor - Project 2: Loan Default Prediction
Streams telemetry logs in O(1) memory across rotated/compressed archives and groups
counts by (traffic_source, model_version, decision_threshold).
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from src.monitoring.observability import (
    iter_telemetry_log_files,
    stream_telemetry_records,
)

DEFAULT_TRAFFIC_SOURCE = "live_production_traffic"
VALID_APPROVALS = {"APPROVE", "APPROVED"}
VALID_DECISIONS = {"APPROVE", "APPROVED", "DECLINE", "REJECTED"}


def _compute_di_95_ci(
    k_prot: int, n_prot: int, k_base: int, n_base: int
) -> list[float] | None:
    """Computes the Katz logarithmic 95% CI with Haldane-Anscombe (+0.5) boundary correction."""
    if n_prot <= 0 or n_base <= 0 or k_base <= 0:
        return None
    if k_prot == 0 or k_prot == n_prot or k_base == n_base:
        kp, np_adj, kb, nb_adj = k_prot + 0.5, n_prot + 1.0, k_base + 0.5, n_base + 1.0
    else:
        kp, np_adj, kb, nb_adj = (
            float(k_prot),
            float(n_prot),
            float(k_base),
            float(n_base),
        )
    p1 = kp / np_adj
    p0 = kb / nb_adj
    di = p1 / p0
    se_log = math.sqrt(max(0.0, (1.0 - p1) / kp + (1.0 - p0) / kb))
    low = 0.0 if k_prot == 0 else round(di * math.exp(-1.96 * se_log), 4)
    high = round(di * math.exp(1.96 * se_log), 4)
    return [low, high]


def _extract_policy_key(rec: dict[str, Any]) -> tuple[str, str]:
    out = rec.get("inference_output") or {}
    mv = str(out.get("model_version") or "unversioned")
    dt = out.get("decision_threshold")
    dt_str = f"{float(dt):.4f}" if isinstance(dt, (int, float)) else "default"
    return mv, dt_str


def _extract_cohort_and_decision(rec: dict[str, Any]) -> tuple[str | None, str | None]:
    snap = rec.get("payload_snapshot") or {}
    out = rec.get("inference_output") or {}
    cohort = snap.get("age_bucket")
    if cohort not in ("UNDER_30", "30_AND_OVER"):
        age_val = snap.get("Age")
        if isinstance(age_val, (int, float)):
            cohort = "UNDER_30" if age_val < 30 else "30_AND_OVER"
        elif out.get("age_group") in ("UNDER_30", "30_AND_OVER"):
            cohort = out.get("age_group")
        else:
            cohort = None
    decision = out.get("model_recommendation") or out.get("underwriting_decision")
    return cohort, decision


def _summarize_counts(
    counts: dict[str, int],
    source_name: str,
    active_mv: str,
    active_dt: str,
    min_cohort_size: int,
) -> dict[str, Any]:
    if counts.get("total_rows", 0) == 0:
        return {
            "data_source": source_name,
            "demographic_parity_status": "INCONCLUSIVE (INSUFFICIENT_DATA)",
            "disparate_impact_ratio": None,
            "di_95_ci": None,
            "sample_size_valid": False,
        }
    if counts.get("cohort_known", 0) == 0:
        return {
            "data_source": source_name,
            "demographic_parity_status": "INCONCLUSIVE (MISSING_COHORT_FIELD)",
            "disparate_impact_ratio": None,
            "di_95_ci": None,
            "sample_size_valid": False,
        }
    if counts.get("decision_known", 0) == 0:
        return {
            "data_source": source_name,
            "demographic_parity_status": "INCONCLUSIVE (MISSING_DECISION_FIELD)",
            "disparate_impact_ratio": None,
            "di_95_ci": None,
            "sample_size_valid": False,
        }
    n_prot = counts["n_prot"]
    n_base = counts["n_base"]
    k_prot = counts["k_prot"]
    k_base = counts["k_base"]
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
    min_synthetic_cohort_size: int = 200,
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

    def _save_blocks(live_b, synth_b, by_src, by_pol) -> dict:
        existing_report["live_telemetry_fairness"] = live_b
        existing_report["synthetic_telemetry_fairness"] = synth_b
        existing_report["live_telemetry_fairness_by_source"] = by_src
        existing_report["live_telemetry_fairness_by_policy"] = by_pol
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(existing_report, f, indent=4)
        return existing_report

    if not iter_telemetry_log_files(log_path):
        print(f"⚠️ Telemetry log ledger not found at: {log_path}")
        saved = _save_blocks(None, None, {}, {})
        if require_live_traffic:
            print(
                "🚨 NOT_EVALUATED: --require-live-traffic set but no telemetry log exists."
            )
            sys.exit(2)
        return saved

    # Streaming O(1) aggregation per (source, mv, dt)
    policy_counts: dict[tuple[str, str, str], dict[str, int]] = {}
    latest_policy_for_source: dict[str, tuple[str, str]] = {}
    source_total_rows: dict[str, int] = {}

    for rec in stream_telemetry_records(log_path):
        if rec.get("status_code") == 200 and rec.get("schema_version") == 2:
            src = str(rec.get("traffic_source") or DEFAULT_TRAFFIC_SOURCE).strip()
            mv, dt_str = _extract_policy_key(rec)
            latest_policy_for_source[src] = (mv, dt_str)
            source_total_rows[src] = source_total_rows.get(src, 0) + 1
            key = (src, mv, dt_str)
            if key not in policy_counts:
                policy_counts[key] = {
                    "total_rows": 0,
                    "cohort_known": 0,
                    "decision_known": 0,
                    "n_prot": 0,
                    "n_base": 0,
                    "k_prot": 0,
                    "k_base": 0,
                }
            st = policy_counts[key]
            st["total_rows"] += 1
            cohort, dec = _extract_cohort_and_decision(rec)
            if cohort is not None:
                st["cohort_known"] += 1
            if dec in VALID_DECISIONS:
                st["decision_known"] += 1
                if cohort == "UNDER_30":
                    st["n_prot"] += 1
                    if dec in VALID_APPROVALS:
                        st["k_prot"] += 1
                elif cohort == "30_AND_OVER":
                    st["n_base"] += 1
                    if dec in VALID_APPROVALS:
                        st["k_base"] += 1

    if not policy_counts:
        print("⚠️ No valid HTTP 200 v2 inference records in telemetry log.")
        saved = _save_blocks(None, None, {}, {})
        if require_live_traffic:
            print(
                "🚨 NOT_EVALUATED: --require-live-traffic set but zero v2 records exist."
            )
            sys.exit(2)
        return saved

    by_source: dict[str, dict] = {}
    for src, (mv, dt_str) in latest_policy_for_source.items():
        by_source[src] = _summarize_counts(
            policy_counts[(src, mv, dt_str)], src, mv, dt_str, min_cohort_size
        )

    by_policy: dict[str, dict] = {}
    for (src, mv, dt_str), counts in policy_counts.items():
        pkey = f"{src}|model={mv}|threshold={dt_str}"
        by_policy[pkey] = _summarize_counts(counts, pkey, mv, dt_str, min_cohort_size)

    if traffic_source:
        live_block = by_source.get(
            traffic_source,
            _summarize_counts(
                {}, traffic_source, "unversioned", "default", min_cohort_size
            ),
        )
    else:
        live_block = by_source.get(DEFAULT_TRAFFIC_SOURCE)

    synthetic_sources = [k for k in by_source if k != DEFAULT_TRAFFIC_SOURCE]
    synthetic_block = (
        by_source[max(synthetic_sources, key=lambda k: source_total_rows.get(k, 0))]
        if synthetic_sources
        else None
    )

    if live_block is None:
        print(
            "⏭️ Live Fairness Gate: NOT_EVALUATED (NO_LIVE_TRAFFIC) — live_telemetry_fairness is null."
        )
    else:
        print(
            f"⚖️ Live Telemetry Fairness ({live_block['data_source']}): "
            f"{live_block['demographic_parity_status']} | DI: {live_block['disparate_impact_ratio']} | 95% CI: {live_block['di_95_ci']}"
        )
    if synthetic_block:
        print(
            f"🧪 Synthetic Telemetry Fairness ({synthetic_block['data_source']}): "
            f"{synthetic_block['demographic_parity_status']} | DI: {synthetic_block['disparate_impact_ratio']} | 95% CI: {synthetic_block['di_95_ci']}"
        )

    saved = _save_blocks(live_block, synthetic_block, by_source, by_policy)
    if require_live_traffic and live_block is None:
        print(
            "🚨 NOT_EVALUATED: --require-live-traffic set and no live_production_traffic records found."
        )
        sys.exit(2)

    if fail_on_breach:
        gated_blocks = [live_block] if live_block is not None else []
        if include_synthetic_breach:
            for src, blk in by_source.items():
                if src != DEFAULT_TRAFFIC_SOURCE:
                    if (
                        blk.get("protected_cohort_count", 0)
                        >= min_synthetic_cohort_size
                        and blk.get("baseline_cohort_count", 0)
                        >= min_synthetic_cohort_size
                    ):
                        gated_blocks.append(blk)
        if any(
            b.get("demographic_parity_status") == "NON-COMPLIANT (BREACH)"
            for b in gated_blocks
        ):
            print("🚨 Demographic parity breach confirmed outside 95% CI bounds.")
            sys.exit(1)
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fail-on-breach", action="store_true")
    parser.add_argument("--include-synthetic", action="store_true")
    parser.add_argument("--require-live-traffic", action="store_true")
    parser.add_argument("--min-synthetic-cohort-size", type=int, default=30)
    parser.add_argument("--traffic-source", type=str, default=None)
    args = parser.parse_args()
    run_production_audit(
        fail_on_breach=args.fail_on_breach,
        traffic_source=args.traffic_source,
        include_synthetic_breach=args.include_synthetic,
        require_live_traffic=args.require_live_traffic,
        min_synthetic_cohort_size=args.min_synthetic_cohort_size,
    )
