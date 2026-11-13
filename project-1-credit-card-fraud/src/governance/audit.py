"""
Governance & Segment Disparity Audit Module - Project 1: Credit Card Fraud Detection
Evaluates out-of-sample transaction-value segment flagging disparity on test.parquet
using the run-ID verified validation-calibrated segment_thresholds.json served by the API.
Reports per-segment recall, precision, flag counts, and Katz 95% CIs.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.utils import (
    CANONICAL_FEATURE_COLS,
    clean_and_transform_features,
    load_scaler_params,
)


def _katz_95_ci(k1: int, n1: int, k0: int, n0: int) -> list[float]:
    if k1 == 0 or k1 == n1 or k0 == 0 or k0 == n0:
        a, m1, b, m0 = k1 + 0.5, n1 + 1.0, k0 + 0.5, n0 + 1.0
    else:
        a, m1, b, m0 = float(k1), float(n1), float(k0), float(n0)
    p1 = a / m1
    p0 = b / m0
    ratio = p1 / p0
    se = math.sqrt(max(0.0, (1.0 - p1) / a + (1.0 - p0) / b))
    return [
        round(ratio * math.exp(-1.96 * se), 4),
        round(ratio * math.exp(1.96 * se), 4),
    ]


def _segment_metrics(y_true: np.ndarray, preds: np.ndarray) -> dict:
    total = int(len(y_true))
    flagged = int(preds.sum())
    actual_pos = int((y_true == 1).sum())
    tp = int(((y_true == 1) & (preds == 1)).sum())
    flag_rate = round(float(flagged / total), 6) if total > 0 else 0.0
    recall = round(float(tp / actual_pos), 4) if actual_pos > 0 else 0.0
    precision = round(float(tp / flagged), 4) if flagged > 0 else 0.0
    return {
        "count": total,
        "flagged_count": flagged,
        "actual_frauds": actual_pos,
        "true_positives": tp,
        "flag_rate": flag_rate,
        "recall": recall,
        "precision": precision,
    }


def _load_verified_segment_policy(expected_run_id: str) -> dict:
    seg_file = BASE_DIR / "artifacts" / "segment_thresholds.json"
    try:
        dl_path = Path(
            mlflow.artifacts.download_artifacts(
                run_id=expected_run_id,
                artifact_path="preprocessing/segment_thresholds.json",
            )
        )
        if dl_path.is_file():
            seg_file = dl_path
    except Exception as dl_err:
        print(f"ℹ️ Using local segment threshold artifact ({dl_err})")
    if not seg_file.is_file():
        raise FileNotFoundError(
            f"Required segment_thresholds.json missing at {seg_file}"
        )
    data = json.loads(seg_file.read_text(encoding="utf-8"))
    file_run_id = str(data.get("mlflow_run_id") or "")
    if file_run_id != str(expected_run_id):
        raise ValueError(
            f"Stale segment_thresholds.json: mlflow_run_id={file_run_id} does not match evaluated run {expected_run_id}"
        )
    return data


def run_fairness_audit(enforce_gate: bool = False) -> dict:
    default_db = BASE_DIR.parent / "mlflow.db"
    if not default_db.exists() and (BASE_DIR / "mlflow.db").exists():
        default_db = BASE_DIR / "mlflow.db"
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{default_db}")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        client = MlflowClient()
        model_name = "fraud_detection_xgb"
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise RuntimeError(f"No registered versions found for model '{model_name}'")
        prod_versions = [v for v in versions if v.current_stage == "Production"]
        selected_mv = (
            max(prod_versions, key=lambda v: int(v.version))
            if prod_versions
            else max(versions, key=lambda v: int(v.version))
        )
        latest_version = int(selected_mv.version)
        expected_run_id = selected_mv.run_id
        model_uri = os.getenv("MODEL_URI", f"models:/{model_name}/{latest_version}")
        model = mlflow.xgboost.load_model(model_uri)
    except Exception as e:
        print(f"❌ Error loading model from {tracking_uri}: {e}")
        sys.exit(1)

    test_data_path = BASE_DIR / "data" / "processed" / "test.parquet"
    fallback_path = BASE_DIR / "data" / "processed_features.parquet"
    target_path = test_data_path if test_data_path.exists() else fallback_path
    if not target_path.exists():
        print(f"❌ Processed test data not found at {target_path}")
        sys.exit(1)

    df = pd.read_parquet(target_path)
    if "scaled_amount" not in df.columns or "scaled_time" not in df.columns:
        scaler_file = BASE_DIR / "artifacts" / "scaler_params.json"
        scaler_params = (
            load_scaler_params(scaler_file) if scaler_file.exists() else None
        )
        processed_df = clean_and_transform_features(
            df, scaler_params=scaler_params, drop_duplicates=False
        )
    else:
        processed_df = df

    seg_policy = _load_verified_segment_policy(expected_run_id)
    X_test = processed_df[CANONICAL_FEATURE_COLS]
    y_test = (
        processed_df["Class"].to_numpy(dtype=int)
        if "Class" in processed_df.columns
        else np.zeros(len(processed_df), dtype=int)
    )
    print(
        f"⏳ Running out-of-sample batch inference across evaluation slices using {model_uri} (run_id={expected_run_id})..."
    )
    probs = model.predict_proba(X_test)[:, 1]
    q80 = float(seg_policy["scaled_amount_q80"])
    is_high = (processed_df["scaled_amount"] >= q80).to_numpy()

    low_cutoff = float(seg_policy["low_value_threshold"])
    high_cutoff = float(seg_policy["high_value_threshold"])
    high_preds = (probs[is_high] >= high_cutoff).astype(int)
    low_preds = (probs[~is_high] >= low_cutoff).astype(int)

    high_stats = _segment_metrics(y_test[is_high], high_preds)
    low_stats = _segment_metrics(y_test[~is_high], low_preds)
    high_flag_rate = high_stats["flag_rate"]
    low_flag_rate = low_stats["flag_rate"]
    disparity_ratio = high_flag_rate / low_flag_rate if low_flag_rate > 0 else 0.0
    di_ci = _katz_95_ci(
        high_stats["flagged_count"],
        high_stats["count"],
        low_stats["flagged_count"],
        low_stats["count"],
    )

    tpr_ratio = (
        round(float(high_stats["recall"] / low_stats["recall"]), 4)
        if low_stats["recall"] > 0
        else 0.0
    )
    tpr_ci = _katz_95_ci(
        high_stats["true_positives"],
        high_stats["actual_frauds"],
        low_stats["true_positives"],
        low_stats["actual_frauds"],
    )
    if di_ci[0] >= 0.80 and di_ci[1] <= 1.25:
        flag_parity_status = "COMPLIANT"
    elif di_ci[1] < 0.80 or di_ci[0] > 1.25:
        flag_parity_status = "NON-COMPLIANT (BREACH)"
    else:
        flag_parity_status = "INCONCLUSIVE (CI_STRADDLES_THRESHOLD)"

    min_flags_ok = (
        high_stats["flagged_count"] >= 10 and low_stats["flagged_count"] >= 10
    )
    recall_floor_ok = high_stats["recall"] >= 0.65 and low_stats["recall"] >= 0.65
    tpr_ci_ok = tpr_ci[0] <= 1.25 and tpr_ci[1] >= 0.80
    no_confirmed_ci_breach = flag_parity_status != "NON-COMPLIANT (BREACH)"
    within_bounds = bool(
        min_flags_ok and recall_floor_ok and tpr_ci_ok and no_confirmed_ci_breach
    )

    uncal_high_preds = (probs[is_high] >= 0.50).astype(int)
    uncal_low_preds = (probs[~is_high] >= 0.50).astype(int)
    uncal_high_rate = float(uncal_high_preds.mean())
    uncal_low_rate = float(uncal_low_preds.mean())
    uncal_ratio = uncal_high_rate / uncal_low_rate if uncal_low_rate > 0 else 0.0

    audit_summary = {
        "model_uri": model_uri,
        "mlflow_run_id": expected_run_id,
        "audit_type": "transaction_amount_segment_disparity",
        "threshold_calibration_split": seg_policy.get(
            "calibration_split", "validation_holdout_20pct"
        ),
        "deployed_segment_thresholds": {
            "low_value_threshold": low_cutoff,
            "high_value_threshold": high_cutoff,
            "scaled_amount_q80": q80,
        },
        "high_value_flag_rate": round(high_flag_rate, 6),
        "low_value_flag_rate": round(low_flag_rate, 6),
        "disparate_impact_ratio": round(disparity_ratio, 4),
        "di_95_ci": di_ci,
        "flag_parity_status": flag_parity_status,
        "equal_opportunity_tpr_ratio": tpr_ratio,
        "tpr_95_ci": tpr_ci,
        "min_flagged_per_segment_required": 10,
        "high_value_segment": high_stats,
        "low_value_segment": low_stats,
        "within_guardrail_bounds": within_bounds,
        "informational_uncalibrated_050": {
            "high_value_flag_rate": round(uncal_high_rate, 6),
            "low_value_flag_rate": round(uncal_low_rate, 6),
            "disparate_impact_ratio": round(uncal_ratio, 4),
        },
    }

    report_path = BASE_DIR / "artifacts" / "fraud_governance_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print("\n" + "=" * 55)
    print("⚖️ TRANSACTION SEGMENT DISPARITY AUDIT (OUT-OF-SAMPLE)")
    print("=" * 55)
    print(
        f" ✅ Deployed High-Value Flag Rate  : {high_flag_rate:.6f} (cutoff={high_cutoff}, recall={high_stats['recall']:.4f}, precision={high_stats['precision']:.4f})"
    )
    print(
        f" ✅ Deployed Low-Value Flag Rate   : {low_flag_rate:.6f} (cutoff={low_cutoff}, recall={low_stats['recall']:.4f}, precision={low_stats['precision']:.4f})"
    )
    print(
        f" ✅ Deployed Segment Disparity     : {disparity_ratio:.4f} (95% CI={di_ci})"
    )
    print(f" ℹ️ Uncalibrated (0.50) Disparity  : {uncal_ratio:.4f}")
    print("=" * 55 + "\n")

    if enforce_gate and not within_bounds:
        print(
            f"🚨 Governance gate tripped: deployed ratio {disparity_ratio:.4f} outside [0.80, 1.25] or insufficient flagged count."
        )
        sys.exit(1)

    return audit_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enforce-gate", action="store_true")
    args = parser.parse_args()
    run_fairness_audit(enforce_gate=args.enforce_gate)
