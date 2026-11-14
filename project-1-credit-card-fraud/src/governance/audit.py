"""
Governance & Operational Segment Telemetry Audit Module - Project 1: Credit Card Fraud Detection
Evaluates out-of-sample classification performance (ROC-AUC, PR-AUC, per-segment recall/precision)
at the served 0.50 threshold and records transaction-amount segment flagging ratios as informational telemetry.
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
from sklearn.metrics import average_precision_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.utils import (
    CANONICAL_FEATURE_COLS,
    clean_and_transform_features,
    load_scaler_params,
)

SERVED_DECISION_THRESHOLD = 0.50


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
    preds = (probs >= SERVED_DECISION_THRESHOLD).astype(int)

    roc_auc = round(float(roc_auc_score(y_test, probs)), 4)
    pr_auc = round(float(average_precision_score(y_test, probs)), 4)

    q80 = round(float(processed_df["scaled_amount"].quantile(0.80)), 6)
    is_high = (processed_df["scaled_amount"] >= q80).to_numpy()
    high_stats = _segment_metrics(y_test[is_high], preds[is_high])
    low_stats = _segment_metrics(y_test[~is_high], preds[~is_high])

    high_flag_rate = high_stats["flag_rate"]
    low_flag_rate = low_stats["flag_rate"]
    disparity_ratio = (
        round(high_flag_rate / low_flag_rate, 4) if low_flag_rate > 0 else 0.0
    )
    di_ci = _katz_95_ci(
        high_stats["flagged_count"],
        high_stats["count"],
        low_stats["flagged_count"],
        low_stats["count"],
    )
    within_80_125_band = bool(0.80 <= disparity_ratio <= 1.25)

    # Gate enforces model discrimination (ROC-AUC >= 0.90, PR-AUC >= 0.75) and per-segment recall >= 0.65.
    # Transaction-amount flag-rate ratio is reported as INFORMATIONAL (amount tier is not a protected demographic class).
    performance_gate_passed = bool(
        roc_auc >= 0.90
        and pr_auc >= 0.75
        and high_stats["recall"] >= 0.65
        and low_stats["recall"] >= 0.65
    )

    audit_summary = {
        "model_uri": model_uri,
        "mlflow_run_id": expected_run_id,
        "served_decision_threshold": SERVED_DECISION_THRESHOLD,
        "audit_type": "operational_segment_performance_and_informational_flag_rate",
        "performance_metrics": {
            "test_roc_auc": roc_auc,
            "test_pr_auc": pr_auc,
        },
        "high_value_segment": high_stats,
        "low_value_segment": low_stats,
        "informational_segment_flag_rate_telemetry": {
            "high_value_flag_rate": round(high_flag_rate, 6),
            "low_value_flag_rate": round(low_flag_rate, 6),
            "disparate_impact_ratio": disparity_ratio,
            "di_95_ci": di_ci,
            "within_80_125_band": within_80_125_band,
            "note": "Informational operational metric only; transaction amount tier is not a protected demographic attribute and higher-amount tiers exhibit higher base fraud prevalence (0.232% vs 0.151%).",
        },
        "performance_gate_passed": performance_gate_passed,
    }

    report_path = BASE_DIR / "artifacts" / "fraud_governance_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print("\n" + "=" * 65)
    print("⚖️ P1 FRAUD PERFORMANCE GATE & INFORMATIONAL SEGMENT TELEMETRY")
    print("=" * 65)
    print(f" ✅ Holdout ROC-AUC / PR-AUC       : {roc_auc:.4f} / {pr_auc:.4f}")
    print(
        f" ✅ High-Value Segment (cutoff=0.5): flag_rate={high_flag_rate:.6f}, recall={high_stats['recall']:.4f}, precision={high_stats['precision']:.4f}"
    )
    print(
        f" ✅ Low-Value Segment  (cutoff=0.5): flag_rate={low_flag_rate:.6f}, recall={low_stats['recall']:.4f}, precision={low_stats['precision']:.4f}"
    )
    print(
        f" ℹ️ Segment Flag-Rate Ratio (Info) : {disparity_ratio:.4f} (95% CI={di_ci}, within_80_125_band={within_80_125_band})"
    )
    print("=" * 65 + "\n")

    if enforce_gate and not performance_gate_passed:
        print("🚨 P1 performance/recall gate tripped.")
        sys.exit(1)

    return audit_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enforce-gate", action="store_true")
    args = parser.parse_args()
    run_fairness_audit(enforce_gate=args.enforce_gate)
