"""
Governance & Segment Disparity Audit Module - Project 1: Credit Card Fraud Detection
Evaluates out-of-sample transaction-value segment flagging disparity on test.parquet
using the validation-calibrated segment_thresholds.json served by the API,
and records uncalibrated (flat 0.50) disparity alongside deployed parity.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
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
        latest_version = max(int(v.version) for v in versions)
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

    seg_file = BASE_DIR / "artifacts" / "segment_thresholds.json"
    if seg_file.is_file():
        seg_policy = json.loads(seg_file.read_text(encoding="utf-8"))
    else:
        seg_policy = {
            "low_value_threshold": 0.50,
            "high_value_threshold": 0.50,
            "scaled_amount_q80": float(processed_df["scaled_amount"].quantile(0.80)),
            "calibration_split": "uncalibrated_default",
        }

    X_test = processed_df[CANONICAL_FEATURE_COLS]
    print(
        f"⏳ Running out-of-sample batch inference across evaluation slices using {model_uri}..."
    )
    probs = model.predict_proba(X_test)[:, 1]
    q80 = float(seg_policy["scaled_amount_q80"])
    is_high = (processed_df["scaled_amount"] >= q80).to_numpy()

    # 1. Deployed policy (validation-calibrated segment thresholds served by FastAPI)
    low_cutoff = float(seg_policy["low_value_threshold"])
    high_cutoff = float(seg_policy["high_value_threshold"])
    high_flag_rate = float((probs[is_high] >= high_cutoff).mean())
    low_flag_rate = float((probs[~is_high] >= low_cutoff).mean())
    disparity_ratio = high_flag_rate / low_flag_rate if low_flag_rate > 0 else 0.0
    within_bounds = bool(0.80 <= disparity_ratio <= 1.25)

    # 2. Informational uncalibrated policy (flat 0.50 cutoff)
    uncal_high_rate = float((probs[is_high] >= 0.50).mean())
    uncal_low_rate = float((probs[~is_high] >= 0.50).mean())
    uncal_ratio = uncal_high_rate / uncal_low_rate if uncal_low_rate > 0 else 0.0

    audit_summary = {
        "model_uri": model_uri,
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
        f" ✅ Deployed High-Value Flag Rate  : {high_flag_rate:.6f} (cutoff={high_cutoff})"
    )
    print(
        f" ✅ Deployed Low-Value Flag Rate   : {low_flag_rate:.6f} (cutoff={low_cutoff})"
    )
    print(f" ✅ Deployed Segment Disparity     : {disparity_ratio:.4f}")
    print(f" ℹ️ Uncalibrated (0.50) Disparity  : {uncal_ratio:.4f}")
    print("=" * 55 + "\n")

    if enforce_gate and not within_bounds:
        print(
            f"🚨 Governance gate tripped: deployed ratio {disparity_ratio:.4f} outside [0.80, 1.25]."
        )
        sys.exit(1)

    return audit_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enforce-gate", action="store_true")
    args = parser.parse_args()
    run_fairness_audit(enforce_gate=args.enforce_gate)
