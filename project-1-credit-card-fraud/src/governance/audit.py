"""
Governance & Segment Disparity Audit Module
Evaluates transaction-value segment flagging disparity using the registered
fraud_detection_xgb model and properly scaled 30-feature inputs.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.utils import (  # noqa: E402
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

    # Transform raw Time/Amount if dataset has not been scaled yet
    if "scaled_amount" not in df.columns or "scaled_time" not in df.columns:
        if "Amount" not in df.columns or "Time" not in df.columns:
            print("❌ Required feature columns missing in test dataset.")
            sys.exit(1)
        scaler_file = BASE_DIR / "artifacts" / "scaler_params.json"
        scaler_params = (
            load_scaler_params(scaler_file) if scaler_file.exists() else None
        )
        processed_df = clean_and_transform_features(
            df, scaler_params=scaler_params, drop_duplicates=False
        )
        amount_series = df["Amount"]
    else:
        processed_df = df
        amount_series = df["scaled_amount"]

    X_test = processed_df[CANONICAL_FEATURE_COLS]
    print(f"⏳ Running batch inference across evaluation slices using {model_uri}...")
    predictions = model.predict(X_test)

    eval_frame = pd.DataFrame(
        {"segment_metric": amount_series.values, "predictions": predictions}
    )
    threshold = eval_frame["segment_metric"].quantile(0.80)
    high_value_group = eval_frame[eval_frame["segment_metric"] >= threshold]
    low_value_group = eval_frame[eval_frame["segment_metric"] < threshold]

    high_flag_rate = float(high_value_group["predictions"].mean())
    low_flag_rate = float(low_value_group["predictions"].mean())

    disparity_ratio = high_flag_rate / low_flag_rate if low_flag_rate > 0 else 0.0
    within_bounds = 0.80 <= disparity_ratio <= 1.25

    audit_summary = {
        "model_uri": model_uri,
        "audit_type": "transaction_amount_segment_disparity",
        "high_value_flag_rate": round(high_flag_rate, 6),
        "low_value_flag_rate": round(low_flag_rate, 6),
        "disparate_impact_ratio": round(disparity_ratio, 4),
        "within_guardrail_bounds": within_bounds,
    }

    report_path = BASE_DIR / "artifacts" / "fraud_governance_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    print("\n" + "=" * 55)
    print("⚖️ TRANSACTION SEGMENT DISPARITY AUDIT (OPERATIONAL)")
    print("=" * 55)
    print(f" ✅ High-Value Segment Flag Rate   : {high_flag_rate:.6f}")
    print(f" ✅ Low-Value Segment Flag Rate    : {low_flag_rate:.6f}")
    print(f" ✅ Segment Disparity Ratio        : {disparity_ratio:.4f}")
    print("=" * 55 + "\n")

    if enforce_gate and not within_bounds:
        print(
            f"🚨 Governance gate tripped: ratio {disparity_ratio:.4f} outside [0.80, 1.25]."
        )
        sys.exit(1)

    return audit_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enforce-gate",
        action="store_true",
        help="Exit non-zero if disparity ratio falls outside [0.80, 1.25]",
    )
    args = parser.parse_args()
    run_fairness_audit(enforce_gate=args.enforce_gate)
