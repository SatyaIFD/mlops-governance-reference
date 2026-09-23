"""
Governance & Ethical Compliance Module
Programmatically computes localized Disparate Impact Ratios to audit model bias.
"""

from pathlib import Path
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient


def run_fairness_audit():
    base_dir = Path(__file__).resolve().parents[2]
    mlflow.set_tracking_uri(f"sqlite:///{base_dir}/mlflow.db")

    try:
        client = MlflowClient()
        model_name = "credit_card_fraud_model"
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest_version = max([int(v.version) for v in versions])
            model_uri = f"models:/{model_name}/{latest_version}"
        else:
            model_uri = f"models:/{model_name}/1"

        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    test_data_path = base_dir / "data" / "processed" / "test.parquet"
    if not test_data_path.exists():
        print(f"❌ Processed test data not found at {test_data_path}")
        return

    df = pd.read_parquet(test_data_path)
    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

    if "Time" not in df.columns or "Amount" not in df.columns:
        print(
            "⚠️ Warning: Data contract columns missing in test data. Cannot compute exact fairness metrics."
        )
        return

    X_test = df[feature_names]
    print(f"⏳ Running batch inference across evaluation slices using {model_uri}...")
    df["predictions"] = model.predict(X_test)

    threshold = df["Amount"].quantile(0.80)
    high_value_group = df[df["Amount"] >= threshold]
    low_value_group = df[df["Amount"] < threshold]

    high_flag_rate = high_value_group["predictions"].mean()
    low_flag_rate = low_value_group["predictions"].mean()

    disparate_impact_ratio = (
        high_flag_rate / low_flag_rate if low_flag_rate > 0 else 0.0
    )

    print("\n" + "=" * 50)
    print("⚖️ PROGRAMMATIC GOVERNANCE FAIRNESS AUDIT")
    print("=" * 50)
    print(f" ✅ High-Value Transaction Flag Rate : {high_flag_rate:.6f}")
    print(f" ✅ Low-Value Transaction Flag Rate  : {low_flag_rate:.6f}")
    print(f" ✅ Calculated Disparate Impact Ratio: {disparate_impact_ratio:.4f}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    run_fairness_audit()
