import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.ingestion.schemas import ProcessedTransactionSchema  # noqa: E402
from src.monitoring.observability import calculate_psi, evaluate_drift_status  # noqa: E402

PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed_features.parquet"
TRAIN_SPLIT_PATH = PROJECT_DIR / "data" / "processed" / "train.parquet"
TEST_SPLIT_PATH = PROJECT_DIR / "data" / "processed" / "test.parquet"
DRIFT_REPORT_PATH = PROJECT_DIR / "artifacts" / "drift_report.json"


def run_statistical_validation() -> dict:
    """
    Applies schema contract checks, null/target integrity checks, and statistical
    Population Stability Index (PSI) drift evaluation across reference vs. evaluation splits.
    """
    print("📊 Initializing Statistical Data Validation & PSI Drift Layer...")

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"❌ Processed features not found at: {PROCESSED_DATA_PATH.resolve()}. Run ingestion first."
        )

    df = pd.read_parquet(PROCESSED_DATA_PATH)
    print("🛡️ Evaluating data profiling expectations...")

    expected_columns = list(ProcessedTransactionSchema.model_fields.keys())
    if list(df.columns) != expected_columns:
        raise ValueError(
            f"🚨 Schema Contract Violation! Mismatch between dataframe and Pydantic model.\n"
            f"Expected: {expected_columns}\nFound: {list(df.columns)}"
        )

    null_counts = int(df.isnull().sum().sum())
    if null_counts > 0:
        raise ValueError(
            f"🚨 Data integrity breach! Found {null_counts} missing values in the dataset."
        )

    unique_classes = set(df["Class"].unique())
    if not unique_classes.issubset({0, 1}):
        raise ValueError(
            f"🚨 Target validation failure! Invalid classes detected: {unique_classes}"
        )

    # Compute PSI across key risk features using reference vs. evaluation partitions
    if TRAIN_SPLIT_PATH.exists() and TEST_SPLIT_PATH.exists():
        ref_df = pd.read_parquet(TRAIN_SPLIT_PATH)
        eval_df = pd.read_parquet(TEST_SPLIT_PATH)
    else:
        midpoint = len(df) // 2
        ref_df = df.iloc[:midpoint]
        eval_df = df.iloc[midpoint:]

    monitored_features = ["V14", "V17", "V12", "scaled_amount", "scaled_time"]
    psi_metrics = {}
    overall_alert = False

    for feat in monitored_features:
        if feat in ref_df.columns and feat in eval_df.columns:
            psi_val = calculate_psi(ref_df[feat].to_numpy(), eval_df[feat].to_numpy())
            status = evaluate_drift_status(psi_val)
            if status == "ALERT":
                overall_alert = True
            psi_metrics[feat] = {"psi": round(psi_val, 6), "status": status}

    report = {
        "schema_valid": True,
        "null_count": null_counts,
        "drift_alert_triggered": overall_alert,
        "feature_psi": psi_metrics,
    }
    DRIFT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("✅ Success! Statistical data validation and PSI checks completed.")
    return report


if __name__ == "__main__":
    try:
        run_statistical_validation()
    except Exception as e:
        print(str(e))
        sys.exit(1)
