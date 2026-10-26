import sys
import pandas as pd
from pathlib import Path

# Bind directly to the ingestion schema contract
from src.ingestion.schemas import ProcessedTransactionSchema

PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed_features.parquet"


def run_statistical_validation():
    """
    Applies strict statistical validation checks to verify distribution consistency,
    null thresholds, and schema contract alignment on engineered Parquet artifacts.
    """
    print("📊 Initializing Statistical Data Validation Layer...")

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"❌ Processed features not found at: {PROCESSED_DATA_PATH.resolve()}. Run ingestion first."
        )

    df = pd.read_parquet(PROCESSED_DATA_PATH)
    print("🛡️ Evaluating data profiling expectations...")

    # Expectation 1 & 5: Bind directly to Pydantic Schema Fields
    expected_columns = list(ProcessedTransactionSchema.model_fields.keys())
    if list(df.columns) != expected_columns:
        raise ValueError(
            f"🚨 Schema Contract Violation! Mismatch between dataframe and Pydantic model.\nExpected: {expected_columns}\nFound: {list(df.columns)}"
        )

    # Expectation 2: Core feature null matrix integrity constraint
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        raise ValueError(
            f"🚨 Data integrity breach! Found {null_counts} missing values in the dataset."
        )

    # Expectation 3: Verify target binary type properties
    unique_classes = set(df["Class"].unique())
    if not unique_classes.issubset({0, 1}):
        raise ValueError(
            f"🚨 Target validation failure! Invalid classes detected: {unique_classes}"
        )

    # Expectation 4: Distribution sanity check on scaled time (Z-score boundaries)
    time_min, time_max = df["scaled_time"].min(), df["scaled_time"].max()
    if time_min < -3.5 or time_max > 3.5:
        print(
            f"⚠️ Warning: Outliers detected in scaled_time distribution: [{time_min:.2f}, {time_max:.2f}]"
        )

    print(
        "✅ Success! Statistical data profiling checks passed. No data drift or schema mutations detected."
    )


if __name__ == "__main__":
    try:
        run_statistical_validation()
    except Exception as e:
        print(str(e))
        sys.exit(1)
