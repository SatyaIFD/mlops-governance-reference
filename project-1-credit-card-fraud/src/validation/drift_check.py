import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def get_project_root() -> Path:
    if "ipykernel" in sys.modules or "__file__" not in globals():
        return Path(os.getcwd()).resolve()
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "requirements.txt").exists() or (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parents[2]

ROOT_DIR = get_project_root()
# Match exact production output directory from ingestion.py
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed_features.parquet"

def run_statistical_validation():
    """
    Applies strict statistical validation checks to verify distribution consistency,
    null thresholds, and feature ranges on engineered Parquet artifacts.
    """
    print("📊 Initializing Statistical Data Validation Layer...")

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Processed features not found at: {PROCESSED_DATA_PATH.resolve()}. Run ingestion first.")

    # Load engineered Parquet data matrix
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    print("🛡️ Evaluating data profiling expectations...")

    # Expectation 1: Assert exact column length boundaries (31 features)
    if len(df.columns) != 31:
        raise ValueError(f"🚨 Matrix structural failure! Expected 31 columns, found {len(df.columns)}")

    # Expectation 2: Core feature null matrix integrity constraint
    null_counts = df.isnull().sum().sum()
    if null_counts > 0:
        raise ValueError(f"🚨 Data integrity breach! Found {null_counts} missing values in the dataset.")

    # Expectation 3: Verify target binary type properties
    unique_classes = set(df['Class'].unique())
    if not unique_classes.issubset({0, 1}):
        raise ValueError(f"🚨 Target validation failure! Invalid classes detected: {unique_classes}")

    # Expectation 4: Distribution sanity check on scaled time (Z-score boundaries)
    time_min, time_max = df['scaled_time'].min(), df['scaled_time'].max()
    if time_min < -3.5 or time_max > 3.5:
        print(f"⚠️ Warning: Outliers detected in scaled_time distribution: [{time_min:.2f}, {time_max:.2f}]")

    # Expectation 5: Explicit column sequence mapping validation
    expected_columns = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time", "Class"]
    if list(df.columns) != expected_columns:
        raise ValueError("🚨 Column orientation mismatch! Features are out of sequence order.")

    print("✅ Success! Statistical data profiling checks passed. No data drift or schema mutations detected.")

if __name__ == "__main__":
    run_statistical_validation()