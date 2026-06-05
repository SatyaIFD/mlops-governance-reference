import os
import sys
import pandas as pd
from pathlib import Path
import great_expectations as ge

def get_project_root() -> Path:
    if "ipykernel" in sys.modules or "__file__" not in globals():
        return Path(os.getcwd()).resolve()
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "requirements.txt").exists() or (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parents[2]

ROOT_DIR = get_project_root()
PROCESSED_DATA_PATH = ROOT_DIR / "project-1-credit-card-fraud" / "data" / "processed_features.parquet"

def run_statistical_validation():
    """
    Applies strict Great Expectations checks to validate statistical distribution consistency,
    null thresholds, and feature ranges on engineered Parquet artifacts.
    """
    print("📊 Initializing Statistical Data Validation Layer...")

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Processed features not found at: {PROCESSED_DATA_PATH.resolve()}. Run ingestion first.")

    # Load data into a Great Expectations Pandas Dataset wrapper
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    ge_df = ge.from_pandas(df)

    print("🛡️ Evaluating data profiling expectations...")

    # Expectation 1: Assert exact column length boundaries (31 features)
    ge_df.expect_table_column_count_to_equal(31)

    # Expectation 2: Core feature null matrix integrity constraint
    for col in ge_df.columns:
        ge_df.expect_column_values_to_not_be_null(col)

    # Expectation 3: Verify target binary type properties
    ge_df.expect_column_values_to_be_in_set("Class", [0, 1])

    # Expectation 4: Distribution sanity check on scaled time (Z-score boundaries)
    ge_df.expect_column_values_to_be_between("scaled_time", min_value=-3.5, max_value=3.5)

    # Expectation 5: Explicit column sequence mapping validation
    expected_columns = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time", "Class"]
    ge_df.expect_table_columns_to_match_ordered_list(expected_columns)

    # Run validation suite
    validation_results = ge_df.validate()

    if validation_results["success"]:
        print("✅ Success! Statistical data profiling checks passed. No data drift or schema mutations detected.")
    else:
        # Collect failed criteria details to throw a descriptive warning/error
        failed_expectations = [
            res["expectation_config"]["kwargs"] 
            for res in validation_results["results"] if not res["success"]
        ]
        raise ValueError(f"🚨 Statistical Profiling Failure! Failed requirements: {failed_expectations}")

if __name__ == "__main__":
    run_statistical_validation()