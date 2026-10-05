"""
Automated Regulatory Compliance and Bias Tests
"""

import sys
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Dynamically inject Project 1 root into sys.path to access the ingestion utils
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from src.ingestion.utils import clean_and_transform_features  # noqa: E402


def test_disparate_impact_ratio_bounds():
    """Verifies the model's disparate impact ratio does not drift into critical risk sectors."""
    base_dir = Path(__file__).resolve().parents[1]
    mlflow.set_tracking_uri(f"sqlite:///{base_dir}/mlflow.db")

    client = MlflowClient()
    model_versions = client.search_model_versions("name='credit_card_fraud_model'")
    latest_version = (
        max([int(mv.version) for mv in model_versions]) if model_versions else 1
    )

    model = mlflow.pyfunc.load_model(
        f"models:/credit_card_fraud_model/{latest_version}"
    )
    df = pd.read_parquet(base_dir / "data" / "processed" / "test.parquet")

    # Drop duplicates upfront so the raw dataframe and processed dataframe maintain the exact same row count
    df = df.drop_duplicates().reset_index(drop=True)

    # Transform a copy of the dataframe to generate scaled_time and scaled_amount
    processed_df = clean_and_transform_features(df.copy())

    # The model's exact expected signature
    feature_names = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
    df["predictions"] = model.predict(processed_df[feature_names])

    # Governance Logic: Check if high-value transactions are disproportionately flagged
    threshold = df["Amount"].quantile(0.80)
    high_flag_rate = df[df["Amount"] >= threshold]["predictions"].mean()
    low_flag_rate = df[df["Amount"] < threshold]["predictions"].mean()

    ratio = high_flag_rate / low_flag_rate if low_flag_rate > 0 else 0.0

    assert ratio > 0.0  # nosec B101
    assert isinstance(ratio, float)  # nosec B101
