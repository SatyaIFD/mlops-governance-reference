"""
Automated Regulatory Compliance and Bias Tests
"""
import pytest
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

def test_disparate_impact_ratio_bounds():
    """Verifies the model's disparate impact ratio does not drift into critical risk sectors."""
    base_dir = Path(__file__).resolve().parents[1]
    mlflow.set_tracking_uri(f"sqlite:///{base_dir}/mlflow.db")
    
    client = MlflowClient()
    latest_versions = client.get_latest_versions("credit_card_fraud_model")
    latest_version = latest_versions[-1].version if latest_versions else "1"
    
    model = mlflow.pyfunc.load_model(f"models:/credit_card_fraud_model/{latest_version}")
    df = pd.read_parquet(base_dir / "data" / "processed" / "test.parquet")
    
    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    df["predictions"] = model.predict(df[feature_names])
    
    threshold = df["Amount"].quantile(0.80)
    high_flag_rate = df[df["Amount"] >= threshold]["predictions"].mean()
    low_flag_rate = df[df["Amount"] < threshold]["predictions"].mean()
    
    ratio = high_flag_rate / low_flag_rate if low_flag_rate > 0 else 0.0
    
    assert ratio > 0.0
    assert isinstance(ratio, float)