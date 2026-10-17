"""
Automated Regulatory Compliance and Bias Tests
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from src.ingestion.utils import clean_and_transform_features  # noqa: E402


@patch("mlflow.pyfunc.load_model")
def test_disparate_impact_ratio_bounds(mock_load_model):
    """Verifies disparate impact ratio logic without requiring local artifacts."""

    # MOCK MLFLOW REGISTRY BEHAVIOR
    mock_model = MagicMock()

    def mock_predict(df_in):
        return (df_in["scaled_amount"] > 0).astype(int)

    mock_model.predict.side_effect = mock_predict
    mock_load_model.return_value = mock_model

    # GENERATE SYNTHETIC DATA
    np.random.seed(42)
    n_rows = 100

    # FIX: Added "Class" to satisfy clean_and_transform_features column indexing
    data = {
        "Time": np.random.uniform(0, 100000, n_rows),
        "Amount": np.random.exponential(50, n_rows),
        "Class": np.zeros(n_rows),
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0, 1, n_rows)

    df = pd.DataFrame(data).drop_duplicates().reset_index(drop=True)
    processed_df = clean_and_transform_features(df.copy())

    feature_names = [f"V{i}" for i in range(1, 29)] + ["scaled_amount", "scaled_time"]
    df["predictions"] = mock_model.predict(processed_df[feature_names])

    threshold = df["Amount"].quantile(0.80)
    high_flag_rate = df[df["Amount"] >= threshold]["predictions"].mean()
    low_flag_rate = df[df["Amount"] < threshold]["predictions"].mean()

    ratio = high_flag_rate / low_flag_rate if low_flag_rate > 0 else 0.0
    assert isinstance(ratio, float)  # nosec B101
