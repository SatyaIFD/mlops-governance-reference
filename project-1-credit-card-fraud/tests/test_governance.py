"""
Automated Regulatory Compliance and Segment Disparity Tests
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
from src.ingestion.utils import split_and_scale_without_leakage  # noqa: E402


def test_leakage_free_split_and_disparity_calculation():
    """Verifies that split_and_scale_without_leakage fits strictly on train and computes exact segment disparity."""
    np.random.seed(42)
    n_rows = 200

    data = {
        "Time": np.linspace(0, 10000, n_rows),
        "Amount": np.concatenate([np.full(160, 20.0), np.full(40, 500.0)]),
        "Class": np.array([0] * 150 + [1] * 10 + [0] * 30 + [1] * 10),
    }
    for i in range(1, 29):
        data[f"V{i}"] = np.random.normal(0, 1, n_rows)

    raw_df = pd.DataFrame(data)
    train_df, test_df, scaler_params = split_and_scale_without_leakage(
        raw_df, test_size=0.25, random_state=42
    )

    assert len(train_df) == 150  # nosec B101
    assert len(test_df) == 50  # nosec B101
    assert scaler_params["amount_center"] == pytest.approx(20.0)  # nosec B101

    threshold = test_df["scaled_amount"].quantile(0.80)
    high_group = test_df[test_df["scaled_amount"] >= threshold]
    low_group = test_df[test_df["scaled_amount"] < threshold]

    high_rate = float(high_group["Class"].mean())
    low_rate = float(low_group["Class"].mean())
    ratio = high_rate / low_rate
    assert ratio > 1.25  # nosec B101
    assert ratio == pytest.approx(10.2857, rel=1e-3)  # nosec B101


def test_segment_weights_and_validation_threshold_calibration():
    """Verifies _compute_segment_weights and calibrate_validation_segment_thresholds balance segment flag rates."""
    from src.training.train import (
        _compute_segment_weights,
        calibrate_validation_segment_thresholds,
    )

    df_x = pd.DataFrame({"scaled_amount": [0.1] * 80 + [5.0] * 20})
    y = pd.Series([0] * 75 + [1] * 5 + [0] * 15 + [1] * 5)
    weights = _compute_segment_weights(df_x, y)
    assert len(weights) == 100  # nosec B101
    probs = np.array([0.1] * 76 + [0.8] * 4 + [0.2] * 16 + [0.9] * 4)
    pol = calibrate_validation_segment_thresholds(df_x, probs, q80=1.0)
    assert 0.05 <= pol["high_value_threshold"] <= 0.95  # nosec B101
    assert pol["calibration_split"] == "validation_holdout_20pct"  # nosec B101
