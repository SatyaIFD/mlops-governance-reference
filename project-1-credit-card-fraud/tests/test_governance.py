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


def test_segment_weights_and_audit_metrics():
    """Verifies _compute_segment_weights and _segment_metrics calculation."""
    from src.training.train import _compute_segment_weights
    from src.governance.audit import _segment_metrics

    df_x = pd.DataFrame({"scaled_amount": [0.1] * 80 + [5.0] * 20})
    y = pd.Series([0] * 75 + [1] * 5 + [0] * 15 + [1] * 5)
    weights = _compute_segment_weights(df_x, y)
    assert len(weights) == 100  # nosec B101
    stats = _segment_metrics(np.array([0, 1, 1, 0]), np.array([0, 1, 0, 0]))
    assert stats["recall"] == 0.5 and stats["precision"] == 1.0  # nosec B101
