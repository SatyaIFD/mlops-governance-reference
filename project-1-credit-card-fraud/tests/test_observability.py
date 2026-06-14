"""
Unit tests for the observability module.
"""

import os
import sys
from pathlib import Path

# Dynamically inject the src container into the runtime path mapping
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Now your original imports will resolve seamlessly across all execution methods
from src.observability import calculate_psi, evaluate_drift_status

def test_calculate_psi_identical_distributions():
    """Ensure that identical distributions return a PSI near zero."""
    np.random.seed(42)
    baseline = np.random.normal(0, 1, 1000)
    target = baseline.copy()
    
    psi = calculate_psi(baseline, target)
    assert psi == pytest.approx(0.0, abs=1e-4)

def test_calculate_psi_significant_drift():
    """Ensure that a heavily drifted distribution triggers a high PSI score."""
    np.random.seed(42)
    baseline = np.random.normal(0, 1, 1000)
    target = np.random.normal(2, 1, 1000) # Heavy distribution mean shift
    
    psi = calculate_psi(baseline, target)
    assert psi > 0.25

def test_evaluate_drift_status_thresholds():
    """Verify that categorical alert levels match regulatory thresholds."""
    assert evaluate_drift_status(0.05) == "STABLE"
    assert evaluate_drift_status(0.18) == "WARNING"
    assert evaluate_drift_status(0.42) == "ALERT"