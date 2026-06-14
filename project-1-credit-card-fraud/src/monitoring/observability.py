"""
Production Observability Module
Calculates Population Stability Index (PSI) to track feature distribution drift.
"""

import numpy as np

def calculate_psi(baseline: np.ndarray, target: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculates the Population Stability Index (PSI) between two distributions.
    
    Args:
        baseline: Training or reference feature data vector.
        target: Live production inference batch vector.
        num_bins: Number of buckets for distribution splitting.
        
    Returns:
        float: Calculated PSI metric value.
    """
    quantiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(baseline, quantiles)
    
    # Handle identical percentiles for concentrated distributions safely
    if len(np.unique(bins)) < len(bins):
        bins = np.linspace(baseline.min(), baseline.max(), num_bins + 1)
        
    bins[0] -= 1e-5
    bins[-1] += 1e-5
    
    baseline_counts, _ = np.histogram(baseline, bins=bins)
    target_counts, _ = np.histogram(target, bins=bins)
    
    # Apply Laplace smoothing to avoid division by zero errors
    baseline_ratios = np.where(baseline_counts / len(baseline) == 0, 1e-4, baseline_counts / len(baseline))
    target_ratios = np.where(target_counts / len(target) == 0, 1e-4, target_counts / len(target))
    
    psi_value = np.sum((target_ratios - baseline_ratios) * np.log(target_ratios / baseline_ratios))
    return float(psi_value)

def evaluate_drift_status(psi_value: float) -> str:
    """Returns the monitoring alerting state based on the calculated PSI threshold."""
    if psi_value < 0.1:
        return "STABLE"
    elif 0.1 <= psi_value < 0.25:
        return "WARNING"
    else:
        return "ALERT"