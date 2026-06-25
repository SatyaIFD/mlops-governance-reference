"""
Production Data Drift Monitoring Engine - Project 2: Loan Default Prediction
Calculates Population Stability Index (PSI) to track production distribution shifts.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

class DataDriftMonitor:
    def __init__(self, reference_path: Path, artifacts_dir: Path):
        self.reference_path = reference_path
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def calculate_psi(self, reference: np.ndarray, target: np.ndarray, num_bins: int = 10) -> float:
        """Computes the Population Stability Index (PSI) between baseline and production arrays."""
        # Enforce clean numerical array dimensions
        reference = reference[~np.isnan(reference)]
        target = target[~np.isnan(target)]
        
        if len(reference) == 0 or len(target) == 0:
            return 0.0
            
        # Determine quantile-based bin split boundaries using the reference baseline split
        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(reference, percentiles)
        bins = np.unique(bins)  # Avoid duplicate edges for highly repeated values
        
        if len(bins) < 2:
            return 0.0
            
        # Adjust outermost boundaries to prevent clipping out-of-bounds outliers
        bins[0] -= 1e-5
        bins[-1] += 1e-5
        
        # Calculate frequency counts across both matrices
        ref_counts, _ = np.histogram(reference, bins=bins)
        target_counts, _ = np.histogram(target, bins=bins)
        
        # Convert absolute counts to target density percentages
        ref_pct = ref_counts / len(reference)
        target_pct = target_counts / len(target)
        
        # Handle zero-count bins using a small smoothing epsilon value to stabilize log properties
        epsilon = 1e-4
        ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
        target_pct = np.where(target_pct == 0, epsilon, target_pct)
        
        # Calculate PSI formula sum components
        psi_value = np.sum((target_pct - ref_pct) * np.log(target_pct / ref_pct))
        return float(psi_value)

    def run_drift_audit(self, production_df: pd.DataFrame) -> bool:
        """Audits live data against historical validation baselines and writes an observability footprint."""
        print("📥 Loading baseline reference split for distribution monitoring...")
        if not self.reference_path.exists():
            raise FileNotFoundError(f"Reference validation data not found at {self.reference_path}")
            
        ref_df = pd.read_parquet(self.reference_path)
        
        # Track core numerical variables that affect loan underwriting risks
        target_features = ["CreditScore", "Income", "LoanAmount", "DTIRatio"]
        drift_metrics = {}
        drift_detected = False
        
        print("📊 Auditing population stability index statistics...")
        for feature in target_features:
            if feature not in ref_df.columns or feature not in production_df.columns:
                print(f"⚠️ Feature '{feature}' missing from payload. Skipping check.")
                continue
                
            psi = self.calculate_psi(ref_df[feature].values, production_df[feature].values)
            
            # Map drift status strings based on standard index ranges
            if psi >= 0.25:
                status = "SEVERE_DRIFT"
                drift_detected = True
            elif psi >= 0.10:
                status = "MODERATE_SHIFT"
            else:
                status = "STABLE"
                
            drift_metrics[feature] = {
                "psi_value": round(psi, 4),
                "status": status
            }
            print(f"   -> {feature}: PSI = {psi:.4f} [{status}]")
            
        report = {
            "drift_detected_alert": drift_detected,
            "metrics": drift_metrics
        }
        
        # Export monitoring report to the production artifacts directory
        report_path = self.artifacts_dir / "data_drift_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        print(f"💾 Observability footprint successfully exported to: {report_path.relative_to(report_path.parents[2])}")
        return drift_detected

if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent
    BASE_DIR = SRC_DIR.parents[1]
    REFERENCE_FILE = BASE_DIR / "data" / "processed" / "train.parquet"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    
    # Simulate a drift scenario by loading test data and injecting an artificial 15% shock to Loan Amounts
    print("🧪 Simulating active production data streaming scenario...")
    test_data_path = BASE_DIR / "data" / "processed" / "test.parquet"
    if test_data_path.exists():
        prod_payload = pd.read_parquet(test_data_path)
        prod_payload["LoanAmount"] = prod_payload["LoanAmount"] * 1.35  # Inject 35% spike to simulate a shift
        
        monitor = DataDriftMonitor(reference_path=REFERENCE_FILE, artifacts_dir=ARTIFACTS_DIR)
        monitor.run_drift_audit(prod_payload)