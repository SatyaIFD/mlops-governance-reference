"""
Production Data Drift Monitoring Engine - Project 2: Loan Default Prediction
Calculates Population Stability Index (PSI) to track production distribution shifts.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


class DataDriftMonitor:
    def __init__(self, reference_path: Path, artifacts_dir: Path):
        self.reference_path = reference_path
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def calculate_psi(
        self, reference: np.ndarray, target: np.ndarray, num_bins: int = 10
    ) -> float:
        """Computes the Population Stability Index (PSI) between baseline and production arrays."""
        reference = reference[~np.isnan(reference)]
        target = target[~np.isnan(target)]

        # CRITICAL FIX: Raise error on empty arrays rather than silently reporting 0.0 (STABLE)
        if len(reference) == 0 or len(target) == 0:
            raise ValueError(
                "🚨 PSI calculation undefined: empty array encountered after NaN filtering."
            )

        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.percentile(reference, percentiles)
        bins = np.unique(bins)

        if len(bins) < 2:
            raise ValueError(
                "🚨 Insufficient variance in reference data to construct PSI bins."
            )

        bins[0] -= 1e-5
        bins[-1] += 1e-5

        ref_counts, _ = np.histogram(reference, bins=bins)
        target_counts, _ = np.histogram(target, bins=bins)

        ref_pct = ref_counts / len(reference)
        target_pct = target_counts / len(target)

        epsilon = 1e-4
        ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
        target_pct = np.where(target_pct == 0, epsilon, target_pct)

        psi_value = np.sum((target_pct - ref_pct) * np.log(target_pct / ref_pct))
        return float(psi_value)

    def run_drift_audit(self, production_df: pd.DataFrame) -> bool:
        """Audits live data against historical validation baselines and writes an observability footprint."""
        print("📥 Loading baseline reference split for distribution monitoring...")
        if not self.reference_path.exists():
            raise FileNotFoundError(
                f"Reference validation data not found at {self.reference_path}"
            )

        ref_df = pd.read_parquet(self.reference_path)

        # These must exist in both frames
        target_features = ["CreditScore", "Income", "LoanAmount", "DTIRatio"]
        drift_metrics = {}
        drift_detected = False

        print("📊 Auditing population stability index statistics...")
        for feature in target_features:
            if feature not in ref_df.columns or feature not in production_df.columns:
                raise ValueError(
                    f"🚨 Schema Contract Violation: Required feature '{feature}' missing from payload."
                )

            psi = self.calculate_psi(
                ref_df[feature].values, production_df[feature].values
            )

            if psi >= 0.25:
                status = "SEVERE_DRIFT"
                drift_detected = True
            elif psi >= 0.10:
                status = "MODERATE_SHIFT"
            else:
                status = "STABLE"

            drift_metrics[feature] = {"psi_value": round(psi, 4), "status": status}
            print(f"   -> {feature}: PSI = {psi:.4f} [{status}]")

        report = {"drift_detected_alert": drift_detected, "metrics": drift_metrics}

        report_path = self.artifacts_dir / "data_drift_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        print(
            f"💾 Observability footprint successfully exported to: {report_path.relative_to(report_path.parents[2])}"
        )
        return drift_detected


if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent
    BASE_DIR = SRC_DIR.parents[1]
    REFERENCE_FILE = BASE_DIR / "data" / "processed" / "train.parquet"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"

    print("🧪 Simulating active production data streaming scenario...")
    test_data_path = BASE_DIR / "data" / "processed" / "test.parquet"

    if test_data_path.exists():
        prod_payload = pd.read_parquet(test_data_path)
        # Inject 35% spike to simulate a shift
        prod_payload["LoanAmount"] = prod_payload["LoanAmount"] * 1.35

        monitor = DataDriftMonitor(
            reference_path=REFERENCE_FILE, artifacts_dir=ARTIFACTS_DIR
        )

        try:
            is_drifting = monitor.run_drift_audit(prod_payload)
            # CRITICAL FIX: Ensure the process fails closed to alert the Airflow DAG
            if is_drifting:
                print(
                    "🚨 System Exit: Severe data drift detected across monitored features."
                )
                sys.exit(1)
        except Exception as e:
            print(str(e))
            sys.exit(1)
