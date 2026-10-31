"""
Production Data Drift Monitoring Engine - Project 2: Loan Default Prediction
Calculates Population Stability Index (PSI) to track feature distribution shifts.
Separates live telemetry monitoring from synthetic drift stress-testing (--simulate-drift).
"""

import argparse
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
        """Computes the Population Stability Index (PSI) between baseline and target arrays."""
        reference = reference[~np.isnan(reference)]
        target = target[~np.isnan(target)]

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

    def run_drift_audit(
        self, production_df: pd.DataFrame, source_label: str = "production_telemetry"
    ) -> bool:
        """Audits target data against historical reference baselines and writes an observability footprint."""
        print(
            f"📥 Loading baseline reference split for distribution monitoring (source={source_label})..."
        )
        if not self.reference_path.exists():
            raise FileNotFoundError(
                f"Reference validation data not found at {self.reference_path}"
            )

        ref_df = pd.read_parquet(self.reference_path)

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
                ref_df[feature].to_numpy(dtype=float),
                production_df[feature].to_numpy(dtype=float),
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

        report = {
            "data_source": source_label,
            "observation_count": int(len(production_df)),
            "drift_detected_alert": drift_detected,
            "overall_status": "DRIFT DETECTED" if drift_detected else "STABLE",
            "metrics": drift_metrics,
        }

        report_path = self.artifacts_dir / "data_drift_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        return drift_detected


def load_live_telemetry_dataframe(
    telemetry_log_path: Path, min_rows: int = 10
) -> pd.DataFrame | None:
    if not telemetry_log_path.exists():
        return None
    snapshots = []
    with open(telemetry_log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line.strip())
                if rec.get("status_code") == 200 and "payload_snapshot" in rec:
                    snapshots.append(rec["payload_snapshot"])
    if len(snapshots) < min_rows:
        return None
    return pd.DataFrame(snapshots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulate-drift",
        action="store_true",
        help="Inject a synthetic 35% LoanAmount shift to test drift alerting gates.",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with non-zero status code when severe drift (PSI >= 0.25) is detected.",
    )
    args = parser.parse_args()

    SRC_DIR = Path(__file__).resolve().parent
    BASE_DIR = SRC_DIR.parents[1]
    REFERENCE_FILE = BASE_DIR / "data" / "processed" / "train.parquet"
    TEST_FILE = BASE_DIR / "data" / "processed" / "test.parquet"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    TELEMETRY_FILE = ARTIFACTS_DIR / "telemetry_logs.jsonl"

    monitor = DataDriftMonitor(
        reference_path=REFERENCE_FILE, artifacts_dir=ARTIFACTS_DIR
    )

    try:
        if args.simulate_drift:
            if not TEST_FILE.exists():
                raise FileNotFoundError(f"Test split missing at {TEST_FILE}")
            prod_payload = pd.read_parquet(TEST_FILE)
            prod_payload["LoanAmount"] = prod_payload["LoanAmount"] * 1.35
            is_drifting = monitor.run_drift_audit(
                prod_payload, source_label="synthetic_drift_simulation"
            )
            if is_drifting:
                print(
                    "🚨 Severe synthetic data drift detected across monitored features."
                )
                sys.exit(1)
        else:
            live_df = load_live_telemetry_dataframe(TELEMETRY_FILE)
            if live_df is not None:
                is_drifting = monitor.run_drift_audit(
                    live_df, source_label="live_telemetry_logs"
                )
            elif TEST_FILE.exists():
                eval_df = pd.read_parquet(TEST_FILE)
                is_drifting = monitor.run_drift_audit(
                    eval_df, source_label="holdout_evaluation_split"
                )
            else:
                print(
                    "⚠️ Neither live telemetry nor holdout test split found; skipping drift run."
                )
                sys.exit(0)

            if args.fail_on_drift and is_drifting:
                sys.exit(1)
    except Exception as e:
        print(str(e))
        sys.exit(1)
