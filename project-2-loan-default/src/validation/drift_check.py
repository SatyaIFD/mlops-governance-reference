"""
Production Data Drift Monitoring Engine - Project 2: Loan Default Prediction
Calculates Population Stability Index (PSI) to track feature distribution shifts.
Evaluates each traffic_source independently (defaulting unlabeled rows to live_production_traffic).
When live telemetry has fewer than min_rows (default 500), reports SKIPPED_INSUFFICIENT_TELEMETRY
without overwriting holdout baseline metrics in data_drift_report.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TRAFFIC_SOURCE = "live_production_traffic"


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

    def compute_drift_summary(
        self, production_df: pd.DataFrame, source_label: str
    ) -> dict:
        if not self.reference_path.exists():
            raise FileNotFoundError(
                f"Reference validation data not found at {self.reference_path}"
            )
        ref_df = pd.read_parquet(self.reference_path)
        target_features = ["CreditScore", "Income", "LoanAmount", "DTIRatio"]
        drift_metrics = {}
        drift_detected = False

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

        return {
            "data_source": source_label,
            "observation_count": len(production_df),
            "drift_detected_alert": drift_detected,
            "overall_status": "DRIFT DETECTED" if drift_detected else "STABLE",
            "metrics": drift_metrics,
        }

    def run_drift_audit(
        self, production_df: pd.DataFrame, source_label: str = "production_telemetry"
    ) -> bool:
        """Audits target data against historical reference baselines and writes an observability footprint."""
        print(
            f"📥 Loading baseline reference split for distribution monitoring (source={source_label})..."
        )
        report = self.compute_drift_summary(production_df, source_label)
        report_path = self.artifacts_dir / "data_drift_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        return bool(report["drift_detected_alert"])


def group_telemetry_by_source(
    telemetry_log_path: Path,
) -> dict[str, pd.DataFrame]:
    if not telemetry_log_path.exists():
        return {}
    grouped: dict[str, list[dict]] = {}
    with open(telemetry_log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line.strip())
                if (
                    rec.get("status_code") == 200
                    and rec.get("schema_version") == 2
                    and "payload_snapshot" in rec
                ):
                    src = str(
                        rec.get("traffic_source") or DEFAULT_TRAFFIC_SOURCE
                    ).strip()
                    grouped.setdefault(src, []).append(rec["payload_snapshot"])
    return {src: pd.DataFrame(rows) for src, rows in grouped.items() if rows}


def record_telemetry_drift_status(artifacts_dir: Path, telemetry_status: dict) -> None:
    """Updates live_telemetry_drift inside data_drift_report.json without overwriting holdout baseline metrics."""
    report_path = artifacts_dir / "data_drift_report.json"
    existing = {}
    if report_path.exists():
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    if not existing:
        existing = {
            "data_source": telemetry_status.get("data_source", "none"),
            "observation_count": telemetry_status.get("observation_count", 0),
            "drift_detected_alert": False,
            "overall_status": telemetry_status.get(
                "status", "SKIPPED_INSUFFICIENT_TELEMETRY"
            ),
            "metrics": {},
        }
    existing["live_telemetry_drift"] = telemetry_status
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulate-drift",
        action="store_true",
        help="Inject a synthetic 35% LoanAmount shift to test drift alerting gates.",
    )
    parser.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="Explicitly compute baseline drift report comparing test.parquet against train.parquet.",
    )
    parser.add_argument(
        "--fail-on-drift",
        action="store_true",
        help="Exit with non-zero status code when severe drift (PSI >= 0.25) is detected.",
    )
    parser.add_argument(
        "--min-telemetry-rows",
        type=int,
        default=500,
        help="Minimum single-source telemetry rows required before computing live PSI.",
    )
    parser.add_argument(
        "--traffic-source",
        type=str,
        default=None,
        help="Optional traffic_source filter for telemetry evaluation.",
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
        elif args.evaluate_holdout:
            if not TEST_FILE.exists():
                raise FileNotFoundError(f"Test split missing at {TEST_FILE}")
            eval_df = pd.read_parquet(TEST_FILE)
            is_drifting = monitor.run_drift_audit(
                eval_df, source_label="holdout_evaluation_split"
            )
            if args.fail_on_drift and is_drifting:
                sys.exit(1)
        else:
            all_grouped_dfs = group_telemetry_by_source(TELEMETRY_FILE)
            target_src = args.traffic_source or DEFAULT_TRAFFIC_SOURCE
            grouped_dfs = {k: v for k, v in all_grouped_dfs.items() if k == target_src}

            counts_by_source = {k: int(len(v)) for k, v in all_grouped_dfs.items()}
            eligible = {
                k: v
                for k, v in grouped_dfs.items()
                if len(v) >= args.min_telemetry_rows
            }

            if not eligible:
                max_obs = max(counts_by_source.values(), default=0)
                print(
                    f"⏭️ SKIPPED_INSUFFICIENT_TELEMETRY: max single-source observations={max_obs} "
                    f"(required>={args.min_telemetry_rows}; counts={counts_by_source}). "
                    f"Preserving existing holdout baseline in data_drift_report.json."
                )
                record_telemetry_drift_status(
                    ARTIFACTS_DIR,
                    {
                        "status": "SKIPPED_INSUFFICIENT_TELEMETRY",
                        "min_rows_required": args.min_telemetry_rows,
                        "observation_count": max_obs,
                        "counts_by_source": counts_by_source,
                    },
                )
                sys.exit(0)

            by_source_reports = {}
            any_drift = False
            for src, df_src in eligible.items():
                print(
                    f"📊 Auditing telemetry PSI for source='{src}' (n={len(df_src)})..."
                )
                rep = monitor.compute_drift_summary(
                    df_src, source_label=f"telemetry:{src}"
                )
                by_source_reports[src] = rep
                if rep["drift_detected_alert"]:
                    any_drift = True

            primary_src = (
                DEFAULT_TRAFFIC_SOURCE
                if DEFAULT_TRAFFIC_SOURCE in by_source_reports
                else next(iter(by_source_reports.keys()))
            )
            record_telemetry_drift_status(
                ARTIFACTS_DIR,
                {
                    "status": by_source_reports[primary_src]["overall_status"],
                    "primary_source": primary_src,
                    "drift_detected_alert": any_drift,
                    "by_source": by_source_reports,
                },
            )
            if args.fail_on_drift and any_drift:
                sys.exit(1)
    except Exception as e:
        print(str(e))
        sys.exit(1)
