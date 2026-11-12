"""
Production Data Drift Monitoring Engine - Project 2: Loan Default Prediction
Calculates Population Stability Index (PSI) across numerical features, categorical
inputs, and age_bucket across rotated/compressed telemetry archives.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from src.monitoring.observability import (
    iter_telemetry_log_files,
    stream_telemetry_records,
)

DEFAULT_TRAFFIC_SOURCE = "live_production_traffic"
CORE_NUMERICAL_FEATURES = ["CreditScore", "Income", "LoanAmount", "DTIRatio"]
EXTENDED_NUMERICAL_FEATURES = [
    "MonthsEmployed",
    "InterestRate",
    "NumCreditLines",
    "LoanTerm",
]
CATEGORICAL_FEATURES = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]


class DataDriftMonitor:
    def __init__(self, reference_path: Path, artifacts_dir: Path):
        self.reference_path = reference_path
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def calculate_psi(
        self, reference: np.ndarray, target: np.ndarray, num_bins: int = 10
    ) -> float:
        """Computes numerical Population Stability Index (PSI) between baseline and target arrays."""
        reference = reference[~np.isnan(reference)]
        target = target[~np.isnan(target)]
        if len(reference) == 0 or len(target) == 0:
            raise ValueError(
                "🚨 PSI calculation undefined: empty array encountered after NaN filtering."
            )
        percentiles = np.linspace(0, 100, num_bins + 1)
        bins = np.unique(np.percentile(reference, percentiles))
        if len(bins) < 2:
            raise ValueError(
                "🚨 Insufficient variance in reference data to construct PSI bins."
            )
        bins[0] -= 1e-5
        bins[-1] += 1e-5
        ref_counts, _ = np.histogram(reference, bins=bins)
        target_counts, _ = np.histogram(target, bins=bins)
        epsilon = 1e-4
        ref_pct = np.where(
            ref_counts / len(reference) == 0, epsilon, ref_counts / len(reference)
        )
        target_pct = np.where(
            target_counts / len(target) == 0, epsilon, target_counts / len(target)
        )
        return float(np.sum((target_pct - ref_pct) * np.log(target_pct / ref_pct)))

    def calculate_categorical_psi(
        self, ref_series: pd.Series, target_series: pd.Series
    ) -> float:
        """Computes categorical PSI across discrete category proportions."""
        ref_clean = ref_series.dropna().astype(str)
        tgt_clean = target_series.dropna().astype(str)
        if len(ref_clean) == 0 or len(tgt_clean) == 0:
            return 0.0
        categories = sorted(set(ref_clean.unique()) | set(tgt_clean.unique()))
        epsilon = 1e-4
        ref_dist = ref_clean.value_counts(normalize=True)
        tgt_dist = tgt_clean.value_counts(normalize=True)
        psi_total = 0.0
        for cat in categories:
            p_ref = max(epsilon, float(ref_dist.get(cat, 0.0)))
            p_tgt = max(epsilon, float(tgt_dist.get(cat, 0.0)))
            psi_total += (p_tgt - p_ref) * np.log(p_tgt / p_ref)
        return float(psi_total)

    def compute_drift_summary(
        self, production_df: pd.DataFrame, source_label: str
    ) -> dict:
        if not self.reference_path.exists():
            raise FileNotFoundError(
                f"Reference validation data not found at {self.reference_path}"
            )
        ref_df = pd.read_parquet(self.reference_path)
        if "Age" in ref_df.columns and "age_bucket" not in ref_df.columns:
            ref_df["age_bucket"] = np.where(
                ref_df["Age"] < 30, "UNDER_30", "30_AND_OVER"
            )
        if "Age" in production_df.columns and "age_bucket" not in production_df.columns:
            production_df = production_df.copy()
            production_df["age_bucket"] = np.where(
                production_df["Age"] < 30, "UNDER_30", "30_AND_OVER"
            )

        drift_metrics = {}
        drift_detected = False

        for feature in CORE_NUMERICAL_FEATURES:
            if feature not in ref_df.columns or feature not in production_df.columns:
                raise ValueError(
                    f"🚨 Schema Contract Violation: Required feature '{feature}' missing from payload."
                )
            psi = self.calculate_psi(
                ref_df[feature].to_numpy(dtype=float),
                production_df[feature].to_numpy(dtype=float),
            )
            status = (
                "SEVERE_DRIFT"
                if psi >= 0.25
                else ("MODERATE_SHIFT" if psi >= 0.10 else "STABLE")
            )
            if psi >= 0.25:
                drift_detected = True
            drift_metrics[feature] = {"psi_value": round(psi, 4), "status": status}

        for feature in EXTENDED_NUMERICAL_FEATURES:
            if (
                feature in ref_df.columns
                and feature in production_df.columns
                and production_df[feature].notna().any()
            ):
                psi = self.calculate_psi(
                    ref_df[feature].to_numpy(dtype=float),
                    production_df[feature].dropna().to_numpy(dtype=float),
                )
                status = (
                    "SEVERE_DRIFT"
                    if psi >= 0.25
                    else ("MODERATE_SHIFT" if psi >= 0.10 else "STABLE")
                )
                if psi >= 0.25:
                    drift_detected = True
                drift_metrics[feature] = {"psi_value": round(psi, 4), "status": status}

        for cat_feat in CATEGORICAL_FEATURES:
            if (
                cat_feat in ref_df.columns
                and cat_feat in production_df.columns
                and production_df[cat_feat].notna().any()
            ):
                psi = self.calculate_categorical_psi(
                    ref_df[cat_feat], production_df[cat_feat]
                )
                status = (
                    "SEVERE_DRIFT"
                    if psi >= 0.25
                    else ("MODERATE_SHIFT" if psi >= 0.10 else "STABLE")
                )
                if psi >= 0.25:
                    drift_detected = True
                drift_metrics[cat_feat] = {"psi_value": round(psi, 4), "status": status}

        info_cohort_psi = None
        if (
            "age_bucket" in ref_df.columns
            and "age_bucket" in production_df.columns
            and production_df["age_bucket"].notna().any()
        ):
            c_psi = self.calculate_categorical_psi(
                ref_df["age_bucket"], production_df["age_bucket"]
            )
            info_cohort_psi = {
                "psi_value": round(c_psi, 4),
                "note": "informational_cohort_mix",
            }
        return {
            "data_source": source_label,
            "observation_count": len(production_df),
            "drift_detected_alert": drift_detected,
            "overall_status": "DRIFT DETECTED" if drift_detected else "STABLE",
            "metrics": drift_metrics,
            "informational_cohort_psi": info_cohort_psi,
        }

    def run_drift_audit(
        self, production_df: pd.DataFrame, source_label: str = "production_telemetry"
    ) -> bool:
        report = self.compute_drift_summary(production_df, source_label)
        report_path = self.artifacts_dir / "data_drift_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
        return bool(report["drift_detected_alert"])


def group_telemetry_by_source(
    telemetry_log_path: Path, max_rows_per_source: int = 10000
) -> dict[str, pd.DataFrame]:
    if not iter_telemetry_log_files(telemetry_log_path):
        return {}
    grouped: dict[str, list[dict]] = {}
    for rec in stream_telemetry_records(telemetry_log_path):
        if (
            rec.get("status_code") == 200
            and rec.get("schema_version") == 2
            and "payload_snapshot" in rec
        ):
            src = str(rec.get("traffic_source") or DEFAULT_TRAFFIC_SOURCE).strip()
            bucket = grouped.setdefault(src, [])
            if len(bucket) < max_rows_per_source:
                bucket.append(rec["payload_snapshot"])
    return {src: pd.DataFrame(rows) for src, rows in grouped.items() if rows}


def record_telemetry_drift_status(artifacts_dir: Path, telemetry_status: dict) -> None:
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
    parser.add_argument("--simulate-drift", action="store_true")
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument("--fail-on-drift", action="store_true")
    parser.add_argument("--min-telemetry-rows", type=int, default=500)
    parser.add_argument("--traffic-source", type=str, default=None)
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
            prod_payload = pd.read_parquet(TEST_FILE)
            prod_payload["LoanAmount"] = prod_payload["LoanAmount"] * 1.35
            if monitor.run_drift_audit(
                prod_payload, source_label="synthetic_drift_simulation"
            ):
                sys.exit(1)
        elif args.evaluate_holdout:
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
                    f"(required>={args.min_telemetry_rows}; counts={counts_by_source})."
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
                    "observation_count": by_source_reports[primary_src][
                        "observation_count"
                    ],
                    "drift_detected_alert": any_drift,
                    "by_source": by_source_reports,
                },
            )
            if args.fail_on_drift and any_drift:
                sys.exit(1)
    except Exception as e:
        print(str(e))
        sys.exit(1)
