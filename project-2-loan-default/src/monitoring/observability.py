"""
Production Observability & Telemetry Engine - Project 2: Loan Default Prediction
Logs runtime inference request summaries, deterministic 16-field payload hashes,
latencies, model recommendations, review states, and age buckets with multi-archive rotation.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

TELEMETRY_SCHEMA_VERSION = 2
MAX_TELEMETRY_LOG_BYTES = int(
    os.getenv("LOAN_TELEMETRY_MAX_BYTES", str(50 * 1024 * 1024))
)
MAX_TELEMETRY_ARCHIVES = int(os.getenv("LOAN_TELEMETRY_MAX_ARCHIVES", "5"))


def compute_payload_hash(payload: dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 digest across all payload attributes without storing raw categorical PII."""
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def iter_telemetry_log_files(
    telemetry_log_path: Path, max_archives: int = MAX_TELEMETRY_ARCHIVES
) -> list[Path]:
    """Returns existing rotated archive files in chronological order (oldest first) followed by the active file."""
    files: list[Path] = []
    for idx in range(max_archives, 0, -1):
        cand = telemetry_log_path.with_suffix(f"{telemetry_log_path.suffix}.{idx}")
        if cand.is_file():
            files.append(cand)
    if telemetry_log_path.is_file():
        files.append(telemetry_log_path)
    return files


class ModelObservabilityLogger:
    def __init__(
        self,
        telemetry_log_path: Path,
        max_bytes: int = MAX_TELEMETRY_LOG_BYTES,
        max_archives: int = MAX_TELEMETRY_ARCHIVES,
    ):
        self.telemetry_log_path = telemetry_log_path
        self.max_bytes = max_bytes
        self.max_archives = max_archives
        self.telemetry_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _rotate_if_needed(self) -> None:
        if (
            self.max_bytes > 0
            and self.max_archives > 0
            and self.telemetry_log_path.exists()
            and self.telemetry_log_path.stat().st_size >= self.max_bytes
        ):
            for idx in range(self.max_archives - 1, 0, -1):
                src = self.telemetry_log_path.with_suffix(
                    f"{self.telemetry_log_path.suffix}.{idx}"
                )
                dst = self.telemetry_log_path.with_suffix(
                    f"{self.telemetry_log_path.suffix}.{idx + 1}"
                )
                if src.exists():
                    os.replace(src, dst)
            first_archive = self.telemetry_log_path.with_suffix(
                f"{self.telemetry_log_path.suffix}.1"
            )
            os.replace(self.telemetry_log_path, first_archive)

    def log_inference_transaction(
        self,
        request_payload: dict[str, Any],
        response_data: dict[str, Any],
        latency_ms: float,
        status_code: int = 200,
        traffic_source: str = "live_production_traffic",
    ):
        """Appends a structured transaction record with bucketed age cohort to the telemetry log."""
        gov = response_data.get("governance", {})
        risk = response_data.get("risk_metrics", {})

        age_val = request_payload.get("Age")
        derived_age_bucket = (
            "UNDER_30"
            if isinstance(age_val, (int, float)) and age_val < 30
            else "30_AND_OVER"
        )
        age_bucket = (
            gov.get("age_group")
            or gov.get("fairness_demographic_proxy_flag")
            or derived_age_bucket
        )
        model_rec = gov.get("model_recommendation") or gov.get("underwriting_decision")

        telemetry_entry = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "traffic_source": traffic_source,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
            "payload_hash": compute_payload_hash(request_payload),
            "payload_snapshot": {
                "age_bucket": age_bucket,
                "Income": request_payload.get("Income"),
                "CreditScore": request_payload.get("CreditScore"),
                "LoanAmount": request_payload.get("LoanAmount"),
                "DTIRatio": request_payload.get("DTIRatio"),
            },
            "inference_output": {
                "default_probability": risk.get("default_probability"),
                "model_recommendation": model_rec,
                "decision_threshold": gov.get("decision_threshold"),
                "model_version": gov.get("model_version"),
                "review_status": gov.get("review_status", "PENDING_HUMAN_REVIEW"),
                "final_decision": gov.get("final_decision"),
                "underwriting_decision": gov.get("underwriting_decision", model_rec),
            },
        }

        self._rotate_if_needed()
        with open(self.telemetry_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(telemetry_entry) + "\n")

    def read_latest_telemetry(self, tail_count: int = 5):
        """Reads back the newest telemetry entries across retained archives and active log."""
        files = iter_telemetry_log_files(
            self.telemetry_log_path, max_archives=self.max_archives
        )
        if not files:
            print("ℹ️ Telemetry ledger file does not exist yet.")
            return []

        records: list[dict] = []
        for fpath in files:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line.strip()))
        return records[-tail_count:]
