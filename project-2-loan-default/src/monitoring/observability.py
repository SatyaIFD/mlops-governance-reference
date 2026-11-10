"""
Production Observability & Telemetry Engine - Project 2: Loan Default Prediction
Logs runtime inference request summaries, deterministic 16-field payload hashes,
latencies, model recommendations, review states, and age buckets with bounded log rotation.
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


def compute_payload_hash(payload: dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 digest across all payload attributes without storing raw categorical PII."""
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class ModelObservabilityLogger:
    def __init__(
        self,
        telemetry_log_path: Path,
        max_bytes: int = MAX_TELEMETRY_LOG_BYTES,
    ):
        self.telemetry_log_path = telemetry_log_path
        self.max_bytes = max_bytes
        self.telemetry_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _rotate_if_needed(self) -> None:
        if (
            self.max_bytes > 0
            and self.telemetry_log_path.exists()
            and self.telemetry_log_path.stat().st_size >= self.max_bytes
        ):
            rotated = self.telemetry_log_path.with_suffix(
                self.telemetry_log_path.suffix + ".1"
            )
            os.replace(self.telemetry_log_path, rotated)

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
        """Reads back the newest telemetry entries to assist audit validation."""
        if not self.telemetry_log_path.exists():
            print("ℹ️ Telemetry ledger file does not exist yet.")
            return []

        with open(self.telemetry_log_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        return [json.loads(line) for line in lines[-tail_count:]]
