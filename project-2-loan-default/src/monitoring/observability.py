"""
Production Observability & Telemetry Engine - Project 2: Loan Default Prediction
Logs runtime inference request summaries, latencies, model recommendations, review states, and age cohorts.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any


class ModelObservabilityLogger:
    def __init__(self, telemetry_log_path: Path):
        self.telemetry_log_path = telemetry_log_path
        self.telemetry_log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_inference_transaction(
        self,
        request_payload: Dict[str, Any],
        response_data: Dict[str, Any],
        latency_ms: float,
        status_code: int = 200,
    ):
        """Appends a structured transaction record to the production system telemetry log."""
        gov = response_data.get("governance", {})
        risk = response_data.get("risk_metrics", {})

        age_val = request_payload.get("Age")
        derived_age_group = (
            "UNDER_30"
            if isinstance(age_val, (int, float)) and age_val < 30
            else "30_AND_OVER"
        )

        model_rec = gov.get("model_recommendation") or gov.get("underwriting_decision")
        age_group = (
            gov.get("age_group")
            or gov.get("fairness_demographic_proxy_flag")
            or derived_age_group
        )

        telemetry_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
            "payload_snapshot": {
                "Age": age_val,
                "Income": request_payload.get("Income"),
                "CreditScore": request_payload.get("CreditScore"),
                "LoanAmount": request_payload.get("LoanAmount"),
                "DTIRatio": request_payload.get("DTIRatio"),
            },
            "inference_output": {
                "default_probability": risk.get("default_probability"),
                "model_recommendation": model_rec,
                "review_status": gov.get("review_status", "PENDING_HUMAN_REVIEW"),
                "final_decision": gov.get("final_decision"),
                "age_group": age_group,
                "underwriting_decision": gov.get("underwriting_decision", model_rec),
                "demographic_cohort": gov.get(
                    "fairness_demographic_proxy_flag", age_group
                ),
            },
        }

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


if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent
    BASE_DIR = SRC_DIR.parents[1]
    LOG_FILE = BASE_DIR / "artifacts" / "telemetry_logs.jsonl"

    logger = ModelObservabilityLogger(telemetry_log_path=LOG_FILE)
    mock_request = {
        "Age": 24,
        "Income": 62000,
        "CreditScore": 710,
        "LoanAmount": 12000,
        "DTIRatio": 0.28,
    }
    mock_response = {
        "risk_metrics": {"default_probability": 0.0475},
        "governance": {
            "action": "MANUAL_REVIEW_REQUIRED",
            "model_recommendation": "APPROVE",
            "review_status": "PENDING_HUMAN_REVIEW",
            "final_decision": None,
            "age_group": "UNDER_30",
        },
    }
    logger.log_inference_transaction(
        request_payload=mock_request, response_data=mock_response, latency_ms=14.25
    )
    print(json.dumps(logger.read_latest_telemetry(1), indent=4))
