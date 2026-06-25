"""
Production Observability & Telemetry Engine - Project 2: Loan Default Prediction
Logs runtime inference request payloads, latencies, decisions, and demographic flags.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any

class ModelObservabilityLogger:
    def __init__(self, telemetry_log_path: Path):
        self.telemetry_log_path = telemetry_log_path
        # Ensure target logging directory infrastructure exists
        self.telemetry_log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_inference_transaction(self, request_payload: Dict[str, Any], response_data: Dict[str, Any], latency_ms: float, status_code: int = 200):
        """Appends a structured transaction record to the production system telemetry log."""
        
        telemetry_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
            "payload_snapshot": {
                "Age": request_payload.get("Age"),
                "Income": request_payload.get("Income"),
                "CreditScore": request_payload.get("CreditScore"),
                "LoanAmount": request_payload.get("LoanAmount"),
                "DTIRatio": request_payload.get("DTIRatio")
            },
            "inference_output": {
                "default_probability": response_data.get("risk_metrics", {}).get("default_probability"),
                "underwriting_decision": response_data.get("governance", {}).get("underwriting_decision"),
                "demographic_cohort": response_data.get("governance", {}).get("fairness_demographic_proxy_flag")
            }
        }
        
        # Write to disk using standard line-delimited JSON (JSON Lines) format
        with open(self.telemetry_log_path, "a") as f:
            f.write(json.dumps(telemetry_entry) + "\n")

    def read_latest_telemetry(self, tail_count: int = 5):
        """Reads back the newest telemetry entries to assist audit validation."""
        if not self.telemetry_log_path.exists():
            print("ℹ️ Telemetry ledger file does not exist yet.")
            return []
            
        with open(self.telemetry_log_path, "r") as f:
            lines = f.readlines()
            
        records = [json.loads(line.strip()) for line in lines[-tail_count:]]
        return records

if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent
    BASE_DIR = SRC_DIR.parents[1]
    LOG_FILE = BASE_DIR / "artifacts" / "telemetry_logs.jsonl"
    
    print("🧪 Simulating real-time endpoint observability logging scenario...")
    logger = ModelObservabilityLogger(telemetry_log_path=LOG_FILE)
    
    # Mock data tracking a live loan request event
    mock_request = {
        "Age": 24,
        "Income": 62000,
        "CreditScore": 710,
        "LoanAmount": 12000,
        "DTIRatio": 0.28
    }
    
    mock_response = {
        "risk_metrics": {"default_probability": 0.0475, "risk_score_tier": "STANDARD RISK"},
        "governance": {"underwriting_decision": "APPROVED", "fairness_demographic_proxy_flag": "Protected Young Cohort"}
    }
    
    # Log the simulated runtime event transaction
    logger.log_inference_transaction(
        request_payload=mock_request,
        response_data=mock_response,
        latency_ms=14.25
    )
    
    print(f"💾 Telemetry event written successfully.")
    latest = logger.read_latest_telemetry(1)
    print(json.dumps(latest, indent=4))