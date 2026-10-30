import json
import logging
from pathlib import Path
from datetime import datetime

# Setup explicit logging for data governance tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StreamingIngestionEngine:
    """
    Production gatekeeper for the AML stream. Validates schema structures 
    in real time and handles Dead-Letter Queue (DLQ) isolation for poison payloads.
    """
    def __init__(self, dlq_output_dir: Path | str):
        self.dlq_path = Path(dlq_output_dir)
        self.dlq_path.mkdir(parents=True, exist_ok=True)
        self.dlq_log_file = self.dlq_path / "dead_letter_queue.jsonl"
        
        # Track pipeline data quality metrics
        self.metrics = {"total_ingested": 0, "clean_events": 0, "dlq_events": 0}

    def validate_and_route(self, raw_payload: dict) -> tuple[bool, dict | None]:
        """
        Validates the schema matrix of an incoming event.
        Returns: (is_valid, processed_payload_or_none)
        """
        self.metrics["total_ingested"] += 1
        errors = []

        # 1. Check for structural completeness (Required Fields)
        required_fields = ['Timestamp', 'Sender_account', 'Receiver_account', 'Amount']
        for field in required_fields:
            if field not in raw_payload or raw_payload[field] is None:
                errors.append(f"Missing mandatory field: {field}")

        if errors:
            self._route_to_dlq(raw_payload, errors)
            return False, None

        # 2. Assert strict type validation and values
        try:
            # Check for non-positive values
            amount = float(raw_payload['Amount'])
            if amount <= 0:
                errors.append("ERR_INVALID_AMOUNT: Amount must be strictly positive.")
            else:
                raw_payload['Amount'] = amount

            # Validate timestamp parsing
            ts_str = raw_payload['Timestamp']
            if isinstance(ts_str, str):
                raw_payload['Timestamp'] = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")

        except ValueError as e:
             errors.append(f"ERR_TYPE_CONVERSION: {str(e)}")

        if errors:
            self._route_to_dlq(raw_payload, errors)
            return False, None
            
        self.metrics["clean_events"] += 1
        return True, raw_payload

    def _route_to_dlq(self, payload: dict, errors: list) -> None:
        """Routes failed payloads to the Dead Letter Queue."""
        self.metrics["dlq_events"] += 1
        dlq_record = {
            "timestamp": datetime.now().isoformat(),
            "errors": errors,
            # Ensure the payload is stringified safely if it contains datetime objects that failed
            "payload": str(payload) 
        }
        with open(self.dlq_log_file, "a") as f:
            f.write(json.dumps(dlq_record) + "\n")
        logging.warning(f"Payload routed to DLQ. Errors: {errors}")

    def get_data_quality_report(self) -> dict:
        """Returns the active data quality footprint for monitoring dashboards."""
        return self.metrics
