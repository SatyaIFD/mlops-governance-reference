import json
import logging
from pathlib import Path
from datetime import datetime, timezone

# Setup explicit logging for data governance tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class StreamingIngestionEngine:
    """
    Production gatekeeper for the AML stream. Validates schema structures 
    in real time and handles Dead-Letter Queue (DLQ) isolation for poison payloads.
    """
    def __init__(self, dlq_output_dir: Path):
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
                errors.append(f"Invalid monetary value: ${amount} (Must be positive)")
        except (ValueError, TypeError):
            errors.append(f"Amount type casting failure: {raw_payload['Amount']}")

        # Ensure account strings are populated
        if not str(raw_payload['Sender_account']).strip():
            errors.append("Sender_account string is blank")
        if not str(raw_payload['Receiver_account']).strip():
            errors.append("Receiver_account string is blank")

        # 3. Handle validation routing decisions
        if errors:
            self._route_to_dlq(raw_payload, errors)
            return False, None

        self.metrics["clean_events"] += 1
        return True, raw_payload

    def _route_to_dlq(self, poisoned_payload: dict, evaluation_errors: list):
        """Isolates bad records out of the stream into local DLQ storage for compliance review."""
        self.metrics["dlq_events"] += 1
        
        dlq_record = {
            # FIX: Use timezone-aware UTC formatting to eliminate the deprecation warning
            "quarantine_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_errors": evaluation_errors,
            "raw_payload": poisoned_payload
        }
        
        # Append to a newline-delimited JSON file (standard streaming format)
        with open(self.dlq_log_file, mode='a', encoding='utf-8') as f:
            f.write(json.dumps(dlq_record) + "\n")
            
        logging.warning(f"⚠️ DLQ TRIGGERED: Payload quarantined due to errors: {evaluation_errors}")

    def get_data_quality_report(self) -> dict:
        """Returns the active data quality footprint for monitoring dashboards."""
        if self.metrics["total_ingested"] == 0:
            return {"data_quality_pct": 100.0, **self.metrics}
        
        dq_pct = (self.metrics["clean_events"] / self.metrics["total_ingested"]) * 100
        return {"data_quality_pct": round(dq_pct, 2), **self.metrics}

if __name__ == '__main__':
    print("Running data governance validation test loop...")
    # Setup test workspace boundary
    test_artifacts = Path("project-3-aml-streaming/artifacts").resolve()
    ingestor = StreamingIngestionEngine(dlq_output_dir=test_artifacts)

    # Mock vectors containing a clean run, a negative money hack, and a missing key
    mock_wire = [
        {'Timestamp': '2026-07-13 12:00:00', 'Sender_account': 'ACC_1', 'Receiver_account': 'ACC_2', 'Amount': 500.0},
        {'Timestamp': '2026-07-13 12:01:00', 'Sender_account': 'ACC_3', 'Receiver_account': 'ACC_4', 'Amount': -250.0},
        {'Timestamp': '2026-07-13 12:02:00', 'Sender_account': '', 'Receiver_account': 'ACC_5', 'Amount': 100.0}
    ]

    for data_tick in mock_wire:
        success, clean_data = ingestor.validate_and_route(data_tick)
        print(f"Processed Tick -> Route Success: {success} | Forwardable Data: {clean_data is not None}")
        
    print("\nData Governance Footprint:")
    print(ingestor.get_data_quality_report())