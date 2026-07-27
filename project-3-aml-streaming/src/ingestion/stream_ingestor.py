import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class StreamingIngestionEngine:
    """Validates streaming transaction payloads against schema contracts and routes failures.

    The engine acts as a quality gatekeeper for the streaming pipeline. It validates field
    presence, data types, and non-negative amounts, converting valid timestamps to datetime
    objects while isolating corrupt or malformed payloads to a Dead-Letter Queue (DLQ).

    Attributes:
        dlq_output_dir (Path): Directory where quarantined DLQ payloads are written.
        clean_event_count (int): Total count of valid events passed to state cache.
        dlq_event_count (int): Total count of corrupted events routed to DLQ.
    """

    def __init__(self, dlq_output_dir: Path):
        """Initializes the ingestion engine and prepares the DLQ artifact directory.

        Args:
            dlq_output_dir (Path): Path to store quarantined JSON payloads.
        """
        self.dlq_output_dir = Path(dlq_output_dir)
        self.dlq_output_dir.mkdir(parents=True, exist_ok=True)
        self.clean_event_count: int = 0
        self.dlq_event_count: int = 0

    def validate_and_route(self, raw_tx: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Validates a raw stream record and routes malformed payloads to the DLQ.

        Args:
            raw_tx (Dict[str, Any]): Dictionary containing unvalidated transaction fields.

        Returns:
            Tuple[bool, Dict[str, Any]]: 
                - bool: True if payload passed validation, False if quarantined.
                - Dict[str, Any]: Processed record (with datetime timestamp) or raw payload.
        """
        errors: List[str] = []

        # 1. Mandatory Field Existence Verification
        required_fields = ['Timestamp', 'Sender_account', 'Receiver_account', 'Amount']
        for field in required_fields:
            if field not in raw_tx or raw_tx[field] is None or raw_tx[field] == '':
                errors.append(f"Missing mandatory field: {field}")

        # 2. Monetary Amount Validation
        if 'Amount' in raw_tx and raw_tx['Amount'] is not None:
            try:
                amt = float(raw_tx['Amount'])
                if amt <= 0.0:
                    errors.append(f"Invalid monetary amount ({amt}): Must be greater than 0.0")
            except (ValueError, TypeError):
                errors.append(f"Non-numeric monetary amount: {raw_tx['Amount']}")

        # 3. Timestamp Parsing & Normalization
        parsed_timestamp = None
        if 'Timestamp' in raw_tx and raw_tx['Timestamp']:
            ts_val = raw_tx['Timestamp']
            if isinstance(ts_val, datetime):
                parsed_timestamp = ts_val
            else:
                ts_str = str(ts_val)
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d", "%Y/%m/%d %H:%M:%S"):
                    try:
                        parsed_timestamp = datetime.strptime(ts_str, fmt)
                        break
                    except ValueError:
                        continue

            if parsed_timestamp is None:
                errors.append(f"Malformed timestamp string format: '{raw_tx['Timestamp']}'")

        # 4. Routing Logic: DLQ Quarantine vs Clean Payload
        if errors:
            self._quarantine_payload(raw_tx, errors)
            self.dlq_event_count += 1
            return False, raw_tx

        # Valid Payload Assembly
        clean_tx = raw_tx.copy()
        clean_tx['Timestamp'] = parsed_timestamp
        clean_tx['Amount'] = float(raw_tx['Amount'])
        self.clean_event_count += 1
        return True, clean_tx

    def _quarantine_payload(self, raw_tx: Dict[str, Any], errors: List[str]) -> None:
        """Serializes and isolates malformed payloads into the DLQ directory.

        Args:
            raw_tx (Dict[str, Any]): The original corrupted transaction record.
            errors (List[str]): List of error messages explaining validation failures.
        """
        logging.warning(f"⚠️ DLQ TRIGGERED: Payload quarantined due to errors: {errors}")
        quarantine_file = self.dlq_output_dir / f"dlq_event_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
        
        quarantine_payload = {
            "isolation_timestamp_utc": datetime.now().isoformat(),
            "validation_errors": errors,
            "raw_payload": raw_tx
        }

        try:
            with open(quarantine_file, "w") as f:
                json.dump(quarantine_payload, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Failed to write DLQ file: {e}")

    def get_data_quality_report(self) -> Dict[str, Any]:
        """Returns aggregated data quality governance statistics for the session.

        Returns:
            Dict[str, Any]: Summary containing total ingested, clean %, and DLQ counts.
        """
        total = self.clean_event_count + self.dlq_event_count
        dq_pct = (self.clean_event_count / total * 100.0) if total > 0 else 100.0
        return {
            "data_quality_pct": round(dq_pct, 2),
            "total_ingested": total,
            "clean_events": self.clean_event_count,
            "dlq_events": self.dlq_event_count
        }