import hashlib
import hmac
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class KeyVersionedPseudonymizer:
    """
    Key-versioned HMAC-SHA256 pseudonymizer for account identifiers.
    Stateless per-call hashing ensures zero in-memory retention of raw account IDs.
    """

    def __init__(
        self,
        key_version: str | None = None,
        secret_key: str | None = None,
    ):
        self.key_version = key_version or os.getenv("AML_HMAC_KEY_VERSION", "v1")
        raw_secret = (secret_key or os.getenv("AML_HMAC_SECRET_KEY") or "").strip()
        self._secret_bytes = raw_secret.encode("utf-8")

    def pseudonymize(self, account_id: Any) -> str:
        if not self._secret_bytes:
            raise RuntimeError(
                "AML_HMAC_SECRET_KEY must be set when account pseudonymization is enabled."
            )
        val = str(account_id).strip()
        if not val:
            return "hmac_empty"
        if val.startswith("hmac_v") and len(val.split("_")) == 3:
            return val
        digest = hmac.new(
            self._secret_bytes, val.encode("utf-8"), hashlib.sha256
        ).hexdigest()[:16]
        return f"hmac_{self.key_version}_{digest}"


class StreamingIngestionEngine:
    """
    Production gatekeeper for the AML stream. Validates schema structures
    in real time, applies consistent HMAC pseudonymization, and handles
    Dead-Letter Queue (DLQ) isolation for poison payloads.
    """

    def __init__(
        self,
        dlq_output_dir: Path | str,
        pseudonymize_accounts: bool = False,
        pseudonymizer: KeyVersionedPseudonymizer | None = None,
    ):
        self.dlq_path = Path(dlq_output_dir)
        self.dlq_path.mkdir(parents=True, exist_ok=True)
        self.dlq_log_file = self.dlq_path / "dead_letter_queue.jsonl"
        self.pseudonymize_accounts = pseudonymize_accounts
        self.pseudonymizer = pseudonymizer or KeyVersionedPseudonymizer()

        self.metrics = {"total_ingested": 0, "clean_events": 0, "dlq_events": 0}

    @staticmethod
    def _compute_event_id(payload: dict) -> str:
        if payload.get("event_id"):
            return str(payload["event_id"])
        ts = payload.get("Timestamp")
        ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)
        raw_key = f"{ts_str}|{payload.get('Sender_account')}|{payload.get('Receiver_account')}|{payload.get('Amount')}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:20]

    def validate_and_route(self, raw_payload: dict) -> tuple[bool, dict | None]:
        """
        Validates the schema matrix of an incoming event.
        Returns: (is_valid, processed_payload_or_none)
        """
        self.metrics["total_ingested"] += 1
        if not isinstance(raw_payload, dict):
            self._route_to_dlq({"raw": str(raw_payload)}, ["ERR_INVALID_PAYLOAD_TYPE"])
            return False, None

        candidate = raw_payload.copy()
        errors = []

        required_fields = ["Timestamp", "Sender_account", "Receiver_account", "Amount"]
        for field in required_fields:
            val = candidate.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                errors.append(
                    f"ERR_MISSING_MANDATORY: Missing mandatory field: {field}"
                )

        if errors:
            self._route_to_dlq(candidate, errors)
            return False, None

        try:
            amount = float(candidate["Amount"])
            if not math.isfinite(amount) or amount <= 0.0:
                errors.append(
                    "ERR_INVALID_AMOUNT: Amount must be a finite positive number."
                )
            else:
                candidate["Amount"] = amount
        except (ValueError, TypeError) as e:
            errors.append(f"ERR_INVALID_AMOUNT: {e!s}")

        try:
            ts_val = candidate["Timestamp"]
            if isinstance(ts_val, str):
                s_ts = ts_val.strip()
                if len(s_ts) == 19 and s_ts[10] == " ":
                    candidate["Timestamp"] = datetime.fromisoformat(s_ts)
                else:
                    candidate["Timestamp"] = datetime.strptime(
                        s_ts, "%Y-%m-%d %H:%M:%S"
                    )
            elif not isinstance(ts_val, datetime):
                errors.append(
                    "ERR_MALFORMED_TIMESTAMP: Timestamp must be string or datetime."
                )
        except (ValueError, TypeError) as e:
            errors.append(f"ERR_MALFORMED_TIMESTAMP: {e!s}")

        if errors:
            self._route_to_dlq(candidate, errors)
            return False, None

        if self.pseudonymize_accounts:
            candidate["Sender_account"] = self.pseudonymizer.pseudonymize(
                candidate["Sender_account"]
            )
            candidate["Receiver_account"] = self.pseudonymizer.pseudonymize(
                candidate["Receiver_account"]
            )

        candidate["event_id"] = self._compute_event_id(candidate)
        self.metrics["clean_events"] += 1
        return True, candidate

    def record_processing_failure(self, payload: dict, errors: list[str]) -> None:
        """Adjusts clean_events counter when a post-validation enrichment or scoring error occurs."""
        if self.metrics["clean_events"] > 0:
            self.metrics["clean_events"] -= 1
        self._route_to_dlq(payload, errors)

    def _route_to_dlq(self, payload: dict, errors: list[str]) -> None:
        """Routes failed payloads to the Dead Letter Queue with deterministic HMAC pseudonymization."""
        self.metrics["dlq_events"] += 1

        sanitized_payload = (
            payload.copy() if isinstance(payload, dict) else {"raw": str(payload)}
        )
        for field in ("Sender_account", "Receiver_account"):
            if field in sanitized_payload and sanitized_payload[field] is not None:
                if self.pseudonymize_accounts:
                    sanitized_payload[field] = self.pseudonymizer.pseudonymize(
                        sanitized_payload[field]
                    )
                else:
                    sanitized_payload[field] = str(sanitized_payload[field])

        if isinstance(sanitized_payload.get("Timestamp"), datetime):
            sanitized_payload["Timestamp"] = sanitized_payload["Timestamp"].strftime(
                "%Y-%m-%d %H:%M:%S"
            )

        if isinstance(sanitized_payload.get("Amount"), float) and not math.isfinite(
            sanitized_payload["Amount"]
        ):
            sanitized_payload["Amount"] = str(sanitized_payload["Amount"])

        dlq_record = {
            "quarantined_at_utc": datetime.now(timezone.utc).isoformat(),
            "key_version": self.pseudonymizer.key_version,
            "errors": errors,
            "raw_payload": sanitized_payload,
        }
        with open(self.dlq_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(dlq_record, default=str) + "\n")
        logging.warning(f"Payload routed to DLQ. Errors: {errors}")

    def get_data_quality_report(self) -> dict:
        """Returns the active data quality footprint including data_quality_pct."""
        total = self.metrics["total_ingested"]
        clean = self.metrics["clean_events"]
        pct = round((clean / total) * 100.0, 2) if total > 0 else 100.0
        return {
            **self.metrics,
            "data_quality_pct": pct,
        }
