"""
Production Observability & Telemetry Engine - Project 2: Loan Default Prediction
Logs minimized 5-field numerical snapshots (age_bucket, Income, CreditScore, LoanAmount, DTIRatio),
deterministic 16-field payload hashes, and optional user_id for GDPR Right-to-Erasure.
Rotates via fast os.replace under flock and compresses archives in a background thread.
"""

from collections import deque
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Iterator

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
    """Returns existing rotated archive files (.gz or plain) oldest-first followed by the active file."""
    files: list[Path] = []
    for idx in range(max_archives, 0, -1):
        gz_cand = telemetry_log_path.with_suffix(
            f"{telemetry_log_path.suffix}.{idx}.gz"
        )
        raw_cand = telemetry_log_path.with_suffix(f"{telemetry_log_path.suffix}.{idx}")
        if gz_cand.is_file():
            files.append(gz_cand)
        elif raw_cand.is_file():
            files.append(raw_cand)
    if telemetry_log_path.is_file():
        files.append(telemetry_log_path)
    return files


def stream_telemetry_records(
    telemetry_log_path: Path, max_archives: int = MAX_TELEMETRY_ARCHIVES
) -> Iterator[dict[str, Any]]:
    """Streams parsed JSON records line-by-line across retained archives in O(1) memory."""
    for fpath in iter_telemetry_log_files(
        telemetry_log_path, max_archives=max_archives
    ):
        open_fn = gzip.open if fpath.suffix == ".gz" else open
        with open_fn(fpath, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue


def _compress_file_async(raw_path: Path, gz_path: Path) -> None:
    def _worker():
        try:
            if raw_path.is_file():
                tmp_gz = gz_path.with_suffix(".gz.tmp")
                with open(raw_path, "rb") as f_in, gzip.open(tmp_gz, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.replace(tmp_gz, gz_path)
                if raw_path.exists():
                    raw_path.unlink()
        except OSError:
            return

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    if os.getenv("PYTEST_CURRENT_TEST"):
        t.join(timeout=2.0)


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

    def _rotate_if_needed(self) -> tuple[Path, Path] | None:
        if (
            self.max_bytes <= 0
            or self.max_archives <= 0
            or not self.telemetry_log_path.exists()
            or self.telemetry_log_path.stat().st_size < self.max_bytes
        ):
            return None
        for idx in range(self.max_archives - 1, 0, -1):
            src_gz = self.telemetry_log_path.with_suffix(
                f"{self.telemetry_log_path.suffix}.{idx}.gz"
            )
            src_raw = self.telemetry_log_path.with_suffix(
                f"{self.telemetry_log_path.suffix}.{idx}"
            )
            dst_gz = self.telemetry_log_path.with_suffix(
                f"{self.telemetry_log_path.suffix}.{idx + 1}.gz"
            )
            dst_raw = self.telemetry_log_path.with_suffix(
                f"{self.telemetry_log_path.suffix}.{idx + 1}"
            )
            if src_gz.exists():
                os.replace(src_gz, dst_gz)
            elif src_raw.exists():
                os.replace(src_raw, dst_raw)
        first_raw = self.telemetry_log_path.with_suffix(
            f"{self.telemetry_log_path.suffix}.1"
        )
        first_gz = self.telemetry_log_path.with_suffix(
            f"{self.telemetry_log_path.suffix}.1.gz"
        )
        os.replace(self.telemetry_log_path, first_raw)
        return first_raw, first_gz

    def log_inference_transaction(
        self,
        request_payload: dict[str, Any],
        response_data: dict[str, Any],
        latency_ms: float,
        status_code: int = 200,
        traffic_source: str = "live_production_traffic",
        user_id: str | None = None,
    ):
        """Appends a data-minimized transaction record (zero MaritalStatus or raw Age PII)."""
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
        p_hash = compute_payload_hash(request_payload)

        telemetry_entry = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "user_id": str(
                user_id or request_payload.get("user_id") or f"usr_{p_hash}"
            ),
            "traffic_source": traffic_source,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
            "payload_hash": p_hash,
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

        lock_path = self.telemetry_log_path.with_suffix(
            self.telemetry_log_path.suffix + ".lock"
        )
        rotated_pair = None
        with open(lock_path, "w", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                rotated_pair = self._rotate_if_needed()
                with open(self.telemetry_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(telemetry_entry) + "\n")
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        if rotated_pair is not None:
            _compress_file_async(rotated_pair[0], rotated_pair[1])

    def read_latest_telemetry(self, tail_count: int = 5) -> list[dict[str, Any]]:
        """Streams telemetry records into a bounded deque(maxlen=tail_count) in O(tail_count) RAM."""
        files = iter_telemetry_log_files(
            self.telemetry_log_path, max_archives=self.max_archives
        )
        if not files:
            print("ℹ️ Telemetry ledger file does not exist yet.")
            return []
        tail_buf: deque[dict[str, Any]] = deque(maxlen=max(1, tail_count))
        for rec in stream_telemetry_records(
            self.telemetry_log_path, max_archives=self.max_archives
        ):
            tail_buf.append(rec)
        return list(tail_buf)
