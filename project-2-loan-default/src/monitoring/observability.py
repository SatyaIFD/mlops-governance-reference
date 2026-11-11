"""
Production Observability & Telemetry Engine - Project 2: Loan Default Prediction
Logs runtime inference request summaries, deterministic 16-field payload hashes,
and full feature snapshots (with bucketed age) using flock-synchronized, gzip-compressed rotation.
"""

from collections import deque
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, Iterator

TELEMETRY_SCHEMA_VERSION = 2
MAX_TELEMETRY_LOG_BYTES = int(
    os.getenv("LOAN_TELEMETRY_MAX_BYTES", str(50 * 1024 * 1024))
)
MAX_TELEMETRY_ARCHIVES = int(os.getenv("LOAN_TELEMETRY_MAX_ARCHIVES", "5"))


def compute_payload_hash(payload: dict[str, Any]) -> str:
    """Computes a deterministic SHA-256 digest across all payload attributes."""
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
    """Streams parsed JSON records line-by-line across retained archives without loading all files into memory."""
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
            self.max_bytes <= 0
            or self.max_archives <= 0
            or not self.telemetry_log_path.exists()
            or self.telemetry_log_path.stat().st_size < self.max_bytes
        ):
            return
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
            if src_gz.exists():
                os.replace(src_gz, dst_gz)
            elif src_raw.exists():
                with open(src_raw, "rb") as f_in, gzip.open(dst_gz, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                src_raw.unlink()
        first_gz = self.telemetry_log_path.with_suffix(
            f"{self.telemetry_log_path.suffix}.1.gz"
        )
        with (
            open(self.telemetry_log_path, "rb") as f_in,
            gzip.open(first_gz, "wb") as f_out,
        ):
            shutil.copyfileobj(f_in, f_out)
        self.telemetry_log_path.unlink()

    def log_inference_transaction(
        self,
        request_payload: dict[str, Any],
        response_data: dict[str, Any],
        latency_ms: float,
        status_code: int = 200,
        traffic_source: str = "live_production_traffic",
    ):
        """Appends a structured transaction record synchronized with fcntl.flock."""
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

        snapshot: dict[str, Any] = {
            "age_bucket": age_bucket,
            "Income": request_payload.get("Income"),
            "CreditScore": request_payload.get("CreditScore"),
            "LoanAmount": request_payload.get("LoanAmount"),
            "DTIRatio": request_payload.get("DTIRatio"),
        }
        for extra_col in (
            "MonthsEmployed",
            "InterestRate",
            "NumCreditLines",
            "LoanTerm",
            "Education",
            "EmploymentType",
            "MaritalStatus",
            "HasMortgage",
            "HasDependents",
            "LoanPurpose",
            "HasCoSigner",
        ):
            if extra_col in request_payload:
                snapshot[extra_col] = request_payload.get(extra_col)

        telemetry_entry = {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "traffic_source": traffic_source,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
            "payload_hash": compute_payload_hash(request_payload),
            "payload_snapshot": snapshot,
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
        with open(lock_path, "w", encoding="utf-8") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                self._rotate_if_needed()
                with open(self.telemetry_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(telemetry_entry) + "\n")
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)

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
