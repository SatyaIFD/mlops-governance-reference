import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import gzip
import hashlib
import hmac
import json
import logging
import os
import shutil
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


class PurgeError(RuntimeError):
    pass


def _journal_path_for(data_file: Path) -> Path:
    return data_file.with_suffix(data_file.suffix + ".purge_journal.json")


def _backup_path_for(data_file: Path) -> Path:
    return data_file.with_suffix(data_file.suffix + ".bak")


def _tmp_path_for(data_file: Path) -> Path:
    return data_file.with_suffix(data_file.suffix + ".tmp")


def _lock_path_for(data_file: Path) -> Path:
    return data_file.with_suffix(data_file.suffix + ".purge.lock")


@contextmanager
def _writer_swap_lock(companion_files: list[Path]):
    """Acquires the telemetry writer lock (.lock) across all active JSONL files strictly for delta catch-up and os.replace."""
    lock_paths = sorted(
        {
            (
                p.with_suffix(".jsonl.lock")
                if p.name.endswith(".jsonl.gz")
                else p.with_suffix(p.suffix + ".lock")
            ).resolve()
            for p in companion_files
        }
    )
    with ExitStack() as stack:
        handles = [
            stack.enter_context(open(lp, "w", encoding="utf-8")) for lp in lock_paths
        ]
        for h in handles:
            fcntl.flock(h.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            for h in reversed(handles):
                fcntl.flock(h.fileno(), fcntl.LOCK_UN)


def _compute_target_tokens(
    target_user: str, require_hmac_key: bool = False
) -> set[str]:
    tokens = {target_user}
    hmac_seed = os.getenv("LOAN_HMAC_SECRET_KEY", "").strip()
    if not hmac_seed:
        if require_hmac_key and not target_user.startswith("usr_hmac_"):
            raise PurgeError(
                "LOAN_HMAC_SECRET_KEY must be set to purge pseudonymized JSONL telemetry."
            )
        return tokens
    hmac_token = (
        "usr_hmac_"
        + hmac.new(
            hmac_seed.encode("utf-8"), target_user.encode("utf-8"), hashlib.sha256
        ).hexdigest()[:16]
    )
    tokens.add(hmac_token)
    return tokens


def _discover_jsonl_targets(data_file: Path) -> list[Path]:
    targets: list[Path] = []
    seen: set[Path] = set()

    def _add(p: Path) -> None:
        if p.is_file() and not p.name.endswith((".lock", ".tmp", ".bak", ".json")):
            rp = p.resolve()
            if rp not in seen:
                targets.append(p)
                seen.add(rp)

    _add(data_file)
    if data_file.suffix == ".jsonl":
        for idx in range(1, 10):
            _add(data_file.with_suffix(f"{data_file.suffix}.{idx}.gz"))
            _add(data_file.with_suffix(f"{data_file.suffix}.{idx}"))
        if data_file.name == "telemetry_logs.jsonl" and data_file.parent.is_dir():
            for pod_f in sorted(data_file.parent.glob("telemetry_*.jsonl*")):
                _add(pod_f)
    return targets


def _write_journal(journal_file: Path, payload: dict) -> None:
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    tmp_journal = journal_file.with_suffix(".json.tmp")
    with open(tmp_journal, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_journal, journal_file)


def _recover_pending_journal(data_file: Path) -> None:
    journal_file = _journal_path_for(data_file)
    if not journal_file.exists():
        return
    companion_paths = [data_file]
    try:
        with open(journal_file, "r", encoding="utf-8") as f:
            journal = json.load(f)
        state = journal.get("state", "INITIATED")
        if isinstance(journal.get("companion_files"), list):
            companion_paths = [Path(p) for p in journal["companion_files"]]
    except Exception:
        state = "FILE_SWAPPED" if _backup_path_for(data_file).exists() else "INITIATED"
    for cpath in companion_paths:
        ctmp = _tmp_path_for(cpath)
        cbak = _backup_path_for(cpath)
        if ctmp.exists():
            ctmp.unlink()
        if state in ("INITIATED", "FILE_STAGED", "FILE_SWAPPED"):
            if cbak.exists():
                os.replace(cbak, cpath)
        elif state in ("DB_COMMITTED", "VERIFIED"):
            if cbak.exists():
                cbak.unlink()
    if journal_file.exists():
        journal_file.unlink()


def _stage_single_jsonl(
    src_file: Path, tmp_file: Path, target_tokens: set[str]
) -> tuple[int, int]:
    removed = 0
    if src_file.suffix == ".gz":
        with (
            gzip.open(src_file, "rt", encoding="utf-8") as f_in,
            gzip.open(tmp_file, "wt", encoding="utf-8") as f_out,
        ):
            for line in f_in:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if str(rec.get("user_id")) in target_tokens:
                    removed += 1
                else:
                    f_out.write(line if line.endswith("\n") else line + "\n")
        return removed, src_file.stat().st_size

    with open(src_file, "rb") as f_in, open(tmp_file, "w", encoding="utf-8") as f_out:
        for raw_line in f_in:
            line = raw_line.decode("utf-8")
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec.get("user_id")) in target_tokens:
                removed += 1
            else:
                f_out.write(line if line.endswith("\n") else line + "\n")
        end_offset = f_in.tell()
    return removed, end_offset


def _catch_up_appended_jsonl_delta(
    cfile: Path,
    bak_file: Path,
    tmp_file: Path,
    staged_offset: int,
    target_tokens: set[str],
) -> int:
    """Reads any bytes appended to a plain .jsonl file since staged_offset under the writer lock."""
    if cfile.suffix == ".gz" or not cfile.is_file():
        return 0
    curr_size = cfile.stat().st_size
    if curr_size <= staged_offset:
        return 0
    removed_delta = 0
    with open(cfile, "rb") as f_in:
        f_in.seek(staged_offset)
        delta_bytes = f_in.read()
    if not delta_bytes:
        return 0
    with open(bak_file, "ab") as f_bak:
        f_bak.write(delta_bytes)
    with open(tmp_file, "a", encoding="utf-8") as f_tmp:
        for line in delta_bytes.decode("utf-8").splitlines(keepends=True):
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec.get("user_id")) in target_tokens:
                removed_delta += 1
            else:
                f_tmp.write(line if line.endswith("\n") else line + "\n")
    return removed_delta


def _stream_stage_file(
    data_file: Path,
    tmp_file: Path,
    target_user: str,
    companion_files: list[Path] | None = None,
    staged_offsets: dict[Path, int] | None = None,
) -> int:
    """Streams filtered records to tmp_file(s), preserving rows where user_id is NULL/NaN."""
    if data_file.suffix == ".jsonl" or data_file.name.endswith(".jsonl.gz"):
        target_tokens = _compute_target_tokens(target_user, require_hmac_key=True)
        targets = companion_files or [data_file]
        total_removed = 0
        for cfile in targets:
            rem, end_off = _stage_single_jsonl(
                cfile, _tmp_path_for(cfile), target_tokens
            )
            total_removed += rem
            if staged_offsets is not None:
                staged_offsets[cfile] = end_off
        return total_removed

    if data_file.suffix == ".parquet":
        dataset = ds.dataset(data_file, format="parquet")
        if "user_id" not in dataset.schema.names:
            raise PurgeError("user_id column missing in dataset")
        original_count = dataset.count_rows()
        kept_rows = 0
        keep_expr = ds.field("user_id").is_null() | (ds.field("user_id") != target_user)
        scanner = dataset.scanner(filter=keep_expr)
        with pq.ParquetWriter(tmp_file, dataset.schema) as writer:
            for batch in scanner.to_batches():
                if batch.num_rows > 0:
                    writer.write_batch(batch)
                    kept_rows += batch.num_rows
            if kept_rows == 0:
                empty_table = pa.Table.from_pydict(
                    {name: [] for name in dataset.schema.names},
                    schema=dataset.schema,
                )
                writer.write_table(empty_table)
        return original_count - kept_rows

    original_count = 0
    kept_rows = 0
    header_written = False
    for chunk in pd.read_csv(data_file, dtype={"user_id": str}, chunksize=10000):
        if "user_id" not in chunk.columns:
            raise PurgeError("user_id column missing in dataset")
        original_count += len(chunk)
        remaining = chunk[chunk["user_id"].isna() | (chunk["user_id"] != target_user)]
        kept_rows += len(remaining)
        remaining.to_csv(
            tmp_file,
            mode="a" if header_written else "w",
            header=not header_written,
            index=False,
        )
        header_written = True
    if not header_written:
        df_empty = pd.read_csv(data_file, dtype={"user_id": str})
        if "user_id" not in df_empty.columns:
            raise PurgeError("user_id column missing in dataset")
        df_empty.to_csv(tmp_file, index=False)
    return original_count - kept_rows


def _verify_purge_completion(
    db_file: Path,
    data_file: Path,
    target_user: str,
    companion_files: list[Path] | None = None,
) -> None:
    """Verifies zero residual records exist for target_user in SQLite and batch/telemetry storage."""
    conn = sqlite3.connect(db_file)
    try:
        cur = conn.cursor()
        tables = {
            r[0]
            for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        for sql in (
            "SELECT COUNT(*) FROM params WHERE key='user_id' AND value=?",
            "SELECT COUNT(*) FROM tags WHERE key='user_id' AND value=?",
        ):
            table_name = sql.split()[3]
            if table_name in tables:
                remaining_db = cur.execute(sql, (target_user,)).fetchone()[0]
                if remaining_db != 0:
                    raise PurgeError(
                        f"Verification failed: {remaining_db} rows remain in {table_name}"
                    )
    finally:
        conn.close()

    if data_file.suffix == ".jsonl" or data_file.name.endswith(".jsonl.gz"):
        target_tokens = _compute_target_tokens(target_user, require_hmac_key=True)
        for cfile in companion_files or [data_file]:
            c_open = gzip.open if cfile.suffix == ".gz" else open
            with c_open(cfile, "rt", encoding="utf-8") as f:
                for line in f:
                    if (
                        line.strip()
                        and str(json.loads(line).get("user_id")) in target_tokens
                    ):
                        raise PurgeError(
                            f"Verification failed: target user still present in {cfile.name}"
                        )
    elif data_file.suffix == ".parquet":
        dataset = ds.dataset(data_file, format="parquet")
        remaining_file = dataset.count_rows(filter=ds.field("user_id") == target_user)
        if remaining_file != 0:
            raise PurgeError(
                f"Verification failed: {remaining_file} rows remain in {data_file.name}"
            )
    else:
        for chunk in pd.read_csv(data_file, dtype={"user_id": str}, chunksize=10000):
            if (chunk["user_id"] == target_user).any():
                raise PurgeError(
                    f"Verification failed: target user still present in {data_file.name}"
                )


def _run_best_effort_vacuum(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("VACUUM")
    except sqlite3.Error as vac_err:
        logging.warning("Best-effort SQLite VACUUM skipped after commit: %s", vac_err)


def execute_right_to_be_forgotten(
    user_id: str,
    db_path: str,
    data_path: str,
    _inject_db_failure: bool = False,
    _on_staged_hook=None,
) -> dict:
    target_user = str(user_id).strip()
    if not target_user:
        raise PurgeError("user_id must be a non-empty string")
    db_file, data_file = Path(db_path), Path(data_path)
    if not db_file.exists():
        raise PurgeError(f"Tracking DB missing: {db_file}")
    if not data_file.exists():
        raise PurgeError(f"Batch data file missing: {data_file}")

    lock_file = _lock_path_for(data_file)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_file, "w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            _recover_pending_journal(data_file)
            journal_file = _journal_path_for(data_file)
            is_jsonl = data_file.suffix == ".jsonl" or data_file.name.endswith(
                ".jsonl.gz"
            )
            companion_files = (
                _discover_jsonl_targets(data_file) if is_jsonl else [data_file]
            )
            staged_offsets: dict[Path, int] = {}
            journal_state = {
                "operation_id": uuid.uuid4().hex,
                "user_id": target_user,
                "db_path": str(db_file.resolve()),
                "data_path": str(data_file.resolve()),
                "companion_files": [str(p.resolve()) for p in companion_files],
                "state": "INITIATED",
            }
            _write_journal(journal_file, journal_state)
            try:
                file_rows_removed = _stream_stage_file(
                    data_file,
                    _tmp_path_for(data_file),
                    target_user,
                    companion_files=companion_files,
                    staged_offsets=staged_offsets,
                )
                # Copy backups BEFORE acquiring the telemetry writer lock
                for cfile in companion_files:
                    shutil.copy2(cfile, _backup_path_for(cfile))
                journal_state["state"] = "FILE_STAGED"
                _write_journal(journal_file, journal_state)
                if _on_staged_hook is not None:
                    _on_staged_hook()
                target_tokens = (
                    _compute_target_tokens(target_user, require_hmac_key=True)
                    if is_jsonl
                    else {target_user}
                )
                with _writer_swap_lock(companion_files):
                    for cfile in companion_files:
                        if is_jsonl and cfile in staged_offsets:
                            file_rows_removed += _catch_up_appended_jsonl_delta(
                                cfile,
                                _backup_path_for(cfile),
                                _tmp_path_for(cfile),
                                staged_offsets[cfile],
                                target_tokens,
                            )
                        os.replace(_tmp_path_for(cfile), cfile)
                journal_state["state"] = "FILE_SWAPPED"
                _write_journal(journal_file, journal_state)
            except Exception as e:
                for cfile in companion_files:
                    ctmp = _tmp_path_for(cfile)
                    cbak = _backup_path_for(cfile)
                    if ctmp.exists():
                        ctmp.unlink()
                    if cbak.exists():
                        os.replace(cbak, cfile)
                if journal_file.exists():
                    journal_file.unlink()
                if isinstance(e, PurgeError):
                    raise
                raise PurgeError(f"Data file purge failed: {e}") from e

            conn = sqlite3.connect(db_file)
            deleted = 0
            try:
                if _inject_db_failure:
                    raise sqlite3.OperationalError(
                        "Simulated database failure after FILE_SWAPPED"
                    )
                conn.execute("PRAGMA secure_delete = ON")
                cur = conn.cursor()
                tables = {
                    r[0]
                    for r in cur.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    )
                }
                for sql in (
                    "DELETE FROM params WHERE key='user_id' AND value=?",
                    "DELETE FROM tags WHERE key='user_id' AND value=?",
                ):
                    target_table = sql.split()[2]
                    if target_table in tables:
                        cur.execute(sql, (target_user,))
                        deleted += cur.rowcount
                conn.commit()
                journal_state["state"] = "DB_COMMITTED"
                _write_journal(journal_file, journal_state)
            except Exception as e:
                conn.rollback()
                with _writer_swap_lock(companion_files):
                    for cfile in companion_files:
                        cbak = _backup_path_for(cfile)
                        if cbak.exists():
                            os.replace(cbak, cfile)
                if journal_file.exists():
                    journal_file.unlink()
                raise PurgeError(
                    f"Database purge failed; data file rolled back: {e}"
                ) from e
            else:
                if deleted > 0:
                    _run_best_effort_vacuum(conn)
            finally:
                conn.close()

            try:
                _verify_purge_completion(
                    db_file, data_file, target_user, companion_files=companion_files
                )
                journal_state["state"] = "VERIFIED"
                _write_journal(journal_file, journal_state)
            finally:
                for cfile in companion_files:
                    cbak = _backup_path_for(cfile)
                    if cbak.exists():
                        cbak.unlink()
                if journal_file.exists():
                    journal_file.unlink()

            result = {
                "db_rows_deleted": deleted,
                "file_rows_removed": file_rows_removed,
            }
            if result["file_rows_removed"] == 0 and result["db_rows_deleted"] == 0:
                raise PurgeError(f"No records found for user_id: {target_user}")
            return result
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            if lock_file.exists():
                try:
                    lock_file.unlink()
                except OSError as cleanup_err:
                    logging.debug("Lock file cleanup skipped: %s", cleanup_err)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--db_path", default="mlflow.db")
    parser.add_argument("--data_path", default="data/processed/train.parquet")
    args = parser.parse_args()
    try:
        res = execute_right_to_be_forgotten(args.user_id, args.db_path, args.data_path)
        logging.info(f"✅ GDPR Purge Complete: {res}")
    except PurgeError as e:
        logging.error(f"❌ GDPR Purge Failed: {e}")
        sys.exit(1)
