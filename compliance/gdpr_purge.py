import argparse
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


def _write_journal(journal_file: Path, payload: dict) -> None:
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    tmp_journal = journal_file.with_suffix(".json.tmp")
    with open(tmp_journal, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_journal, journal_file)


def _recover_pending_journal(data_file: Path) -> None:
    """
    Recovers from an interrupted prior purge run using the durable state journal:
    - INITIATED / FILE_STAGED: original data_file is untouched; remove .tmp and journal.
    - FILE_SWAPPED: data_file was replaced before DB commit finished; restore .bak -> data_file.
    - DB_COMMITTED / VERIFIED: both file and DB succeeded; remove .bak, .tmp, and journal.
    """
    journal_file = _journal_path_for(data_file)
    bak_file = _backup_path_for(data_file)
    tmp_file = _tmp_path_for(data_file)

    if not journal_file.exists():
        return

    try:
        with open(journal_file, "r", encoding="utf-8") as f:
            journal = json.load(f)
        state = journal.get("state", "INITIATED")
    except Exception:
        state = "FILE_SWAPPED" if bak_file.exists() else "INITIATED"

    if tmp_file.exists():
        tmp_file.unlink()

    if state in ("INITIATED", "FILE_STAGED", "FILE_SWAPPED"):
        if bak_file.exists():
            os.replace(bak_file, data_file)
    elif state in ("DB_COMMITTED", "VERIFIED"):
        if bak_file.exists():
            bak_file.unlink()

    if journal_file.exists():
        journal_file.unlink()


def _stream_stage_file(data_file: Path, tmp_file: Path, target_user: str) -> int:
    """Streams filtered records to tmp_file without loading the entire dataset into RAM."""
    if data_file.suffix == ".parquet":
        dataset = ds.dataset(data_file, format="parquet")
        if "user_id" not in dataset.schema.names:
            raise PurgeError("user_id column missing in dataset")

        original_count = dataset.count_rows()
        kept_rows = 0
        scanner = dataset.scanner(filter=ds.field("user_id") != target_user)

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

    # Stream CSV in bounded chunks
    original_count = 0
    kept_rows = 0
    header_written = False

    for chunk in pd.read_csv(data_file, dtype={"user_id": str}, chunksize=10000):
        if "user_id" not in chunk.columns:
            raise PurgeError("user_id column missing in dataset")
        original_count += len(chunk)
        remaining = chunk[chunk["user_id"] != target_user]
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


def _verify_purge_completion(db_file: Path, data_file: Path, target_user: str) -> None:
    """Verifies zero residual records exist for target_user in SQLite and batch storage."""
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

    if data_file.suffix == ".parquet":
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


def execute_right_to_be_forgotten(
    user_id: str,
    db_path: str,
    data_path: str,
    _inject_db_failure: bool = False,
) -> dict:
    target_user = str(user_id).strip()
    if not target_user:
        raise PurgeError("user_id must be a non-empty string")

    db_file, data_file = Path(db_path), Path(data_path)

    if not db_file.exists():
        raise PurgeError(f"Tracking DB missing: {db_file}")
    if not data_file.exists():
        raise PurgeError(f"Batch data file missing: {data_file}")

    # Recover any interrupted state from a previous crash before starting
    _recover_pending_journal(data_file)

    journal_file = _journal_path_for(data_file)
    bak_file = _backup_path_for(data_file)
    tmp_file = _tmp_path_for(data_file)
    op_id = uuid.uuid4().hex

    journal_state = {
        "operation_id": op_id,
        "user_id": target_user,
        "db_path": str(db_file.resolve()),
        "data_path": str(data_file.resolve()),
        "state": "INITIATED",
    }
    _write_journal(journal_file, journal_state)

    # Phase 1: Stream filtered dataset to .tmp (FILE_STAGED)
    try:
        file_rows_removed = _stream_stage_file(data_file, tmp_file, target_user)
        journal_state["state"] = "FILE_STAGED"
        _write_journal(journal_file, journal_state)

        # Phase 2: Backup original file to .bak and atomically swap .tmp -> data_file (FILE_SWAPPED)
        shutil.copy2(data_file, bak_file)
        os.replace(tmp_file, data_file)
        journal_state["state"] = "FILE_SWAPPED"
        _write_journal(journal_file, journal_state)
    except Exception as e:
        if tmp_file.exists():
            tmp_file.unlink()
        if bak_file.exists():
            os.replace(bak_file, data_file)
        if journal_file.exists():
            journal_file.unlink()
        if isinstance(e, PurgeError):
            raise
        raise PurgeError(f"Data file purge failed: {e}") from e

    # Phase 3: Scrub MLflow SQLite DB (params and tags strictly scoped to key='user_id')
    conn = sqlite3.connect(db_file)
    deleted = 0
    try:
        if _inject_db_failure:
            raise sqlite3.OperationalError(
                "Simulated database failure after FILE_SWAPPED"
            )

        cur = conn.cursor()
        tables = {
            r[0]
            for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
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
        # Compensating rollback: restore original data file from .bak
        if bak_file.exists():
            os.replace(bak_file, data_file)
        if journal_file.exists():
            journal_file.unlink()
        raise PurgeError(f"Database purge failed; data file rolled back: {e}") from e
    finally:
        conn.close()

    # Phase 4: Verify zero residual records and clean up backup/journal (VERIFIED)
    try:
        _verify_purge_completion(db_file, data_file, target_user)
        journal_state["state"] = "VERIFIED"
        _write_journal(journal_file, journal_state)
    finally:
        if bak_file.exists():
            bak_file.unlink()
        if journal_file.exists():
            journal_file.unlink()

    result = {"db_rows_deleted": deleted, "file_rows_removed": file_rows_removed}
    if result["file_rows_removed"] == 0 and result["db_rows_deleted"] == 0:
        raise PurgeError(f"No records found for user_id: {target_user}")

    return result


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
