import sqlite3
import pandas as pd
from pathlib import Path
import logging
import argparse
import sys


class PurgeError(RuntimeError):
    pass


def execute_right_to_be_forgotten(user_id: str, db_path: str, data_path: str) -> dict:
    db_file, data_file = Path(db_path), Path(data_path)

    if not db_file.exists():
        raise PurgeError(f"Tracking DB missing: {db_file}")
    if not data_file.exists():
        raise PurgeError(f"Batch data file missing: {data_file}")

    # 1. Scrub MLflow SQLite DB (Both params and tags tables)
    conn = sqlite3.connect(db_file)
    try:
        cur = conn.cursor()
        tables = {
            r[0]
            for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        deleted = 0

        for sql in (
            "DELETE FROM params WHERE key='user_id' AND value=?",
            "DELETE FROM tags WHERE value=?",
        ):
            target_table = sql.split()[2]
            if target_table in tables:
                cur.execute(sql, (user_id,))
                deleted += cur.rowcount
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise PurgeError(f"Database purge failed: {e}") from e
    finally:
        conn.close()

    # 2. Scrub Batch Data File (CSV or Parquet)
    try:
        if data_file.suffix == ".parquet":
            df = pd.read_parquet(data_file)
        else:
            df = pd.read_csv(data_file, dtype={"user_id": str})

        if "user_id" not in df.columns:
            raise PurgeError("user_id column missing in dataset")

        remaining = df[df["user_id"] != str(user_id)]
        tmp = data_file.with_suffix(data_file.suffix + ".tmp")

        # Atomic replacement to prevent corruption during writes
        if data_file.suffix == ".parquet":
            remaining.to_parquet(tmp, index=False)
        else:
            remaining.to_csv(tmp, index=False)

        tmp.replace(data_file)
        file_rows_removed = len(df) - len(remaining)
    except Exception as e:
        raise PurgeError(f"Data file purge failed: {e}") from e

    result = {"db_rows_deleted": deleted, "file_rows_removed": file_rows_removed}
    if result["file_rows_removed"] == 0 and result["db_rows_deleted"] == 0:
        raise PurgeError(f"No records found for user_id: {user_id}")

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
