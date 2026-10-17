import sqlite3
import pandas as pd
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def execute_right_to_be_forgotten(user_id: str, db_path: str, data_path: str):
    """
    GDPR Compliance: Permanently purges a specific user's PII
    from MLflow tracking databases and raw batch data files.
    """
    logging.info(f"Initiating GDPR protocol for User ID: {user_id}")

    if Path(db_path).exists():
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM params WHERE key='user_id' AND value=?", (user_id,)
            )
            conn.commit()
            conn.close()
            logging.info("✅ User scrubbed from MLflow tracking database.")
        except Exception as e:
            logging.error(f"Failed to scrub database: {e}")

    if Path(data_path).exists():
        try:
            # CRITICAL FIX: Enforce string type to prevent ID '00000' becoming integer 0
            df = pd.read_csv(data_path, dtype={"user_id": str})
            if "user_id" in df.columns:
                initial_count = len(df)
                df = df[df["user_id"] != str(user_id)]
                df.to_csv(data_path, index=False)
                logging.info(
                    f"✅ Removed {initial_count - len(df)} records from {data_path}."
                )
        except Exception as e:
            logging.error(f"Failed to scrub data file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDPR Data Purge Utility")
    parser.add_argument("--user_id", required=True)
    parser.add_argument("--db_path", required=True)
    parser.add_argument("--data_path", required=True)
    args = parser.parse_args()
    execute_right_to_be_forgotten(args.user_id, args.db_path, args.data_path)
