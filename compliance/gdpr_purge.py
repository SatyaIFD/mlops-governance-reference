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
    logging.info(f"Initiating GDPR Right to be Forgotten protocol for User ID: {user_id}")
    
    # Purge from MLflow SQLite Database
    if Path(db_path).exists():
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            # Targets the MLflow params table where user_id might be logged
            cursor.execute("DELETE FROM params WHERE key='user_id' AND value=?", (user_id,))
            conn.commit()
            conn.close()
            logging.info("✅ User scrubbed from MLflow tracking database.")
        except Exception as e:
            logging.error(f"Failed to scrub database: {e}")
    else:
        logging.warning(f"Database not found at {db_path}. Skipping.")
    
    # Purge from raw Batch CSV data
    if Path(data_path).exists():
        try:
            df = pd.read_csv(data_path)
            if 'user_id' in df.columns:
                initial_count = len(df)
                df = df[df['user_id'] != user_id]
                df.to_csv(data_path, index=False)
                logging.info(f"✅ Removed {initial_count - len(df)} records from {data_path}.")
            else:
                logging.info("No 'user_id' column found in target data.")
        except Exception as e:
            logging.error(f"Failed to scrub data file: {e}")
    else:
        logging.warning(f"Data file not found at {data_path}. Skipping.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GDPR Data Purge Utility")
    parser.add_argument("--user_id", required=True, help="The exact User ID to purge")
    parser.add_argument("--db_path", required=True, help="Path to the MLflow SQLite DB")
    parser.add_argument("--data_path", required=True, help="Path to the raw batch CSV")
    args = parser.parse_args()
    
    execute_right_to_be_forgotten(args.user_id, args.db_path, args.data_path)
