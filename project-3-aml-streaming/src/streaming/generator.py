import csv
import time
import shutil
from pathlib import Path

def ensure_dataset_exists(csv_path: Path):
    """Automatically fetches the SAML-D dataset from Kaggle if missing."""
    if csv_path.exists():
        return

    print(f"\n📦 Dataset missing at {csv_path}. Initiating automatic download...")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Try kagglehub auto-fetch
    try:
        import kagglehub
        print("Downloading SAML-D dataset via kagglehub...")
        download_dir = kagglehub.dataset_download("berkanoztas/synthetic-transaction-monitoring-dataset-aml")
        extracted_csv = Path(download_dir) / "SAML-D.csv"
        if extracted_csv.exists():
            shutil.copy(extracted_csv, csv_path)
            print(f"✅ Successfully downloaded SAML-D.csv to {csv_path}")
            return
    except Exception as e:
        print(f"ℹ️ kagglehub download skipped/failed ({e}). Trying Kaggle API...")

    # 2. Try Kaggle API fallback
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files("berkanoztas/synthetic-transaction-monitoring-dataset-aml", path=str(csv_path.parent), unzip=True)
        print(f"✅ Successfully downloaded SAML-D.csv to {csv_path}")
        return
    except Exception as e:
        print(f"⚠️ Kaggle API download failed ({e}).")

    raise FileNotFoundError(
        f"Missing base transaction data ledger at: {csv_path}\n"
        f"Automatic fetch requires 'kagglehub' or Kaggle credentials (~/.kaggle/kaggle.json).\n"
        f"Please run 'uv pip install kagglehub' or place SAML-D.csv manually into {csv_path.parent}."
    )

def transaction_stream_generator(csv_path: Path, delay_seconds: float = 0.0):
    """Yields live transactions from SAML-D dataset row-by-row as a streaming simulation."""
    ensure_dataset_exists(csv_path)

    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if delay_seconds > 0:
                time.sleep(delay_seconds)

            # Normalize raw CSV schema for the ingestion contract
            normalized_row = row.copy()

            # SAML-D stores timestamp in separate 'Date' and 'Time' columns
            if 'Timestamp' not in normalized_row or not normalized_row['Timestamp']:
                date_val = normalized_row.get('Date', '')
                time_val = normalized_row.get('Time', '')
                if date_val and time_val:
                    normalized_row['Timestamp'] = f"{date_val} {time_val}"
                elif date_val:
                    normalized_row['Timestamp'] = str(date_val)
                elif time_val:
                    normalized_row['Timestamp'] = str(time_val)

            yield normalized_row