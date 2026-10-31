import csv
import os
import time
import shutil
from datetime import datetime, timedelta
from pathlib import Path


def _write_synthetic_saml_fallback(csv_path: Path, num_rows: int = 1000) -> None:
    """Generates a deterministic SAML-D schema CSV when running in headless containers without Kaggle keys."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    base_ts = datetime(2025, 1, 1, 8, 0, 0)
    fieldnames = [
        "Time",
        "Date",
        "Sender_account",
        "Receiver_account",
        "Amount",
        "Payment_currency",
        "Received_currency",
        "Sender_bank_location",
        "Receiver_bank_location",
        "Payment_type",
        "Is_laundering",
    ]
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(num_rows):
            ts = base_ts + timedelta(seconds=i * 15)
            is_laundering = 1 if (i % 50 == 0 and i > 0) else 0
            amount = 9250.00 if is_laundering else round(75.0 + (i % 25) * 40.0, 2)
            writer.writerow(
                {
                    "Time": ts.strftime("%H:%M:%S"),
                    "Date": ts.strftime("%Y-%m-%d"),
                    "Sender_account": f"ACC_{(i % 40) + 1000}",
                    "Receiver_account": f"ACC_{((i + 7) % 40) + 2000}",
                    "Amount": amount,
                    "Payment_currency": "USD",
                    "Received_currency": "USD",
                    "Sender_bank_location": "US",
                    "Receiver_bank_location": "US",
                    "Payment_type": "WIRE" if is_laundering else "ACH",
                    "Is_laundering": is_laundering,
                }
            )


def ensure_dataset_exists(csv_path: Path):
    """Automatically fetches the SAML-D dataset from Kaggle if missing."""
    if csv_path.is_file():
        return

    print(f"\n📦 Dataset missing at {csv_path}. Initiating automatic download...")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Try kagglehub auto-fetch
    try:
        import kagglehub

        print("Downloading SAML-D dataset via kagglehub...")
        download_dir = kagglehub.dataset_download(
            "berkanoztas/synthetic-transaction-monitoring-dataset-aml"
        )
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
        api.dataset_download_files(
            "berkanoztas/synthetic-transaction-monitoring-dataset-aml",
            path=str(csv_path.parent),
            unzip=True,
        )
        if csv_path.is_file():
            print(f"✅ Successfully downloaded SAML-D.csv to {csv_path}")
            return
    except Exception as e:
        print(f"⚠️ Kaggle API download failed ({e}).")

    # 3. Opt-in synthetic fallback for headless container environments
    if os.getenv("AML_SYNTHETIC_STREAM_FALLBACK", "0") == "1":
        print(
            f"⚙️ AML_SYNTHETIC_STREAM_FALLBACK=1 enabled: generating deterministic SAML-D stream at {csv_path}..."
        )
        _write_synthetic_saml_fallback(csv_path)
        return

    raise FileNotFoundError(
        f"Missing base transaction data ledger at: {csv_path}\n"
        f"Automatic fetch requires 'kagglehub' or Kaggle credentials (~/.kaggle/kaggle.json).\n"
        f"Please run 'uv pip install kagglehub' or place SAML-D.csv manually into {csv_path.parent}."
    )


def transaction_stream_generator(csv_path: Path, delay_seconds: float = 0.0):
    """Yields live transactions from SAML-D dataset row-by-row as a streaming simulation."""
    ensure_dataset_exists(csv_path)

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if delay_seconds > 0:
                time.sleep(delay_seconds)

            normalized_row = row.copy()

            if "Timestamp" not in normalized_row or not normalized_row["Timestamp"]:
                date_val = normalized_row.get("Date", "")
                time_val = normalized_row.get("Time", "")
                if date_val and time_val:
                    normalized_row["Timestamp"] = f"{date_val} {time_val}"
                elif date_val:
                    normalized_row["Timestamp"] = str(date_val)
                elif time_val:
                    normalized_row["Timestamp"] = str(time_val)

            yield normalized_row
