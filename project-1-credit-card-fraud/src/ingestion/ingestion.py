import os
import sys
import zipfile
from pathlib import Path
import pandas as pd

# 1. Capture the exact terminal workspace root directory
cwd = Path.cwd()
if (cwd / "project-1-credit-card-fraud").exists():
    ROOT_DIR = cwd
else:
    ROOT_DIR = Path(__file__).resolve().parents[3]

# 2. Derive the explicit Project 1 directory boundary path
PROJECT_DIR = ROOT_DIR / "project-1-credit-card-fraud"
DATA_DIR = PROJECT_DIR / "data"

# 3. Inject PROJECT_DIR into sys.path so Python finds 'src/' immediately inside it
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# 4. Map data file targets explicitly
KAGGLE_JSON = DATA_DIR / "kaggle.json"
RAW_DATA_PATH = DATA_DIR / "creditcard.csv"
ZIP_DATA_PATH = DATA_DIR / "creditcardfraud.zip"
PROCESSED_OUTPUT_PATH = DATA_DIR / "processed_features.parquet"

print("\n================== PATH ALIGNMENT SUCCESS ==================")
print(f"Repository Root Directory:   {ROOT_DIR}")
print(f"Python Package Injection:    {PROJECT_DIR}")
print(f"Data Target Directory Path:  {DATA_DIR}")
print("============================================================\n")

# =====================================================================
# IMPORTS ENFORCED SAFELY NOW THAT SYS.PATH TARGETS PROJECT_DIR
# =====================================================================
from src.ingestion.schemas import TransactionInputSchema, ProcessedTransactionSchema
from src.ingestion.utils import clean_and_transform_features

def download_raw_data():
    """Robust download layer with strict absolute path routing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists() and RAW_DATA_PATH.stat().st_size > 0:
        print(f"📦 Valid raw dataset found at: {RAW_DATA_PATH.name}. Skipping download.")
        return

    print("🔍 Target data missing. Initializing system checks...")

    if not KAGGLE_JSON.exists():
        raise FileNotFoundError(
            f"\n🚨 [AUTH ERROR] Kaggle credentials missing!\n"
            f"Please place your 'kaggle.json' file here: {KAGGLE_JSON.resolve()}"
        )

    try:
        if os.name != 'nt':  # Linux / macOS
            os.chmod(KAGGLE_JSON, 0o600)
    except Exception as e:
        print(f"⚠️ Warning: Could not adjust permissions on kaggle.json: {e}")

    os.environ['KAGGLE_CONFIG_DIR'] = str(DATA_DIR.resolve())

    try:
        import kaggle
        print("📥 Requesting 'mlg-ulb/creditcardfraud' from Kaggle servers...")
        
        kaggle.api.dataset_download_files(
            "mlg-ulb/creditcardfraud", 
            path=str(DATA_DIR.resolve()), 
            unzip=False
        )

        if ZIP_DATA_PATH.exists():
            print(f"🔩 Extracting {ZIP_DATA_PATH.name} payload...")
            with zipfile.ZipFile(ZIP_DATA_PATH, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            ZIP_DATA_PATH.unlink()
            print("🗑️ Cleaned up compressed archive downloads.")

        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError("Dataset downloaded, but target 'creditcard.csv' not found inside archive.")

    except Exception as error:
        if ZIP_DATA_PATH.exists():
            ZIP_DATA_PATH.unlink()
        raise RuntimeError(f"❌ Ingestion Pipeline Failure: {error}") from error

def run_ingestion_pipeline():
    """Executes the ingestion lifecycle stages."""
    print("🚀 Starting Production Ingestion Pipeline Sequence...")
    download_raw_data()
        
    print(f"📖 Reading raw matrix data from: {RAW_DATA_PATH.resolve()}...")
    raw_df = pd.read_csv(RAW_DATA_PATH)
    
    print("🛡️ Evaluating raw ingestion payloads against TransactionInputSchema parameters...")
    sample_payloads = raw_df.drop('Class', axis=1, errors='ignore').head(100).to_dict(orient='records')
    try:
        for payload in sample_payloads:
            TransactionInputSchema(**payload)
        print("✅ Entry payload schema contract: VALID.")
    except Exception as validation_err:
        raise ValueError(f"🚨 Ingestion block structural validation breach: {validation_err}")

    print("⚙️ Initiating feature scaling transformations...")
    processed_df = clean_and_transform_features(raw_df)
    
    print("🛡️ Evaluating processed dataset weights against ProcessedTransactionSchema parameters...")
    output_sample = processed_df.head(1).to_dict(orient='records')[0]
    try:
        ProcessedTransactionSchema(**output_sample)
        print("✅ Post-processing schema compliance: VALID.")
    except Exception as final_validation_err:
        raise ValueError(f"🚨 Downstream model configuration schema mismatch anomaly: {final_validation_err}")

    print(f"💾 Serializing optimized features to: {PROCESSED_OUTPUT_PATH.resolve()}")
    processed_df.to_parquet(PROCESSED_OUTPUT_PATH, index=False, engine='pyarrow')
    print("🏁 Ingestion Pipeline execution sequence completed cleanly with zero exceptions.")

if __name__ == "__main__":
    run_ingestion_pipeline()