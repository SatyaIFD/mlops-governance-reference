import os
import sys
import zipfile
import pandas as pd
from pathlib import Path

# Force Python execution space to dynamically recognize our root package structure
def get_project_root() -> Path:
    if "ipykernel" in sys.modules or "__file__" not in globals():
        return Path(os.getcwd()).resolve()
    current_dir = Path(__file__).resolve().parent
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "requirements.txt").exists() or (parent / ".git").exists():
            return parent
    return Path(__file__).resolve().parents[2]

ROOT_DIR = get_project_root()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Absolute internal package path mapping
from src.ingestion.schemas import TransactionInputSchema, ProcessedTransactionSchema
from src.ingestion.utils import clean_and_transform_features

# Base operational layout paths matching the repository structure
DATA_DIR = ROOT_DIR / "project-1-credit-card-fraud" / "data"
KAGGLE_JSON = DATA_DIR / "kaggle.json"
RAW_DATA_PATH = DATA_DIR / "creditcard.csv"
ZIP_DATA_PATH = DATA_DIR / "creditcardfraud.zip"
PROCESSED_OUTPUT_PATH = DATA_DIR / "processed_features.parquet"

def download_raw_data():
    """
    Robust download layer with path checking, dependency management, 
    and environment cleaning.
    """
    # Force creation of data folder if it doesn't exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Fast-pass if raw file is unzipped and ready
    if RAW_DATA_PATH.exists() and RAW_DATA_PATH.stat().st_size > 0:
        print(f"📦 Valid raw dataset found at: {RAW_DATA_PATH.name}. Skipping download.")
        return

    print("🔍 Target data missing. Initializing system checks...")

    # 2. Hard Stop if Credentials are missing
    if not KAGGLE_JSON.exists():
        raise FileNotFoundError(
            f"\n🚨 [AUTH ERROR] Kaggle credentials missing!\n"
            f"Please place your 'kaggle.json' file here: {KAGGLE_JSON.resolve()}"
        )

    # 3. Secure File Permissions (Crucial requirement for Kaggle API)
    try:
        if os.name != 'nt':  # Linux / macOS
            os.chmod(KAGGLE_JSON, 0o600)
    except Exception as e:
        print(f"⚠️ Warning: Could not adjust permissions on kaggle.json: {e}")

    # 4. Inject Configurations Safely into Session memory
    os.environ['KAGGLE_CONFIG_DIR'] = str(DATA_DIR)

    # 5. Execute API Call with manual extraction handling
    try:
        import kaggle
        print("📥 Requesting 'mlg-ulb/creditcardfraud' from Kaggle servers...")
        
        # Download without unzipping first to cleanly catch mid-download disruptions
        kaggle.api.dataset_download_files(
            "mlg-ulb/creditcardfraud", 
            path=str(DATA_DIR), 
            unzip=False
        )

        # 6. Safe Zip Extraction Loop
        if ZIP_DATA_PATH.exists():
            print(f"🔩 Extracting {ZIP_DATA_PATH.name} payload...")
            with zipfile.ZipFile(ZIP_DATA_PATH, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            
            # Clean up the zip file to save storage space inside your repository
            ZIP_DATA_PATH.unlink()
            print("🗑️ Cleaned up compressed archive downloads.")

        # Final verification check
        if RAW_DATA_PATH.exists() and RAW_DATA_PATH.stat().st_size > 0:
            print(f"✅ Success! Ingestion target configured: {RAW_DATA_PATH.resolve()}")
        else:
            raise FileNotFoundError("Dataset downloaded, but target 'creditcard.csv' not found inside archive.")

    except Exception as error:
        # Self-healing: Delete corrupted fragments on failure so the next run starts fresh
        if ZIP_DATA_PATH.exists():
            ZIP_DATA_PATH.unlink()
        raise RuntimeError(f"❌ Ingestion Pipeline Failure: {error}") from error

def run_ingestion_pipeline():
    """
    Executes the ingestion lifecycle stages: download, schema enforcement, 
    feature transformation, and structured output serialization.
    """
    print("🚀 Starting Production Ingestion Pipeline Sequence...")
    
    # 1. Trigger the Secure Data Ingestion/Download Engine
    download_raw_data()
        
    # 2. Data stream ingestion
    print(f"📖 Reading raw matrix data: {RAW_DATA_PATH.name}...")
    raw_df = pd.read_csv(RAW_DATA_PATH)
    
    # 3. Governance Schema Validation Contract (Input Check)
    print("🛡️ Evaluating raw ingestion payloads against TransactionInputSchema parameters...")
    sample_payloads = raw_df.drop('Class', axis=1, errors='ignore').head(100).to_dict(orient='records')
    try:
        for payload in sample_payloads:
            TransactionInputSchema(**payload)
        print("✅ Entry payload schema contract: VALID.")
    except Exception as validation_err:
        raise ValueError(f"🚨 Ingestion block structural validation breach: {validation_err}")

    # 4. Feature Extraction, duplicate cleansing, and scaling mutations
    print("⚙️ Initiating feature scaling transformations...")
    processed_df = clean_and_transform_features(raw_df)
    
    # 5. Downstream Compliance Contract Check (Output Check)
    print("🛡️ Evaluating processed dataset weights against ProcessedTransactionSchema parameters...")
    output_sample = processed_df.head(1).to_dict(orient='records')[0]
    try:
        ProcessedTransactionSchema(**output_sample)
        print("✅ Post-processing schema compliance: VALID.")
    except Exception as final_validation_err:
        raise ValueError(f"🚨 Downstream model configuration schema mismatch anomaly: {final_validation_err}")

    # 6. Save State via Engineered Parquet Serialization
    PROCESSED_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"💾 Serializing optimized features to: {PROCESSED_OUTPUT_PATH.resolve()}")
    processed_df.to_parquet(PROCESSED_OUTPUT_PATH, index=False, engine='pyarrow')
    
    print("🏁 Ingestion Pipeline execution sequence completed cleanly with zero exceptions.")

if __name__ == "__main__":
    run_ingestion_pipeline()