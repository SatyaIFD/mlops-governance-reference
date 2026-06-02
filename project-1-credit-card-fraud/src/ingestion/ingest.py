import os
import sys
import zipfile
import shutil
from pathlib import Path

def get_project_root() -> Path:
    """Dynamically determines the project root across scripts, notebooks, and Docker."""
    # Handle interactive environments like Jupyter notebooks where __file__ is missing
    if "ipykernel" in sys.modules or "__file__" not in globals():
        current_dir = Path(os.getcwd()).resolve()
    else:
        current_dir = Path(__file__).resolve().parent

    # Traverse upward until finding a unique structural anchor file
    for parent in [current_dir] + list(current_dir.parents):
        if (parent / "requirements.txt").exists() or (parent / ".git").exists() or (parent / "src").exists():
            return parent
            
    # Fallback to safe structural routing if no anchor file is found
    return Path(__file__).resolve().parents[2]

# 1. Unified Directory Routing
ROOT_DIR = get_project_root()
DATA_DIR = ROOT_DIR / "data"
KAGGLE_JSON = DATA_DIR / "kaggle.json"
RAW_DATA_PATH = DATA_DIR / "creditcard.csv"
ZIP_DATA_PATH = DATA_DIR / "creditcardfraud.zip"  # Standard Kaggle download name

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
    # Sets file permissions to read/write only for the owner (600)
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
        
        # We download without unzipping first to cleanly catch mid-download disruptions
        kaggle.api.dataset_download_files(
            "mlg-ulb/creditcardfraud", 
            path=str(DATA_DIR), 
            unzip=False  # Handled manually next for greater reliability
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
