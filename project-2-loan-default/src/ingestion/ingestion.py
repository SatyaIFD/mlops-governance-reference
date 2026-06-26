"""
Ingestion Orchestrator - Project 2: Loan Default Prediction
Coordinates data extraction and verifies columns/types against the strict schema contract.
"""

import sys
from pathlib import Path

# CRITICAL: Append path BEFORE running any internal src imports
INGEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INGEST_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.ingestion.extract import LoanDataIngestionEngine
from src.ingestion.schemas import RAW_LOAN_DATA_SCHEMA
from src.ingestion.utils import initialize_directories, get_project_root

# ... (rest of the file remains exactly the same)

def validate_raw_dataset(df: pd.DataFrame) -> bool:
    """Validates structural types and structural column alignments against our schema."""
    print("🧐 Checking data contract constraints...")
    
    # 1. Assert column completeness
    expected = set(RAW_LOAN_DATA_SCHEMA["expected_columns"])
    actual = set(df.columns)
    
    missing = expected - actual
    if missing:
        print(f"❌ Structural Validation Failure: Missing columns -> {missing}")
        return False
        
    # 2. Assert data type compliance
    for col, expected_type in RAW_LOAN_DATA_SCHEMA["numerical_types"].items():
        if col in df.columns:
            if not pd.api.types.is_dtype_equal(df[col].dtype, expected_type):
                try:
                    df[col] = df[col].astype(expected_type)
                except Exception as e:
                    print(f"❌ Type Cast Failure on column '{col}' to {expected_type}: {e}")
                    return False
                
    print("✅ Raw dataset matches data contract rules.")
    return True

def run_ingestion_pipeline():
    """Runs the full structural ingestion workflow."""
    initialize_directories()
    root = get_project_root()
    
    raw_path = root / "data" / "loan-default.csv"
    
    # If raw CSV is missing, invoke the object-oriented extractor class
    if not raw_path.exists():
        print("📥 Raw CSV data missing. Invoking extraction engine...")
        extractor = LoanDataIngestionEngine(data_dir=root / "data")
        extractor.fetch_and_extract()
        
    # Load raw file and validate schema
    df = pd.read_csv(raw_path)
    if not validate_raw_dataset(df):
        raise ValueError("❌ Ingestion halted: Data validation contract failed.")
        
    print(f"🎉 Data ingestion pipeline run complete! Total rows loaded: {len(df)}")

if __name__ == "__main__":
    run_ingestion_pipeline()