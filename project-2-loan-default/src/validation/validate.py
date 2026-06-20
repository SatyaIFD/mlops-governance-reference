"""
Data Validation Layer - Project 2: Loan Default Prediction
Validates incoming data schemas, continuous boundary limits, and schema structural alignment.
"""

import sys
from pathlib import Path
import pandas as pd

class LoanDataValidator:
    def __init__(self, expected_schema: dict):
        self.expected_schema = expected_schema

    def validate_dataset(self, file_path: Path) -> bool:
        """Validates schema structure, missing data bounds, and real business rules."""
        print(f"🧐 Validating structural integrity of: {file_path.name}")
        
        if not file_path.exists():
            print(f"❌ Validation Failed: Target file {file_path} does not exist.")
            return False
            
        df = pd.read_parquet(file_path)
        
        # 1. Structural Check: Verify all columns exist and match expected types
        for col, expected_type in self.expected_schema.items():
            if col not in df.columns:
                print(f"❌ Validation Failed: Missing required column '{col}'")
                return False
                
        # 2. Completeness Check: Ensure zero missing data profiles exist
        null_counts = df.isnull().sum().sum()
        if null_counts > 0:
            print(f"❌ Validation Failed: {null_counts} unexpected null values detected.")
            return False
            
        # 3. Domain Business Boundary Checks
        # Credit Score must strictly respect financial boundaries
        if (df["CreditScore"].min() < 300) or (df["CreditScore"].max() > 850):
            print(f"❌ Validation Failed: Out-of-bounds Credit Scores found: [{df['CreditScore'].min()}, {df['CreditScore'].max()}]")
            return False
            
        # Age should correspond to legal applicant profiles
        if df["Age"].min() < 18:
            print(f"❌ Validation Failed: Underage applicant record detected: Min age is {df['Age'].min()}")
            return False
            
        print(f"✅ Schema contract valid! {len(df):,} rows verified with zero anomalies.\n")
        return True

if __name__ == "__main__":
    # Resolve project root pathing structure
    SRC_DIR = Path(__file__).resolve().parent
    BASE_DIR = SRC_DIR.parents[1]
    PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
    
    # Strictly define the feature data contract mapping derived during our EDA
    core_schema_contract = {
        "Age": "int8",
        "Income": "int32",
        "CreditScore": "int16",
        "LoanAmount": "int32",
        "DTIRatio": "float32",
        "MonthsEmployed": "int16",
        "Default": "int8"
    }
    
    validator = LoanDataValidator(expected_schema=core_schema_contract)
    
    # Execute validation audits across both data partitions
    train_ok = validator.validate_dataset(PROCESSED_DATA_DIR / "train.parquet")
    test_ok = validator.validate_dataset(PROCESSED_DATA_DIR / "test.parquet")
    
    if not (train_ok and test_ok):
        print("🚨 Data pipeline broken. Data contract checks failed.")
        sys.exit(1)
    else:
        print("🎉 All data partitions successfully cleared for downstream model ingestion.")