"""
Ingestion & Data Engineering Engine - Project 2: Loan Default Prediction
Handles native KaggleApi authentication, extraction, schema downcasting, and Parquet serialization.
"""

import os
import sys
import zipfile
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi

class LoanDataIngestionEngine:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.raw_data_path = self.data_dir / "loan-default.csv"
        self.zip_path = self.data_dir / "loan-default.zip"
        
        # Enforce Kaggle environment tracking parameters locally
        os.environ["KAGGLE_CONFIG_DIR"] = str(self.data_dir)

    def fetch_and_extract(self):
        """Authenticates with Kaggle, streams the raw zip payload, and handles system extraction."""
        print("🔌 Authenticating via native Kaggle API context...")
        api = KaggleApi()
        api.authenticate()
        
        print(f"📥 Downloading dataset nikhil1e9/loan-default to: {self.data_dir.name}/")
        api.dataset_download_files("nikhil1e9/loan-default", path=self.data_dir, quiet=False)
        
        print("🔓 Extracting compressed archive layers...")
        with zipfile.ZipFile(self.zip_path, "r") as z:
            z.extractall(self.data_dir)
            csv_candidates = [name for name in z.namelist() if name.lower().endswith(".csv")]
            
        if csv_candidates:
            src = self.data_dir / csv_candidates[0]
            if src.exists():
                src.rename(self.raw_data_path)
                
        # Clean up scratch storage artifacts immediately
        if self.zip_path.exists():
            self.zip_path.unlink()
            
        print(f"🚀 Success: {self.raw_data_path.name} standard footprint finalized.")

    def optimize_and_partition(self):
        """Optimizes numerical data types and exports stratified Parquet train/test splits."""
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"❌ Ingestion target missing at: {self.raw_data_path}")
            
        print(f"📖 Reading raw source data: {self.raw_data_path.name}")
        df = pd.read_csv(self.raw_data_path)
        
        print("⚡ Executing type-downcasting constraints...")
        # Optimize types to balance data memory gravity
        df['Age'] = df['Age'].astype('int8')
        df['Income'] = df['Income'].astype('int32')
        df['CreditScore'] = df['CreditScore'].astype('int16')
        df['LoanAmount'] = df['LoanAmount'].astype('int32')
        df['DTIRatio'] = df['DTIRatio'].astype('float32')
        df['MonthsEmployed'] = df['MonthsEmployed'].astype('int16')
        df['Default'] = df['Default'].astype('int8')
        
        # Stratified Train/Test partitioning based on the default target signature
        print("✂️ Generating stratified train/test partitions...")
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df['Default']
        )
        
        processed_dir = self.data_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Serialize to optimized columnar storage files
        train_df.to_parquet(processed_dir / "train.parquet", index=False)
        test_df.to_parquet(processed_dir / "test.parquet", index=False)
        print(f"💾 Optimized splits successfully serialized to: {processed_dir.name}/")

if __name__ == "__main__":
    # Dynamic home path resolution matching execution context
    SRC_DIR = Path(__file__).resolve().parent
    BASE_DIR = SRC_DIR.parents[1]
    DATA_TARGET_DIR = BASE_DIR / "data"
    
    engine = LoanDataIngestionEngine(data_dir=DATA_TARGET_DIR)
    engine.fetch_and_extract()
    engine.optimize_and_partition()