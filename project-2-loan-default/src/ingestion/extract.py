"""
Ingestion & Inbound Normalization Module - Project 2: Loan Default Prediction
Handles applicant profile matrix ingestion and data type downcasting.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

class LoanDataIngestionEngine:
    def __init__(self, raw_data_path: str = None):
        self.raw_data_path = raw_data_path
        print("🏦 Initializing Loan Default Ingestion & Responsible AI Preprocessing Engine...")

    def generate_mock_applicant_pool(self, sample_size: int = 1000) -> pd.DataFrame:
        """Synthesizes an applicant matrix to establish data contracts without data gravity gaps."""
        np.random.seed(42)
        print(f"📊 Synthesizing {sample_size} applicant evaluation profiles...")
        
        data = {
            "applicant_id": range(1000, 1000 + sample_size),
            "credit_score": np.random.randint(300, 850, size=sample_size),
            "annual_income": np.random.uniform(20000, 250000, size=sample_size),
            "debt_to_income_ratio": np.random.uniform(0.1, 0.6, size=sample_size),
            "employment_length_years": np.random.randint(0, 25, size=sample_size),
            "age_proxy_group": np.random.choice([0, 1], size=sample_size, p=[0.3, 0.7]), # Protected Proxy Attribute
            "loan_default_status": np.random.choice([0, 1], size=sample_size, p=[0.85, 0.15])
        }
        return pd.DataFrame(data)

    def optimize_and_split(self, df: pd.DataFrame, output_dir: Path):
        """Downcasts types for memory efficiency and exports splits to columnar storage."""
        print("⚡ Executing type downcasting optimizations...")
        
        # Optimize schemas
        df["credit_score"] = df["credit_score"].astype("int16")
        df["annual_income"] = df["annual_income"].astype("float32")
        df["debt_to_income_ratio"] = df["debt_to_income_ratio"].astype("float32")
        df["employment_length_years"] = df["employment_length_years"].astype("int8")
        df["age_proxy_group"] = df["age_proxy_group"].astype("int8")
        df["loan_default_status"] = df["loan_default_status"].astype("int8")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Stratified split simulation
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        
        train_df.to_parquet(output_dir / "train.parquet", index=False)
        test_df.to_parquet(output_dir / "test.parquet", index=False)
        print(f"💾 Clean partitions successfully serialized to: {output_dir}")

if __name__ == "__main__":
    base_path = Path(__file__).resolve().parents[2]
    engine = LoanDataIngestionEngine()
    raw_df = engine.generate_mock_applicant_pool()
    engine.optimize_and_split(raw_df, base_path / "data" / "processed")