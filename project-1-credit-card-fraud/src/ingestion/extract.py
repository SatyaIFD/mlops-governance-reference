"""
Data Ingestion Module
Handles raw transactional database streaming splits into optimized Parquet formats.
"""

import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def run_ingestion(source_path: str, output_dir: str, test_size: float = 0.2, seed: int = 42):
    """Loads a raw dataset, optimizes types, and saves deterministic train/test parquet files."""
    print(f"📦 Extracting raw dataset from {source_path}...")
    
    # In a real pipeline, this connects to an upstream database or lake
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Missing raw data storage reference at {source_path}")
        
    df = pd.read_csv(source_path)
    
    # Basic data type downcasting to optimize memory footprints
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
        
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df['Class'] if 'Class' in df else None)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(out_path / "train.parquet", index=False)
    test_df.to_parquet(out_path / "test.parquet", index=False)
    print(f"🟢 Ingestion complete. Train/Test tensors compiled in: {output_dir}")

if __name__ == "__main__":
    # Standard entry point execution
    BASE_DIR = Path(__file__).resolve().parents[2]
    # Adjust this path if your raw creditcard.csv is named or stored differently
    run_ingestion(
        source_path=str(BASE_DIR / "data/creditcard.csv"),
        output_dir=str(BASE_DIR / "data/processed")
    )