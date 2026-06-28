"""
Data Splitting Utility - Project 2: Loan Default Prediction
Handles reproducible training, validation, and testing dataset partitioning.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_and_save_dataset(raw_csv_path: Path, output_dir: Path, test_size: float = 0.2, random_state: int = 42):
    """Splits a raw dataset into train and test splits, exporting them as Parquet format."""
    print(f"📦 Reading dataset for partitioning: {raw_csv_path}")
    df = pd.read_csv(raw_csv_path)
    
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"💾 Split complete: Train rows={len(train_df)}, Test rows={len(test_df)}")
    return train_path, test_path

if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_DIR.parents[1]
    
    split_and_save_dataset(
        raw_csv_path=PROJECT_ROOT / "data" / "loan-default.csv",
        output_dir=PROJECT_ROOT / "data" / "processed"
    )