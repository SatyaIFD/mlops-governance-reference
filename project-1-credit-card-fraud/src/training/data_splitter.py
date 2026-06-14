import os
import sys
import pandas as pd
from typing import Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_stratified_splits(
    df: pd.DataFrame, 
    target_column: str = "Class", 
    test_size: float = 0.2, 
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the processed dataset into train and test sets while preserving
    the exact minority class ratio across both matrices to prevent data leakage.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print("✂️ Data segmented via production splitting layer:")
    print(f" - Train Shape: {X_train.shape[0]:,} rows | Fraud Ratio: {(y_train.sum()/len(y_train))*100:.3f}%")
    print(f" - Test Shape:  {X_test.shape[0]:,} rows | Fraud Ratio: {(y_test.sum()/len(y_test))*100:.3f}%")
    
    return X_train, X_test, y_train, y_test