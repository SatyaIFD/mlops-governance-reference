import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

def clean_and_transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies production-grade scaling adjustments to raw transactional inputs.
    Removes structural duplicates and normalizes skewed parameters.
    """
    # 1. Self-Healing Copy
    processed_df = df.copy()
    
    # 2. Production Duplicate Purge (Matching exploration notebook)
    initial_rows = len(processed_df)
    processed_df.drop_duplicates(inplace=True)
    purged_rows = initial_rows - len(processed_df)
    if purged_rows > 0:
        print(f"🧹 Purged {purged_rows:,} exact duplicate entries from execution block.")
        
    # 3. Robust Scaling for Amount (Protects downstream XGBoost from heavy financial tails)
    robust_scaler = RobustScaler()
    processed_df['scaled_amount'] = robust_scaler.fit_transform(processed_df['Amount'].values.reshape(-1, 1))
    
    # 4. Standard Scaling for Time 
    standard_scaler = StandardScaler()
    processed_df['scaled_time'] = standard_scaler.fit_transform(processed_df['Time'].values.reshape(-1, 1))
    
    # 5. Column Pruning and Alignment Invariants
    processed_df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # Enforce exact column positioning: PCA space variables first, engineered weights next, target at the tail
    feature_cols = [col for col in processed_df.columns if col != 'Class']
    ordered_cols = feature_cols + ['Class']
    
    return processed_df[ordered_cols]