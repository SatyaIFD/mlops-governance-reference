"""
Governance & Ethical Compliance Module
Programmatically computes localized Disparate Impact Ratios to audit model bias.
"""

import os
from pathlib import Path
import pandas as pd
import mlflow.pyfunc

def run_fairness_audit():
    # 1. Resolve master database path
    base_dir = Path(__file__).resolve().parents[2]
    mlflow.set_tracking_uri(f"sqlite:///{base_dir}/mlflow.db")
    
    # 2. Load the registered production model binary
    try:
        model_uri = "models:/credit_card_fraud_model/1"
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Load processed test data splits
    test_data_path = base_dir / "data" / "processed" / "test.parquet"
    if not test_data_path.exists():
        print(f"❌ Processed test data not found at {test_data_path}")
        return
        
    df = pd.read_parquet(test_data_path)
    
    # Separate features for model input matching exact naming contract
    feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    
    # Ensure the columns exist in the test file or handle fallback
    if "Time" not in df.columns or "Amount" not in df.columns:
        print("⚠️ Warning: Data contract columns missing in test data. Cannot compute exact fairness metrics.")
        return
        
    X_test = df[feature_names]
    
    # 4. Generate live predictions across the data frame
    print("⏳ Running batch inference across evaluation slices...")
    df["predictions"] = model.predict(X_test)
    
    # 5. Segment populations by proxy definitions (e.g., High-Value vs Low-Value transactions)
    # Define High-Value transactions as those above the 80th percentile of your dataset
    threshold = df["Amount"].quantile(0.80)
    
    high_value_group = df[df["Amount"] >= threshold]
    low_value_group = df[df["Amount"] < threshold]
    
    # Calculate selection (flagging) rates
    high_flag_rate = high_value_group["predictions"].mean()
    low_flag_rate = low_value_group["predictions"].mean()
    
    # Calculate Disparate Impact Ratio (Selection Rate of Unfavorable / Selection Rate of Baseline)
    # Handling potential division by zero safely
    if low_flag_rate == 0:
        disparate_impact_ratio = 0.0
    else:
        disparate_impact_ratio = high_flag_rate / low_flag_rate
        
    print("\n" + "="*50)
    print("⚖️ PROGRAMMATIC GOVERNANCE FAIRNESS AUDIT")
    print("="*50)
    print(f" ✅ High-Value Transaction Flag Rate : {high_flag_rate:.6f}")
    print(f" ✅ Low-Value Transaction Flag Rate  : {low_flag_rate:.6f}")
    print(f" ✅ Calculated Disparate Impact Ratio: {disparate_impact_ratio:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_fairness_audit()