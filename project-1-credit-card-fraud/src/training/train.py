import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import mlflow
import mlflow.xgboost

# 1. Capture the exact working directory to align system variables
cwd = Path.cwd()
if (cwd / "project-1-credit-card-fraud").exists():
    ROOT_DIR = cwd
else:
    ROOT_DIR = Path(__file__).resolve().parents[3]

PROJECT_DIR = ROOT_DIR / "project-1-credit-card-fraud"
PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed_features.parquet"

# 2. Inject PROJECT_DIR into sys.path so Python maps 'src/' instantly
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# =====================================================================
# IMPORTS ENFORCED SAFELY NOW THAT SYS.PATH IS ALIGNED
# =====================================================================
from src.training.data_splitter import prepare_stratified_splits

def train_production_model():
    print("🏋️ Initializing Automated ML Training Pipeline Engine...")

    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Processed data absent at: {PROCESSED_DATA_PATH.resolve()}. Run ingestion first.")

    # 1. Load Parquet State Matrix
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    
    # 2. Segment Data via Production Splitter Module
    X_train, X_test, y_train, y_test = prepare_stratified_splits(df)

    # 3. Calculate Objective Class Balancing Weight (Scale Pos Weight)
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_weight = neg_count / pos_count
    print(f"⚖️ Loss optimization multiplier locked: {scale_weight:.4f}")

    # 4. Define Hyperparameters evaluated in Experimentation Notebook
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_weight,
        "max_depth": 5,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "random_state": 42
    }

    # 5. Initialize MLflow Experiment Context
    mlflow.set_experiment("credit_card_fraud_governance")
    
    with mlflow.start_run(run_name="xgboost_balanced_production"):
        print("🚀 Executing XGBoost training routine with active MLflow tracing...")
        
        # Log Hyperparameters & Operational Lineage Metadata
        mlflow.log_params(params)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("test_samples", len(X_test))

        # Train Classifier
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        # Generate Test Metrics
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Calculate Compliance Evaluation Matrix Metrics
        metrics = {
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        print("\n📈 Evaluation Matrix Metrics successfully calculated:")
        for name, score in metrics.items():
            print(f" - {name.upper()}: {score:.4f}")
            mlflow.log_metric(name, score)

        # Log Model Artifact with Input Schema Signature to Registry
        mlflow.xgboost.log_model(
            xgb_model=model, 
            artifact_path="model", 
            registered_model_name="fraud_detection_xgb"
        )
        print("\n📦 Model architecture binary successfully locked in MLflow Central Registry.")

if __name__ == "__main__":
    train_production_model()