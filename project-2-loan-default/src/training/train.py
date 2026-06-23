"""
Production Training Engine - Project 2: Loan Default Prediction
Trains the mitigated fair model using sample re-weighting and registers metrics via MLflow.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
import mlflow

class LoanModelTrainer:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.train_path = self.data_dir / "train.parquet"
        self.test_path = self.data_dir / "test.parquet"

    def calculate_mitigation_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Computes sample weights to neutralize historical age-based demographic parity bias."""
        N = len(df)
        n_fav = len(df[df['Default'] == 0])
        n_unfav = len(df[df['Default'] == 1])
        
        n_protected = len(df[df['Age'] < 30])
        n_baseline = len(df[df['Age'] >= 30])
        
        n_prot_fav = len(df[(df['Age'] < 30) & (df['Default'] == 0)])
        n_prot_unfav = len(df[(df['Age'] < 30) & (df['Default'] == 1)])
        n_base_fav = len(df[(df['Age'] >= 30) & (df['Default'] == 0)])
        n_base_unfav = len(df[(df['Age'] >= 30) & (df['Default'] == 1)])
        
        w_prot_fav = (n_protected * n_fav) / (N * n_prot_fav) if n_prot_fav > 0 else 1.0
        w_prot_unfav = (n_protected * n_unfav) / (N * n_prot_unfav) if n_prot_unfav > 0 else 1.0
        w_base_fav = (n_baseline * n_fav) / (N * n_base_fav) if n_base_fav > 0 else 1.0
        w_base_unfav = (n_baseline * n_unfav) / (N * n_base_unfav) if n_base_unfav > 0 else 1.0
        
        weights = np.ones(N)
        weights[(df['Age'] < 30) & (df['Default'] == 0)] = w_prot_fav
        weights[(df['Age'] < 30) & (df['Default'] == 1)] = w_prot_unfav
        weights[(df['Age'] >= 30) & (df['Default'] == 0)] = w_base_fav
        weights[(df['Age'] >= 30) & (df['Default'] == 1)] = w_base_unfav
        return weights

    def run_pipeline(self):
        """Processes matrices, executes training, and writes to the MLflow tracking registry."""
        print("📥 Ingesting validated Parquet frames...")
        train_df = pd.read_parquet(self.train_path)
        test_df = pd.read_parquet(self.test_path)
        
        # Strip unneeded columns
        for df in [train_df, test_df]:
            if 'LoanID' in df.columns:
                df.drop(columns=['LoanID'], inplace=True)
                
        X_train = train_df.drop(columns=['Default'])
        y_train = train_df['Default']
        X_test = test_df.drop(columns=['Default'])
        y_test = test_df['Default']
        
        # Establish protected indicators
        X_train['is_young'] = (X_train['Age'] < 30).astype(int)
        X_test['is_young'] = (X_test['Age'] < 30).astype(int)
        
        # Categorical processing
        cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
            X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
            X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
            
        X_train = X_train.astype({col: 'int8' for col in X_train.select_dtypes(include=['bool']).columns})
        X_test = X_test.astype({col: 'int8' for col in X_test.select_dtypes(include=['bool']).columns})
        
        # Core Mitigation Weights
        sample_weights = self.calculate_mitigation_weights(train_df)
        
        # Initialize MLflow tracking context
        mlflow.set_experiment("loan_default_risk_governance")
        with mlflow.start_run(run_name="mitigated_sample_reweighting"):
            print("🌲 Fitting Fair XGBoost pipeline...")
            model = XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=100)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            # Performance tracking
            train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
            test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            
            # Fairness Metric calculation
            preds = model.predict(X_test)
            df_audit = pd.DataFrame({'is_young': X_test['is_young'], 'approved': (preds == 0).astype(int)})
            approval_rates = df_audit.groupby('is_young')['approved'].mean()
            di_ratio = approval_rates[1] / approval_rates[0] if approval_rates[0] > 0 else 0
            
            print(f"📊 Run Completed: Test AUC = {test_auc:.4f} | Disparate Impact = {di_ratio:.4f}")
            
            # Log metrics to MLflow dashboard
            mlflow.log_param("model_type", "XGBClassifier")
            mlflow.log_param("mitigation_technique", "sample_reweighting")
            mlflow.log_metric("train_roc_auc", train_auc)
            mlflow.log_metric("test_roc_auc", test_auc)
            mlflow.log_metric("disparate_impact_ratio", di_ratio)
            
            # Register model binary artifact safely
            mlflow.xgboost.log_model(model, "fair_loan_model")
            print("💾 Production run completely tracked in system registry.")

if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent
    BASE_DIR = SRC_DIR.parents[1]
    DATA_DIR = BASE_DIR / "data" / "processed"
    
    trainer = LoanModelTrainer(data_dir=DATA_DIR)
    trainer.run_pipeline()