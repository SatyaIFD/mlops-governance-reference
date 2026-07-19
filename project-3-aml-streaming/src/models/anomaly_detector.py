import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

class AMLAnomalyDetector:
    """
    Supervised Streaming Classifier using a Controlled Downsampled Random Forest.
    Prevents high minority class weights from distorting decision tree splits.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_cols = [
            'Amount', 
            'tx_count_1h', 
            'tx_amount_sum_1h', 
            'tx_count_24h', 
            'beneficiary_diversity_24h', 
            'pass_through_ratio_1h',
            'receiver_inflow_count_1h',
            'receiver_inflow_sum_1h'
        ]
        self.model = None
        self.is_trained = False

    def _preprocess(self, df_or_dict) -> pd.DataFrame:
        """Applies mathematical log-scaling to all monetary volume metrics."""
        if isinstance(df_or_dict, dict):
            X = pd.DataFrame([df_or_dict])[self.feature_cols].copy()
        else:
            X = df_or_dict[self.feature_cols].copy()
            
        X['Amount'] = np.log1p(X['Amount'].astype(float))
        X['tx_amount_sum_1h'] = np.log1p(X['tx_amount_sum_1h'].astype(float))
        X['receiver_inflow_sum_1h'] = np.log1p(X['receiver_inflow_sum_1h'].astype(float))
        return X

    def train(self, feature_dataframe) -> None:
        if 'Is_laundering' not in feature_dataframe.columns:
            raise KeyError("Supervised training requires the target label column 'Is_laundering'.")
            
        print(f"Training Supervised Classifier across features: {self.feature_cols}")
        
        # 1. Implement Controlled Downsampling to balance the feature boundaries naturally
        normal_df = feature_dataframe[feature_dataframe['Is_laundering'] == 0]
        laundering_df = feature_dataframe[feature_dataframe['Is_laundering'] == 1]
        
        # Maintain a clean 10:1 ratio of normal-to-laundering data to stabilize node splits
        sample_size = min(len(normal_df), len(laundering_df) * 10)
        normal_sampled = normal_df.sample(n=sample_size, random_state=self.random_state)
        
        balanced_df = pd.concat([laundering_df, normal_sampled]).sample(frac=1.0, random_state=self.random_state)
        
        X_scaled = self._preprocess(balanced_df)
        y = balanced_df['Is_laundering'].astype(int)
        
        # 2. Re-initialize the model with strict depth constraints to eliminate cold-start memorization
        self.model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8, 
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print("✅ Supervised AML model trained successfully with structural downsampling balancing.")

    def score_transaction(self, enriched_tx: dict) -> tuple:
        if not self.is_trained:
            raise RuntimeError("Model must be trained on a baseline before running real-time scoring.")
        
        X_row = self._preprocess(enriched_tx)
        prob = float(self.model.predict_proba(X_row)[0][1])
        
        # ADJUSTMENT: Calibrate threshold to 0.40 to fit the downsampled distribution
        pred = 1 if prob >= 0.40 else 0
        
        return prob, pred

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, output_path)
        print(f"Artifact successfully saved to {output_path}")

    def load(self, input_path: Path) -> None:
        if not input_path.exists():
            raise FileNotFoundError(f"No model artifact found at {input_path}")
        self.model = joblib.load(input_path)
        self.is_trained = True
        print(f"✅ Model loaded from {input_path}")