import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

class AMLAnomalyDetector:
    """
    Supervised Streaming Classifier using a Balanced Random Forest architecture.
    Learns explicit behavioral signatures of deceptive laundering profiles.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_cols = [
            'Amount', 
            'tx_count_1h', 
            'tx_amount_sum_1h', 
            'tx_count_24h', 
            'beneficiary_diversity_24h', 
            'pass_through_ratio_1h'
        ]
        # class_weight='balanced' forces the trees to heavily penalize missing the rare laundering cases
        self.model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_trained = False

    def _preprocess(self, df_or_dict) -> pd.DataFrame:
        """Applies mathematical log-scaling to monetary volumes."""
        if isinstance(df_or_dict, dict):
            X = pd.DataFrame([df_or_dict])[self.feature_cols].copy()
        else:
            X = df_or_dict[self.feature_cols].copy()
            
        X['Amount'] = np.log1p(X['Amount'].astype(float))
        X['tx_amount_sum_1h'] = np.log1p(X['tx_amount_sum_1h'].astype(float))
        return X

    def train(self, feature_dataframe) -> None:
        """Trains the supervised engine using feature vectors and historical labels."""
        if 'Is_laundering' not in feature_dataframe.columns:
            raise KeyError("Supervised training requires the target label column 'Is_laundering'.")
            
        print(f"Training Supervised Classifier across features: {self.feature_cols}")
        X_scaled = self._preprocess(feature_dataframe)
        y = feature_dataframe['Is_laundering'].astype(int)
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print("✅ Supervised AML model trained successfully.")

    def score_transaction(self, enriched_tx: dict) -> tuple:
        """Evaluates a single stateful transaction payload. Returns: (laundering_probability, prediction)"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained on a baseline before running real-time scoring.")
        
        X_row = self._preprocess(enriched_tx)
        
        # Extract the probability score of the transaction being laundering (class 1)
        prob = float(self.model.predict_proba(X_row)[0][1])
        pred = int(self.model.predict(X_row)[0])
        
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