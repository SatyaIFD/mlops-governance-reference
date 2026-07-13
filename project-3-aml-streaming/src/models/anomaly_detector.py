import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from pathlib import Path

class AMLAnomalyDetector:
    """
    Unsupervised Anomaly Detection wrapper using an Isolation Forest.
    Evaluates spatial geometry of transactions to flag structural outliers.
    """
    def __init__(self, contamination: float = 0.005, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.feature_cols = [
            'Amount', 
            'tx_count_1h', 
            'tx_amount_sum_1h', 
            'tx_count_24h', 
            'beneficiary_diversity_24h', 
            'pass_through_ratio_1h'
        ]
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_trained = False

    def train(self, feature_dataframe) -> None:
        """Trains the unsupervised engine on a baseline matrix of engineered features."""
        print(f"Training Isolation Forest across features: {self.feature_cols}")
        X = feature_dataframe[self.feature_cols]
        self.model.fit(X)
        self.is_trained = True
        print("✅ Anomaly detection model trained successfully.")

    def score_transaction(self, enriched_tx: dict) -> tuple:
        if not self.is_trained:
            raise RuntimeError("Model must be trained on a baseline before running real-time scoring.")
        
        # FIX: Construct a single-row DataFrame with explicit feature names to eliminate warnings
        X_row = pd.DataFrame([enriched_tx])[self.feature_cols]
        
        # decision_function returns lower values for high anomalies
        score = float(self.model.decision_function(X_row)[0])
        pred = self.model.predict(X_row)[0]
        
        # Sklearn maps anomalies to -1; we conform this to 1 for Anomaly, 0 for Normal
        is_anomaly = 1 if pred == -1 else 0
        
        return score, is_anomaly

    def save(self, output_path: Path) -> None:
        """Serializes the trained model artifacts to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, output_path)
        print(f"Artifact successfully saved to {output_path}")

    def load(self, input_path: Path) -> None:
        """Loads a pre-trained model artifact from disk."""
        if not input_path.exists():
            raise FileNotFoundError(f"No model artifact found at {input_path}")
        self.model = joblib.load(input_path)
        self.is_trained = True
        print(f"✅ Model loaded from {input_path}")