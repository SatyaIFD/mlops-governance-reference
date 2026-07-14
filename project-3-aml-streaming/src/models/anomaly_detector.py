import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from pathlib import Path

class AMLAnomalyDetector:
    """
    Unsupervised Anomaly Detection wrapper using an Isolation Forest.
    Evaluates spatial geometry of transactions to flag structural outliers.
    """
    def __init__(self, contamination: float = 0.01, random_state: int = 42):
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

    def _preprocess(self, df_or_dict) -> pd.DataFrame:
        """
        Applies mathematical log-scaling to monetary volumes to prevent 
        high-value clean transactions from dominating tree isolation paths.
        """
        if isinstance(df_or_dict, dict):
            X = pd.DataFrame([df_or_dict])[self.feature_cols].copy()
        else:
            X = df_or_dict[self.feature_cols].copy()
            
        # Compress heavy right-skewed cash metrics using natural log
        X['Amount'] = np.log1p(X['Amount'].astype(float))
        X['tx_amount_sum_1h'] = np.log1p(X['tx_amount_sum_1h'].astype(float))
        return X

    def train(self, feature_dataframe) -> None:
        """Trains the unsupervised engine on a preprocessed baseline matrix."""
        print(f"Training Isolation Forest across features: {self.feature_cols}")
        X_scaled = self._preprocess(feature_dataframe)
        self.model.fit(X_scaled)
        self.is_trained = True
        print("✅ Anomaly detection model trained successfully.")

    def score_transaction(self, enriched_tx: dict) -> tuple:
        """Evaluates a single stateful transaction payload using scaled vectors."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained on a baseline before running real-time scoring.")
        
        X_row = self._preprocess(enriched_tx)
        
        # Extract decisions
        score = float(self.model.decision_function(X_row)[0])
        pred = self.model.predict(X_row)[0]
        
        is_anomaly = 1 if pred == -1 else 0
        return score, is_anomaly

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