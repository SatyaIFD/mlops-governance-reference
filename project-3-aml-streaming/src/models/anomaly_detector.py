import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestClassifier


class AMLAnomalyDetector:
    """Supervised anomaly detector tuned for high-recall streaming AML interception."""

    def __init__(self):
        """Initializes detector with expanded stateful feature schema."""
        self.feature_cols: List[str] = [
            'Amount',
            'pass_through_ratio_1h',
            'is_structuring',
            'velocity_acceleration',
            'fan_out_count_24h',
            'receiver_inflow_count_1h',
            'receiver_inflow_amount_1h'
        ]
        self.model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight='balanced',
            random_state=42
        )

    def train(self, df: pd.DataFrame) -> None:
        """Trains Random Forest model on enriched training records."""
        X = df[self.feature_cols]
        y = df['Is_laundering'].astype(int)
        self.model.fit(X, y)

    def score_transaction(self, enriched_tx: Dict[str, Any]) -> Tuple[float, int]:
        """Scores an enriched transaction event in real-time."""
        feature_dict = {col: [float(enriched_tx.get(col, 0.0))] for col in self.feature_cols}
        X_single = pd.DataFrame(feature_dict)

        prob = float(self.model.predict_proba(X_single)[0][1])
        # Alert threshold set to 0.40 to optimize recall
        is_anomaly = 1 if prob >= 0.40 else 0
        return round(prob, 4), is_anomaly

    def save(self, artifact_path: Path) -> None:
        """Serializes trained model artifact to disk."""
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, artifact_path)

    def load(self, artifact_path: Path) -> None:
        """Loads pre-trained model artifact from disk."""
        self.model = joblib.load(artifact_path)