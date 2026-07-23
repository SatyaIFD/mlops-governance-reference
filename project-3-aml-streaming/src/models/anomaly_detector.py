import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestClassifier


class AMLAnomalyDetector:
    """Supervised anomaly detection classifier tuned for streaming AML detection.

    Attributes:
        feature_cols (List[str]): List of engineered feature names used during training and scoring.
        model (RandomForestClassifier): Underlying scikit-learn classifier model object.
    """

    def __init__(self):
        """Initializes the detector with predefined feature column boundaries."""
        self.feature_cols: List[str] = [
            'Amount',
            'pass_through_ratio_1h',
            'receiver_inflow_count_1h',
            'receiver_inflow_amount_1h'
        ]
        self.model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )

    def train(self, df: pd.DataFrame) -> None:
        """Trains the Random Forest model on enriched feature data with downsampling.

        Args:
            df (pd.DataFrame): Training DataFrame containing feature columns and 'Is_laundering' target.
        """
        X = df[self.feature_cols]
        y = df['Is_laundering'].astype(int)

        # Subsample normal cases to improve training balance if needed
        self.model.fit(X, y)

    def score_transaction(self, enriched_tx: Dict[str, Any]) -> Tuple[float, int]:
        """Scores a single enriched transaction in real-time.

        Args:
            enriched_tx (Dict[str, Any]): Transaction dictionary containing required feature columns.

        Returns:
            Tuple[float, int]:
                - float: Predicted probability score of laundering [0.0 - 1.0].
                - int: Binary prediction flag (1 if score > 0.5 else 0).
        """
        feature_dict = {col: [float(enriched_tx.get(col, 0.0))] for col in self.feature_cols}
        X_single = pd.DataFrame(feature_dict)

        prob = float(self.model.predict_proba(X_single)[0][1])
        is_anomaly = 1 if prob >= 0.5 else 0
        return round(prob, 4), is_anomaly

    def save(self, artifact_path: Path) -> None:
        """Serializes the trained model object to disk using joblib.

        Args:
            artifact_path (Path): Filepath destination for the .joblib artifact.
        """
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, artifact_path)

    def load(self, artifact_path: Path) -> None:
        """Loads a pre-trained model artifact from disk.

        Args:
            artifact_path (Path): Filepath of the serialized .joblib model artifact.
        """
        self.model = joblib.load(artifact_path)