import joblib
import mlflow
import mlflow.pyfunc
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
from mlflow.tracking import MlflowClient


class AMLAnomalyDetector:
    """Supervised anomaly detection classifier tuned for streaming AML detection."""

    def __init__(self):
        """Initializes the detector with predefined feature column boundaries."""
        self.feature_cols: List[str] = [
            "Amount",
            "pass_through_ratio_1h",
            "receiver_inflow_count_1h",
            "receiver_inflow_amount_1h",
        ]
        self.model: RandomForestClassifier = None

    def train(self, df):
        pass  # Training logic omitted for inference refactor

    def score_transaction(self, enriched_tx: Dict[str, Any]) -> Tuple[float, int]:
        """Scores a single enriched transaction in real-time."""
        if not self.model:
            raise ValueError("Inference engine is uninitialized.")

        features = [[enriched_tx.get(col, 0.0) for col in self.feature_cols]]

        # Extract underlying sklearn model if wrapped by MLflow PyFunc
        model_to_use = self.model
        if hasattr(self.model, "_model_impl"):
            model_to_use = self.model._model_impl.sklearn_model

        score = model_to_use.predict_proba(features)[0][1]
        flag = 1 if score > 0.5 else 0
        return float(score), int(flag)

    def save(self, artifact_path: Path) -> None:
        """Serializes the trained model object to disk using joblib."""
        joblib.dump(self.model, artifact_path)

    def load(self, model_name: str = "aml_streaming_model") -> None:
        """Loads the latest production model artifact from the central MLflow registry."""
        # Resolve the absolute path to the root mlops-governance-reference directory
        root_dir = Path(__file__).resolve().parents[4]
        mlflow_db_path = root_dir / "mlflow.db"

        mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path}")

        client = MlflowClient()
        model_versions = client.search_model_versions(f"name='{model_name}'")
        latest_version = (
            max([int(mv.version) for mv in model_versions]) if model_versions else 1
        )

        # Load the PyFunc wrapper from the central registry
        self.model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version}")
