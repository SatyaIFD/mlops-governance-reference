"""
Model Training & Orchestration Module
Executes hyperparameter runs and registers state models to the SQLite metadata registry.
"""

import sys
from pathlib import Path
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# Inject project root to resolve src
BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.training.data_splitter import prepare_stratified_splits  # noqa: E402


def execute_training_pipeline(processed_data_file: str, tracking_uri: str):
    """Loads preprocessed parquet file, splits it, trains XGBoost, and registers to MLflow."""
    mlflow.set_tracking_uri(tracking_uri)

    data_path = Path(processed_data_file).resolve()
    project_root = (
        data_path.parents[1] if "data" in data_path.parts else data_path.parent
    )

    mlruns_dir = project_root.parent / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = "credit_card_fraud_governance"
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        client.create_experiment(experiment_name, artifact_location=mlruns_dir.as_uri())
    mlflow.set_experiment(experiment_name)

    if not data_path.exists():
        raise FileNotFoundError(f"🚨 FATAL: Processed data missing at {data_path}.")

    # Load the single processed dataset
    df = pd.read_parquet(data_path)

    # Split the data in-memory
    X_train, X_test, y_train, y_test = prepare_stratified_splits(df)

    with mlflow.start_run() as run:
        print(f"🚀 Training scaled model inside active run: {run.info.run_id}")

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
        }

        mlflow.log_params(params)

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        accuracy = float(model.score(X_test, y_test))
        mlflow.log_metric("accuracy", accuracy)
        print(f"📊 Scaled Run Accuracy: {accuracy:.4f}")

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="fraud_detection_xgb",
        )
        print("🟢 Model binaries successfully logged and registered.")


if __name__ == "__main__":
    execute_training_pipeline(
        processed_data_file=str(BASE_DIR / "data/processed_features.parquet"),
        tracking_uri=f"sqlite:///{BASE_DIR.parent}/mlflow.db",
    )
