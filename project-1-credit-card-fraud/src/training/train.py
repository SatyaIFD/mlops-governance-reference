"""
Model Training & Orchestration Module
Executes hyperparameter runs and registers state models to the SQLite metadata registry.
"""

from pathlib import Path
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient


def execute_training_pipeline(processed_data_dir: str, tracking_uri: str):
    """Loads preprocessed parquet files, trains an XGBoost model, and registers it to MLflow."""
    mlflow.set_tracking_uri(tracking_uri)

    data_path = Path(processed_data_dir).resolve()
    project_root = (
        data_path.parents[1] if "data" in data_path.parts else data_path.parent
    )

    mlruns_dir = project_root / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = "credit_card_fraud_governance"
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        client.create_experiment(experiment_name, artifact_location=mlruns_dir.as_uri())
    mlflow.set_experiment(experiment_name)

    # DEFENSIVE CI/CD FIX: Hard fail if data is missing instead of silently mocking it
    train_file = data_path / "train.parquet"
    test_file = data_path / "test.parquet"
    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"🚨 FATAL: Processed data missing at {data_path}. Upstream ingestion failed."
        )

    train_df = pd.read_parquet(train_file)
    test_df = pd.read_parquet(test_file)

    X_train = train_df.drop(columns=["Class"])
    y_train = train_df["Class"]
    X_test = test_df.drop(columns=["Class"])
    y_test = test_df["Class"]

    with mlflow.start_run() as run:
        print(f"🚀 Training model inside active run: {run.info.run_id}")

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
        print(f"📊 Run Accuracy: {accuracy:.4f}")

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="credit_card_fraud_model",
        )
        print("🟢 Model binaries successfully logged and registered.")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[2]
    execute_training_pipeline(
        processed_data_dir=str(BASE_DIR / "data/processed"),
        tracking_uri=f"sqlite:///{BASE_DIR}/mlflow.db",
    )
