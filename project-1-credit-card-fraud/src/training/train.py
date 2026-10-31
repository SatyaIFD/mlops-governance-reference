"""
Model Training & Orchestration Module
Fits preprocessing strictly on the training split when raw data is available,
persists scaler parameters, trains XGBoost, and registers to MLflow.
"""

import sys
from pathlib import Path
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from mlflow.tracking import MlflowClient

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.training.data_splitter import prepare_stratified_splits  # noqa: E402
from src.ingestion.utils import (  # noqa: E402
    CANONICAL_FEATURE_COLS,
    split_and_scale_without_leakage,
    save_scaler_params,
)


def execute_training_pipeline(processed_data_file: str, tracking_uri: str):
    """Loads dataset, prevents train/test scaling leakage, trains XGBoost, and registers to MLflow."""
    mlflow.set_tracking_uri(tracking_uri)

    data_path = Path(processed_data_file).resolve()
    project_root = (
        data_path.parents[1] if "data" in data_path.parts else data_path.parent
    )
    raw_csv_path = project_root / "data" / "creditcard.csv"
    scaler_artifact_path = project_root / "artifacts" / "scaler_params.json"
    processed_dir = project_root / "data" / "processed"

    mlruns_dir = project_root.parent / "mlruns"
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = "credit_card_fraud_governance"
    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        client.create_experiment(experiment_name, artifact_location=mlruns_dir.as_uri())
    mlflow.set_experiment(experiment_name)

    if raw_csv_path.exists():
        print(
            f"🔒 Splitting raw dataset ({raw_csv_path.name}) prior to scaling to prevent test leakage..."
        )
        raw_df = pd.read_csv(raw_csv_path)
        train_df, test_df, scaler_params = split_and_scale_without_leakage(raw_df)
        save_scaler_params(scaler_params, scaler_artifact_path)

        processed_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(processed_dir / "train.parquet", index=False)
        test_df.to_parquet(processed_dir / "test.parquet", index=False)

        X_train = train_df[CANONICAL_FEATURE_COLS]
        y_train = train_df["Class"]
        X_test = test_df[CANONICAL_FEATURE_COLS]
        y_test = test_df["Class"]
    else:
        if not data_path.exists():
            raise FileNotFoundError(f"🚨 FATAL: Processed data missing at {data_path}.")
        df = pd.read_parquet(data_path)
        X_train, X_test, y_train, y_test = prepare_stratified_splits(df)
        X_train = X_train[CANONICAL_FEATURE_COLS]
        X_test = X_test[CANONICAL_FEATURE_COLS]

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

        if scaler_artifact_path.exists():
            mlflow.log_artifact(
                str(scaler_artifact_path), artifact_path="preprocessing"
            )

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
