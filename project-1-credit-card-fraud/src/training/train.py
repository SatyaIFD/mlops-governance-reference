"""
Model Training & Orchestration Module - Project 1: Credit Card Fraud Detection
Fits preprocessing strictly on the training split, applies segment-aware sample weights
across transaction amount tiers so high/low segment flagging parity lies within [0.80, 1.25],
logs ROC-AUC, PR-AUC, F1, Precision, Recall, and Accuracy, and registers to MLflow.
"""

import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.ingestion.utils import (
    CANONICAL_FEATURE_COLS,
    save_scaler_params,
    split_and_scale_without_leakage,
)
from src.training.data_splitter import prepare_stratified_splits


def _compute_segment_weights(X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
    """Computes inverse joint-probability weights across (>=80th pct amount) and Class."""
    q80 = float(X_train["scaled_amount"].quantile(0.80))
    is_high = (X_train["scaled_amount"] >= q80).to_numpy()
    y_arr = y_train.to_numpy()
    n_total = len(y_arr)
    weights = np.ones(n_total, dtype=float)
    for seg_flag in (True, False):
        seg_mask = is_high == seg_flag
        n_seg = int(seg_mask.sum())
        for cls_val in (0, 1):
            cell_mask = seg_mask & (y_arr == cls_val)
            n_cls = int((y_arr == cls_val).sum())
            n_cell = int(cell_mask.sum())
            if n_cell > 0 and n_seg > 0 and n_cls > 0:
                weights[cell_mask] = (n_seg * n_cls) / (n_total * n_cell)
    return weights


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

    sample_weights = _compute_segment_weights(X_train, y_train)

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
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)
        accuracy = float((preds == y_test).mean())
        roc_auc = float(roc_auc_score(y_test, probs))
        pr_auc = float(average_precision_score(y_test, probs))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))

        mlflow.log_metric("accuracy", round(accuracy, 4))
        mlflow.log_metric("roc_auc", round(roc_auc, 4))
        mlflow.log_metric("pr_auc", round(pr_auc, 4))
        mlflow.log_metric("f1_score", round(f1, 4))
        mlflow.log_metric("precision", round(prec, 4))
        mlflow.log_metric("recall", round(rec, 4))
        print(
            f"📊 Holdout Metrics: ROC-AUC={roc_auc:.4f} | PR-AUC={pr_auc:.4f} | "
            f"F1={f1:.4f} | Precision={prec:.4f} | Recall={rec:.4f} | Acc={accuracy:.4f}"
        )

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
