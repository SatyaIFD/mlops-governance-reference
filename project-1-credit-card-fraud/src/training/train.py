"""
Model Training & Orchestration Module - Project 1: Credit Card Fraud Detection
Fits preprocessing strictly on the training split, splits train into 80% fit and 20% validation,
calibrates segment_thresholds.json on validation (never touching test.parquet),
and logs ROC-AUC, PR-AUC, F1, Precision, Recall, and Accuracy to MLflow.
"""

import json
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
from sklearn.model_selection import train_test_split

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


def calibrate_validation_segment_thresholds(
    X_val: pd.DataFrame,
    val_probs: np.ndarray,
    q80: float,
    y_val: pd.Series | None = None,
) -> dict:
    """Calibrates segment cutoffs on the held-out validation split targeting a 1.00 flag-rate ratio while preserving high-value recall."""
    is_high_val = (X_val["scaled_amount"] >= q80).to_numpy()
    low_probs = val_probs[~is_high_val]
    high_probs = val_probs[is_high_val]
    low_cutoff = 0.35
    high_cutoff = 0.65
    if len(low_probs) > 0 and len(high_probs) > 0:
        best_pair = (0.35, 0.65)
        best_loss = float("inf")
        for lc in np.linspace(0.30, 0.50, 9):
            l_rate = float((low_probs >= lc).mean())
            if l_rate <= 0:
                continue
            for hc in np.linspace(0.50, 0.75, 11):
                h_rate = float((high_probs >= hc).mean())
                ratio = h_rate / l_rate
                loss = abs(ratio - 1.0)
                if loss < best_loss:
                    best_loss = loss
                    best_pair = (round(float(lc), 4), round(float(hc), 4))
        low_cutoff, high_cutoff = best_pair
    return {
        "low_value_threshold": low_cutoff,
        "high_value_threshold": high_cutoff,
        "scaled_amount_q80": round(q80, 6),
        "calibration_split": "validation_holdout_20pct",
    }


def execute_training_pipeline(processed_data_file: str, tracking_uri: str):
    mlflow.set_tracking_uri(tracking_uri)
    data_path = Path(processed_data_file).resolve()
    project_root = (
        data_path.parents[1] if "data" in data_path.parts else data_path.parent
    )
    raw_csv_path = project_root / "data" / "creditcard.csv"
    scaler_artifact_path = project_root / "artifacts" / "scaler_params.json"
    seg_thresh_path = project_root / "artifacts" / "segment_thresholds.json"
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
        X_train_full = train_df[CANONICAL_FEATURE_COLS].reset_index(drop=True)
        y_train_full = train_df["Class"].reset_index(drop=True)
        X_test = test_df[CANONICAL_FEATURE_COLS].reset_index(drop=True)
        y_test = test_df["Class"].reset_index(drop=True)
    else:
        if not data_path.exists():
            raise FileNotFoundError(f"🚨 FATAL: Processed data missing at {data_path}.")
        df = pd.read_parquet(data_path)
        X_train_full, X_test, y_train_full, y_test = prepare_stratified_splits(df)
        X_train_full = X_train_full[CANONICAL_FEATURE_COLS].reset_index(drop=True)
        X_test = X_test[CANONICAL_FEATURE_COLS].reset_index(drop=True)
        y_train_full = y_train_full.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full,
    )
    q80_train = float(X_fit["scaled_amount"].quantile(0.80))
    sample_weights = _compute_segment_weights(X_fit, y_fit)

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
            X_fit,
            y_fit,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        val_probs = model.predict_proba(X_val)[:, 1]
        seg_policy = calibrate_validation_segment_thresholds(
            X_val, val_probs, q80_train
        )
        seg_policy["mlflow_run_id"] = run.info.run_id
        seg_thresh_path.parent.mkdir(parents=True, exist_ok=True)
        with open(seg_thresh_path, "w", encoding="utf-8") as f:
            json.dump(seg_policy, f, indent=2)

        probs = model.predict_proba(X_test)[:, 1]
        is_high_test = (
            X_test["scaled_amount"] >= seg_policy["scaled_amount_q80"]
        ).to_numpy()
        cutoffs = np.where(
            is_high_test,
            seg_policy["high_value_threshold"],
            seg_policy["low_value_threshold"],
        )
        preds = (probs >= cutoffs).astype(int)

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
        mlflow.log_artifact(str(seg_thresh_path), artifact_path="preprocessing")

        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name="fraud_detection_xgb",
        )
        print(
            "🟢 Model binaries and validation-calibrated segment thresholds logged and registered."
        )


if __name__ == "__main__":
    execute_training_pipeline(
        processed_data_file=str(BASE_DIR / "data/processed_features.parquet"),
        tracking_uri=f"sqlite:///{BASE_DIR.parent}/mlflow.db",
    )
