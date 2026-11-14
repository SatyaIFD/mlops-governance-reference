import os
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

SRC_DIR = Path(__file__).resolve().parents[1]
SERVICE_ROOT = SRC_DIR.parent
REPO_ROOT = (
    SERVICE_ROOT.parent
    if (SERVICE_ROOT.parent / "docker-compose.yml").exists()
    else SERVICE_ROOT
)
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from src.features.velocity import (
    CANONICAL_AML_FEATURES,
    StreamingStateManager,
)
from src.models.anomaly_detector import (
    TRUSTED_SKOPS_TYPES,
    AMLAnomalyDetector,
)
from src.streaming.generator import ensure_dataset_exists


def _extract_label(tx: dict) -> int:
    for key in ("Is_laundering", "Is_Laundering", "is_laundering", "Target"):
        if key in tx and tx[key] is not None and str(tx[key]).strip() != "":
            return 1 if int(float(tx[key])) == 1 else 0
    raise ValueError(
        f"Missing required ground-truth label ('Is_laundering' or 'Is_Laundering') in record: {list(tx.keys())}"
    )


def prepare_training_data(
    csv_path: Path, warmup_rows: int = 50000, train_rows: int = 50000
) -> pd.DataFrame:
    total_rows = warmup_rows + train_rows
    print(
        f"🧠 Building historical 7-feature state matrix ({warmup_rows} warmup + {train_rows} training rows)..."
    )
    ensure_dataset_exists(csv_path)

    df = pd.read_csv(csv_path, nrows=total_rows)

    if "Date" in df.columns and "Time" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str)
        )
    elif "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    else:
        raise ValueError(
            "Dataset must contain 'Timestamp' or ('Date' and 'Time') columns."
        )

    df = df.sort_values("Timestamp").reset_index(drop=True)

    manager = StreamingStateManager()
    enriched_records = []

    for idx, tx in enumerate(df.to_dict("records")):
        target = _extract_label(tx)
        enriched_tx = manager.update_and_enrich(tx)
        enriched_tx["Is_laundering"] = target
        if idx >= warmup_rows or len(df) <= warmup_rows:
            enriched_records.append(enriched_tx)

    return pd.DataFrame(enriched_records)


def select_empirical_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Selects the probability threshold maximizing F1 on validation data subject to recall >= 0.25 (falling back to unconstrained F1)."""
    if len(np.unique(y_true)) < 2:
        return 0.40
    best_thresh = 0.40
    best_f1 = -1.0
    for candidate in np.linspace(0.25, 0.75, 21):
        preds = (probs >= candidate).astype(int)
        pos = int((y_true == 1).sum())
        rec = float(((preds == 1) & (y_true == 1)).sum() / pos) if pos > 0 else 0.0
        score = float(f1_score(y_true, preds, zero_division=0))
        if rec >= 0.25 and score >= best_f1:
            best_f1 = score
            best_thresh = float(round(candidate, 2))
    if best_f1 < 0:
        for candidate in np.linspace(0.25, 0.75, 11):
            preds = (probs >= candidate).astype(int)
            score = float(f1_score(y_true, preds, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_thresh = float(round(candidate, 2))
    return best_thresh


def train_and_register():
    db_path = REPO_ROOT / "mlflow.db"
    data_path = (
        REPO_ROOT / "data" / "SAML-D.csv"
        if (REPO_ROOT / "data" / "SAML-D.csv").exists()
        else SERVICE_ROOT / "data" / "SAML-D.csv"
    )
    skops_artifact_path = SERVICE_ROOT / "artifacts" / "aml_streaming_model"

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{db_path}")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("aml_streaming_detection")

    warmup_rows = 50000
    train_rows = 50000
    df = prepare_training_data(
        data_path, warmup_rows=warmup_rows, train_rows=train_rows
    )
    if df["Is_laundering"].nunique() < 2:
        raise RuntimeError(
            "Training dataset contains only a single class; cannot fit a valid binary classifier."
        )

    split_idx = int(len(df) * 0.8)
    train_df, val_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train = train_df[CANONICAL_AML_FEATURES]
    y_train = train_df["Is_laundering"].astype(int)
    X_val = val_df[CANONICAL_AML_FEATURES]
    y_val = val_df["Is_laundering"].astype(int)

    with mlflow.start_run(run_name="random_forest_7feature_calibrated") as run:
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        val_probs = model.predict_proba(X_val)[:, 1]
        val_auc = (
            float(roc_auc_score(y_val, val_probs))
            if y_val.nunique() > 1
            else float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
        )
        optimal_threshold = select_empirical_threshold(y_val.to_numpy(), val_probs)

        mlflow.log_metric("val_roc_auc", val_auc)
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("feature_count", len(CANONICAL_AML_FEATURES))
        mlflow.log_param("calibrated_threshold", optimal_threshold)

        detector = AMLAnomalyDetector(threshold=optimal_threshold)
        detector.model = model
        detector.training_window_end_row = warmup_rows + train_rows
        detector.save(skops_artifact_path, mlflow_run_id=run.info.run_id)

        signature = infer_signature(X_train, model.predict_proba(X_train)[:, 1])
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="aml_model",
                registered_model_name="aml_streaming_model",
                signature=signature,
                skops_trusted_types=TRUSTED_SKOPS_TYPES,
            )
        except TypeError:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="aml_model",
                registered_model_name="aml_streaming_model",
                skops_trusted_types=TRUSTED_SKOPS_TYPES,
            )
        print(
            f"💾 7-feature AML model registered to MLflow and saved to {skops_artifact_path} (val_auc={val_auc:.4f}, threshold={optimal_threshold})."
        )


if __name__ == "__main__":
    train_and_register()
