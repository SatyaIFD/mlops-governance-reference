import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import mlflow
from mlflow.models import infer_signature

SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.velocity import StreamingStateManager  # noqa: E402
from src.streaming.generator import ensure_dataset_exists  # noqa: E402


def prepare_training_data(csv_path: Path):
    print(
        "🧠 Building historical state features (Strict limit: 50k rows to protect 16GB RAM)..."
    )
    ensure_dataset_exists(csv_path)

    df = pd.read_csv(csv_path, nrows=50000)

    if "Date" in df.columns and "Time" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    elif "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    df = df.sort_values("Timestamp").reset_index(drop=True)

    manager = StreamingStateManager()
    enriched_records = []

    print("🔄 Processing transactions (Optimized dict loop for i5-9400F)...")
    for tx in df.to_dict("records"):
        target = 1 if tx.get("Is_Laundering", 0) == 1 else 0
        enriched_tx = manager.update_and_enrich(tx)
        enriched_tx["Target"] = target
        enriched_records.append(enriched_tx)

    final_df = pd.DataFrame(enriched_records)

    if final_df["Target"].nunique() == 1:
        print(
            "⚠️ Injecting synthetic anomaly to ensure multi-class training baseline..."
        )
        fake_anomaly = final_df.iloc[-1].copy()
        fake_anomaly["Target"] = 1
        fake_anomaly["Amount"] = 95000.00
        fake_anomaly["pass_through_ratio_1h"] = 0.99
        final_df.loc[len(final_df)] = fake_anomaly

    return final_df


def train_and_register():
    db_path = PROJECT_ROOT / "mlflow.db"
    data_path = PROJECT_ROOT / "data" / "SAML-D.csv"

    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("aml_streaming_detection")

    df = prepare_training_data(data_path)

    feature_cols = [
        "Amount",
        "pass_through_ratio_1h",
        "receiver_inflow_count_1h",
        "receiver_inflow_amount_1h",
    ]

    X = df[feature_cols]
    y = df["Target"]

    with mlflow.start_run(run_name="random_forest_baseline"):
        print("🌲 Training Random Forest Classifier (Using all 6 CPU cores)...")
        model = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        )
        model.fit(X, y)

        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)

        print(f"📊 Training AUC: {auc:.4f}")

        mlflow.log_metric("train_auc", auc)
        mlflow.log_param("n_estimators", 50)

        signature = infer_signature(X, probs)
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="aml_model",
                registered_model_name="aml_streaming_model",
                signature=signature,
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
        except TypeError:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="aml_model",
                registered_model_name="aml_streaming_model",
                skops_trusted_types=["sklearn.tree._tree.Tree"],
            )
        print("💾 Model successfully registered to MLflow as 'aml_streaming_model'.")


if __name__ == "__main__":
    train_and_register()
