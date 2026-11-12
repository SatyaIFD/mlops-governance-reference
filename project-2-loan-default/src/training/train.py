"""
Production Training Engine - Project 2: Loan Default Prediction
Splits train.parquet into fit (80%) and validation (20%) partitions, fits the
sample-reweighted XGBoost model on the fit split, calibrates a single unified
cohort-blind decision threshold on the validation split, and logs to MLflow.
"""

import json
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def calibrate_unified_validation_threshold(
    y_val: pd.Series,
    val_probs: np.ndarray,
    is_young_val: pd.Series,
    min_recall: float = 0.40,
    max_recall: float = 0.55,
) -> float:
    """
    Selects a single cohort-blind probability threshold on the validation split
    that achieves target default recall in [min_recall, max_recall] while minimizing
    the relative TPR gap between (<30) and (>=30) cohorts.
    """
    pos_probs = val_probs[y_val == 1]
    if len(pos_probs) == 0:
        return 0.20

    best_cutoff = float(np.quantile(pos_probs, 0.55))
    best_loss = float("inf")

    y_arr = y_val.to_numpy()
    young_mask = (is_young_val == 1).to_numpy()
    base_mask = ~young_mask

    for q in np.linspace(1.0 - max_recall, 1.0 - min_recall, 31):
        cand = float(np.quantile(pos_probs, q))
        preds = (val_probs >= cand).astype(int)

        prot_pos = (y_arr == 1) & young_mask
        base_pos = (y_arr == 1) & base_mask
        prot_neg = (y_arr == 0) & young_mask
        base_neg = (y_arr == 0) & base_mask

        tpr_prot = float(preds[prot_pos].mean()) if prot_pos.any() else 0.0
        tpr_base = float(preds[base_pos].mean()) if base_pos.any() else 0.0
        fpr_prot = float(preds[prot_neg].mean()) if prot_neg.any() else 0.0
        fpr_base = float(preds[base_neg].mean()) if base_neg.any() else 0.0

        if tpr_base <= 0.0:
            continue
        tpr_ratio = tpr_prot / tpr_base
        fpr_gap = abs(fpr_prot - fpr_base)
        loss = abs(tpr_ratio - 1.0) + 0.5 * fpr_gap
        if loss < best_loss:
            best_loss = loss
            best_cutoff = cand

    return round(max(0.05, min(0.50, best_cutoff)), 4)


class LoanModelTrainer:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.train_path = self.data_dir / "train.parquet"
        self.test_path = self.data_dir / "test.parquet"

    def calculate_mitigation_weights(self, df: pd.DataFrame) -> np.ndarray:
        """Computes sample weights to neutralize historical age-based demographic parity bias."""
        N = len(df)
        n_fav = len(df[df["Default"] == 0])
        n_unfav = len(df[df["Default"] == 1])

        n_protected = len(df[df["Age"] < 30])
        n_baseline = len(df[df["Age"] >= 30])

        n_prot_fav = len(df[(df["Age"] < 30) & (df["Default"] == 0)])
        n_prot_unfav = len(df[(df["Age"] < 30) & (df["Default"] == 1)])
        n_base_fav = len(df[(df["Age"] >= 30) & (df["Default"] == 0)])
        n_base_unfav = len(df[(df["Age"] >= 30) & (df["Default"] == 1)])

        w_prot_fav = (n_protected * n_fav) / (N * n_prot_fav) if n_prot_fav > 0 else 1.0
        w_prot_unfav = (
            (n_protected * n_unfav) / (N * n_prot_unfav) if n_prot_unfav > 0 else 1.0
        )
        w_base_fav = (n_baseline * n_fav) / (N * n_base_fav) if n_base_fav > 0 else 1.0
        w_base_unfav = (
            (n_baseline * n_unfav) / (N * n_base_unfav) if n_base_unfav > 0 else 1.0
        )

        weights = np.ones(N)
        weights[(df["Age"] < 30) & (df["Default"] == 0)] = w_prot_fav
        weights[(df["Age"] < 30) & (df["Default"] == 1)] = w_prot_unfav
        weights[(df["Age"] >= 30) & (df["Default"] == 0)] = w_base_fav
        weights[(df["Age"] >= 30) & (df["Default"] == 1)] = w_base_unfav
        return weights

    def run_pipeline(self):
        """Fits on 80% of train.parquet, calibrates unified cutoff on 20% validation, and logs to MLflow."""
        print("📥 Ingesting validated Parquet frames...")
        full_train_df = pd.read_parquet(self.train_path)
        test_df = pd.read_parquet(self.test_path)

        for df in [full_train_df, test_df]:
            if "LoanID" in df.columns:
                df.drop(columns=["LoanID"], inplace=True)

        strat_key = (
            full_train_df["Default"].astype(str)
            + "_"
            + (full_train_df["Age"] < 30).astype(int).astype(str)
        )
        fit_df, val_df = train_test_split(
            full_train_df,
            test_size=0.20,
            random_state=42,
            stratify=strat_key,
        )
        fit_df = fit_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        def _encode_features(
            df_in: pd.DataFrame,
        ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
            is_young = (df_in["Age"] < 30).astype(int).copy()
            # ECOA / Reg B: exclude Age and MaritalStatus from model predictors
            drop_cols = [c for c in ("Default", "MaritalStatus") if c in df_in.columns]
            X = df_in.drop(columns=drop_cols).copy()
            X["is_young"] = is_young
            y = df_in["Default"].copy()
            cat_cols = X.select_dtypes(
                include=["object", "category", "string"]
            ).columns.tolist()
            if cat_cols:
                X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
            X = X.astype(
                {col: "int8" for col in X.select_dtypes(include=["bool"]).columns}
            )
            return X, y, is_young

        X_fit, y_fit, _ = _encode_features(fit_df)
        X_val, y_val, is_young_val = _encode_features(val_df)
        X_test, y_test, is_young_test = _encode_features(test_df)

        X_fit, X_val = X_fit.align(X_val, join="left", axis=1, fill_value=0)
        X_fit, X_test = X_fit.align(X_test, join="left", axis=1, fill_value=0)

        sample_weights = self.calculate_mitigation_weights(fit_df)
        artifacts_dir = self.data_dir.parent.parent / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        feature_cols_path = artifacts_dir / "feature_columns.json"
        with open(feature_cols_path, "w", encoding="utf-8") as f:
            json.dump(list(X_fit.columns), f, indent=2)

        mlflow.set_experiment("loan_default_risk_governance")
        with mlflow.start_run(run_name="mitigated_sample_reweighting") as run:
            print("🌲 Fitting Fair XGBoost pipeline on 80% fit partition...")
            model = XGBClassifier(
                random_state=42, eval_metric="logloss", n_estimators=100
            )
            model.fit(X_fit, y_fit, sample_weight=sample_weights)

            fit_probs = model.predict_proba(X_fit)[:, 1]
            val_probs = model.predict_proba(X_val)[:, 1]
            test_probs = model.predict_proba(X_test)[:, 1]

            unified_cutoff = calibrate_unified_validation_threshold(
                y_val, val_probs, is_young_val
            )
            thresholds_payload = {
                "decision_threshold": unified_cutoff,
                "calibration_split": "validation_holdout_20pct",
                "mlflow_run_id": run.info.run_id,
            }
            import shutil
            import tempfile

            tmp_thresh_dir = Path(tempfile.mkdtemp())
            thresholds_path = tmp_thresh_dir / "decision_thresholds.json"
            with open(thresholds_path, "w", encoding="utf-8") as f:
                json.dump(thresholds_payload, f, indent=2)

            preds = (test_probs >= unified_cutoff).astype(int)
            train_auc = float(roc_auc_score(y_fit, fit_probs))
            val_auc = float(roc_auc_score(y_val, val_probs))
            test_auc = float(roc_auc_score(y_test, test_probs))
            test_acc = float(accuracy_score(y_test, preds))
            test_logloss = float(log_loss(y_test, test_probs))

            df_audit = pd.DataFrame(
                {"is_young": is_young_test, "approved": (preds == 0).astype(int)}
            )
            approval_rates = df_audit.groupby("is_young")["approved"].mean()
            di_ratio = (
                float(approval_rates[1] / approval_rates[0])
                if approval_rates.get(0, 0) > 0
                else 0.0
            )

            print(
                f"📊 Run Completed: Val AUC = {val_auc:.4f} | Test AUC = {test_auc:.4f} | "
                f"Unified Cutoff = {unified_cutoff:.4f} | Test DI = {di_ratio:.4f}"
            )

            mlflow.log_param("model_type", "XGBClassifier")
            mlflow.log_param("mitigation_technique", "sample_reweighting")
            mlflow.log_param("decision_threshold", unified_cutoff)
            mlflow.log_metric("train_roc_auc", train_auc)
            mlflow.log_metric("val_roc_auc", val_auc)
            mlflow.log_metric("test_roc_auc", test_auc)
            mlflow.log_metric("test_accuracy", test_acc)
            mlflow.log_metric("test_log_loss", test_logloss)
            mlflow.log_metric("disparate_impact_ratio", di_ratio)
            mlflow.log_artifact(str(feature_cols_path), artifact_path="preprocessing")
            mlflow.log_artifact(str(thresholds_path), artifact_path="preprocessing")
            shutil.rmtree(tmp_thresh_dir, ignore_errors=True)

            signature = infer_signature(X_fit, fit_probs)

            try:
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    name="fair_loan_model",
                    registered_model_name="loan_default_production_model",
                    signature=signature,
                )
            except TypeError:
                mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path="fair_loan_model",
                    registered_model_name="loan_default_production_model",
                    signature=signature,
                )
            final_thresh_path = artifacts_dir / "decision_thresholds.json"
            with open(final_thresh_path, "w", encoding="utf-8") as f:
                json.dump(thresholds_payload, f, indent=2)
            print("💾 Production run completely tracked in system registry.")


if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent
    BASE_DIR = SRC_DIR.parents[1]
    DATA_DIR = BASE_DIR / "data" / "processed"

    db_path = BASE_DIR.parent / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")

    trainer = LoanModelTrainer(data_dir=DATA_DIR)
    trainer.run_pipeline()
