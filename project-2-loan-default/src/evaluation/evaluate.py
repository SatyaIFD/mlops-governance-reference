"""
Production Evaluation & Governance Engine - Project 2: Loan Default Prediction
Evaluates registered models against data contracts and exports compliance audit signatures.
"""

import json
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
import mlflow
import mlflow.xgboost


class LoanModelEvaluator:
    def __init__(self, data_dir: Path, artifacts_dir: Path):
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.test_path = self.data_dir / "test.parquet"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def run_evaluation(self):
        """Loads latest registered model, runs performance & fairness audits, and dumps a sign-off report."""
        print("📥 Loading validated test partition for governance audit...")

        db_path = self.data_dir.parents[2] / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        test_df = pd.read_parquet(self.test_path)

        if "LoanID" in test_df.columns:
            test_df.drop(columns=["LoanID"], inplace=True)

        X_test = test_df.drop(columns=["Default"])
        y_test = test_df["Default"]

        X_test["is_young"] = (X_test["Age"] < 30).astype(int)

        cat_cols = X_test.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)

        X_test = X_test.astype(
            {col: "int8" for col in X_test.select_dtypes(include=["bool"]).columns}
        )

        experiment = mlflow.get_experiment_by_name("loan_default_risk_governance")
        if not experiment:
            raise ValueError(
                "❌ No active experiment found. Please run train.py first!"
            )

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["attributes.start_time DESC"],
        )
        if runs.empty:
            raise ValueError(
                "❌ No logged runs found within the experiment tracking space."
            )

        latest_run_id = runs.iloc[0]["run_id"]
        model_uri = f"runs:/{latest_run_id}/fair_loan_model"

        print(f"🌲 Loading model binary from MLflow run context: {latest_run_id}")
        model = mlflow.xgboost.load_model(model_uri)

        if hasattr(model, "get_booster") and model.get_booster().feature_names:
            X_test = X_test.reindex(
                columns=list(model.get_booster().feature_names), fill_value=0
            )

        preds_proba = model.predict_proba(X_test)[:, 1]
        preds_labels = model.predict(X_test)

        auc = float(roc_auc_score(y_test, preds_proba))
        acc = float(accuracy_score(y_test, preds_labels))
        bce_loss = float(log_loss(y_test, preds_proba))

        df_audit = pd.DataFrame(
            {
                "is_young": X_test["is_young"],
                "approved": (preds_labels == 0).astype(int),
            }
        )
        approval_rates = df_audit.groupby("is_young")["approved"].mean()
        di_ratio = (
            float(approval_rates[1] / approval_rates[0])
            if approval_rates.get(0, 0) > 0
            else 0.0
        )

        fairness_passed = bool(0.80 <= di_ratio <= 1.25)
        performance_passed = bool(auc >= 0.70)
        governance_signoff = bool(fairness_passed and performance_passed)

        audit_report = {
            "mlflow_run_id": latest_run_id,
            "performance_metrics": {
                "test_roc_auc": round(auc, 4),
                "test_accuracy": round(acc, 4),
                "log_loss": round(bce_loss, 4),
            },
            "fairness_metrics": {
                "demographic_parity_status": "COMPLIANT (FAIR)"
                if fairness_passed
                else "NON-COMPLIANT (BREACH)",
                "disparate_impact_ratio": round(di_ratio, 4),
            },
            "compliance": {
                "four_fifths_rule_passed": fairness_passed,
                "minimum_performance_passed": performance_passed,
                "governance_signoff_status": "APPROVED"
                if governance_signoff
                else "REJECTED",
            },
        }

        report_path = self.artifacts_dir / "compliance_audit_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=4)

        print(
            f"📊 Audit Completed: Sign-off Status -> {audit_report['compliance']['governance_signoff_status']}"
        )

        if not governance_signoff:
            print("🚨 Model fails compliance validation gates.")
            sys.exit(1)


if __name__ == "__main__":
    SRC_DIR = Path(__file__).resolve().parent
    BASE_DIR = SRC_DIR.parents[1]
    DATA_DIR = BASE_DIR / "data" / "processed"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"

    evaluator = LoanModelEvaluator(data_dir=DATA_DIR, artifacts_dir=ARTIFACTS_DIR)
    evaluator.run_evaluation()
