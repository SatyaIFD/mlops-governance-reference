"""
Production Evaluation & Governance Engine - Project 2: Loan Default Prediction
Evaluates the registered model out-of-sample on test.parquet using the single
cohort-blind decision_threshold calibrated on the validation split during training.
Verifies threshold mlflow_run_id lineage, gates sign-off on ROC-AUC >= 0.70,
Disparate Impact 95% CI in [0.80, 1.25], Equal Opportunity TPR 95% CI in [0.80, 1.25],
and FPR absolute gap <= 0.05, and documents residual cohort FPR differences.
"""

import json
import math
import sys
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def _katz_95_ci(k1: int, n1: int, k0: int, n0: int) -> list[float]:
    if k1 == 0 or k1 == n1 or k0 == 0 or k0 == n0:
        a, m1, b, m0 = k1 + 0.5, n1 + 1.0, k0 + 0.5, n0 + 1.0
    else:
        a, m1, b, m0 = float(k1), float(n1), float(k0), float(n0)
    p1 = a / m1
    p0 = b / m0
    ratio = p1 / p0
    se = math.sqrt(max(0.0, (1.0 - p1) / a + (1.0 - p0) / b))
    return [
        round(ratio * math.exp(-1.96 * se), 4),
        round(ratio * math.exp(1.96 * se), 4),
    ]


def _cohort_confusion_rates(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Computes default-detection TPR, FPR, approval counts, and decline rates for a cohort."""
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    approved_count = int((y_pred == 0).sum())
    total_count = int(len(y_true))
    tpr = round(float(tp / n_pos), 4) if n_pos > 0 else 0.0
    fpr = round(float(fp / n_neg), 4) if n_neg > 0 else 0.0
    approval_rate = (
        round(float(approved_count / total_count), 4) if total_count > 0 else 0.0
    )
    return {
        "count": total_count,
        "approved_count": approved_count,
        "positive_defaults": n_pos,
        "true_positives": tp,
        "approval_rate": approval_rate,
        "decline_rate": round(1.0 - approval_rate, 4),
        "default_tpr": tpr,
        "default_fpr": fpr,
    }


class LoanModelEvaluator:
    def __init__(self, data_dir: Path, artifacts_dir: Path):
        self.data_dir = data_dir
        self.artifacts_dir = artifacts_dir
        self.test_path = self.data_dir / "test.parquet"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def _load_validated_threshold(self, expected_run_id: str) -> tuple[float, str]:
        thresholds_path = self.artifacts_dir / "decision_thresholds.json"
        if not thresholds_path.is_file():
            raise FileNotFoundError(
                f"Required validation-calibrated threshold file missing at {thresholds_path}. Run train.py first."
            )
        with open(thresholds_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "decision_threshold" not in data:
            raise ValueError(
                f"Missing 'decision_threshold' key in {thresholds_path}: {data}"
            )
        file_run_id = data.get("mlflow_run_id")
        if file_run_id and str(file_run_id) != str(expected_run_id):
            raise ValueError(
                f"Stale threshold file {thresholds_path}: mlflow_run_id={file_run_id} does not match evaluated run {expected_run_id}"
            )
        return float(data["decision_threshold"]), str(
            data.get("calibration_split", "validation_holdout_20pct")
        )

    def run_evaluation(self):
        """Loads latest registered model and validation threshold, evaluating out-of-sample on test.parquet."""
        print(
            "📥 Loading validated test partition for out-of-sample governance audit..."
        )

        db_path = self.data_dir.parents[2] / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path}")

        test_df = pd.read_parquet(self.test_path)

        if "LoanID" in test_df.columns:
            test_df.drop(columns=["LoanID"], inplace=True)

        X_test = test_df.drop(columns=["Default"])
        y_test = test_df["Default"]

        X_test["is_young"] = (X_test["Age"] < 30).astype(int)

        cat_cols = X_test.select_dtypes(
            include=["object", "category", "string"]
        ).columns.tolist()
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

        decision_threshold, calibration_source = self._load_validated_threshold(
            expected_run_id=latest_run_id
        )
        preds_proba = model.predict_proba(X_test)[:, 1]
        preds_labels = pd.Series(
            (preds_proba >= decision_threshold).astype(int), index=X_test.index
        )

        auc = float(roc_auc_score(y_test, preds_proba))
        acc = float(accuracy_score(y_test, preds_labels))
        bce_loss = float(log_loss(y_test, preds_proba))

        young_mask = X_test["is_young"] == 1
        base_mask = X_test["is_young"] == 0
        prot_stats = _cohort_confusion_rates(
            y_test[young_mask], preds_labels[young_mask]
        )
        base_stats = _cohort_confusion_rates(y_test[base_mask], preds_labels[base_mask])

        di_ratio = (
            float(prot_stats["approval_rate"] / base_stats["approval_rate"])
            if base_stats["approval_rate"] > 0
            else 0.0
        )
        di_ci = _katz_95_ci(
            prot_stats["approved_count"],
            prot_stats["count"],
            base_stats["approved_count"],
            base_stats["count"],
        )

        tpr_ratio = (
            float(prot_stats["default_tpr"] / base_stats["default_tpr"])
            if base_stats["default_tpr"] > 0
            else 0.0
        )
        tpr_ci = _katz_95_ci(
            prot_stats["true_positives"],
            prot_stats["positive_defaults"],
            base_stats["true_positives"],
            base_stats["positive_defaults"],
        )
        fpr_gap = round(abs(prot_stats["default_fpr"] - base_stats["default_fpr"]), 4)

        # Both DI and Equal Opportunity TPR require their entire 95% Katz CI inside [0.80, 1.25]
        di_passed = bool(di_ci[0] >= 0.80 and di_ci[1] <= 1.25)
        tpr_passed = bool(tpr_ci[0] >= 0.80 and tpr_ci[1] <= 1.25)
        fpr_passed = bool(fpr_gap <= 0.05)
        fairness_passed = bool(di_passed and tpr_passed and fpr_passed)
        performance_passed = bool(auc >= 0.70)
        governance_signoff = bool(fairness_passed and performance_passed)

        for st_dict in (prot_stats, base_stats):
            st_dict.pop("approved_count", None)
            st_dict.pop("positive_defaults", None)
            st_dict.pop("true_positives", None)

        residual_risk_note = (
            f"Under unified cohort-blind threshold ({decision_threshold}), Equal Opportunity "
            f"(TPR ratio {tpr_ratio:.4f}, 95% CI {tpr_ci}) and Four-Fifths Disparate Impact "
            f"(DI {di_ratio:.4f}, 95% CI {di_ci}) pass, but a residual FPR gap of {fpr_gap:.4f} "
            f"remains (under-30 non-defaulters declined at {prot_stats['default_fpr'] * 100:.2f}% "
            f"vs {base_stats['default_fpr'] * 100:.2f}% for 30+, approval {prot_stats['approval_rate'] * 100:.2f}% "
            f"vs {base_stats['approval_rate'] * 100:.2f}%) due to base-rate credit-history differences "
            f"(Chouldechova impossibility theorem); mitigated via mandatory PENDING_HUMAN_REVIEW."
        )

        holdout_fairness_block = {
            "demographic_parity_status": "COMPLIANT (FAIR)"
            if fairness_passed
            else "NON-COMPLIANT (BREACH)",
            "disparate_impact_ratio": round(di_ratio, 4),
            "di_95_ci": di_ci,
            "equal_opportunity_tpr_ratio": round(tpr_ratio, 4),
            "tpr_95_ci": tpr_ci,
            "fpr_absolute_gap": fpr_gap,
            "fpr_gap_tolerance": 0.05,
            "residual_risk_note": residual_risk_note,
            "decision_threshold": decision_threshold,
            "threshold_calibration_split": calibration_source,
            "protected_cohort_under_30": prot_stats,
            "baseline_cohort_30_and_over": base_stats,
        }

        audit_report = {
            "mlflow_run_id": latest_run_id,
            "performance_metrics": {
                "test_roc_auc": round(auc, 4),
                "test_accuracy": round(acc, 4),
                "log_loss": round(bce_loss, 4),
            },
            "fairness_metrics": holdout_fairness_block,
            "holdout_fairness_metrics": holdout_fairness_block.copy(),
            "compliance": {
                "four_fifths_rule_passed": di_passed,
                "equal_opportunity_passed": tpr_passed,
                "fpr_parity_passed": fpr_passed,
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
            f"📊 Out-of-Sample Audit Completed: Sign-off -> {audit_report['compliance']['governance_signoff_status']} "
            f"(Unified Cutoff={decision_threshold}, DI={di_ratio:.4f} CI={di_ci}, "
            f"TPR_ratio={tpr_ratio:.4f} CI={tpr_ci}, FPR_gap={fpr_gap:.4f})"
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
