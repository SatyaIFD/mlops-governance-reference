# Model Card: Credit Card Fraud Detection Classifier (Version 1.0.0)

## 1. Model Details
- **Developer:** Enterprise Risk MLOps Team
- **Model Architecture:** XGBoost Classifier (`XGBClassifier`)
- **Version Binary:** `models:/credit_card_fraud_model/1` (SQLite Ledger Backend Tracking ID)
- **Release Date:** June 2026
- **License:** Internal Financial Operations Use Only

## 2. Intended Use
- **Primary Use Case:** Real-time scoring and interception of card-not-present transaction fraud events.
- **Out-of-Scope Uses:** Application credit scoring, long-term credit risk assessment, or currency laundering tracing pipelines.

## 3. Quantitative Performance Factors
Evaluated across stratified transaction vectors containing extreme minority class imbalances (0.17% baseline presence):

- **Global Accuracy:** 99.94%
- **Precision (Positive Predictive Value):** 0.5891
- **Recall (Sensitivity):** 0.8000
- **F1-Score Metrics:** 0.6786
- **ROC-AUC Score:** 0.9734

## 4. Ethical & Training Constraints
- **Training Constraints:** Numerical floating-point integrity is structurally locked using localized PyArrow columnar Parquet datasets.
- **Fairness Metrics:** Evaluated across transactional scale buckets to satisfy anti-bias regulations.

## 5. Model Maintenance & Monitoring Thresholds
- **Data Drift Watchdog:** Custom Population Stability Index (PSI) tracker with Laplace adjustment configurations ($1e^{-4}$).
- **Warning Threshold:** `PSI > 0.10` (Triggers operational inspection logs).
- **Critical Breaker Threshold:** `PSI > 0.25` (Triggers automatic retraining pipeline generation).
- **Calculated Disparate Impact Ratio:** 1.8135 (🚨 Critical Production Blocker: Exceeds 1.25 Threshold)