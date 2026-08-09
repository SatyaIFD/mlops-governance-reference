# Model Card: Credit Card Fraud Detection Classifier (Version 1.0.0)

## 1. Model Details
- **Developer:** Enterprise Risk MLOps Team
- **Model Architecture:** XGBoost Classifier (`XGBClassifier`)
- **Engine Dependencies:** `xgboost==3.3.0`, `mlflow==3.15.0`, `python 3.13`
- **Version Binary:** `models:/credit_card_fraud_model/1` (SQLite Ledger Backend Tracking ID)
- **License:** Internal Financial Operations Use Only

## 2. Intended Use
- **Primary Scope:** Baseline model binary for validating end-to-end MLOps pipeline architecture, containerization, CI/CD automation, and automated compliance circuit breakers.
- **Out-of-Scope Uses:** Direct production credit decisioning without fairness remediation or post-processing logit calibration.

## 3. Quantitative Performance & Observability Factors
- **Data Drift Watchdog:** Custom Population Stability Index (PSI) tracker with Laplace adjustment ($1e^{-4}$).
- **Warning Threshold:** `0.10 <= PSI <= 0.25` (Triggers operational inspection logs).
- **Critical Breaker Threshold:** `PSI > 0.25` (Triggers automatic retraining pipeline generation).

## 4. Ethical & Governance Audit Results
- **High-Value Transaction Flag Rate:** `0.002446` (0.245%)
- **Low-Value Transaction Flag Rate:** `0.001296` (0.130%)
- **Calculated Disparate Impact Ratio:** **`1.8868`** (🚨 Automated Governance Breaker: Exceeds mandatory `0.80–1.25` regulatory parity envelope)
