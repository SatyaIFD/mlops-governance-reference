# Model Card: Credit Card Fraud Detection Classifier (Version 1.0.0)

## 1. Model Details
* **Developer:** Enterprise Risk MLOps Team
* **Model Architecture:** XGBoost Classifier (`XGBClassifier`)
* **Engine Dependencies:** `xgboost==3.4.1`, `mlflow==3.18.0`, `python 3.13.14`
* **Registered Model URI:** `models:/fraud_detection_xgb/9` (SQLite Backend Tracking Registry)
* **Served Decision Threshold:** `0.50`
* **License:** Internal Financial Operations Use Only

## 2. Intended Use
* **Primary Scope:** Baseline model binary for validating end-to-end MLOps pipeline architecture, leakage-free train/test scaling, containerization, CI/CD automation, and DORA circuit-breaker resilience.
* **Out-of-Scope Uses:** Direct autonomous account termination without downstream human review.

## 3. Quantitative Holdout Performance & Observability Factors
* **Holdout ROC-AUC / PR-AUC:** `0.9626` / `0.8192`
* **High-Value Segment ($\ge 80\text{th}$ Percentile) Recall / Precision:** `0.6538` / `0.9444` (`17/26` frauds detected)
* **Low-Value Segment ($< 80\text{th}$ Percentile) Recall / Precision:** `0.7246` / `1.0` (`50/69` frauds detected)
* **Data Drift Watchdog:** Custom Population Stability Index (PSI) tracker (`src/monitoring/observability.py`) with Laplace smoothing ($10^{-4}$).

## 4. Informational Operational Transaction Segment Telemetry (`src/governance/audit.py`)
* **High-Value Segment ($\ge 80\text{th}$ Percentile) Flag Rate:** `0.001607` (`18/11203`)
* **Low-Value Segment ($< 80\text{th}$ Percentile) Flag Rate:** `0.001098` (`50/45543`)
* **Informational Segment Flag-Rate Ratio:** **`1.4636`** (95% Katz CI: `[0.8542, 2.5072]`, `within_80_125_band: false` — informational operational telemetry only; transaction size is not a protected demographic attribute and high-value transactions exhibit a higher base fraud prevalence of 0.232% vs. 0.151%. `--enforce-gate` gates on `ROC-AUC >= 0.90`, `PR-AUC >= 0.75`, and per-segment recall `>= 0.65`.)
