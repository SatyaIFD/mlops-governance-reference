# Model Card: Credit Card Fraud Detection Classifier (Version 1.0.0)

## 1. Model Details
* **Developer:** Enterprise Risk MLOps Team
* **Model Architecture:** XGBoost Classifier (`XGBClassifier`)
* **Engine Dependencies:** `xgboost==3.4.1`, `mlflow==3.18.0`, `python 3.13.14`
* **Registered Model URI:** `models:/fraud_detection_xgb/10` (SQLite Backend Tracking Registry)
* **Served Decision Threshold:** `0.50`
* **License:** Internal Financial Operations Use Only

## 2. Intended Use
* **Primary Scope:** Baseline model binary for validating end-to-end MLOps pipeline architecture, leakage-free train/test scaling, containerization, CI/CD automation, and DORA circuit-breaker resilience.
* **Out-of-Scope Uses:** Direct autonomous account termination without downstream human review.

## 3. Quantitative Holdout Performance & Observability Factors
* **Holdout ROC-AUC / PR-AUC:** `0.9659` / `0.8166`
* **Overall Holdout Recall / Precision:** `0.7158` (95% Wilson CI: `[0.6181, 0.7967]`) / `0.9855`
* **High-Value Segment ($\ge 80\text{th}$ Percentile) Recall / Precision:** `0.6538` (95% Wilson CI: `[0.4622, 0.8059]`) / `0.9444` (`17/26` frauds detected)
* **Low-Value Segment ($< 80\text{th}$ Percentile) Recall / Precision:** `0.7391` (95% Wilson CI: `[0.6249, 0.8281]`) / `1.0` (`51/69` frauds detected)
* **Data Drift Watchdog:** Custom Population Stability Index (PSI) tracker (`src/monitoring/observability.py`) with Laplace smoothing ($10^{-4}$).

## 4. Informational Operational Transaction Segment Telemetry (`src/governance/audit.py`)
* **High-Value Segment ($\ge 80\text{th}$ Percentile) Flag Rate:** `0.001607` (`18/11203`, fraud prevalence `0.232%`)
* **Low-Value Segment ($< 80\text{th}$ Percentile) Flag Rate:** `0.001120` (`51/45543`, fraud prevalence `0.152%`)
* **Informational Segment Flag-Rate Ratio:** **`1.4348`** (95% Katz CI: `[0.8387, 2.4546]`, `within_80_125_band: false` — Informational operational metric only; transaction amount tier is not a protected demographic attribute and higher-amount tiers exhibit higher base fraud prevalence (0.232% vs 0.152%). `--enforce-gate` gates on `ROC-AUC >= 0.90`, `PR-AUC >= 0.75`, `overall_recall >= 0.70`, and per-segment recall 95% Wilson CI lower bounds.)
