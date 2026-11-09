# Model Card: Credit Card Fraud Detection Classifier (Version 1.0.0)

## 1. Model Details
* **Developer:** Enterprise Risk MLOps Team
* **Model Architecture:** XGBoost Classifier (`XGBClassifier`)
* **Engine Dependencies:** `xgboost==3.4.1`, `mlflow==3.18.0`, `python 3.13.14`
* **Registered Model URI:** `models:/fraud_detection_xgb/3` (SQLite Backend Tracking Registry)
* **License:** Internal Financial Operations Use Only

## 2. Intended Use
* **Primary Scope:** Baseline model binary for validating end-to-end MLOps pipeline architecture, leakage-free train/test scaling, containerization, CI/CD automation, and DORA circuit-breaker resilience.
* **Out-of-Scope Uses:** Direct autonomous account termination without downstream human review or segment threshold calibration.

## 3. Quantitative Performance & Observability Factors
* **Data Drift Watchdog:** Custom Population Stability Index (PSI) tracker (`src/monitoring/observability.py`) with Laplace smoothing ($10^{-4}$).
* **Warning Threshold:** `0.10 <= PSI < 0.25` (Logs `"WARNING"` status in `artifacts/drift_report.json`).
* **Alert Threshold:** `PSI >= 0.25` (Logs `"ALERT"` status and sets `drift_alert_triggered: true` in `artifacts/drift_report.json`).

## 4. Operational Transaction Segment Disparity Audit (`src/governance/audit.py`)
* **High-Value Segment ($\ge 80\text{th}$ Percentile) Flag Rate:** `0.001670` (0.167%)
* **Low-Value Segment ($< 80\text{th}$ Percentile) Flag Rate:** `0.001190` (0.119%)
* **Segment Disparity Ratio:** **`1.4036`** (`within_guardrail_bounds: false` — exceeds the `[0.80, 1.25]` operational guardrail envelope on `data/processed/test.parquet` when `--enforce-gate` is enabled)
