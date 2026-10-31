
# Project 1: Credit Card Fraud Detection Pipeline

An end-to-end MLOps pipeline for credit card fraud classification, transaction-segment disparity auditing, and Population Stability Index (PSI) drift monitoring.

---

## Architecture Overview

1. **Ingestion & Schema Validation (`src/ingestion/`)**
Validates incoming payloads against `TransactionInputSchema` and `ProcessedTransactionSchema` (canonical order: `V1`–`V28`, `scaled_amount`, `scaled_time`, `Class`) with strict finite-number constraints (`allow_inf_nan=False`).
2. **Leakage-Free Preprocessing & Training (`src/training/train.py`)**
Performs a stratified train/test split on deduplicated raw data *before* fitting `RobustScaler` (`Amount`) and `StandardScaler` (`Time`) exclusively on the training partition, persisting `scaler_params.json` and registering `fraud_detection_xgb` in MLflow.
3. **Segment Disparity Auditing (`src/governance/audit.py`)**
Evaluates flagging rate disparity between high-value ($\ge 80\text{th}$ percentile) and low-value transaction segments (`--enforce-gate` exits non-zero when outside `[0.80, 1.25]`). Note: this is an operational transaction-size segment check, not a demographic protected-class audit.
4. **Statistical Validation & PSI Drift (`src/validation/drift_check.py`)**
Verifies schema ordering, null counts, and binary class integrity, and computes Population Stability Index (PSI) scores across `V14`, `V17`, `V12`, `scaled_amount`, and `scaled_time`.

---

## Technical Specifications

* **Core Frameworks:** Python 3.13, XGBoost 3.4.1, MLflow 3.16.1, FastAPI, Uvicorn, SHAP, Pytest, Pandas, PyArrow
* **Container Environment:** `python:3.13-slim` running as non-root `UID 10001`

---

## 🚀 Quickstart & Reproduction Guide

### 1. Execute Unit & Auth Boundary Tests

```bash
PYTHONPATH=project-1-credit-card-fraud pytest project-1-credit-card-fraud/tests/ -v

```

### 2. Run via Docker Compose & Nginx Gateway

```bash
sudo docker compose up -d fraud-detection-api api-gateway

# Health probe via Nginx Gateway
curl http://localhost/api/v1/fraud/health

# Authenticated prediction probe (30 features: V1..V28, scaled_amount, scaled_time)
curl -X POST "http://localhost/api/v1/fraud/predict" \
-H "Content-Type: application/json" \
-H "X-API-Key: mlops-dev-secret-key" \
-d '{"features": [-1.35, -0.07, 2.53, 1.37, -0.33, 0.46, 0.23, 0.09, 0.36, 0.09, -0.55, -0.61, -0.99, -0.31, 1.46, -0.47, 0.20, 0.02, 0.40, 0.25, -0.01, 0.27, -0.11, 0.06, 0.12, -0.18, 0.13, -0.02, 0.24, -0.99]}'

```
