# Project 1: Production-Grade Credit Card Fraud Detection Pipeline

An end-to-end, reproducible Machine Learning Operations (MLOps) pipeline built for credit card fraud classification, risk auditing, and automated data drift monitoring. 

This project transitions exploratory modeling code into modular, contract-validated software sub-packages to implement Explainable AI (XAI) compliance audits, fairness evaluations, and real-time REST API endpoints suitable for highly regulated financial environments.

---

## 🎯 Primary Scope & Pipeline Intent

> **Note on Scope:** The primary objective of this project is to build and demonstrate production-grade **MLOps infrastructure, CI/CD automation, containerization, and compliance auditing pipelines**. Model training steps produce baseline artifacts to validate end-to-end telemetry and automated governance circuit breakers rather than state-of-the-art predictive performance or model tuning.

---

## Architecture Overview

The system moves away from raw, manual notebook execution toward automated software components segmented by clear structural boundaries:

1. **Ingestion & Data Engineering (`src/ingestion/`)**
   Loads transactional telemetry records, handles memory-efficient type downcasting, and exports deterministic Train/Test data splits into compressed columnar storage.
2. **Experiment Tracking & Orchestration (`src/training/`)**
   Automates hyperparameter sweep matrices via XGBoost. All optimization paths, performance weights, and training metrics are registered directly into a centralized relational metadata database.
3. **Fairness Auditing & Compliance (`src/governance/`)**
   Leverages game-theoretic Shapley values (`SHAP`) to map exact feature risk attribution and calculates demographic/financial disparity metrics across explicit transaction sub-segments.
4. **Production Observability & Monitoring (`src/monitoring/`)**
   Implements a statistical Population Stability Index (PSI) watchdog component to mathematically quantify profile distribution shifts on incoming production inference batches.

---

## Technical Specifications & Environment

- **Core Frameworks:** Python 3.13, XGBoost 3.3.0 (strictly pinned), MLflow 3.15.0 (strictly pinned), FastAPI, Uvicorn, SHAP, Pytest, Pandas, PyArrow
- **Container Environment:** `python:3.13-slim` base image
- **Isolation Strategy:** All data engine tracks are decoupled from physical absolute path variables using runtime package resolution to ensure seamless containerized migrations and automated CI/CD execution.

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup & Dependency Installation

```bash
pip install --upgrade pip
pip install -r project-1-credit-card-fraud/requirements.txt

```

### 2. Train and Register Model to MLflow

```bash
PYTHONPATH=project-1-credit-card-fraud python3 project-1-credit-card-fraud/src/training/train.py

```

### 3. Execute Integration Tests

```bash
PYTHONPATH=project-1-credit-card-fraud pytest project-1-credit-card-fraud/tests/

```

### 4. Build and Run Docker Container

> **Note:** Execute `docker build` from the repository root so build context captures all code and artifacts.

```bash
# Build image from repository root
sudo docker build -t fraud-detection-api:v1 -f project-1-credit-card-fraud/docker/Dockerfile .

# Launch container on port 5001
sudo docker run -d --name fraud-api -p 5001:5001 fraud-detection-api:v1

# Health probe
curl http://localhost:5001/health

# Prediction probe
curl -X POST "http://localhost:5001/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [0.0, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, -1.35, 85.00]}'

```

---

## Why This Architecture Was Chosen (Design Decisions)

### 1. Relational SQLite Backend Configuration over Flat File Tracking

Standard MLflow logging defaults to flat-file directory structures (`mlruns/`) that easily degrade, lack native concurrency handling, and fail ACID compliance requirements. Forcing a dedicated relational scheme (`sqlite:///mlflow.db`) keeps the enterprise audit ledger structured, indexable, and positioned to scale transparently into cloud environments like PostgreSQL.

### 2. Columnar Parquet Serialization over CSV Formats

Financial fraud datasets involve deep numerical precision across multiple Principal Component Analysis (PCA) dimensions (`V1`–`V28`). Storing data in Parquet format yields massive advantages:

* Deep columnar compression reducing local disk I/O bottlenecks.
* Strict data type schema enforcement, completely neutralizing floating-point rounding errors during downstream mathematical evaluations.

### 3. Population Stability Index (PSI) Tracking with Laplace Smoothing

Traditional data drift tests (such as Kolmogorov-Smirnov) struggle with high-velocity production data streams or heavily concentrated categorical distributions. The custom implementation utilizes fixed-quantile bucket allocations combined with a Laplace smoothing coefficient ($1e^{-4}$) to completely eliminate division-by-zero or logarithmic runtime infinity errors when zero-count buckets occur in live inference tracking windows.

---

## 🧪 Benchmarking & Production Validation

This pipeline is validated against the **ULB Credit Card Fraud Detection benchmark dataset** (284,807 transactions, 0.172% class imbalance) and synthetic CI matrices to verify enterprise scalability and mathematical compliance.

1. **Data Gravity Optimization:** The data engine optimizes raw records into compressed, type-downcasted Parquet partitions (`train.parquet` / `test.parquet`) to minimize disk I/O bottlenecks and enforce strict schema types.
2. **Ephemeral Fallback Engine:** For headless CI/CD contexts where raw dataset partitions are absent, `train.py` includes a defensive mock synthesis fallback to ensure test suite execution without blocking pipelines.
3. **Production Lifespan Latency:** Model binaries are loaded via an active FastAPI ASGI lifespan context, achieving sub-millisecond inference latencies on incoming prediction payloads.
4. **CI/CD End-to-End Validation:** The GitHub Actions pipeline verifies code quality via `pytest`, trains model artifacts, and executes a full `docker build` + container startup probe (`/health` and `/predict`) on `ubuntu-24.04` runners using Python 3.13.

---

## ⚖️ Governance, Risk, & Ethical Compliance Audit Framework

Operating an AI model within financial risk sectors requires strict compliance tracking. This pipeline fulfills regulatory guidelines across two critical pillars:

### 1. Explainability & Interpretability Framework

* **Mechanism:** Game-theoretic Shapley Values via the `SHAP` TreeExplainer engine.
* **Audit Findings:** The compliance audit verified that the model relies heavily on hidden latent transaction signatures—specifically **`V14`**, **`V17`**, and **`V12`**—to isolate fraudulent patterns. Features like `Amount` carry a lower global priority weight, minimizing the risk that an ordinary high-value purchase is flagged as fraud purely due to its scale.

### 2. Quantitative Fairness & Anti-Bias Audit (Proxy Segment Review)

* **High-Value Transaction Flagging Rate:** `0.002446` (0.245%)
* **Low-Value Transaction Flagging Rate:** `0.001296` (0.130%)
* **Calculated Disparate Impact Ratio:** **`1.8868`** (exceeds mandatory 0.80–1.25 regulatory window).

*Automated Circuit Breaker Result:* This baseline model version is flagged by the compliance pipeline and blocked from production deployment. In a production setting, this automated check triggers remediation (such as loss re-weighting or decision threshold adjustment) prior to artifact release.

### 3. Data Drift Observability Validation

Simulated live production batches featuring distribution shifts are evaluated against strict regulatory thresholds:

* `PSI < 0.10`: **STABLE** (No action needed)
* `0.10 <= PSI <= 0.25`: **WARNING** (Monitor closely)
* `PSI > 0.25`: **ALERT** (Trigger automated retraining pipeline)

---

## Project Structure

```text
project-1-credit-card-fraud/
├── data/
│   └── processed/          
├── docs/
│   ├── gdpr_controls.md    
│   ├── model_card.md        
│   └── risk_assessment.md  
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_fairness_and_governance.ipynb
│   └── 04_production_observability.ipynb
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   └── extract.py
│   ├── training/
│   │   └── train.py
│   ├── inference/
│   │   └── app.py
│   ├── monitoring/
│   │   └── observability.py
│   └── governance/
│       └── audit.py
├── tests/
│   ├── test_inference.py
│   ├── test_ingestion.py
│   ├── test_observability.py
│   └── test_governance.py
└── README.md

```

