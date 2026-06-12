# Project 1: Production-Grade Credit Card Fraud Detection Pipeline

An end-to-end, reproducible Machine Learning Operations (MLOps) pipeline built for credit card fraud classification, risk auditing, and automated data drift monitoring. 

This project transitions exploratory modeling code into modular, contract-validated software sub-packages to implement Explainable AI (XAI) compliance audits, fairness evaluations, and real-time REST API endpoints suitable for highly regulated financial environments.

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

- **Core Frameworks:** Python 3.12+, XGBoost, MLflow, FastAPI, Uvicorn, SHAP, Pytest, Pandas, PyArrow
- **Environment Context:** Configured for local Ubuntu Linux workspaces (`/media/storage/mlops-governance-reference`)
- **Isolation Strategy:** All data engine tracks are entirely decoupled from physical absolute path variables using runtime package resolution (`sys.path` injection) to ensure seamless containerized migrations or automated CI/CD runs.

---

## Why This Architecture Was Chosen (Design Decisions)

### 1. Relational SQLite Backend Configuration over Flat File Tracking
Standard MLflow logging defaults to flat-file directory structures (`mlruns/`) that easily degrade, lack native concurrency handling, and fail ACID compliance requirements. Forcing a dedicated relational scheme (`sqlite:///mlflow.db`) keeps the enterprise audit ledger structured, indexable, and positioned to scale transparently into cloud environments like PostgreSQL.

### 2. Columnar Parquet Serialization over CSV Formats
Financial fraud datasets involve deep numerical precision across multiple Principal Component Analysis (PCA) dimensions (`V1`–`V28`). Storing data in Parquet format yields massive advantages:
- Deep columnar compression reducing local disk I/O bottlenecks.
- Strict data type schema enforcement, completely neutralizing floating-point rounding errors during downstream mathematical evaluations.

### 3. Population Stability Index (PSI) Tracking with Laplace Smoothing
Traditional data drift tests (such as Kolmogorov-Smirnov) struggle with high-velocity production data streams or heavily concentrated categorical distributions. The custom implementation utilizes fixed-quantile bucket allocations combined with a Laplace smoothing coefficient ($1e^{-4}$) to completely eliminate division-by-zero or logarithmic runtime infinity errors when zero-count buckets occur in live inference tracking windows.

---

## ⚖️ Governance, Risk, & Ethical Compliance Audit Framework

Operating an AI model within financial risk sectors requires strict compliance tracking. This pipeline fulfills regulatory guidelines (such as the EU AI Act, Fair Credit Reporting Act, and "Right to Explanation" laws) across two critical pillars:

### 1. Explainability & Interpretability Framework (The "How" and "Why")
- **Mechanism:** Game-theoretic Shapley Values via the `SHAP` TreeExplainer engine.
- **Why it matters:** Traditional feature importance metrics (like Gini split criteria) are fundamentally biased toward high-cardinality features and lack directionality. SHAP calculates the exact, mathematically sound marginal contribution of each variable toward an individual transaction's fraud score.
- **Audit Findings:** The compliance audit verified that the model relies heavily on hidden latent transaction signatures—specifically **`V14`**, **`V17`**, and **`V12`**—to isolate fraudulent patterns. Features like `Amount` carry a lower global priority weight, minimizing the risk that an ordinary high-value purchase is flagged as fraud purely due to its scale.

### 2. Quantitative Fairness & Anti-Bias Audit (Proxy Segment Review)
To verify that the model treats different financial sub-segments equitably, we ran a proxy metric audit based on transaction scale (`scaled_amount`), tracking if affluent user profiles are penalized at disproportionate rates.

- **The Metric (Disparate Impact Ratio):** Calculated by dividing the selection rate (flagging rate) of the protected/unfavorable group by the selection rate of the baseline group.
- **Regulatory Guardrail:** Industry compliance standards enforce the **Four-Fifths Rule**. A calculated ratio between **0.80 and 1.25** indicates equitable statistical parity. Anything outside this window demonstrates systematic bias.

#### Critical Audit Discovery:
- **High-Value Transaction Flagging Rate:** 0.0057 (0.57%)
- **Low-Value Transaction Flagging Rate:** 0.0012 (0.12%)
- **Calculated Disparate Impact Ratio:** **0.2129**

**Audit Conclusion (🚨 CRITICAL PRODUCTION BLOCKER):** With a metric of **0.2129**, the model exhibits severe **Disparate Impact**. It flags high-value transactions for fraud at nearly **5 times** the rate of low-value transactions. In a live banking system, this represents an unacceptable operational risk that would disproportionately disrupt high-net-worth customers, trigger a massive spike in false-positive disputes, and fail an external regulatory compliance audit. 

*Remediation Plan:* This model version is blocked from production migration. The next development phase requires implementing adversarial debiasing, sample loss re-weighting during the `src/training/train.py` phase, or applying post-processing decision threshold adjustments to bring the Disparate Impact Ratio back within the mandatory **0.80–1.25** compliance envelope.

### 3. Data Drift Observability Validation
Simulated live production batches featuring an adversarial shift on key features successfully tripped our tracking thresholds:
- **Calculated PSI Metric Value:** 1.7066
- **System Guardrail Triggered:** `🔴 ALERT! Significant Data Drift detected. Trigger Automated Retraining.`

The custom module handles zero-width bin boundaries safely via uniform linear space distributions if feature concentration collapses percentiles, making it highly reliable for high-density transactional tracking.

---

## Project Structure

```text
project-1-credit-card-fraud/
├── notebooks/
│   ├── 01_data_ingestion.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_fairness_and_governance.ipynb
│   └── 04_production_observability.ipynb
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── extract.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py
│   ├── inference/
│   │   ├── __init__.py
│   │   └── app.py
│   └── monitoring/
│       ├── __init__.py
│       └── observability.py
├── tests/
│   ├── __init__.py
│   └── test_observability.py
└── README.md