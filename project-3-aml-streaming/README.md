# Project 3: Real-Time Stateful AML Streaming Observability Engine

An enterprise-grade, stateful streaming inference engine built to detect complex money-laundering typologies (smurfing, layering, structuring, pass-through mules) in real-time transaction streams with zero data loss guarantees and automated schema governance.

---

## 🏛️ System Architecture Overview

```text
[ Live Transaction Stream ]
│
▼
┌─────────────────────────┐
│ StreamingIngestionEngine│ ─── (Corrupted / Poisoned) ───► [ Dead-Letter Queue (DLQ) ]
└─────────────────────────┘                                          │
│ (Clean Payload)                                       ▼
▼                                             [ scripts/replay_dlq.py ]
┌─────────────────────────┐
│  StreamingStateManager  │ ─── (In-Memory Lookback Cache + 24h TTL Key Eviction)
└─────────────────────────┘
│ (7 Graph & Velocity Features)
▼
┌─────────────────────────┐
│   AMLAnomalyDetector    │ ─── (Calibrated Random Forest Classifier)
└─────────────────────────┘
│
▼
[ 🚨 Real-Time AML Alert Output & Telemetry Logging ]

```

---

## 🔑 Core Features & Mathematical Formulations

Standard transaction-level models fail on money laundering because individual events look normal. This engine calculates **stateful graph-like features** over rolling lookback windows:

1. **Pass-Through Ratio (1-Hour Window):**
Detects rapid "mule" layering where an account receives funds and immediately transfers them out.

$$\text{Pass-Through Ratio}_{1h} = \min\left(1.0, \frac{\text{Outgoing Volume}_{1h}}{\text{Incoming Volume}_{1h} + \epsilon}\right)$$


2. **Structuring Flag ($8k–$10k CTR Range):**
Flags payments in the $\$8,000$–$\$9,999$ corridor designed to evade mandatory $\$10,000$ Currency Transaction Reporting (CTR) limits.
3. **Velocity Acceleration:**
Compares 1-hour transaction frequency against the account's 24-hour average hourly rate to catch sudden volume spikes.
4. **Fan-Out Dispersion Count (24-Hour Window):**
Tracks unique beneficiary accounts receiving funds to detect fan-out dispersion networks.
5. **Receiver Inflow Count & Volume (1-Hour Window):**
Measures incoming transaction frequency and total volume at beneficiary accounts to catch smurfing aggregation.

---

## 🛡️ Governance & Data Reliability Features

* **5-Pillar Governance Suite:** Complete regulatory and operational documentation under `docs/` (`data_dictionary.md`, `model_card.md`, `gdpr_compliance.md`, `fairness_and_bias_audit.md`, `incident_response_runbook.md`, `monitoring_and_slas.md`).
* **Automated API Documentation:** Generated directly from Python docstrings via `pdoc` in CI/CD (`.github/workflows/project3-ci.yml`).
* **24-Hour TTL Key Eviction:** Automated memory garbage collection (`_evict_dormant_accounts`) wipes dormant account keys to prevent OOM memory leaks.
* **Dead-Letter Queue (DLQ) & Replay:** Isolates malformed records with quarantine metadata and includes `scripts/replay_dlq.py` for automated re-injection.

---

## 📊 Performance Benchmarks

| Metric | Benchmark Value |
| --- | --- |
| **Total Stream Ingestion** | 200,000 records |
| **Data Quality Compliance** | 100.0% clean events (0 DLQ drops) |
| **Real-Time Streaming Recall** | **30.33%** (37 / 122 laundering events intercepted) |
| **Stateful Feature Matrix** | 7 graph & velocity features |
| **State Memory Bounds** | Safe under 24h TTL key eviction |

---

# Enterprise MLOps Governance & Observability Monorepo

A reference architecture for production Machine Learning Operations (MLOps) across batch, shadow deployment, and real-time stateful streaming paradigms.

This repository demonstrates enterprise-grade data quality enforcement, drift detection, stateful feature engineering, automated CI/CD validation, and audit-ready artifact lineage.

---

## 🏛️ Portfolio Architecture & System Pillars

```text
                    ┌─────────────────────────────────────────┐
                    │   MLOps Governance Reference Platform   │
                    └────────────────────┬────────────────────┘
                                         │
  ┌──────────────────────────────────────┼──────────────────────────────────────┐
  │                                      │                                      │
  ▼                                      ▼                                      ▼
┌───────────────────────────┐  ┌───────────────────────────┐  ┌───────────────────────────┐
│         PROJECT 1         │  │         PROJECT 2         │  │         PROJECT 3         │
│  Batch Governance & Drift │  │ Shadow Deployment Engine  │  │ Real-Time AML Streaming   │
├───────────────────────────┼──────────────────────────────┼───────────────────────────┤
│ • Evidently AI Profiling  │  │ • Dual-Model Routing      │  │ • 7 Stateful Graph Features│
│ • Automated Data Quality  │  │ • Non-Intrusive Monitoring│  │ • 24h TTL Key Eviction    │
│ • Model Retraining Triggers│ │ • Drift & Bias Scoring    │  │ • DLQ & Replay Tooling    │
└───────────────────────────┘  └───────────────────────────┘  └───────────────────────────┘

```

---

## 🔬 Core Projects Breakdown

### 1. Project 1: Credit Card Fraud — Batch Governance & Drift Tracking

* **Scope:** Batch ML pipeline with automated drift monitoring and retrain trigger evaluation.
* **Directory:** [`/project-1-credit-card-fraud/`](https://www.google.com/search?q=./project-1-credit-card-fraud/)

### 2. Project 2: Loan Default — Shadow Deployment & Near Real-Time Monitoring

* **Scope:** Production shadow deployment engine comparing champion vs. candidate models live without impacting business decisions.
* **Directory:** [`/project-2-loan-default/`](https://www.google.com/search?q=./project-2-loan-default/)

### 3. Project 3: Anti-Money Laundering (AML) — Real-Time Stateful Streaming Observability

* **Scope:** High-throughput streaming inference engine intercepting financial crime patterns.
* **Key Achievements:** 30.33% streaming recall, 100% data quality compliance across 200k records, 7 stateful graph/velocity features, 24h TTL memory eviction, and a 5-pillar governance suite.
* **Directory:** [`/project-3-aml-streaming/`](https://www.google.com/search?q=./project-3-aml-streaming/)

---

## 🛠️ Cross-Project Governance Standards

* **Automated CI/CD:** GitHub Actions executes unit tests and auto-generates API docs on every commit (`.github/workflows/project3-ci.yml`).
* **Environment Portability:** Dynamic relative path resolution via `pathlib` across Linux, macOS, and containerized runners.
* **Full Auditability:** Complete 5-pillar governance framework (`data_dictionary.md`, `model_card.md`, `gdpr_compliance.md`, `fairness_and_bias_audit.md`, `incident_response_runbook.md`, `monitoring_and_slas.md`).

---

## 🚀 Quickstart & Setup

```bash
# Clone repository
git clone [https://github.com/SatyaIFD/mlops-governance-reference.git](https://github.com/SatyaIFD/mlops-governance-reference.git)
cd mlops-governance-reference

# Activate environment and install dependencies
conda activate mlops-lab
uv pip install pandas numpy scikit-learn joblib pytest faker time-machine anyio hydra-core pdoc kagglehub

# Run test suite
PYTHONPATH=project-3-aml-streaming pytest project-3-aml-streaming/tests/

# Launch real-time streaming pipeline
PYTHONPATH=project-3-aml-streaming python project-3-aml-streaming/src/streaming/pipeline.py

```