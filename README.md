# Enterprise MLOps Governance & Observability Monorepo

[![Python](https://img.shields.io/badge/Python-3.13--slim-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/GitHub_Actions-Automated-success.svg)](https://github.com/features/actions)

A reference architecture for production Machine Learning Operations (MLOps) across batch, shadow deployment, and real-time stateful streaming paradigms. This repository demonstrates enterprise-grade data quality enforcement, drift detection, stateful feature engineering, automated CI/CD validation, and audit-ready artifact lineage.

## 🏛️ Portfolio Architecture & System Pillars

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
├───────────────────────────┤  ├───────────────────────────┤  ├───────────────────────────┤
│ • Evidently AI Profiling  │  │ • Dual-Model Routing      │  │ • 7 Stateful Graph Features│
│ • Automated Data Quality  │  │ • Non-Intrusive Monitoring│  │ • 24h TTL Key Eviction    │
│ • Model Retraining Triggers│ │ • Drift & Bias Scoring    │  │ • DLQ & Replay Tooling    │
└───────────────────────────┘  └───────────────────────────┘  └───────────────────────────┘

## 🔬 Core Projects Breakdown

### 1. Project 1: Credit Card Fraud — Batch Governance & Drift Tracking
* **Scope:** Batch ML pipeline with automated drift monitoring and retrain trigger evaluation.
* **Directory:** `/project-1-credit-card-fraud/`

### 2. Project 2: Loan Default — Shadow Deployment & Near Real-Time Monitoring
* **Scope:** Production shadow deployment engine comparing champion vs. candidate models live without impacting business decisions.
* **Directory:** `/project-2-loan-default/`

### 3. Project 3: Anti-Money Laundering (AML) — Real-Time Stateful Streaming Observability
* **Scope:** High-throughput streaming inference engine intercepting financial crime patterns.
* **Key Achievements:** 30.33% streaming recall, 100% data quality compliance across 200k records, 7 stateful graph/velocity features, 24h TTL memory eviction, and a 5-pillar governance suite.
* **Directory:** `/project-3-aml-streaming/`

## 🛠️ Cross-Project Governance Standards

* **Automated CI/CD & Containerization:** GitHub Actions executes unit tests, auto-generates API docs, and builds optimized Docker containers (`python:3.13-slim`) on every commit via path filters.
* **Continuous Deployment (CD):** Merges to `main` automatically publish production-ready images to GitHub Container Registry (GHCR) with strict Git commit SHA tagging for absolute traceability.
* **Environment Portability:** Dynamic relative path resolution via `pathlib` across Linux, macOS, and containerized deployment environments.
* **Full Auditability:** Complete 5-pillar governance framework (`data_dictionary.md`, `model_card.md`, `gdpr_compliance.md`, `fairness_and_bias_audit.md`, `incident_response_runbook.md`, `monitoring_and_slas.md`).


## ⚖️ Regulatory Compliance & Legal Defensibility

This architecture is designed to satisfy the strictest global regulatory frameworks for High-Risk AI systems. A dedicated `/compliance` directory provides centralized programmatic enforcement mechanisms:

* **GDPR (Right to be Forgotten):** Programmatic purge utilities (`gdpr_purge.py`) to permanently scrub PII from MLflow tracking databases and batch datasets.
* **EU AI Act (Human-in-the-Loop):** Decision routing gateways (`eu_ai_act_hitl_router.py`) that intercept borderline predictions and push them to manual auditor queues, preventing autonomous high-risk decisions.
* **DORA (Operational Resilience):** Circuit breaker patterns (`dora_circuit_breaker.py`) ensuring inference degradation fails safely to rules-based fallbacks without causing cascading system outages.

## 🚀 Quickstart & Setup

### Option 1: Run Production Container (Recommended)
You can instantly deploy the latest production build of the AML Streaming Engine directly from the GitHub Container Registry without installing any local Python dependencies:

```bash
# Pull the latest production image
docker pull ghcr.io/satyaifd/mlops-governance-reference/aml-stream-api:latest

# Run the real-time streaming pipeline
docker run -d -p 8000:8000 --name aml-streaming-api ghcr.io/satyaifd/mlops-governance-reference/aml-stream-api:latest

```
### Option 2: Local Development Setup
If you wish to run the test suite or modify the architecture locally:

# Clone repository
git clone [https://github.com/SatyaIFD/mlops-governance-reference.git](https://github.com/SatyaIFD/mlops-governance-reference.git)
cd mlops-governance-reference

# Activate environment and install dependencies
conda activate mlops-lab
uv pip install pandas numpy scikit-learn joblib pytest faker time-machine anyio hydra-core pdoc kagglehub

# Run test suite
PYTHONPATH=project-3-aml-streaming pytest project-3-aml-streaming/tests/

# Launch real-time streaming pipeline locally
PYTHONPATH=project-3-aml-streaming python project-3-aml-streaming/src/streaming/pipeline.py

---

## 🤖 Acknowledgments

This enterprise reference architecture was conceptualized, structured, and developed with the assistance of **Google Gemini Pro**. AI was utilized to accelerate CI/CD boilerplate generation, refine MLOps architectural patterns, and implement robust, enterprise-grade compliance frameworks.
