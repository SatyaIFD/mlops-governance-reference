# Enterprise MLOps Governance & Observability Monorepo

[![Python](https://img.shields.io/badge/Python-3.13--slim-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/GitHub_Actions-Automated-success.svg)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reference architecture for Machine Learning Operations (MLOps) across batch fraud classification, fair-lending underwriting with human-in-the-loop (HITL) routing, and stateful streaming anomaly detection. This repository demonstrates schema validation, automated DevSecOps scanning, and programmatic governance engineering controls.

## 🏛️ Portfolio Architecture & System Pillars

This repository is structured as a multi-service MLOps platform housing three deployment paradigms:

* **Project 1: Credit Card Fraud** (Leakage-Free Batch Training, PSI Drift Validation & Segment Disparity Auditing)
* **Project 2: Loan Default** (Sample-Reweighted Fair Lending API, HITL Routing & Live Telemetry Monitoring)
* **Project 3: AML Streaming** (Headless Stateful Consumer with 7 Event-Time Watermarked Features & DLQ Replay)

## 🛡️ Automated DevSecOps Pipeline

* **SAST & Linting:** Automated vulnerability scanning via **Bandit** (`make security`) and formatting baselines via **Ruff**.
* **Container Security:** Automated OS-level and package vulnerability scanning across all three microservice images using **Trivy** (`aquasecurity/trivy-action@0.28.0`).
* **Artifact Publishing:** Docker images are built and published to the **GitHub Container Registry (GHCR)** after passing unit and compliance test gates.

## ⚖️ Programmatic Governance & Resilience Controls (`/compliance`)

The `/compliance` package implements technical building blocks that support regulatory engineering workflows:

* **GDPR Erasure Utility (`gdpr_purge.py`):** Journaled, crash-recoverable erasure utility (`INITIATED` $\rightarrow$ `FILE_STAGED` $\rightarrow$ `FILE_SWAPPED` $\rightarrow$ `DB_COMMITTED` $\rightarrow$ `VERIFIED`) that streams Parquet/CSV batch rewrites via PyArrow and scrubs scoped `user_id` records from SQLite tracking tables with automatic `.bak` rollback on failure.
* **EU AI Act HITL Router (`eu_ai_act_hitl_router.py`):** Decision-routing helper that tags credit underwriting recommendations (`APPROVE` / `DECLINE`) with `MANUAL_REVIEW_REQUIRED` and `PENDING_HUMAN_REVIEW` status so downstream case-management systems can queue human adjudication.
* **DORA Circuit Breaker (`dora_circuit_breaker.py`):** Thread-safe circuit breaker with exclusive `HALF_OPEN` single-probe locking that fails closed (`HTTP 503`) when repeated inference exceptions occur.
* **Automated Verification (`test_compliance.py`):** Property-based (`hypothesis`), concurrency, and failure-injection tests verifying circuit-breaker exclusivity and journaled purge recovery.

## 🚀 Quickstart & Setup

### 1. Run Multi-Project Fleet

```bash
git clone [https://github.com/SatyaIFD/mlops-governance-reference.git](https://github.com/SatyaIFD/mlops-governance-reference.git)
cd mlops-governance-reference
cp .env.example .env
sudo docker compose up -d

```

### 2. Centralized Infrastructure Access

* **API Gateway (Nginx on port 80):**
* Project 1 (Fraud) Swagger UI: `http://localhost/api/v1/fraud/docs`
* Project 2 (Loan) Swagger UI: `http://localhost/api/v1/loan/docs`


* **Headless AML Streaming Worker:** Runs continuously in `mlops-aml-streaming`, writing execution reports and DLQ JSONL records to `project-3-aml-streaming/artifacts/`.
* **MLflow Tracking UI:** Accessible at `http://localhost:5000`.

### 3. Local Development & Verification

```bash
conda activate mlops-lab
make test
make security

```

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
