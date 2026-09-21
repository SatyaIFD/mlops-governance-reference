# Enterprise MLOps Governance & Observability Monorepo

[![Python](https://img.shields.io/badge/Python-3.13--slim-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/GitHub_Actions-Automated-success.svg)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reference architecture for production Machine Learning Operations (MLOps) across batch, shadow deployment, and real-time stateful streaming paradigms. This repository demonstrates enterprise-grade data quality enforcement, automated DevSecOps validation, and audit-ready artifact lineage.

## 🏛️ Portfolio Architecture & System Pillars

This repository is structured as a unified MLOps platform housing three distinct deployment paradigms:

*   **Project 1: Credit Card Fraud** (Batch Governance & Drift Tracking)
*   **Project 2: Loan Default** (Shadow Deployment Engine)
*   **Project 3: AML Streaming** (Real-Time Stateful Observability)

## 🛡️ Automated DevSecOps Pipeline

This repository enforces strict CI/CD and DevSecOps governance for every commit merged into the `main` branch.

*   **Branch Governance:** Direct pushes to `main` are disabled. All changes require a Pull Request and successful status checks.
*   **SAST & Linting:** Automated vulnerability scanning via **Bandit** and strict formatting baselines enforced by **Ruff**.
*   **Container Security:** Automated OS-level and package vulnerability scanning using **Trivy** (`.trivyignore` used for triaged CVEs).
*   **Artifact Publishing:** Secure Docker images are automatically built and published to the **GitHub Container Registry (GHCR)**.
*   **Release Automation:** Semantic versioning and changelog generation are handled entirely by Google's **Release Please** via Conventional Commits.
*   **Dependency Management:** **Dependabot** automatically monitors and patches Python packages and GitHub Actions.

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

## ⚖️ Regulatory Compliance & Legal Defensibility

This architecture is designed to satisfy the strictest global regulatory frameworks for High-Risk AI systems. A dedicated `/compliance` directory provides centralized programmatic enforcement mechanisms:

*   **GDPR (Right to be Forgotten):** Programmatic purge utilities (`gdpr_purge.py`) to permanently scrub PII from MLflow tracking databases and batch datasets.
*   **EU AI Act (Human-in-the-Loop):** Decision routing gateways (`eu_ai_act_hitl_router.py`) that intercept borderline predictions and push them to manual auditor queues.
*   **DORA (Operational Resilience):** Circuit breaker patterns (`dora_circuit_breaker.py`) ensuring inference degradation fails safely to rules-based fallbacks.
*   **Automated Compliance Gates:** Property-based dynamic testing via `hypothesis` running in GitHub Actions to mathematically prove regulatory enforcement.

## 🚀 Quickstart & Setup

This repository uses a `Makefile` to simplify orchestration. View all available commands by running `make help`.

### Option 1: Run Multi-Project Fleet (Recommended)
Deploy the fully containerized, three-project architecture simultaneously using unified Docker Compose orchestration:


```

bash
git clone [https://github.com/SatyaIFD/mlops-governance-reference.git](https://github.com/SatyaIFD/mlops-governance-reference.git?utm_source=gemini)
cd mlops-governance-reference
make up
sudo docker ps
make down

```

### Option 2: Local Development Setup
If you wish to run the test suites or modify the architecture natively:


```

bash
conda activate mlops-lab
uv pip install pandas numpy scikit-learn joblib pytest faker time-machine anyio hydra-core pdoc kagglehub hypothesis
make test
make security

```

## 🤖 Acknowledgments

This enterprise reference architecture was conceptualized, structured, and developed with the assistance of Google Gemini. AI was utilized to accelerate CI/CD boilerplate generation, refine MLOps architectural patterns, and implement robust, enterprise-grade compliance frameworks.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.
