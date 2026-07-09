# 📑 Project 2: Responsible AI Loan Default Risk Underwriting Engine

This production-grade MLOps repository implements a fair, bias-mitigated XGBoost underwriting engine designed to evaluate consumer loan application default risks. The architecture features a production-ready workflow that integrates automated real-time ingestion, feature alignment, statistical data drift detection, and algorithmic fairness monitoring orchestrated cleanly via Apache Airflow.

---

## 🏗️ Repository Architecture

The project space is partitioned into structured micro-modules separating infrastructure orchestration states, pipeline components, and documentation assets:

```text
project-2-loan-default/
├── airflow/                 # Apache Airflow Local Home (Logs, DB, and configs Git Ignored)
│   ├── airflow.cfg
│   ├── dags/
│   │   └── loan_underwriting_dag.py  # Production Automated Directed Acyclic Graph (DAG)
│   └── webserver_config.py
├── artifacts/               # Local ephemeral telemetry and statistical reports (Git Ignored)
│   ├── compliance_audit_report.json
│   └── data_drift_report.json
├── data/                    # Storage partition for raw data inputs and parquet splits
│   ├── loan-default.csv
│   └── processed/
├── docker/                  # Isolated deployment container configurations
│   └── Dockerfile
├── docs/                    # Regulatory living compliance ledgers
│   ├── gdpr_controls.md     # Privacy controls, right to explanation, and SHAP mappings
│   ├── model_card.md        # Dynamic performance, fairness, and feature PSI data sheet
│   └── risk_assessment.md   # Model failure mitigation protocols and incident response thresholds
├── kubernetes/              # Structural Infrastructure-as-Code (IaC) serving blueprints
│   ├── deployment.yaml      # Multi-replica deployment with health-check probes
│   └── service.yaml         # NodePort internal load balancing configuration
└── src/                     # Core Application Source Code
    ├── evaluation/          # Offline evaluation and test matrix validation engines
    │   └── evaluate.py
    ├── governance/          # Markdown compliance compilation and assembly components
    │   └── audit.py
    ├── inference/           # FastAPI application service layers
    │   └── app.py
    ├── ingestion/           # Data extraction, schema validation, and loading utilities
    │   ├── extract.py
    │   ├── ingestion.py
    │   ├── schemas.py
    │   └── utils.py
    ├── monitoring/          # Production telemetry logging and fairness scanners
    │   ├── drift_monitor.py
    │   ├── generate_traffic.py
    │   └── observability.py
    ├── training/            # Bias-mitigating training pipelines using sample re-weighting
    │   ├── data_splitter.py
    │   └── train.py
    └── validation/          # Statistical distribution check engines (PSI)
        ├── drift_check.py
        └── validate.py


---

## ⚖️ MLOps Governance & Fairness Framework

To satisfy strict compliance mandates for fair lending practices, this system integrates automated algorithmic protections targeting age discrimination (comparing the Protected Young Cohort `<30` against the Baseline Mature Cohort `>=30`):

* **Bias Mitigation:** Employs an offline sample re-weighting algorithm during training to calculate sample weights that penalize historical demographic parity disparities before fitting the XGBoost classifier.
* **The 4/5ths Rule Mandate:** Production traffic outputs are continuously audited to track the **Disparate Impact (DI) Ratio**:

$$\text{Disparate Impact Ratio} = \frac{\text{Approval Rate of Protected Cohort}}{\text{Approval Rate of Baseline Cohort}}$$

* **Compliance Boundaries:** The system enforces a strict regulatory boundary. If the calculated rolling Disparate Impact Ratio falls outside the standard $[0.80, 1.25]$ range, the monitoring engine flags an adverse breach alert to initiate automated retraining.

---

## 📡 Production Drift Detection

Population stability is verified by tracking the **Population Stability Index (PSI)** across continuous numerical financial features (`CreditScore`, `Income`, `LoanAmount`, `DTIRatio`) to determine if production feature distributions have shifted significantly away from the baseline model training data distribution:

* **$\text{PSI} < 0.10$**: Stable population footprint; no modifications required.
* **$0.10 \le \text{PSI} \le 0.25$**: Moderate shift detected; system surfaces warning diagnostics.
* **$\text{PSI} > 0.25$**: Significant data drift occurred; system triggers data-refresh alerts.

---

## 🌪️ Production Orchestration Engine (Apache Airflow)

The operational pipeline is entirely automated via an Apache Airflow Directed Acyclic Graph (DAG) running locally inside an isolated repository environment. The scheduler coordinates tasks sequentially, parsing execution results and exporting compliance states without manual shell intervention.

```
[stream_production_traffic] ──> [audit_fairness_metrics] ──> [check_population_data_drift] ──> [compile_regulatory_model_card]

```

### ⚙️ Environment Initialization

To manage the local Airflow system cleanly without introducing cross-talk or leaking state files, all core CLI actions must pass the explicit local home path environment variable:

```bash
# Pin the environment variable path to your local workspace partition
export AIRFLOW_HOME=$(pwd)/project-2-loan-default/airflow

# Run database migrations to prepare the local SQLite instance
airflow db migrate

# Create the primary administrative user profile
airflow users create --username admin --firstname Satya --lastname Linux --role Admin --email admin@example.com --password adminpwd

```

### 🚀 Running the Core Orchestration Daemons

Run the background service daemons directly to process your pipelines asynchronously:

```bash
# Start the scheduler loops to monitor directory graphs
AIRFLOW_HOME=$(pwd)/project-2-loan-default/airflow airflow scheduler > scheduler_debug.log 2>&1 &

# Start the web UI dashboard server instance on local port 8080
AIRFLOW_HOME=$(pwd)/project-2-loan-default/airflow airflow webserver --port 8080 > webserver_debug.log 2>&1 &

```

---

## 🚀 Manual Execution & Testing Playbook

For standalone diagnostic verification or explicit debugging of independent pipeline layers, execute these commands sequentially from the repository workspace root:

### 1. Launch the Local Containerized Microservice

Boot the application serving image context natively, mapping port 8000 for FastAPI application traffic:

```bash
docker run -d --name loan-api-server -p 8000:8000 loan-prediction-service:v1

```

### 2. Stream Production Stress-Test Traffic

Execute the simulation stream to push 40 ultra-high-risk financially stressed applicant payloads to the endpoint, testing the engine's behavior under economic strain:

```bash
PYTHONPATH=. python project-2-loan-default/src/monitoring/generate_traffic.py

```

### 3. Run the Real-Time Fairness Scanner

Parse the line-delimited telemetry log ledger to analyze the rolling Disparate Impact Ratio and check for regulatory bias threshold breaches:

```bash
PYTHONPATH=. python project-2-loan-default/src/monitoring/drift_monitor.py

```

### 4. Execute the Population Drift Engine

Evaluate distribution stability using the validation engine to generate fresh PSI metrics and update data status snapshots:

```bash
PYTHONPATH=. python project-2-loan-default/src/validation/drift_check.py

```

### 5. Assemble the Living Compliance Model Card

Compile the collective drift telemetry and fairness statuses into the formal markdown documentation sheet for regulatory audit reviews:

```bash
PYTHONPATH=. python project-2-loan-default/src/governance/audit.py

```

---

*Report auto-generated by the Automated Governance Engine on an active MLOps deployment.*