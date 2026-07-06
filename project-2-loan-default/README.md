# 📑 Project 2: Responsible AI Loan Default Risk Underwriting Engine

This production-grade MLOps repository implements a fair, bias-mitigated XGBoost underwriting engine designed to evaluate consumer loan application default risks. The architecture features an operational pipeline that integrates automated real-time ingestion, feature alignment, statistical data drift detection, and algorithmic fairness monitoring using regulatory governance constraints.

---

## 🏗️ Repository Architecture

The project space is partitioned into structured micro-modules separating infrastructure states, pipeline components, and documentation assets:

```text
project-2-loan-default/
├── artifacts/               # Local ephemeral telemetry and statistical reports (Git Ignored)
│   ├── compliance_audit_report.json
│   └── data_drift_report.json
├── data/                    # Storage partition for raw data inputs and parquet splits
│   ├── loan-default.csv
│   └── processed/
├── docs/                    # Regulatory living compliance ledgers
│   └── model_card.md
├── src/                     # Core Application Source Code
│   ├── evaluation/          # Offline evaluation and test matrix validation engines
│   │   └── evaluate.py
│   ├── governance/          # Markdown compliance compilation and assembly components
│   │   └── audit.py
│   ├── inference/           # FastAPI application service layers
│   │   └── app.py
│   ├── ingestion/           # Data extraction, schema validation, and loading utilities
│   │   ├── extract.py
│   │   ├── ingestion.py
│   │   ├── schemas.py
│   │   └── utils.py
│   ├── monitoring/          # Production telemetry logging and fairness scanners
│   │   ├── drift_monitor.py
│   │   ├── generate_traffic.py
│   │   └── observability.py
│   ├── training/            # Bias-mitigating training pipelines using sample re-weighting
│   │   ├── data_splitter.py
│   │   └── train.py
│   └── validation/          # Statistical distribution check engines (PSI)
│       ├── drift_check.py
│       └── validate.py
```

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

## 🚀 Production Operational Playbook

Run these execution commands sequentially from the workspace root (`mlops-governance-reference`) to operate the complete pipeline end-to-end:

### 1. Launch the Live Inference Service
Boot the FastAPI application container mapping your internal MLflow model registry artifacts:
```bash
docker start loan-default-container
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
