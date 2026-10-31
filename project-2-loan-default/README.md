
# 📑 Project 2: Fair Loan Default Risk Underwriting Engine

An MLOps underwriting pipeline implementing sample-reweighted XGBoost training, single-row categorical alignment, authenticated FastAPI serving, EU AI Act human-review routing metadata, and live telemetry drift/fairness monitoring.

---

## 🏗️ Repository Structure

```text
project-2-loan-default/
├── Dockerfile # Rootless (UID 10001) production container definition
├── artifacts/ # Runtime telemetry logs and JSON audit reports
├── data/ # Raw CSV and stratified Parquet splits (train.parquet, test.parquet)
├── docs/ # Model card, risk assessment, and GDPR control documentation
├── kubernetes/ # Restricted PSS deployment.yaml and NodePort service.yaml
├── src/
│ ├── evaluation/ # Holdout ROC-AUC, accuracy, log-loss & demographic parity evaluator (evaluate.py)
│ ├── governance/ # Automated Markdown model card compiler (audit.py)
│ ├── inference/ # Authenticated FastAPI service (main.py)
│ ├── ingestion/ # Kaggle extraction, schema validation & Parquet partitioning
│ ├── monitoring/ # Telemetry logger (observability.py), traffic generator & fairness scanner
│ ├── training/ # Sample-reweighted XGBoost trainer (train.py)
│ └── validation/ # Parquet validator (validate.py) & PSI drift monitor (drift_check.py)
└── tests/ # Pytest unit and contract test suites

```

---

## ⚖️ Decision Semantics & Fairness Monitoring

1. **Separated Decision Semantics:** The `/predict` endpoint separates model scoring from final human adjudication:
* `model_recommendation`: `"APPROVE"` or `"DECLINE"`
* `review_status`: `"PENDING_HUMAN_REVIEW"`
* `final_decision`: `null` (until recorded by an upstream case review system)
* `age_group`: `"UNDER_30"` or `"30_AND_OVER"`


2. **Sample Reweighting:** `src/training/train.py` computes pre-processing sample weights across age cohorts (`Age < 30` vs. `Age >= 30`) and default labels to mitigate historical demographic parity disparities.
3. **Honest Telemetry Auditing:** `src/monitoring/drift_monitor.py` filters for valid `HTTP 200` observations and reports `INCONCLUSIVE (INSUFFICIENT_DATA)` whenever cohort sample sizes or baseline approval rates are zero—never defaulting to `1.0` (`COMPLIANT`).

---

## 🚀 Execution Playbook

```bash
# 1. Start the containerized service and Nginx gateway
sudo docker compose up -d loan-default-api api-gateway

# 2. Stream representative authenticated traffic through Nginx
EXPECTED_API_KEY=mlops-dev-secret-key PYTHONPATH=project-2-loan-default \
python project-2-loan-default/src/monitoring/generate_traffic.py --mode representative --count 40

# 3. Run the fairness scanner on collected telemetry
PYTHONPATH=project-2-loan-default python project-2-loan-default/src/monitoring/drift_monitor.py

# 4. Run population drift checks on live telemetry (or pass --simulate-drift for synthetic stress test)
PYTHONPATH=project-2-loan-default python project-2-loan-default/src/validation/drift_check.py

# 5. Compile the Markdown Model Card
PYTHONPATH=project-2-loan-default python project-2-loan-default/src/governance/audit.py

```
