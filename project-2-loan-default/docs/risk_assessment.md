
# ⚖️ Model Risk Assessment & Algorithmic Mitigation Protocol

**Project 2: Loan Default Risk Underwriting Engine**

---

## 1. Core Model Failure Modes & Operational Risk

This model scores retail loan applications to assist human credit underwriting. Unmitigated failures introduce two primary risks:

* **False Negatives (Under-prediction of Default Risk):** Recommending approval for high-risk applicants, increasing Non-Performing Loan (NPL) ratios.
* **False Positives & Demographic Disparity:** Systematically recommending rejection for younger applicants (`Age < 30`) due to shorter credit histories.

## 2. Quantitative Guardrails Implemented in Code

The pipeline enforces automated checks across evaluation (`src/evaluation/evaluate.py`), drift validation (`src/validation/drift_check.py`), and live telemetry auditing (`src/monitoring/drift_monitor.py`):

| Check Layer | Metric / Indicator | Implemented Threshold | Pipeline Behavior in Code |
| --- | --- | --- | --- |
| **Holdout Performance** | Test ROC-AUC (`evaluate.py`) | $\text{ROC-AUC} \ge 0.70$ | Sets `minimum_performance_passed = True`; exits non-zero if breached. |
| **Holdout & Live Fairness** | Disparate Impact Ratio (`Age < 30` vs. `Age >= 30`) | $0.80 \le \text{DI} \le 1.25$ | Reports `COMPLIANT` within bounds, `NON-COMPLIANT (BREACH)` outside bounds, or `INCONCLUSIVE (INSUFFICIENT_DATA)` when cohort counts are below `min_cohort_size = 30` or baseline approval rates are zero. |
| **Population Drift** | Population Stability Index (`drift_check.py`) | $\text{PSI} < 0.10$ (`STABLE`), $0.10 \le \text{PSI} < 0.25$ (`MODERATE_SHIFT`), $\text{PSI} \ge 0.25$ (`SEVERE_DRIFT`) | Evaluates `CreditScore`, `Income`, `LoanAmount`, and `DTIRatio`; exits non-zero when `--fail-on-drift` or `--simulate-drift` detects severe drift. |
| **Operational Resilience** | Consecutive Inference Exceptions (`dora_circuit_breaker.py`) | $\ge 3$ failures (`recovery_timeout_s = 30.0`) | Trips `MLSystemCircuitBreaker` to `OPEN`, returning `HTTP 503` with single-probe `HALF_OPEN` recovery locking. |

## 3. Implemented Fair Lending & Resilience Controls

* **Pre-Processing Sample Reweighting (`src/training/train.py`):** `calculate_mitigation_weights()` computes inverse joint-probability weights across age cohorts (`Age < 30` vs. `Age >= 30`) and default labels (`Default == 0` vs. `Default == 1`) prior to fitting `XGBClassifier`.
* **Single-Row Categorical Preservation (`src/inference/main.py`):** `prepare_single_row_features()` encodes incoming requests with `drop_first=False` before reindexing to the model's exact training feature columns so single-row payloads never lose their active category level.
* **Mandatory Human-in-the-Loop Routing (`src/inference/main.py`):** Every `/predict` call invokes `enforce_human_in_the_loop()`, setting `action: "MANUAL_REVIEW_REQUIRED"`, `review_status: "PENDING_HUMAN_REVIEW"`, and `final_decision: null`.
