# 🇪🇺 GDPR Compliance & Automated Decision-Making Control Matrix
**Project 2: Loan Default Risk Underwriting Engine**

---

## 1. Principle of Data Minimization (Article 5)
The underwriting engine validates and processes a bounded 16-feature credit application schema (`LoanApplicationPayload` in `src/inference/main.py`):

* **Numerical Features (9):** `Age`, `Income`, `CreditScore`, `LoanAmount`, `DTIRatio`, `MonthsEmployed`, `InterestRate`, `NumCreditLines`, `LoanTerm` (plus the derived binary cohort indicator `is_young` for `Age < 30`).
* **Categorical Features (7):** `Education`, `EmploymentType`, `MaritalStatus`, `HasMortgage`, `HasDependents`, `LoanPurpose`, `HasCoSigner`.
* **Prohibited Attributes:** Direct personal identifiers (`LoanID`, names, national IDs, race, gender, religion, or postal codes) are dropped prior to training (`src/training/train.py`) and excluded from inference payloads.

## 2. Human-in-the-Loop Decision Routing (Article 22 & EU AI Act Art. 14)
To prevent solely automated credit denials or approvals from taking legal effect without human oversight, `/predict` (`src/inference/main.py`) routes every scored application through `enforce_human_in_the_loop` (`compliance/eu_ai_act_hitl_router.py`):

```text
[Incoming Application] ──> [FastAPI /predict Scoring] ──> [EU AI Act HITL Router]
│
▼
action: "MANUAL_REVIEW_REQUIRED"
model_recommendation: "APPROVE" | "DECLINE"
review_status: "PENDING_HUMAN_REVIEW"
final_decision: null

```

* **Separated Decision Semantics:** The API emits `model_recommendation` (`"APPROVE"` or `"DECLINE"`) alongside `review_status: "PENDING_HUMAN_REVIEW"` and `final_decision: null`, ensuring final credit adjudication remains with a human credit officer.

## 3. Telemetry Minimization & Right to Erasure (Article 17)

* **Minimized Telemetry Snapshots:** `ModelObservabilityLogger` (`src/monitoring/observability.py`) appends a versioned (`schema_version: 2`) 5-field numerical snapshot (`Age`, `Income`, `CreditScore`, `LoanAmount`, `DTIRatio`), latency, HTTP status, and governance routing state to `artifacts/telemetry_logs.jsonl`.
* **Journaled Right to be Forgotten:** Batch dataset and MLflow SQLite (`params`/`tags`) erasure for `user_id`-keyed datasets is enforced via the crash-recoverable state journal in `compliance/gdpr_purge.py`.
