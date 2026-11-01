
# GDPR & Data Privacy Framework Compliance Control Matrix

This matrix documents the technical controls implemented across the Project 1 Credit Card Fraud pipeline and monorepo compliance modules to support European General Data Protection Regulation (GDPR) engineering requirements.

| GDPR Mandate | Pipeline Implementation Control | Operational Safeguard |
| --- | --- | --- |
| **Article 5: Data Minimization** | Model training and inference operate strictly on 28 PCA-transformed components (`V1`–`V28`) plus `scaled_amount` and `scaled_time`. | Direct PII (names, card numbers, CVVs) is absent from training Parquet files and `TransactionData` API payloads. |
| **Article 17: Right to Erasure** | Monorepo erasure utility (`compliance/gdpr_purge.py`) supports journaled, crash-recoverable record removal across Parquet/CSV datasets and SQLite tracking tables. | Ensures atomic `.bak` rollback if database scrubbing fails mid-operation and verifies zero residual rows post-purge. |
| **Article 22: Automated Decisions** | The `/predict` endpoint (`src/inference/main.py`) emits a binary class prediction (`prediction`) and probability (`risk_score`) without executing autonomous account closures. | Downstream payment switches can combine `risk_score` with step-up verification or human review (`compliance/eu_ai_act_hitl_router.py`). |
| **Article 32: Security of Processing** | API endpoints enforce constant-time `X-API-Key` verification (`secrets.compare_digest`), strict finite-float Pydantic validation (`allow_inf_nan=False`), and DORA circuit breaking (`MLSystemCircuitBreaker`). | Prevents NaN/Inf payload injection, unauthorized access (`HTTP 401`/`500`), and cascading downstream failures (`HTTP 503`). |
|


# 🇪🇺 GDPR Compliance & Automated Decision-Making Control Matrix

**Project 2: Loan Default Risk Underwriting Engine**

---

## 1. Principle of Data Minimization (Article 5)

The underwriting engine validates and processes a bounded 16-feature credit application schema (`LoanApplicationPayload` in `src/inference/main.py`):

* **Numerical Features (9):** `Age`, `Income`, `CreditScore`, `LoanAmount`, `DTIRatio`, `MonthsEmployed`, `InterestRate`, `NumCreditLines`, `LoanTerm` (plus the derived binary cohort indicator `is_young` for `Age < 30`).
* **Categorical Features (7):** `Education`, `EmploymentType`, `MaritalStatus`, `HasMortgage`, `HasDependents`, `LoanPurpose`, `HasCoSigner`.
* **Prohibited Attributes:** Direct personal identifiers (`LoanID`, names, national IDs, race, gender, religion, or postal codes) are excluded from model training matrices (`src/training/train.py`) and live inference payloads.

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
* **Offline Explainability:** Model feature attributions are evaluated during governance audits, while runtime telemetry logs capture structured probability and cohort metadata.

## 3. Telemetry Minimization & Right to Erasure (Article 17)

* **Minimized Telemetry Snapshots:** `ModelObservabilityLogger` (`src/monitoring/observability.py`) appends only a 5-field numerical snapshot (`Age`, `Income`, `CreditScore`, `LoanAmount`, `DTIRatio`), latency, HTTP status, and governance routing state to `artifacts/telemetry_logs.jsonl`—omitting applicant identifiers.
* **Journaled Right to be Forgotten:** Batch dataset and MLflow SQLite (`params`/`tags`) erasure for `user_id`-keyed records is enforced via the crash-recoverable state journal in `compliance/gdpr_purge.py`.
