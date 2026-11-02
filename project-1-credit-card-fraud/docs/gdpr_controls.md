# GDPR & Data Privacy Framework Compliance Control Matrix

This matrix documents the technical controls implemented across the Project 1 Credit Card Fraud pipeline to support European General Data Protection Regulation (GDPR) engineering requirements.

| GDPR Mandate | Pipeline Implementation Control | Operational Safeguard |
| :--- | :--- | :--- |
| **Article 5: Data Minimization** | Model training and inference operate strictly on 28 PCA-transformed components (`V1`–`V28`) plus `scaled_amount` and `scaled_time`. | Direct PII (names, card numbers, CVVs, or user IDs) is absent from training Parquet files and `TransactionData` API payloads. |
| **Article 22: Automated Decisions** | The `/predict` endpoint (`src/inference/main.py`) emits a binary class prediction (`prediction`) and probability (`risk_score`) without executing autonomous account closures. | Downstream payment switches can combine `risk_score` with step-up verification or human review (`compliance/eu_ai_act_hitl_router.py`). |
| **Article 32: Security of Processing** | API endpoints enforce constant-time `X-API-Key` verification (`secrets.compare_digest`), strict finite-float Pydantic validation (`allow_inf_nan=False`), and DORA circuit breaking (`MLSystemCircuitBreaker`). | Prevents NaN/Inf payload injection, unauthorized access (`HTTP 401`/`500`), and cascading downstream failures (`HTTP 503`). |
