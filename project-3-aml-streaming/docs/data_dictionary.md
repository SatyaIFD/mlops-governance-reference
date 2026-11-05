# Data Dictionary & Schema Governance Contract

This document specifies the data contracts, state feature definitions, mathematical formulas, and error taxonomy for the **Project 3 Real-Time AML Streaming Observability Engine**.

---

## 📥 1. Raw Stream Ingestion Schema Contract

Every event entering the pipeline must satisfy the following schema rules before it is passed to the state cache. Events violating these rules are routed to the **Dead-Letter Queue (DLQ)**.

| Field Name | Type | Required | Valid Range / Format | Description |
|---|---|---|---|---|
| `Timestamp` | `string` / `datetime` | **Yes** | `YYYY-MM-DD HH:MM:SS` | Event timestamp (derived from `Date` + `Time` in SAML-D). |
| `Sender_account` | `string` | **Yes** | Non-empty string | Unique identifier for the sending account (HMAC-pseudonymized when enabled). |
| `Receiver_account` | `string` | **Yes** | Non-empty string | Unique identifier for the receiving account (HMAC-pseudonymized when enabled). |
| `Amount` | `float` | **Yes** | $> 0.0$ (finite) | Nominal transaction amount in the record's `Payment_currency` (`UK pounds`, `US dollar`, `Euro`, etc.). |
| `Payment_type` | `string` | No | `Credit card`, `Debit card`, `Cheque`, `ACH`, `Cross-border`, `Cash Deposit`, `Cash Withdrawal`, `WIRE` | Transfer medium in SAML-D (not used as a model feature). |
| `Is_laundering` | `integer` | No | `0` or `1` | Binary ground-truth target label (`0` = Normal, `1` = Laundering). |

---

## 🧮 2. Stateful & Graph Feature Definitions

These 7 features (`CANONICAL_AML_FEATURES`) are computed dynamically in-memory by `StreamingStateManager` using rolling event-time lookback windows (`1h` and `24h`):

1. **`Amount` (`float`):** Raw positive transaction amount.
2. **`pass_through_ratio_1h` (`float`, $[0.0, 1.0]$):** $\min(1.0, \text{Outgoing Volume}_{1h} / (\text{Incoming Volume}_{1h} + \epsilon))$.
3. **`is_structuring` (`int`, 0 or 1):** Nominal band indicator (`1` if $8000.0 \le \text{Amount} < 10000.0$ else `0`; applied directly to raw `Amount` without FX conversion).
4. **`velocity_acceleration` (`float`, $[0.0, \infty)$):** $\text{Outgoing Count}_{1h} / ((\text{Outgoing Count}_{24h} / 24.0) + \epsilon)$.
5. **`fan_out_count_24h` (`int`, $[0, \infty)$):** Distinct beneficiary accounts paid by the sender within the 24-hour event-time window.
6. **`receiver_inflow_count_1h` (`int`, $[1, \infty)$):** Inbound transaction count at the beneficiary account within 1 hour.
7. **`receiver_inflow_amount_1h` (`float`, $(0.0, \infty)$):** Cumulative inbound transaction amount at the beneficiary account within 1 hour.

---

## 🚨 3. Data Quality & DLQ Error Taxonomy

| Error Code | Trigger Condition | Severity | DLQ Action |
|---|---|---|---|
| `ERR_INVALID_PAYLOAD_TYPE` | Incoming payload is not a Python `dict`. | **High** | Quarantine immediately in `dead_letter_queue.jsonl`. |
| `ERR_MISSING_MANDATORY` | `Timestamp`, `Sender_account`, `Receiver_account`, or `Amount` is null/empty. | **High** | Quarantine immediately; log missing field. |
| `ERR_INVALID_AMOUNT` | `Amount <= 0.0`, `NaN`/`Inf`, or non-numeric string. | **High** | Quarantine immediately. |
| `ERR_MALFORMED_TIMESTAMP` | String cannot be parsed as `YYYY-MM-DD HH:MM:SS`. | **Medium** | Quarantine immediately. |
| `ERR_ENRICHMENT_FAILURE` / `ERR_SCORING_FAILURE` | Runtime exception during feature enrichment or model scoring. | **High** | Decrements `clean_events` via `record_processing_failure()` and quarantines record. |
