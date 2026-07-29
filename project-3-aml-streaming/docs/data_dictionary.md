# Data Dictionary & Schema Governance Contract

This document specifies the data contracts, state feature definitions, mathematical formulas, and error taxonomy for the **Project 3 Real-Time AML Streaming Observability Engine**.

---

## 📥 1. Raw Stream Ingestion Schema Contract

Every event entering the pipeline must satisfy the following schema rules before it is passed to the state cache. Events violating these rules are routed to the **Dead-Letter Queue (DLQ)**.

| Field Name | Type | Required | Valid Range / Format | Description |
|---|---|---|---|---|
| `Timestamp` | `string` / `datetime` | **Yes** | `YYYY-MM-DD HH:MM:SS` | UTC timestamp of the transaction event. |
| `Sender_account` | `string` | **Yes** | Non-empty alphanumeric string | Unique identifier for the sending entity/account. |
| `Receiver_account` | `string` | **Yes** | Non-empty alphanumeric string | Unique identifier for the receiving beneficiary account. |
| `Amount` | `float` | **Yes** | $> 0.0$ | Transaction currency amount (must be positive). |
| `Payment_type` | `string` | No | `CHEQUE`, `CREDIT`, `ACH`, `CASH`, `WIRE` | Medium used to initiate transfer. |
| `Is_laundering` | `integer` | No | `0` or `1` | Binary ground-truth target label (0 = Normal, 1 = Laundering). |

---

## 🧮 2. Stateful & Graph Feature Definitions

These features are computed dynamically in-memory by `StreamingStateManager` using rolling time-based lookback windows (1h and 24h) with automated TTL key eviction.

### A. Pass-Through Ratio (1-Hour Lookback)
* **Variable Name:** `pass_through_ratio_1h` | **Type:** `float` | **Range:** $[0.0, 1.0]$
* **Purpose:** Detects "mule" layering where incoming funds are rapidly transferred out within 60 minutes.
* **Formula:**

$$\text{Pass-Through Ratio}_{1h} = \min\left(1.0, \frac{\text{Outgoing Volume}_{1h}}{\text{Incoming Volume}_{1h} + \epsilon}\right)$$

---

### B. Structuring Indicator Flag
* **Variable Name:** `is_structuring` | **Type:** `integer` | **Range:** $\{0, 1\}$
* **Purpose:** Flags transaction amounts falling in the $\$8,000.00$ to $\$9,999.99$ corridor designed to evade mandatory $\$10,000$ Currency Transaction Reporting (CTR) thresholds.
* **Formula:** $\text{is\_structuring} = 1 \text{ if } 8000.0 \le \text{Amount} < 10000.0 \text{ else } 0$.

---

### C. Velocity Acceleration Rate
* **Variable Name:** `velocity_acceleration` | **Type:** `float` | **Range:** $[0.0, \infty)$
* **Purpose:** Measures sudden transaction frequency spikes by comparing the sender's 1-hour outgoing transaction count against their 24-hour average hourly rate.
* **Formula:**

$$\text{Velocity Acceleration} = \frac{\text{Outgoing Count}_{1h}}{\left(\frac{\text{Outgoing Count}_{24h}}{24.0}\right) + \epsilon}$$

---

### D. Fan-Out Dispersion Count (24-Hour Lookback)
* **Variable Name:** `fan_out_count_24h` | **Type:** `integer` | **Range:** $[0, \infty)$
* **Purpose:** Tracks the count of distinct beneficiary accounts receiving funds from the sender over a 24-hour window to catch fan-out dispersion networks.

---

### E. Receiver Inflow Count & Amount (1-Hour Lookback)
* **Variable Names:** `receiver_inflow_count_1h` (`integer`), `receiver_inflow_amount_1h` (`float`)
* **Purpose:** Tracks inbound transaction volume and count at beneficiary accounts within 60 minutes to catch smurfing aggregation.

---

## 🚨 3. Data Quality & DLQ Error Taxonomy

| Error Code | Trigger Condition | Severity | DLQ Action |
|---|---|---|---|
| `ERR_MISSING_MANDATORY` | Field `Timestamp`, `Sender_account`, `Receiver_account`, or `Amount` is null/missing. | **High** | Quarantine immediately; log field name. |
| `ERR_INVALID_AMOUNT` | `Amount <= 0.0` or non-numeric type conversion failure. | **High** | Quarantine immediately; reject negative/zero transfers. |
| `ERR_MALFORMED_TIMESTAMP` | Date string cannot be parsed into a valid UTC `datetime` object. | **Medium** | Quarantine immediately; inspect date format string. |