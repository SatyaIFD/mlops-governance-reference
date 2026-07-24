# Model Card: Real-Time AML Streaming Anomaly Detector

## 1. Model Details
* **Model Name:** Streaming Random Forest AML Classifier
* **Version:** 1.1.0
* **Model Type:** Supervised Ensemble Classifier (`sklearn.ensemble.RandomForestClassifier`)
* **Target Variable:** `Is_laundering` (Binary: `0` = Normal Transaction, `1` = Money Laundering)
* **Release Date:** July 2026
* **Maintainers:** MLOps Governance & Financial Crime Engineering Team

---

## 2. Intended Use & Scope
* **Primary Intended Use:** Real-time scoring of streaming transaction events to intercept complex money laundering typologies (smurfing, layering, structuring, and pass-through mules).
* **Out-of-Scope Uses:** Autonomous account blocking without human investigator review; credit risk scoring; individual consumer behavioral profiling.

---

## 3. Training & Validation Data
* **Base Dataset:** Synthetic AML Transaction Data (SAML-D)
* **Training Window:** 50,000 mature streaming transaction events following lookback cache warmup.
* **Sampling Strategy:** Dynamic class-weight balancing combined with downsampling to handle extreme class imbalance (~0.1% baseline laundering prevalence).

---

## 4. Model Feature Matrix
The model intentionally excludes demographic or personal attributes, operating strictly on transactional amount and stateful graph dynamics:

| Feature Name | Feature Type | Description |
|---|---|---|
| `Amount` | Continuous | Numerical transaction value in USD. |
| `pass_through_ratio_1h` | Continuous $[0, 1]$ | Ratio of 1-hour outgoing volume to incoming volume. |
| `is_structuring` | Binary $\{0, 1\}$ | Flag for transactions in the $\$8,000$–$\$9,999$ CTR threshold evasion range. |
| `velocity_acceleration` | Continuous | Ratio of 1-hour transaction frequency vs. 24-hour average hourly rate. |
| `fan_out_count_24h` | Integer | Count of unique beneficiary accounts receiving funds in 24 hours. |
| `receiver_inflow_count_1h` | Integer | Count of inbound transactions to the beneficiary in 1 hour. |
| `receiver_inflow_amount_1h` | Continuous | Cumulative inbound volume to the beneficiary in 1 hour. |

---

## 5. Performance & Operational Guardrails
* **Alert Threshold:** Prediction probability cutoff tuned to $0.40$ to optimize streaming recall while containing False Positive Rates (FPR).
* **Latency SLA:** Sub-10ms per-event inference time under rolling state lookback lookups.
* **Fallback Strategy:** If model inference fails or payload is malformed, payload routes to Dead-Letter Queue (DLQ) while maintaining zero data loss on stream flow.

---

## 6. Ethical Considerations & Fairness
* **Demographic Neutrality:** Features depend solely on network transaction flow and velocity, completely avoiding protected demographic attributes (age, location, gender, nationality).
* **Human-in-the-Loop (HITL):** High-risk alerts generate compliance queue items for human SAR (Suspicious Activity Report) review rather than executing automated funds confiscation.