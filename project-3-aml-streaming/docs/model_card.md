# Model Card: Real-Time AML Streaming Anomaly Detector

## 1. Model Details
* **Model Name:** Streaming Random Forest AML Classifier
* **Version:** 1.2.0
* **Model Type:** Supervised Ensemble Classifier (`sklearn.ensemble.RandomForestClassifier`)
* **Target Variable:** `Is_laundering` (Binary: `0` = Normal Transaction, `1` = Money Laundering)
* **Maintainers:** MLOps Governance & Financial Crime Engineering Team

---

## 2. Intended Use & Scope
* **Primary Intended Use:** Real-time scoring of streaming transaction events to flag suspicious money laundering typologies (smurfing, layering, structuring, and pass-through mules) into `artifacts/aml_alerts.jsonl`.
* **Out-of-Scope Uses:** Autonomous account blocking without human investigator review; credit risk scoring; individual consumer behavioral profiling.

---

## 3. Training & Out-of-Sample Evaluation Windows
* **Base Dataset:** Synthetic AML Transaction Data (SAML-D)
* **Training Window (`src/models/train.py`):** Ingests the first `100,000` events—using events `1–50,000` strictly to warm up the rolling 1h/24h state caches and events `50,001–100,000` (`40,000` train / `10,000` validation) to fit the classifier and select an empirical F1 probability cutoff (`0.68`). Note: because base laundering prevalence is ~0.1%, the 10,000-row validation slice contains roughly a dozen positives, so `0.68` is a lightweight heuristic cutoff rather than a probability-calibrated threshold.
* **Out-of-Sample Streaming Window (`src/streaming/pipeline.py`):** Warms up `StreamingStateManager` across the entire `100,000`-event training prefix (`warmup_events_count: 100000`) before scoring events `100,001–200,000` (`total_processed: 100000`) strictly out-of-sample.

---

## 4. Model Feature Matrix
The model excludes demographic attributes, operating strictly on transactional amount and stateful graph dynamics:

| Feature Name | Feature Type | Description |
|---|---|---|
| `Amount` | Continuous | Nominal transaction value in the record's `Payment_currency` (`UK pounds`, `US dollar`, `Euro`, etc.). |
| `pass_through_ratio_1h` | Continuous $[0, 1]$ | Ratio of 1-hour outgoing volume to incoming volume. |
| `is_structuring` | Binary (0 or 1) | Nominal indicator for amounts in the $8,000–$9,999.99 band (applied without FX conversion). |
| `velocity_acceleration` | Continuous | Ratio of 1-hour transaction frequency vs. 24-hour average hourly rate. |
| `fan_out_count_24h` | Integer | Count of unique beneficiary accounts receiving funds in 24 hours. |
| `receiver_inflow_count_1h` | Integer | Count of inbound transactions to the beneficiary in 1 hour. |
| `receiver_inflow_amount_1h` | Continuous | Cumulative inbound volume to the beneficiary in 1 hour. |

---

## 5. Out-of-Sample Performance & Operational Guardrails
* **Alert Threshold:** `0.68` (persisted in `aml_streaming_model.meta.json` and SHA-256 verified on load).
* **Out-of-Sample Recall:** `23.77%` (`29 / 122` laundering events intercepted; target SLO $\ge 20.0\%$).
* **Out-of-Sample Alert Precision & Volume:** `0.48%` (`6,068` alerts written to `artifacts/aml_alerts.jsonl` across 100,000 scored events, or ~209 alerts per true positive).
* **Fallback Strategy:** Malformed or failed events route to `artifacts/dlq/dead_letter_queue.jsonl` via `record_processing_failure()`.
