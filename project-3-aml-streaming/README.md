# Project 3: Real-Time Stateful AML Streaming Observability Engine

A stateful streaming inference engine built to detect money-laundering typologies (smurfing, layering, structuring, pass-through mules) in transaction streams with Dead-Letter Queue (DLQ) isolation and schema validation.

---

## 🏛️ System Architecture Overview

```text
[ Live Transaction Stream ]
│
▼
┌─────────────────────────┐
│ StreamingIngestionEngine│ ─── (Corrupted / Poisoned) ───► [ Dead-Letter Queue (DLQ) ]
└─────────────────────────┘                                   │
│ (Clean Payload)                                             ▼
▼                                                   [ scripts/replay_dlq.py ]
┌─────────────────────────┐
│ StreamingStateManager   │ ─── (In-Memory Lookback Cache + 24h Event-Time Watermark Eviction)
└─────────────────────────┘
│ (7 Graph & Velocity Features)
▼
┌─────────────────────────┐
│ AMLAnomalyDetector      │ ─── (Random Forest Classifier + SHA-256 Verified .meta.json)
└─────────────────────────┘
│
▼
[ 🚨 Real-Time Alert Sink (artifacts/aml_alerts.jsonl) & Execution Report ]
```

---

## 🔑 Core Features & Mathematical Formulations

1. **Pass-Through Ratio (1-Hour Window):**
   $$\text{Pass-Through Ratio}_{1h} = \min\left(1.0, \frac{\text{Outgoing Volume}_{1h}}{\text{Incoming Volume}_{1h} + \epsilon}\right)$$
2. **Structuring Flag (8k–10k Nominal Range):** Flags payments in the $8,000$–$9,999.99$ nominal band.
3. **Velocity Acceleration:** Compares 1-hour transaction frequency against the account's 24-hour average hourly rate.
4. **Fan-Out Dispersion Count (24-Hour Window):** Tracks unique beneficiary accounts receiving funds.
5. **Receiver Inflow Count & Volume (1-Hour Window):** Measures incoming transaction frequency and volume at beneficiary accounts.

---

## 📊 Out-of-Sample Performance Benchmarks

| Metric | Benchmark Value |
| --- | --- |
| **Total Stream Ingestion** | 200,000 records (100,000 warmup + 100,000 out-of-sample scored) |
| **Data Quality Compliance** | 100.0% clean events (0 DLQ drops) |
| **Out-of-Sample Streaming Recall** | **58.2%** (71 / 122 laundering events intercepted) |
| **Out-of-Sample Alert Precision** | **0.87%** (8,119 alerts written to `artifacts/aml_alerts.jsonl` at threshold `0.68`) |
| **Stateful Feature Matrix** | 7 graph & velocity features |
| **State Memory Bounds** | Swept every 25,000 events under 24h event-time watermark eviction |

---

## 🔒 Operational & MLflow Engineering Notes

* **100k Warmup + 100k Out-of-Sample Separation:** `train.py` ingests 100,000 events (50,000 warmup + 40,000 train + 10,000 validation). `pipeline.py` warms up state across the 100,000-event training prefix before scoring events 100,001–200,000 strictly out-of-sample.
* **Scikit-Learn `skops` Persistence & SHA-256 Verification:** Model artifacts are persisted via `skops.io` (`TRUSTED_SKOPS_TYPES`) alongside `aml_streaming_model.meta.json` and SHA-256 verified at load time.
