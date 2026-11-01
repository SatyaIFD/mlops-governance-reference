
# GDPR & Regulatory Compliance Framework: AML Streaming Engine

This document outlines how the **Project 3 Real-Time AML Streaming Observability Engine** balances Anti-Money Laundering (AML) regulatory obligations under the Bank Secrecy Act (BSA) and EU Anti-Money Laundering Directives (4AMLD/5AMLD) with strict compliance under the **General Data Protection Regulation (GDPR)**.

---

## ⚖️ Key Regulatory Balancing Act

| Regulatory Mandate | Compliance Requirement | Engine Implementation |
| --- | --- | --- |
| **AML Directives (BSA / 5AMLD)** | Mandatory detection and reporting of suspicious transaction patterns. | Stateful rolling window features track multi-account pass-through and velocity (`src/features/velocity.py`). |
| **GDPR Data Minimization (Art. 5(1)(c))** | Personal data collected must be adequate, relevant, and limited to necessity. | State queues hold only minimal transaction metadata (`Timestamp`, `Amount`, Account IDs). |
| **GDPR Storage Limitation (Art. 5(1)(e))** | Personal data must not be kept longer than necessary. | Automated 24-hour event-time watermark eviction (`_evict_dormant_accounts`) wipes expired lookback states and seen `event_id` caches. |
| **GDPR Automated Decision-Making (Art. 22)** | Right not to be subject to purely automated legal decisions. | Anomaly flags (`is_anomaly == 1`) emit investigation telemetry rather than executing autonomous account freezes. |

---

## 🔐 1. Key-Versioned HMAC Account Pseudonymization

* **Deterministic HMAC Pseudonymizer (`KeyVersionedPseudonymizer`):** `src/ingestion/stream_ingestor.py` provides HMAC-SHA256 pseudonymization (`hmac_<key_version>_<16-char-digest>`) controlled via `AML_HMAC_KEY_VERSION` and `AML_HMAC_SECRET_KEY`.
* **DLQ Replay Continuity:** Pseudonymization is idempotent (`hmac_v*` tokens are never double-hashed), ensuring quarantined records in `dead_letter_queue.jsonl` map to the exact same account state during `scripts/replay_dlq.py` recovery.
* **No Direct PII Storage:** Raw names, addresses, social security numbers, or IP addresses are never ingested into `StreamingStateManager`.

---

## 🧹 2. Data Minimization & Event-Time Watermark Eviction

* **Rolling Memory Bounds:** Transaction records stored in `StreamingStateManager` deques are pruned relative to the stream watermark ($T_{\text{watermark}} - 24\text{h}$).
* **Dormant Key Garbage Collection:** `src/streaming/pipeline.py` invokes `_evict_dormant_accounts()` every 500 events, deleting empty account keys and expired deduplication entries from memory.

---

## 📋 3. Auditability & Lineage Reporting (GDPR Recital 71)

* **Deterministic Feature Enrichment:** Every transaction is enriched with 7 interpretable graph/velocity metrics (`CANONICAL_AML_FEATURES`) prior to scoring.
* **Lineage Logging:** Every pipeline pass exports a structured JSON summary (`artifacts/pipeline_execution_report.json`) tracking `data_quality_pct`, `dlq_events`, and `recall_percentage`.
