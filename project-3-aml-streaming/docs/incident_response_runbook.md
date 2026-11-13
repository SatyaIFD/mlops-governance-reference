# Operational Incident Response Runbook

Operational procedures for responding to pipeline failures, memory spikes, recall drops, and Dead-Letter Queue (DLQ) bursts in the streaming environment.

---

## 🚨 Incident Playbooks

### Playbook A: DLQ Spike / High Ingestion Failure Rate
* **Trigger:** Data Quality Governance metric (`data_quality_pct`) drops below $99.0\%$ clean events, or DLQ write rate spikes.
* **Root Cause Analysis:**
  1. Inspect quarantined records in `artifacts/dlq/dead_letter_queue.jsonl`.
  2. Identify error codes: `ERR_INVALID_PAYLOAD_TYPE`, `ERR_MISSING_MANDATORY`, `ERR_INVALID_AMOUNT`, `ERR_MALFORMED_TIMESTAMP`, `ERR_ENRICHMENT_FAILURE`, or `ERR_SCORING_FAILURE`.
* **Remediation:**
  * Update `StreamingIngestionEngine` parsing rules in `src/ingestion/stream_ingestor.py` if upstream schema shifted.
  * Re-inject repaired DLQ payloads using `scripts/replay_dlq.py`.

---

### Playbook B: In-Memory State Manager OOM / High RAM Usage
* **Trigger:** Process memory exceeds $80\%$ of container limit during continuous stream processing.
* **Root Cause Analysis:** High cardinality of active accounts within the 24-hour window.
* **Remediation:**
  1. Trigger event-time watermark garbage collection (`self.watermark - timedelta(hours=24)`):
     ```python
     state_manager._evict_dormant_accounts()
     ```
  2. Reduce the periodic eviction interval (currently every `25,000` events) in `src/streaming/pipeline.py`.

---

### Playbook C: Out-of-Sample Recall Drop Below SLO (< 20%)
* **Trigger:** Out-of-sample interception recall (`recall_percentage`) falls below the $20.0\%$ operational target (baseline `58.2%` on events 100,001–200,000).
* **Root Cause Analysis:** Emerging laundering typologies or distribution shift across the stream.
* **Remediation:**
  1. Inspect `artifacts/pipeline_execution_report.json` and verify `model_sha256` matches `aml_streaming_model.meta.json`.
  2. Trigger model retraining (`python -m src.models.train`) on recent mature stream records.
