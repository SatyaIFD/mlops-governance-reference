
# Operational Incident Response Runbook

Operational procedures for responding to pipeline failures, memory spikes, recall drops, and Dead-Letter Queue (DLQ) bursts in the streaming environment.

---

## 🚨 Incident Playbooks

### Playbook A: DLQ Spike / High Ingestion Failure Rate

* **Trigger:** Data Quality Governance metric (`data_quality_pct`) drops below $99.0\%$ clean events, or DLQ write rate spikes.
* **Root Cause Analysis:**
1. Inspect recent quarantined records in `artifacts/dlq/dead_letter_queue.jsonl` (and any legacy `dlq_event_*.json` files).
2. Identify error codes: `ERR_INVALID_PAYLOAD_TYPE`, `ERR_MISSING_MANDATORY`, `ERR_INVALID_AMOUNT`, or `ERR_MALFORMED_TIMESTAMP`.


* **Remediation:**
* If upstream schema shifted (e.g., new date format string), update `StreamingIngestionEngine` parsing rules in `src/ingestion/stream_ingestor.py`.
* Re-inject repaired DLQ payloads using `scripts/replay_dlq.py` (which preserves HMAC pseudonymization and `event_id` deduplication).



---

### Playbook B: In-Memory State Manager OOM / High RAM Usage

* **Trigger:** Process memory exceeds $80\%$ of container limit during continuous stream processing.
* **Root Cause Analysis:** High cardinality of active accounts within the 24-hour window or infrequent watermark sweeps.
* **Remediation:**
1. Trigger event-time watermark garbage collection (defaults to `self.watermark - timedelta(hours=24)`):
```python
state_manager._evict_dormant_accounts()
```


2. Adjust the periodic eviction cadence (`processed_count % 500 == 0`) in `src/streaming/pipeline.py`.



---

### Playbook C: Streaming Recall Drop Below SLA (< 30%)

* **Trigger:** Interception recall (`recall_percentage`) falls below operational target over a streaming window.
* **Root Cause Analysis:** Emerging laundering typologies bypassing historical training features (concept drift).
* **Remediation:**
1. Inspect `artifacts/pipeline_execution_report.json`.
2. Trigger model retraining (`python -m src.models.train`) on recent mature stream records, which automatically re-calibrates the probability threshold via `select_empirical_threshold()`.
