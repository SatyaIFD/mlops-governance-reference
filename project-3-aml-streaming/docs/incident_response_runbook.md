# Operational Incident Response Runbook

Operational procedures for responding to pipeline failures, memory spikes, recall drops, and Dead-Letter Queue (DLQ) bursts in the streaming environment.

---

## 🚨 Incident Playbooks

### Playbook A: DLQ Spike / High Ingestion Failure Rate
* **Trigger:** Data Quality Governance metric drops below $99.0\%$ clean events, or DLQ write rate exceeds $50 \text{ events/min}$.
* **Root Cause Analysis:**
  1. Inspect recent quarantine payloads in `artifacts/dlq_event_*.json`.
  2. Identify error codes: `ERR_MISSING_MANDATORY`, `ERR_INVALID_AMOUNT`, or `ERR_MALFORMED_TIMESTAMP`.
* **Remediation:**
  * If upstream schema shifted (e.g., new date format string), update `StreamingIngestionEngine` parsing rules in `stream_ingestor.py`.
  * Re-inject repaired DLQ payloads using `scripts/replay_dlq.py`.

---

### Playbook B: In-Memory State Manager OOM / High RAM Usage
* **Trigger:** Process memory exceeds $80\%$ of system limit during continuous stream processing.
* **Root Cause Analysis:** Dormant account keys accumulating in `StreamingStateManager.state` without eviction.
* **Remediation:**
  1. Manually trigger key garbage collection:
     ```python
     state_manager._evict_dormant_accounts(cutoff_time=datetime.now() - timedelta(hours=24))
     ```
  2. Verify that `_evict_dormant_accounts` frequency is configured in `velocity.py`.

---

### Playbook C: Streaming Recall Drop Below SLA (< 30%)
* **Trigger:** Interception recall falls below operational target over a 100,000-event window.
* **Root Cause Analysis:** Emerging laundering typologies bypassing historical training features (concept drift).
* **Remediation:**
  1. Export latest execution report (`artifacts/pipeline_execution_report.json`).
  2. Trigger model retraining pipeline with recent 50k mature stream records.
  3. Re-tune prediction probability alert threshold (e.g., lower cutoff from $0.50$ to $0.40$).