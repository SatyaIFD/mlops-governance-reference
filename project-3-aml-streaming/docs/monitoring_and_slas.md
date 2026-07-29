# Service Level Agreements (SLAs) & Telemetry Metrics

This document specifies the operational SLAs, Service Level Objectives (SLOs), and telemetry monitoring standards for the streaming pipeline.

---

## 🎯 Service Level Objectives (SLOs)

| Metric | Target SLA / SLO | Measurement Window | Action on Breach |
|---|---|---|---|
| **Inference Latency (p95)** | $< 15\text{ ms}$ per transaction | Rolling 10,000 events | Scale state manager memory / optimize deque pruning |
| **Inference Latency (p99)** | $< 50\text{ ms}$ per transaction | Rolling 10,000 events | Inspect garbage collection pauses |
| **Data Quality Pass Rate** | $\ge 99.9\%$ clean payloads | Continuous session | Trigger Playbook A (DLQ Alert) |
| **State Memory Limit** | $< 2.0\text{ GB}$ RAM consumption | Continuous session | Trigger Playbook B (TTL Eviction) |
| **Streaming Laundering Recall** | $\ge 30.0\%$ detection rate | Rolling 100,000 events | Trigger Playbook C (Threshold Re-tuning) |

---

## 📈 Telemetry & Health Monitoring
The pipeline outputs structured JSON metadata (`artifacts/pipeline_execution_report.json`) tracking session statistics:

```json
{
  "execution_timestamp_utc": "2026-07-28T12:00:00Z",
  "data_governance": {
    "data_quality_pct": 100.0,
    "total_ingested": 200000,
    "clean_events": 200000,
    "dlq_events": 0
  },
  "streaming_performance": {
    "total_processed": 100000,
    "anomalies_flagged": 8772,
    "actual_laundering_events": 122,
    "laundering_intercepted": 37,
    "recall_percentage": 30.33
  }
}