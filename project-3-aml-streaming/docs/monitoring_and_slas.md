# Service Level Agreements (SLAs) & Telemetry Metrics

This document specifies the operational SLAs, Service Level Objectives (SLOs), and telemetry monitoring standards for the streaming pipeline.

---

## 🎯 Service Level Objectives (SLOs)

| Metric | Target SLA / SLO | Measurement Window | Action on Breach |
| --- | --- | --- | --- |
| **Inference Latency (p95)** | $< 15\text{ ms}$ per transaction | Rolling 10,000 events | Scale state manager memory / optimize deque pruning |
| **Inference Latency (p99)** | $< 50\text{ ms}$ per transaction | Rolling 10,000 events | Inspect garbage collection pauses |
| **Data Quality Pass Rate** | $\ge 99.9\%$ clean payloads | Continuous session | Trigger Playbook A (DLQ Alert) |
| **State Memory Limit** | $< 2.0\text{ GB}$ RAM consumption | Continuous session | Trigger Playbook B (Watermark Eviction) |
| **Streaming Laundering Recall** | $\ge 30.0\%$ detection rate | Out-of-sample 100,000 events | Trigger Playbook C (Threshold Re-tuning) |

---

## 📈 Telemetry & Health Monitoring

The pipeline outputs structured JSON metadata (`artifacts/pipeline_execution_report.json`) tracking out-of-sample session statistics:

```json
{
  "execution_timestamp_utc": "2026-11-02T07:04:42.645505+00:00",
  "data_governance": {
    "total_ingested": 200000,
    "clean_events": 200000,
    "dlq_events": 0,
    "data_quality_pct": 100.0
  },
  "model_metadata": {
    "artifact_path": "aml_streaming_model",
    "calibrated_threshold": 0.65,
    "features_used": [
      "Amount",
      "pass_through_ratio_1h",
      "is_structuring",
      "velocity_acceleration",
      "fan_out_count_24h",
      "receiver_inflow_count_1h",
      "receiver_inflow_amount_1h"
    ],
    "warmup_events_count": 100000
  },
  "streaming_performance": {
    "total_processed": 100000,
    "anomalies_flagged": 3720,
    "actual_laundering_events": 122,
    "laundering_intercepted": 37,
    "recall_percentage": 30.33,
    "precision_percentage": 0.99
  }
}
