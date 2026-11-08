# Service Level Agreements (SLAs) & Telemetry Metrics

This document specifies the operational SLAs, Service Level Objectives (SLOs), and telemetry monitoring standards for the streaming pipeline.

---

## 🎯 Service Level Objectives (SLOs)

| Metric | Target SLA / SLO | Measurement Window | Action on Breach |
| --- | --- | --- | --- |
| **Batch Inference Latency** | $< 15\text{ ms}$ per transaction | Micro-batches of 5,000 events | Inspect state manager deque pruning |
| **Heartbeat Freshness** | $< 120\text{ s}$ (`heartbeat.json` mtime) | Updated every 5,000 events during active pass | Container healthcheck marks service `unhealthy` |
| **Data Quality Pass Rate** | $\ge 99.9\%$ clean payloads | Per 200,000-event pass | Trigger Playbook A (DLQ Alert) |
| **State Memory Limit** | $< 2.0\text{ GB}$ RAM consumption | Swept every 25,000 events | Trigger Playbook B (Watermark Eviction) |
| **Out-of-Sample Laundering Recall** | $\ge 20.0\%$ detection rate | Out-of-sample 100,000 events | Trigger Playbook C (Model Retraining) |

---

## 📈 Telemetry & Health Monitoring

The pipeline outputs structured JSON metadata (`artifacts/pipeline_execution_report.json`) with lifecycle `status` values (`"IN_PROGRESS"`, `"COMPLETE"`, `"FAILED_MODEL_LOAD"`, `"FAILED_ZERO_SCORED"`, `"INTERRUPTED"`, or `"FAILED"`) and heartbeat states (`"WARMING_UP"`, `"HEALTHY"`, `"FAILED_MODEL_LOAD"`, `"FAILED_ZERO_SCORED"`, `"INTERRUPTED"`, `"FAILED"`). At ~0.12% base laundering prevalence, `0.47%` alert precision corresponds to ~213 alerts per true positive across `6,172` flagged events:

```json
{
  "execution_timestamp_utc": "2026-11-08T05:19:28.424713+00:00",
  "status": "COMPLETE",
  "data_governance": {
    "total_ingested": 200000,
    "clean_events": 200000,
    "dlq_events": 0,
    "data_quality_pct": 100.0
  },
  "model_metadata": {
    "artifact_path": "aml_streaming_model",
    "calibrated_threshold": 0.65,
    "model_sha256": "428d294b2b1eda4397a33c883ce452ed777a80e3a0eff71471a2e966b3832324",
    "mlflow_run_id": "e565ff5ce09142ad9fec2cedf336c5cb",
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
    "anomalies_flagged": 6172,
    "actual_laundering_events": 122,
    "laundering_intercepted": 29,
    "recall_percentage": 23.77,
    "precision_percentage": 0.47
  }
}
```
