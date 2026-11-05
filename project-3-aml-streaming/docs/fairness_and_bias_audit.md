# Fairness & Bias Audit: AML Streaming Anomaly Detector

This document defines the bias prevention design and operational fairness controls for the **Project 3 Real-Time AML Streaming Engine**.

---

## 1. Bias Prevention Principles

Financial crime monitoring models carry inherent risk of indirect bias or demographic profiling. To ensure objective behavioral scoring:

* **Zero Demographic Predictors:** The 7-feature matrix (`CANONICAL_AML_FEATURES`) strictly excludes age, gender, nationality, ethnicity, postal code, and bank location strings (`Sender_bank_location` and `Receiver_bank_location` are ignored during feature extraction).
* **Behavioral Graph Topology:** Features measure relative transaction flow dynamics (`pass_through_ratio_1h`, `velocity_acceleration`, `fan_out_count_24h`, `receiver_inflow_count_1h`, `receiver_inflow_amount_1h`, `is_structuring`, `Amount`) rather than static entity identity.

---

## 2. Proxy Risk & Feature Normalization Controls

| Potential Proxy Risk | Assessment | Implemented Control in Code |
| --- | --- | --- |
| **Geographic / Country Corridors** | High-risk cross-border corridors can be over-flagged if location columns are fed to the classifier. | `AMLAnomalyDetector._align_features()` filters strictly to `CANONICAL_AML_FEATURES`, dropping country/currency fields prior to training and inference. |
| **Transaction Frequency Differences** | High-volume accounts could be over-flagged by raw transaction counts. | `velocity_acceleration` normalizes each account's 1-hour outgoing count against its own 24-hour average hourly baseline (`out_count_24h / 24.0`). |
| **Class Imbalance Distortion** | Laundering events represent ~0.1% of stream volume (~12 positives in the 10,000-row validation slice). | `RandomForestClassifier` uses `class_weight="balanced"` and `select_empirical_threshold()` (`src/models/train.py`) selects a heuristic F1 cutoff (`0.65` persisted in `.meta.json`). |

---

## 3. Human-In-The-Loop (HITL) Operational Protections

* **Non-Autonomous Blocking:** Transactions scoring at or above the persisted threshold (`prob >= 0.65`) are written to `artifacts/aml_alerts.jsonl` (`is_anomaly = 1`) for human investigator review rather than triggering automated account freezes.
