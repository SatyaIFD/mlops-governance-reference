# Fairness & Bias Audit: AML Streaming Anomaly Detector

This document defines the bias auditing criteria, fairness metrics, and anti-discrimination controls for the **Project 3 Real-Time AML Streaming Engine**.

---

## 1. Bias Prevention Principles
Financial crime monitoring models carry inherent risk of indirect bias or demographic profiling. To guarantee ethical operation:

* **No Demographic Predictors:** The feature matrix strictly excludes age, gender, nationality, ethnicity, postal code, or socio-economic indicators.
* **Neutral Network Topology:** Features measure transaction flow dynamics (`pass_through_ratio_1h`, `velocity_acceleration`, `fan_out_count_24h`) rather than entity identity.
* **Disparate Impact Threshold:** Model scoring must maintain a Disparate Impact Ratio between $0.80$ and $1.25$ across all transaction amount corridors.

---

## 2. Protected Class & Proxy Discrimination Audit

| Potential Proxy Risk | Assessment | Mitigation Strategy |
|---|---|---|
| **Geographic / Country Corridors** | High-risk cross-border corridors can be over-flagged. | Scoring weights reflect account velocity acceleration rather than origin region. |
| **Transaction Size Disparity** | Small-business accounts with high volume may resemble "smurfing" patterns. | `velocity_acceleration` normalizes 1-hour activity against the account's own 24-hour baseline. |
| **New Account Cold Start** | Accounts without historical state risk artificially high pass-through scores. | Warmup lookback threshold delays high-severity auto-escalation until baseline state establishes. |

---

## 3. Human-In-The-Loop (HITL) Fairness Protections
* **Non-Automated Freezing:** High anomaly scores ($> 0.80$) trigger compliance queue items for human investigation. No account is frozen purely by machine decision (GDPR Article 22 compliance).
* **Feedback Loop Auditing:** Investigator overrides (False Positives) are logged to detect systemic feature bias during quarterly model retraining cycles.