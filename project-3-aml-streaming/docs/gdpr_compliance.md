# GDPR & Regulatory Compliance Framework: AML Streaming Engine

This document outlines how the **Project 3 Real-Time AML Streaming Observability Engine** balances Anti-Money Laundering (AML) regulatory obligations under the Bank Secrecy Act (BSA) and EU Anti-Money Laundering Directives (4AMLD/5AMLD) with strict compliance under the **General Data Protection Regulation (GDPR)**.

---

## ⚖️ Key Regulatory Balancing Act

| Regulatory Mandate | Compliance Requirement | Engine Implementation |
|---|---|---|
| **AML Directives (BSA / 5AMLD)** | Mandatory detection and reporting of suspicious transaction patterns. | Stateful rolling window features track multi-account pass-through and velocity. |
| **GDPR Data Minimization (Art. 5(1)(c))** | Personal data collected must be adequate, relevant, and limited to necessity. | State queues hold only minimal transaction metadata (`Timestamp`, `Amount`, Account IDs). |
| **GDPR Storage Limitation (Art. 5(1)(e))** | Personal data must not be kept longer than necessary. | Automated 24-hour TTL key eviction (`_evict_dormant_accounts`) wipes expired lookback states. |
| **GDPR Automated Decision-Making (Art. 22)** | Right not to be subject to purely automated legal decisions. | Alerts route to human compliance investigators for manual verification. |

---

## 🔐 1. Account Anonymization & Pseudonymization
* **Opaque Identifiers:** Account keys (`Sender_account`, `Receiver_account`) are treated as pseudonymous identifiers.
* **No Direct PII Storage:** Raw names, addresses, social security numbers, or IP addresses are never ingested into or stored within the in-memory feature state (`StreamingStateManager`).

---

## 🧹 2. Data Minimization & In-Memory TTL Eviction
* **Rolling Memory Bounds:** Transaction records stored in `StreamingStateManager` deques are strictly pruned once they exceed 24 hours ($T_{\text{event}} - 24\text{h}$).
* **Dormant Key Garbage Collection:** The engine periodically executes `_evict_dormant_accounts()`, removing inactive account keys completely from system memory to prevent indefinite data retention.

---

## 📋 3. Auditability & Right to Explanation (GDPR Recital 71)
* **Deterministic Scoring:** Every flagged alert outputs an explicit log containing feature values (`pass_through_ratio_1h`, `velocity_acceleration`, etc.) and exact prediction probability.
* **Lineage Logging:** Every pipeline execution generates an immutable JSON audit artifact (`pipeline_execution_report.json`) tracking feature schemas, data quality metrics, and performance counters.