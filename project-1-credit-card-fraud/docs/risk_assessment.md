# Algorithmic Risk Assessment & Bias Compliance Report

## Executive Summary
This report documents the quantitative compliance validation for the Credit Card Fraud Detection Classifier (v1.0.0). Based on strict regulatory audit thresholds (including the EU AI Act and standard consumer protection fair credit evaluations), **this model version has tripped the automated governance circuit breaker and is blocked from automated production deployment due to disparate impact.**

---

## 1. Interpretability & Explainability Audit (XAI)
- **Mechanism:** Game-theoretic marginal contributions verified via `SHAP` TreeExplainer tracking.
- **Findings:** The model decision boundaries are structurally aligned with hidden latent transaction component signatures (**`V14`**, **`V17`**, and **`V12`**). Global attribute scales demonstrate that transaction magnitude (`Amount`) holds lower prioritization weight, insulating standard high-value checkouts from arbitrary systemic flagging.

---

## 2. Quantitative Fairness & Anti-Bias Audit
To ensure compliance with the **Four-Fifths Rule** (Demographic/Statistical Parity), we executed a proxy audit segmented across transaction scale categories (`Amount`), tracking whether high-value transaction profiles are targeted at disproportionate rates.

### Audit Metrics Summary:
- **Baseline Group (Low-Value Transactions):** Flagging Rate = **`0.001296` (0.130%)**
- **Protected/Proxy Group (High-Value Transactions):** Flagging Rate = **`0.002446` (0.245%)**
- **Calculated Disparate Impact Ratio:** **`1.8868`**

### Regulatory Compliance Evaluation:
The industry-mandated compliance guardrail envelope requires a Disparate Impact Ratio between **0.80 and 1.25**. 

A calculated score of **1.8868** demonstrates substantial systemic divergence. High-value transactions are flagged at nearly **1.89 times** the frequency of baseline transactions, tripping the automated CI/CD compliance circuit breaker.

---

## 3. Remediation & Hardening Roadmap
To resolve the production blocker in future iterations:
1. **Loss Function Re-weighting:** Apply sample re-weighting directly inside `src/training/train.py` to balance minority high-value representation penalties.
2. **Post-Processing Threshold Calibration:** Adjust classification logit boundaries independently across segment vectors to bring the Disparate Impact Metric within the mandated **0.80–1.25** compliance envelope.
