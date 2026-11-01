
# Algorithmic Risk Assessment & Segment Disparity Report

## Executive Summary

This report documents the quantitative validation for the Credit Card Fraud Detection Classifier (`fraud_detection_xgb`). Using an operational segment disparity audit comparing high-value ($\ge 80\text{th}$ percentile) against low-value ($< 80\text{th}$ percentile) transactions, **the baseline model exhibits a 1.8868 segment flagging disparity ratio, which trips the optional `--enforce-gate` check in `src/governance/audit.py`.**

---

## 1. Interpretability & Offline Explainability Analysis (XAI)

* **Mechanism:** Offline game-theoretic marginal contributions analyzed via `SHAP` (`TreeExplainer`).
* **Findings:** Model decision boundaries are primarily driven by PCA-transformed latent transaction components (**`V14`**, **`V17`**, and **`V12`**), which are continuously monitored for Population Stability Index (PSI) drift alongside `scaled_amount` and `scaled_time` in `src/validation/drift_check.py`.

---

## 2. Operational Transaction Segment Disparity Audit

Because the public ULB credit card dataset contains only PCA-anonymized features (`V1`–`V28`), `Time`, and `Amount` (with zero demographic attributes), `src/governance/audit.py` evaluates operational flagging parity across transaction size segments rather than demographic protected classes.

### Audit Metrics Summary:

* **Baseline Segment (Low-Value Transactions, $< 80\text{th}$ percentile):** Flagging Rate = **`0.001296` (0.130%)**
* **Comparison Segment (High-Value Transactions, $\ge 80\text{th}$ percentile):** Flagging Rate = **`0.002446` (0.245%)**
* **Calculated Segment Disparity Ratio:** **`1.8868`**

### Operational Guardrail Evaluation:

When `src/governance/audit.py --enforce-gate` is invoked, the pipeline enforces a target segment parity envelope of **0.80 to 1.25**. A score of **1.8868** indicates that high-value transactions are flagged at **1.89 times** the rate of lower-value transactions.

---

## 3. Remediation & Hardening Roadmap

1. **Segment Sample Re-weighting:** Apply segment-aware sample weights inside `src/training/train.py` to balance false-positive penalties across transaction amount tiers.
2. **Segment-Specific Threshold Calibration:** Calibrate classification probability thresholds across transaction amount quantiles to bring segment flagging parity within `[0.80, 1.25]`.
