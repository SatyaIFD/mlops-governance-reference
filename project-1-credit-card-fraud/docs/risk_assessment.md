# Algorithmic Risk Assessment & Operational Segment Telemetry Report

## Executive Summary
This report documents the out-of-sample quantitative validation for the Credit Card Fraud Detection Classifier (`models:/fraud_detection_xgb/10`) on `data/processed/test.parquet`. At the served `0.50` probability threshold, the model achieves **`ROC-AUC = 0.9659`**, **`PR-AUC = 0.8166`**, and **`overall recall = 71.58%`** (95% Wilson CI `[0.6181, 0.7967]`), with **`65.38%` recall (`94.44%` precision)** on high-value transactions and **`73.91%` recall (`100.00%` precision)** on low-value transactions (`performance_gate_passed: true`). The informational flag-rate ratio between high-value ($\ge 80\text{th}$ percentile) and low-value ($< 80\text{th}$ percentile) segments is **`1.4348`** (95% CI `[0.8387, 2.4546]`, `within_80_125_band: false`), reflecting the higher empirical fraud prevalence in large transactions (`0.232%` vs. `0.152%`).

---

## 1. Interpretability & Offline Explainability Analysis (XAI)
* **Mechanism:** Offline game-theoretic marginal contributions analyzed via `SHAP` (`TreeExplainer`).
* **Findings:** Model decision boundaries are primarily driven by PCA-transformed latent transaction components (**`V14`**, **`V17`**, and **`V12`**), which are monitored for Population Stability Index (PSI) drift alongside `scaled_amount` and `scaled_time` in `src/validation/drift_check.py`.

---

## 2. Operational Transaction Segment Telemetry (Informational)
Because the public ULB credit card dataset contains only PCA-anonymized features (`V1`–`V28`), `Time`, and `Amount` (with zero demographic attributes), transaction amount tiers are not a legal protected class. `src/governance/audit.py` enforces a hard performance and recall confidence floor (`ROC-AUC >= 0.90`, `PR-AUC >= 0.75`, `overall_recall >= 0.70`, and Wilson 95% CI lower bounds `>= 0.45` / `>= 0.55`) while recording segment flag-rate ratios as informational telemetry:

* **Low-Value Segment ($< 80\text{th}$ percentile):** Flag Rate = **`0.001120`**, Recall = **`0.7391`** (95% CI `[0.6249, 0.8281]`), Precision = **`1.0`**
* **High-Value Segment ($\ge 80\text{th}$ percentile):** Flag Rate = **`0.001607`**, Recall = **`0.6538`** (95% CI `[0.4622, 0.8059]`), Precision = **`0.9444`**
* **Informational Flag-Rate Ratio:** **`1.4348`** (`within_80_125_band: false`, `performance_gate_passed: true`)
