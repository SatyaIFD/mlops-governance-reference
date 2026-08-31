# Security Policy

## 🛡️ Supported Versions

As this is an Enterprise Reference Architecture, only the latest commits on the `main` branch are actively monitored for security patches and dependency upgrades.

| Version | Supported          |
| ------- | ------------------ |
| `main`  | :white_check_mark: |
| `< 1.0` | :x:                |

## 🚨 Reporting a Vulnerability

We take the security and integrity of this MLOps platform seriously. If you discover a vulnerability, please **do not** open a public issue. 

Instead, please report it via [GitHub Security Advisories](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability) or reach out directly to the repository maintainer. 

We will acknowledge receipt of your vulnerability report within **48 hours** and strive to provide a timeline for a patch within 5 business days.

## 🎯 Scope of Security

Because this architecture processes simulated financial and PII data, we are particularly interested in vulnerability reports regarding the following AI/ML threat vectors:

1. **Model Evasion & Poisoning:** Vulnerabilities allowing malicious inputs to bypass the EU AI Act HITL router or manipulate the AML streaming stateful graph.
2. **Data Extraction:** Flaws that would allow unauthorized access to the MLflow backend or bypass the GDPR Right to be Forgotten protocol.
3. **Pipeline Integrity:** Supply chain attacks or vulnerabilities in the automated GitHub Actions CI/CD pipeline or Docker base images (`python:3.13-slim`).
4. **Denial of Service (DoS):** Attacks capable of bypassing the DORA circuit breaker pattern and crashing the streaming inference engine.

## 🔒 Best Practices Implemented
This repository natively enforces:
* **Least Privilege:** Docker containers run on hardened Alpine/Slim layers without unnecessary root utilities.
* **Secret Management:** No API keys, database credentials, or real customer PII are hardcoded in this repository.
* **Dependency Scanning:** CI/CD automatically installs the latest secure versions of `pytest`, `scikit-learn`, and `pandas`.
