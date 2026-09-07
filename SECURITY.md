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

## 🤖 Automated DevSecOps Pipeline
This repository enforces strict, automated security gates on every Pull Request and merge to `main`:
* **SAST (Static Application Security Testing):** `Bandit` scans all Python code for hardcoded credentials, injection flaws, and unsafe ML deserialization.
* **Container Vulnerability Scanning:** `Trivy` scans the Docker layers (`python:3.13-slim`) for OS-level and Python-level CVEs. Builds will hard-fail (`exit code 1`) if `HIGH` or `CRITICAL` vulnerabilities are detected.
* **Supply Chain Monitoring:** GitHub `Dependabot` automatically opens PRs to patch outdated dependencies.

## 👻 Vulnerability Mitigation & Ghost Packages (.trivyignore)
Because Docker builds in immutable layers, security scanners occasionally flag "ghost packages"—vulnerable files trapped in inert base layers, even after the active environment has been successfully patched. 

Our policy dictates that `.trivyignore` is **strictly reserved for mitigated ghost packages or accepted risks with no upstream patch**. Any entry in this file must include a written mitigation record proving the active container runtime is secure.

## 🎯 Scope of Security
Because this architecture processes simulated financial and PII data, we are particularly interested in vulnerability reports regarding the following AI/ML threat vectors:
1. **Model Evasion & Poisoning:** Vulnerabilities allowing malicious inputs to bypass the EU AI Act HITL router.
2. **Data Extraction:** Flaws that would bypass the GDPR Right to be Forgotten protocol.
