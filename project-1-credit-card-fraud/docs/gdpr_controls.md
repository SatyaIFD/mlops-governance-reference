# GDPR & Data Privacy Framework Compliance Control Matrix

This matrix documents the explicit architectural constraints built into the pipeline to satisfy European General Data Protection Regulation (GDPR) mandates, ensuring consumer privacy rights are maintained across continuous lifecycle tracking processes.

| GDPR Mandate | Pipeline Structural Implementation Control | Operational Safeguard |
| :--- | :--- | :--- |
| **Article 5:<br>Data Minimization** | Raw transaction vectors are processed through dimensionality reduction transformations prior to infrastructure landing. | Direct identity attributes (PII) such as Names, Account Numbers, and Card CVVs are completely absent from model arrays. |
| **Article 22:<br>Automated Decisions** | Human-in-the-loop escalation paths are maintained via API status response structures. | High-risk transactions trigger intermediate security verification holds rather than permanent account terminations. |
| **Article 15:<br>Right to Explanation** | Local explainability values are generated via localized SHAP attribution. | Customer support systems can query specific feature contributions to explain precisely why an isolated transaction was flagged. |
| **Article 32:<br>Security of Processing** | Data persistence layers utilize highly compressed, immutable Apache Parquet formats. | Strict file schema constraints prevent SQL injection scripts or unstructured memory-bloat payloads from penetrating storage buckets. |