.PHONY: bootstrap test lint security inspect-mlflow up down

bootstrap:
	@echo "Initializing directory structures, permissions, and model artifacts..."
	mkdir -p mlruns \
		project-1-credit-card-fraud/data/processed project-1-credit-card-fraud/artifacts \
		project-2-loan-default/data/processed project-2-loan-default/artifacts \
		project-3-aml-streaming/data project-3-aml-streaming/artifacts/dlq
	find . -name "kaggle.json" -exec chmod 600 {} + 2>/dev/null || true
	PYTHONPATH=project-1-credit-card-fraud python project-1-credit-card-fraud/src/training/train.py
	PYTHONPATH=project-1-credit-card-fraud python project-1-credit-card-fraud/src/governance/audit.py
	PYTHONPATH=project-1-credit-card-fraud python project-1-credit-card-fraud/src/validation/drift_check.py
	PYTHONPATH=project-2-loan-default python project-2-loan-default/src/training/train.py
	PYTHONPATH=project-2-loan-default python project-2-loan-default/src/evaluation/evaluate.py
	PYTHONPATH=project-2-loan-default python project-2-loan-default/src/validation/drift_check.py
	PYTHONPATH=project-2-loan-default python project-2-loan-default/src/governance/audit.py
	PYTHONPATH=project-3-aml-streaming python project-3-aml-streaming/src/models/train.py
	chmod 775 mlruns project-1-credit-card-fraud/artifacts project-2-loan-default/artifacts project-3-aml-streaming/artifacts project-3-aml-streaming/data
	chmod 664 mlflow.db 2>/dev/null || true

lint:
	ruff check .
	ruff format --check .

test:
	@echo "Running rigorous validation suite across all pipelines..."
	PYTHONPATH=project-1-credit-card-fraud python -m pytest project-1-credit-card-fraud/tests/ -v
	PYTHONPATH=project-2-loan-default python -m pytest project-2-loan-default/tests/ -v
	PYTHONPATH=project-3-aml-streaming python -m pytest project-3-aml-streaming/tests/ -v
	python -m pytest compliance/test_compliance.py -v

security:
	@echo "Running Bandit SAST security scan..."
	bandit -r compliance/ project-1-credit-card-fraud/src/ project-2-loan-default/src/ project-3-aml-streaming/src/ -ll

inspect-mlflow:
	@python -c 'import sqlite3, os; db="mlflow.db"; \
	print("DB exists:", os.path.exists(db)); \
	conn=sqlite3.connect(db) if os.path.exists(db) else None; \
	print("Registered Models:", conn.execute("SELECT name, version, current_stage, source, run_id FROM model_versions").fetchall() if conn and conn.execute("SELECT name FROM sqlite_master WHERE type=\"table\" AND name=\"model_versions\"").fetchone() else "None")'

up:
	sudo docker compose up -d

down:
	sudo docker compose down
