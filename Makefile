.PHONY: bootstrap test lint security inspect-mlflow up down

bootstrap:
	@echo "Initializing directory structures, permissions, and model artifacts..."
	mkdir -p mlruns \
		project-1-credit-card-fraud/data/processed project-1-credit-card-fraud/artifacts \
		project-2-loan-default/data/processed project-2-loan-default/artifacts \
		project-3-aml-streaming/data project-3-aml-streaming/artifacts/dlq
	touch mlflow.db
	find . -name "kaggle.json" -exec chmod 600 {} + 2>/dev/null || true
	@if [ ! -f project-1-credit-card-fraud/data/creditcard.csv ]; then \
		PYTHONPATH=project-1-credit-card-fraud python project-1-credit-card-fraud/src/ingestion/ingestion.py; \
	fi
	@if [ ! -f project-2-loan-default/data/processed/train.parquet ]; then \
		PYTHONPATH=project-2-loan-default python -c "from pathlib import Path; from src.ingestion.extract import LoanDataIngestionEngine; eng = LoanDataIngestionEngine(Path('project-2-loan-default/data')); eng.fetch_and_extract() if not eng.raw_data_path.exists() else None; eng.optimize_and_partition()"; \
	fi
	PYTHONPATH=project-1-credit-card-fraud python project-1-credit-card-fraud/src/training/train.py
	PYTHONPATH=project-1-credit-card-fraud python project-1-credit-card-fraud/src/governance/audit.py
	PYTHONPATH=project-1-credit-card-fraud python project-1-credit-card-fraud/src/validation/drift_check.py
	PYTHONPATH=project-2-loan-default python project-2-loan-default/src/training/train.py
	PYTHONPATH=project-2-loan-default python project-2-loan-default/src/evaluation/evaluate.py
	PYTHONPATH=project-2-loan-default python project-2-loan-default/src/validation/drift_check.py
	PYTHONPATH=project-2-loan-default python project-2-loan-default/src/governance/audit.py
	PYTHONPATH=project-3-aml-streaming python project-3-aml-streaming/src/models/train.py
	PYTHONPATH=project-3-aml-streaming AML_DAEMON_MODE=0 AML_WARMUP_EVENTS=100000 AML_MAX_EVENTS=100000 AML_STREAM_DELAY_SECONDS=0.0 python project-3-aml-streaming/src/streaming/pipeline.py
	chmod 2775 mlruns project-1-credit-card-fraud/artifacts project-2-loan-default/artifacts project-2-loan-default/docs project-3-aml-streaming/artifacts project-3-aml-streaming/data
	chmod 664 mlflow.db project-1-credit-card-fraud/artifacts/* project-2-loan-default/artifacts/* project-2-loan-default/docs/* project-3-aml-streaming/artifacts/*.json* 2>/dev/null || true
	find . -name "kaggle.json" -exec chmod 600 {} + 2>/dev/null || true

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
	touch mlflow.db
	sudo docker compose up -d

down:
	sudo docker compose down
