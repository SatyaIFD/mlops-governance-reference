.PHONY: test security inspect-mlflow up down

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
