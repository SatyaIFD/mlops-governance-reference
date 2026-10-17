.PHONY: help test security up down

help:
	@echo "Enterprise MLOps Reference Architecture Commands:"
	@echo "  make test      - Run all rigorous functional, fairness, and compliance tests"
	@echo "  make security  - Run Bandit SAST scanning locally"
	@echo "  make up        - Spin up the entire multi-project fleet via Docker Compose"
	@echo "  make down      - Tear down the Docker Compose fleet"

test:
	@echo "Running rigorous validation suite across all pipelines..."
	PYTHONPATH=project-1-credit-card-fraud python -m pytest project-1-credit-card-fraud/tests/ -v
	PYTHONPATH=project-2-loan-default python -m pytest project-2-loan-default/tests/ -v
	PYTHONPATH=project-3-aml-streaming python -m pytest project-3-aml-streaming/tests/ -v
	python -m pytest compliance/test_compliance.py -v

security:
	bandit -r . -ll -ii -x '*/tests/*,*/.pytest_cache/*'

up:
	sudo docker compose up --build -d

down:
	sudo docker compose down -v
