.PHONY: help test security up down

help:
	@echo "Enterprise MLOps Reference Architecture Commands:"
	@echo "  make test      - Run the automated compliance and dynamic Hypothesis tests"
	@echo "  make security  - Run Bandit SAST scanning locally"
	@echo "  make up        - Spin up the entire multi-project fleet via Docker Compose"
	@echo "  make down      - Tear down the Docker Compose fleet"

test:
	python -m pytest compliance/test_compliance.py -v

security:
	bandit -r . -ll -ii -x '*/tests/*,*/.pytest_cache/*'

up:
	sudo docker-compose up --build -d

down:
	sudo docker-compose down
