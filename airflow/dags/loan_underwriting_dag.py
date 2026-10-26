from pathlib import Path
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

# Dynamically resolve the absolute path to the repository root
# DAG is in airflow/dags. Parent 1 is airflow/. Parent 2 is the repo root.
DAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = str(DAG_DIR.parents[1])

default_args = {
    "owner": "mlops_engine",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "loan_default_underwriting_governance",
    default_args=default_args,
    description="Automated traffic streaming, data drift analysis, and fairness auditing",
    schedule_interval="@daily",
    catchup=False,
    max_active_runs=1,
) as dag:
    # Wrap commands in 'set -e' to ensure python script failures trigger a non-zero exit code in bash
    stream_traffic = BashOperator(
        task_id="stream_production_traffic",
        bash_command=f"set -e; cd {PROJECT_ROOT} && PYTHONPATH=. python project-2-loan-default/src/monitoring/generate_traffic.py",
    )

    audit_fairness = BashOperator(
        task_id="audit_fairness_metrics",
        bash_command=f"set -e; cd {PROJECT_ROOT} && PYTHONPATH=. python project-2-loan-default/src/monitoring/drift_monitor.py",
    )

    check_data_drift = BashOperator(
        task_id="check_population_data_drift",
        bash_command=f"set -e; cd {PROJECT_ROOT} && PYTHONPATH=. python project-2-loan-default/src/validation/drift_check.py",
    )

    compile_model_card = BashOperator(
        task_id="compile_regulatory_model_card",
        bash_command=f"set -e; cd {PROJECT_ROOT} && PYTHONPATH=. python project-2-loan-default/src/governance/audit.py",
    )

    stream_traffic >> audit_fairness >> check_data_drift >> compile_model_card
