# 1. Base Image
FROM python:3.12-slim

# 2. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. App Directory setup
WORKDIR /app

# 4. Dependency Layer
COPY project-1-credit-card-fraud/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Application Code Layer
COPY project-1-credit-card-fraud/src/ /app/src/

# 6. MLflow Database Layer (Copying the real DB from the repo root)
COPY mlflow.db /app/mlflow.db

# 7. THE FIX: Recreate your host machine's exact directory structure inside the container
# This ensures the absolute paths stored in the SQLite database still work!
RUN mkdir -p /media/storage/mlops-governance-reference/
COPY mlruns/ /media/storage/mlops-governance-reference/mlruns/

# 8. Runtime Configuration
ENV PYTHONPATH=/app
EXPOSE 5001

# 9. Entrypoint
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "5001"]