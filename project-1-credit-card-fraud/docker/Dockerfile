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

# 6. MLflow Database & Artifacts Layer (Self-contained in /app)
COPY project-1-credit-card-fraud/mlflow.db /app/mlflow.db
COPY project-1-credit-card-fraud/mlruns/ /app/mlruns/

# 7. Runtime Configuration
ENV PYTHONPATH=/app
EXPOSE 5001

# 8. Entrypoint
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "5001"]