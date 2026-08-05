# 1. Base Image (Aligned with host Python 3.13)
FROM python:3.13-slim

# 2. System Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# 3. App Directory setup
WORKDIR /app

# 4. Environment Variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# 5. Dependency Layer
COPY project-1-credit-card-fraud/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Application Code Layer
COPY project-1-credit-card-fraud/src/ /app/src/

# 7. MLflow Database & Artifacts Layer
COPY project-1-credit-card-fraud/mlflow.db /app/mlflow.db
COPY project-1-credit-card-fraud/mlruns/ /app/mlruns/

EXPOSE 5001

# 8. Entrypoint
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "5001"]