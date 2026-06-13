# 1. Base Image: Use a lightweight, official Python 3.12 runtime
FROM python:3.12-slim

# 2. System Dependencies: Install build tools required for compiling C-based ML libraries (like XGBoost)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3. Environment Setup: Define the working directory inside the isolated container
WORKDIR /app

# 4. Dependency Layer: Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Application Layer: Copy the core service code and the MLflow SQLite database
COPY src/ /app/src/
COPY mlflow.db /app/mlflow.db
# Also copy the MLflow artifact directory where the XGBoost binary is stored
# (Assuming standard mlflow local storage. Adjust if your artifacts are elsewhere)
COPY mlruns/ /app/mlruns/

# 6. Runtime Configuration: Set dynamic environment paths for internal execution
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

# 7. Network Config: Expose the operational port
EXPOSE 5001

# 8. Entrypoint: Launch the FastAPI gateway via Uvicorn, binding to all network interfaces
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "5001"]