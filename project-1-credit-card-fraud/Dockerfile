FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY project-1-credit-card-fraud/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && pip install -U mlflow
COPY project-1-credit-card-fraud/ .
COPY compliance/ ./compliance/
RUN useradd -m -u 10001 mlopsuser && chown -R mlopsuser /app
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "src.inference.main:app", "--host", "0.0.0.0", "--port", "8000"]
