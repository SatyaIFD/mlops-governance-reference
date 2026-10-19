# ---------------------------------------------------------
# Production Dockerfile for Streaming AML Microservice
# ---------------------------------------------------------

FROM python:3.13.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

RUN mkdir -p /app/data /app/artifacts

# This is a headless streaming consumer, not a web API.
# We remove EXPOSE and HTTP HEALTHCHECK, and execute the pipeline directly.
CMD ["python", "-m", "src.streaming.pipeline"]
