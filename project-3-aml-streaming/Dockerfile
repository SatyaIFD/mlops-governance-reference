FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies (and fix apt cache cleanup path)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade setuptools>=78.1.1 msgpack>=1.2.1 && pip install --no-cache-dir -r requirements.txt pytest

# Copy application source code, scripts, tests, data, and baseline artifacts
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
RUN mkdir -p data/
RUN mkdir -p artifacts/

# Default command: run unit tests first, then launch the streaming pipeline
CMD ["sh", "-c", "pytest tests/ && python src/streaming/pipeline.py"]
