# ---------------------------------------------------------
# Production Dockerfile for Streaming AML Microservice
# ---------------------------------------------------------

# Use the official slim Debian-based Python image for a smaller footprint
FROM python:3.14.7-slim

# Set environment variables to optimize Python for containerized execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set the working directory
WORKDIR /app

# Install critical system dependencies and force upgrade to patch OS vulnerabilities
# specifically targeting OpenSSL CVE-2026-75804 and CVE-2026-84782
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code
COPY src/ /app/src/

# Create necessary directories that might be excluded by .dockerignore
RUN mkdir -p /app/data /app/artifacts

# Expose the API port
EXPOSE 8000

# Define the healthcheck to ensure the container is ready before receiving traffic
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the FastAPI application via Uvicorn
CMD ["uvicorn", "src.inference.stream_api:app", "--host", "0.0.0.0", "--port", "8000"]
