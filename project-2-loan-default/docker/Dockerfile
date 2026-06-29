# Use the exact major/minor Python match from your environment
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable real-time logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install basic system library dependencies (CORRECTED FLAG HERE)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency file and install via layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the complete project directory structure into the image context
COPY . .

# Expose port 8000 for FastAPI application endpoint traffic
EXPOSE 8000

ENV PYTHONPATH=/app

# Start the application server natively pointing to our service engine
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]