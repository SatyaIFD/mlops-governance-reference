FROM python:3.13-slim

WORKDIR /app

# Copy dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt mlflow

# Copy the rest of the project codebase
COPY . .

# Default command for a batch container
CMD ["/bin/bash"]
