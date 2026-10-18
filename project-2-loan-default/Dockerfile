FROM python:3.14-slim

WORKDIR /app

# Copy dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project codebase
COPY . .

# Expose the API port
EXPOSE 8000

# Start the FastAPI application via Uvicorn
CMD ["uvicorn", "src.inference.main:app", "--host", "0.0.0.0", "--port", "8000"]
