FROM python:3.13.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    procps \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

COPY project-3-aml-streaming/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade "pip>=26.2.2" "msgpack>=1.2.3" "setuptools>=78.1.1" "urllib3>=2.8.0" && \
    find /usr/local -name "*.spdx.json" -delete && \
    rm -rf /usr/local/lib/python3.13/ensurepip/_bundled

COPY project-3-aml-streaming/src/ /app/src/
COPY project-3-aml-streaming/scripts/ /app/scripts/
COPY compliance/ ./compliance/

RUN useradd -m -u 10001 mlopsuser
RUN mkdir -p /app/data /app/artifacts /app/mlruns && \
    chown -R mlopsuser:mlopsuser /app/data /app/artifacts /app/mlruns
USER 10001

CMD ["python", "-m", "src.streaming.pipeline"]
