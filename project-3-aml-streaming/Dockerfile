FROM python:3.13.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libssl-dev \
    procps && \
    rm -rf /var/lib/apt/lists/*

COPY project-3-aml-streaming/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade "msgpack>=1.2.3" "setuptools>=78.1.1" "urllib3>=2.8.0" && \
    rm -rf /usr/local/lib/python3.13/site-packages/pip* \
           /usr/local/lib/python3.13/site-packages/*.dist-info/sboms \
           /usr/local/lib/python3.13/ensurepip && \
    find /usr \( -iname "*spdx*" -o -iname "*cdx*" -o -iname "*sbom*" \) -delete 2>/dev/null || true

COPY project-3-aml-streaming/src/ /app/src/
COPY project-3-aml-streaming/scripts/ /app/scripts/
COPY compliance/ ./compliance/

RUN useradd -m -u 10001 mlopsuser
RUN mkdir -p /app/data /app/artifacts /app/mlruns && \
    chown -R mlopsuser:mlopsuser /app/data /app/artifacts /app/mlruns
USER 10001

CMD ["python", "-m", "src.streaming.pipeline"]
