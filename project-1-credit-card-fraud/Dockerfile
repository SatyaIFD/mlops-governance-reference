FROM python:3.13.14-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY project-1-credit-card-fraud/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade "msgpack>=1.2.3" "setuptools>=78.1.1" "urllib3>=2.8.0" && \
    rm -rf /usr/local/lib/python3.13/site-packages/pip* \
           /usr/local/lib/python3.13/site-packages/*.dist-info/sboms \
           /usr/local/lib/python3.13/ensurepip && \
    find /usr - \( -iname "*spdx*" -o -iname "*cdx*" -o -iname "*sbom*" \) -delete 2>/dev/null || true
COPY project-1-credit-card-fraud/ .
COPY compliance/ ./compliance/
RUN useradd -m -u 10001 mlopsuser
RUN mkdir -p /app/data /app/artifacts /app/mlruns && \
    chown -R mlopsuser:mlopsuser /app/data /app/artifacts /app/mlruns
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
  CMD curl -f http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "src.inference.main:app", "--host", "0.0.0.0", "--port", "8000"]
