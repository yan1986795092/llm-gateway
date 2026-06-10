FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml ./
COPY gateway ./gateway
COPY README.md ./README.md

RUN pip install --upgrade pip setuptools wheel && \
    pip install .

EXPOSE 9010

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9010/health')"

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "9010"]
