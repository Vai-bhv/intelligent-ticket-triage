FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install -r requirements.txt \
    && useradd --create-home --uid 10001 triage

COPY app ./app
COPY src ./src

USER triage
ENV SLA_MODEL_BUNDLE=/app/model_bundle

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3).close()"]

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
