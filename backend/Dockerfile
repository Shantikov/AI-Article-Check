FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_ENVIRONMENT=production \
    PRELOAD_MODEL=true \
    LOCAL_MODEL_CACHE_DIR=/models

WORKDIR /app

COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir --require-hashes -r requirements.lock

COPY app ./app
COPY calibration.json ./calibration.json

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /models \
    && chown -R appuser:appuser /app /models

USER appuser

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import json,os,urllib.request; json.load(urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', os.environ.get('APP_PORT', '8787')) + '/health', timeout=4))"

CMD ["python", "-m", "app.run"]
