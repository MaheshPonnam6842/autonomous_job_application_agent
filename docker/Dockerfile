# Image for the Flask web app. The LLM runs in a separate `ollama` container
# (see docker-compose.yml). Served with gunicorn (single worker so the in-memory
# async-rewrite job store stays consistent; threads handle concurrency).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    HOST=0.0.0.0 \
    OLLAMA_HOST=http://ollama:11434 \
    JOBAGENT_ARTIFACTS_DIR=/app/artifacts

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt gunicorn

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "1", "--threads", "8", "--timeout", "300", "-b", "0.0.0.0:5000", "web.app:app"]
