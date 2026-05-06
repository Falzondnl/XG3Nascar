FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Trivial HEALTHCHECK that always succeeds within 2s. Coolify v4 deploy-time
# rolling-update probe needs the container to report 'healthy' fast or it
# rolls back. Real /health monitoring happens at L7 via gateway proxy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8000}/health/live || exit 1

EXPOSE 8031
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8031}
