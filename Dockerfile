FROM python:3.14.3-slim

ARG VERSION=dev
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_VERSION=${VERSION}

WORKDIR /app

# Create a non-root user with specific UID for easier host permission matching
RUN groupadd -r -g 1000 mediasageappuser && useradd -r -u 1000 -g mediasageappuser mediasageappuser

# Create data directory with correct ownership (for volume mounts)
RUN mkdir -p /app/data && chown mediasageappuser:mediasageappuser /app/data

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code with ownership
COPY --chown=mediasageappuser:mediasageappuser backend/ ./backend/
COPY --chown=mediasageappuser:mediasageappuser frontend/ ./frontend/
COPY --chown=mediasageappuser:mediasageappuser model_catalog.yaml ./

# Expose port
EXPOSE 5765

# Switch to non-root user
USER mediasageappuser

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; import sys; code = urllib.request.urlopen('http://localhost:5765/api/health').getcode(); sys.exit(0 if code == 200 else 1)"

# Run the application
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port 5765 --workers ${UVICORN_WORKERS:-1}"]
