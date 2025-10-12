# VERA Cloud - Multi-stage Dockerfile
# Optimized for Azure Container Apps deployment

# Stage 1: Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    WORKERS=${WORKERS:-1}

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create app user and group
RUN groupadd -r vera && \
    useradd -r -g vera -d /app -s /bin/bash vera

# Create app directory and necessary subdirectories
WORKDIR /app
RUN mkdir -p /app/logs /app/data/sessions /app/temp && \
    chown -R vera:vera /app

# Copy application code
COPY --chown=vera:vera api/ ./api/
COPY --chown=vera:vera websocket/ ./websocket/
COPY --chown=vera:vera scenarios/ ./scenarios/
COPY --chown=vera:vera config/ ./config/
COPY --chown=vera:vera frontend/ ./frontend/

# Switch to non-root user
USER vera

# Expose port
EXPOSE 8000

# Health check with better error handling
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start command with configurable workers
CMD ["sh", "-c", "python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers ${WORKERS}"]
