# Multi-stage production Dockerfile for MaBuy ML Engine API
FROM python:3.11-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu


# Production image
FROM python:3.11-slim

WORKDIR /app

# Install runtime system dependencies (curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY . .

# Create non-root system user and configure directory permissions
RUN groupadd -r mabuygroup && useradd -r -g mabuygroup -d /app -s /sbin/nologin mabuyuser \
    && mkdir -p /app/data /app/logs /app/models \
    && chown -R mabuyuser:mabuygroup /app

# Switch to non-root user for security (C6 FIX)
USER mabuyuser

# Native container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment settings
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

EXPOSE 8000

# Run FastAPI REST server via uvicorn
CMD ["uvicorn", "src.presentation.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
