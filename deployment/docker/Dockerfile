# LiteAds Ad Server Dockerfile
# Multi-stage build for smaller image size

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy source code for building wheel
COPY pyproject.toml ./
COPY liteads ./liteads

# Build wheel
RUN pip install --no-cache-dir build hatchling && \
    pip wheel --no-cache-dir --wheel-dir=/app/wheels .

# Runtime stage
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash liteads

# Copy wheels from builder
COPY --from=builder /app/wheels /app/wheels

# Install application
RUN pip install --no-cache-dir /app/wheels/*.whl && \
    rm -rf /app/wheels

# Copy application code
COPY --chown=liteads:liteads . .

# Switch to non-root user
USER liteads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["python", "-m", "liteads.ad_server.main"]
