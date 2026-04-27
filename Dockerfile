# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build-time deps (gcc needed by some pip packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.12-slim

LABEL maintainer="MemHub Team"
LABEL description="MemHub — Centralized Memory-as-a-Service for Multi-Agent Systems"

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY api/        ./api/
COPY core/       ./core/
COPY agents/     ./agents/
COPY eval/       ./eval/
COPY scripts/    ./scripts/

# Create persistent storage directory
RUN mkdir -p /data/chroma_db

# ── Environment defaults ─────────────────────────────────────────────────────
ENV MEMHUB_DB_PATH=/data/memhub.db \
    MEMHUB_CHROMA_PATH=/data/chroma_db \
    MEMHUB_DISABLE_AUTH=0 \
    MEMHUB_CORS_ORIGINS=* \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    LOG_LEVEL=info \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check — polls /v1/health every 30 s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')" || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]
