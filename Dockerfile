# ──────────────────────────────────────────────────────────────
#  FraudLens — Unified Docker image (train + serve)
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir --upgrade "pip<26" \
    && pip install --no-cache-dir . \
    && rm -rf /root/.cache

# Copy application code
COPY main.py app.py entrypoint.sh ./
COPY configs/ ./configs/
COPY dashboard/ ./dashboard/

# Copy data for training (baked in so one command works)
COPY data/ ./data/

# Create output directories
RUN mkdir -p checkpoints runs

# Expose the FastAPI port
EXPOSE 8000

# Health check (only relevant after training finishes and server starts)
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Train → Serve
ENTRYPOINT ["bash", "entrypoint.sh"]
