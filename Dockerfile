# ============================================================
# Dockerfile — PyroShield AI (Multi-stage Production Build)
# ============================================================
# Stage 1: Builder — install dependencies
# Stage 2: Runtime — minimal image for deployment
# ============================================================

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

LABEL maintainer="PyroShield AI Team"
LABEL description="Wildfire Containment RL Prediction Service"
LABEL version="2.0.0"

# Security: non-root user
RUN groupadd -r pyroshield && useradd -r -g pyroshield pyroshield

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY sim/ sim/
COPY src/ src/
COPY api/ api/
COPY models/ models/
COPY configs/ configs/
COPY train.py .
COPY evaluate.py .

# Create log directory
RUN mkdir -p logs results data && chown -R pyroshield:pyroshield /app

# Switch to non-root user
USER pyroshield

# Environment
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=models/policy_exp-qlearning-1_final.pkl
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Default: run API server
CMD ["python", "-m", "api.app"]
