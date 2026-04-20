# ── Base: Python 3.11 slim ────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps: git for pip installs from git, curl for healthchecks,
# build-essential for Rust-backed packages (solders)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required for solders — Rust-backed Solana types)
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# ── Builder: install Python deps ──────────────────────────────────────────
FROM base AS builder

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Runtime: final image ──────────────────────────────────────────────────
FROM builder AS runtime

COPY . .

# Create logs directory
RUN mkdir -p logs

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default: print help (actual command set by docker-compose)
CMD ["python", "main.py", "--help"]
