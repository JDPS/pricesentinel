# Copyright (c) 2025 Soares
# SPDX-License-Identifier: Apache-2.0

# Stage 1: Build with uv
FROM python:3.13-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
ARG EXTRAS=ml
RUN uv sync --frozen --extra ${EXTRAS} --no-dev

# Stage 2: Runtime
FROM python:3.13-slim AS runtime

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

ENTRYPOINT ["python", "run_pipeline.py"]
