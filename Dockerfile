# One image for rag-foundry-api and rag-foundry-worker. Built from uv.lock; runs as non-root.
FROM ghcr.io/astral-sh/uv:0.13.6 AS uv

FROM python:3.12.16-slim-trixie AS build
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

# Dependency layer: only manifests and the lockfile, so it caches until they change.
COPY pyproject.toml uv.lock ./
COPY packages/core/pyproject.toml packages/core/pyproject.toml
COPY apps/api/pyproject.toml apps/api/pyproject.toml
COPY apps/worker/pyproject.toml apps/worker/pyproject.toml
RUN uv sync --locked --no-dev --all-packages --no-install-workspace

# Project layer: install workspace members non-editable into the venv.
COPY packages/ packages/
COPY apps/ apps/
RUN uv sync --locked --no-dev --all-packages --no-editable

FROM python:3.12.16-slim-trixie AS runtime
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin app
COPY --from=build --chown=root:root /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
USER 10001
EXPOSE 8000
CMD ["rag-foundry-api"]
