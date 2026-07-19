# EasyStudy image. Build context is the REPOSITORY ROOT (so pyproject.toml is visible):
#   docker build -f server/Dockerfile -t easy-study .
# or just use `docker compose up` (recommended).
#
# By default this installs the LIGHTWEIGHT core. To include heavy backends, pass extras:
#   docker build -f server/Dockerfile --build-arg EASYSTUDY_EXTRAS="tensorflow,lenskit" -t easy-study .
#
# Some heavy extras pin OLD transitive deps without wheels for the newest Python (e.g. the
# `recbole` extra needs `ray`, which only ships wheels up to cp311). Build those on 3.11:
#   PYTHON_VERSION=3.11 EASYSTUDY_EXTRAS=recbole docker compose up --build
ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# comma-separated pyproject extras, e.g. "recbole,tensorflow,lenskit,redis"; empty => core only
ARG EASYSTUDY_EXTRAS=""

# uv-managed environment (fast, reproducible from uv.lock). Put it on PATH so gunicorn/python
# resolve to it.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# uv binary (pinned via the official distroless image).
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/

# System build deps some scientific wheels may need; kept minimal.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better layer caching). uv.lock pins the exact resolution.
COPY pyproject.toml uv.lock README.md ./
RUN EXTRA_ARGS=""; for e in $(echo "$EASYSTUDY_EXTRAS" | tr ',' ' '); do EXTRA_ARGS="$EXTRA_ARGS --extra $e"; done; \
    uv sync --frozen --no-dev $EXTRA_ARGS

# App source
COPY server/ ./server/
COPY scripts/ ./scripts/

WORKDIR /app/server

EXPOSE 8000

# NOTE: default is a SINGLE worker on purpose — the current fine-tuning keeps per-participant
# model state in worker memory, so raising GUNICORN_WORKERS requires the shared-state rework
# (see 03_V2_TECHNICAL.md A3/B1). Timeout/worker count are overridable via env.
CMD gunicorn -w ${GUNICORN_WORKERS:-1} \
    --bind 0.0.0.0:8000 \
    --timeout ${GUNICORN_TIMEOUT:-0} \
    --preload \
    --log-level ${LOG_LEVEL:-info} \
    "app:create_app()"
