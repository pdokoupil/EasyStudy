# EasyStudy image. Build context is the REPOSITORY ROOT (so pyproject.toml is visible):
#   docker build -f server/Dockerfile -t easy-study .
# or just use `docker compose up` (recommended).
#
# By default this installs the LIGHTWEIGHT core. To include heavy backends, pass extras:
#   docker build -f server/Dockerfile --build-arg EASYSTUDY_EXTRAS="tensorflow,lenskit" -t easy-study .
FROM python:3.12-slim

# comma-separated pyproject extras, e.g. "tensorflow,lenskit,redis"; empty => core only
ARG EASYSTUDY_EXTRAS=""

WORKDIR /app

# System build deps some scientific wheels may need; kept minimal.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better layer caching). pyproject.toml is the single manifest.
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip \
    && if [ -n "$EASYSTUDY_EXTRAS" ]; then pip install --no-cache-dir ".[$EASYSTUDY_EXTRAS]"; else pip install --no-cache-dir .; fi

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
