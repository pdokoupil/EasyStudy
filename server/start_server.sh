#!/usr/bin/env bash
# Run the EasyStudy image. Prefer `docker compose up`; this is the plain-docker path.
set -euo pipefail
cd "$(dirname "$0")/.."
docker run -d -p "${PORT:-5000}:5000" \
  --mount type=bind,source="$(pwd)/server",target=/app/server \
  --mount type=bind,source="$(pwd)/scripts",target=/app/scripts \
  --name easy-study easy-study
