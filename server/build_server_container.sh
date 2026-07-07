#!/usr/bin/env bash
# Build the EasyStudy image (lightweight core). Run from the repo root.
# For heavy backends: EASYSTUDY_EXTRAS="tensorflow,lenskit" ./server/build_server_container.sh
set -euo pipefail
cd "$(dirname "$0")/.."
docker build -f server/Dockerfile \
  --build-arg EASYSTUDY_EXTRAS="${EASYSTUDY_EXTRAS:-}" \
  -t easy-study .
