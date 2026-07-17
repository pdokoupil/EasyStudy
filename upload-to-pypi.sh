#!/usr/bin/env bash
#
# Build and upload EasyStudy to PyPI (or TestPyPI) using twine.
#
# Usage:
#   export PYPI_TOKEN=pypi-AgE...          # your PyPI API token (or TestPyPI token)
#   ./upload-to-pypi.sh                    # -> uploads to PyPI
#   ./upload-to-pypi.sh --test             # -> uploads to TestPyPI (rehearsal)
#   ./upload-to-pypi.sh --check            # -> build + twine check only, no upload
#
# ⚠️  Packaging status: until backlog item B3 lands (make `server/` an importable
#     `easystudy` package + a console entry point), this publishes a metadata/deps-only
#     wheel. That's fine to RESERVE the name `easystudy`, but it is not yet an importable
#     package. See PUBLISHING.local.md for the full plan.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

MODE="upload"
REPO_ARGS=(--repository pypi)
for arg in "$@"; do
  case "$arg" in
    --test)  REPO_ARGS=(--repository testpypi); ;;
    --check) MODE="check"; ;;
    *) echo "Unknown option: $arg" >&2; exit 2; ;;
  esac
done

if [[ "$MODE" == "upload" && -z "${PYPI_TOKEN:-}" ]]; then
  echo "ERROR: PYPI_TOKEN is not set. Run: export PYPI_TOKEN=pypi-..." >&2
  exit 1
fi

echo ">> Installing build tooling…"
python -m pip install --quiet --upgrade build twine

echo ">> Cleaning previous artifacts…"
rm -rf dist build ./*.egg-info

echo ">> Building sdist + wheel…"
python -m build

echo ">> Validating metadata / README rendering…"
twine check dist/*

if [[ "$MODE" == "check" ]]; then
  echo ">> --check only: skipping upload. Artifacts in dist/."
  exit 0
fi

echo ">> Uploading with twine (${REPO_ARGS[*]})…"
# twine reads token auth from env: username __token__, password = the token.
TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_TOKEN" twine upload "${REPO_ARGS[@]}" dist/*

echo ">> Done. Verify at https://pypi.org/project/easystudy/"
