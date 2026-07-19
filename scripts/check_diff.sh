#!/usr/bin/env bash
# Lint/format-check only the Python files that differ from a base ref (default: origin/main).
#
# Why diff-scoped: `ruff check .` / `black --check .` over the WHOLE repo fails today (years of
# pre-existing style debt), which makes a blocking whole-repo check useless — it would always
# be red. Scoping to the diff means a PR is judged on the code IT touches, not on history.
#
# Used both locally (before you push) and in CI (.github/workflows/lint-diff.yml), calling this
# same script, so "does my PR pass?" means the same thing in both places.
#
# Usage:
#   scripts/check_diff.sh                  # diff against origin/main (fetches it first)
#   scripts/check_diff.sh <base-ref>       # diff against an explicit ref/SHA
#   scripts/check_diff.sh --fix            # apply black + ruff --fix to the changed files
#   scripts/check_diff.sh --no-tests       # skip the (full, non-diff-scoped) pytest run
#
# Only *.py files are linted — templates/HTML/JSON are excluded on purpose (legacy templates
# are not in scope for this check and are considerably messier).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BASE=""
FIX=0
RUN_TESTS=1
for arg in "$@"; do
  case "$arg" in
    --fix) FIX=1 ;;
    --no-tests) RUN_TESTS=0 ;;
    *) BASE="$arg" ;;
  esac
done

if [ -z "$BASE" ]; then
  git fetch origin main --quiet 2>/dev/null || true
  BASE="origin/main"
fi

MERGE_BASE=$(git merge-base HEAD "$BASE")
echo ">> Diffing against $BASE (merge-base $MERGE_BASE)"

# --diff-filter=ACMR: added/copied/modified/renamed files only — skip deleted ones (nothing to
# lint) and never touch anything outside *.py.
FILES=$(git diff --name-only --diff-filter=ACMR "$MERGE_BASE" -- '*.py')

if [ -z "$FILES" ]; then
  echo ">> No changed Python files vs $BASE — nothing to lint."
else
  echo ">> Changed Python files:"
  echo "$FILES" | sed 's/^/     /'
  if [ "$FIX" = "1" ]; then
    black $FILES
    ruff check --fix $FILES
  else
    echo ">> black --check"
    black --check --diff $FILES
    echo ">> ruff check"
    ruff check $FILES
  fi
fi

if [ "$RUN_TESTS" = "1" ]; then
  echo ">> Running the full test suite (not diff-scoped — partial test runs aren't meaningful)"
  cd server
  if [ -x ../.venv/bin/pytest ]; then
    ../.venv/bin/pytest -q
  else
    python3 -m pytest -q
  fi
fi

echo ">> OK"
