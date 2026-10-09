#!/usr/bin/env bash
# Installs a built wheel into a fresh virtualenv outside the source tree and
# checks that it runs, to catch files missing from the wheel.
#
# Usage: scripts/smoke_test.sh dist/claude_wrap-*.whl
set -euo pipefail

if [ $# -ne 1 ] || [ ! -f "$1" ]; then
  echo "usage: $0 <path to a built .whl>" >&2
  exit 2
fi

wheel="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
check="$(cd "$(dirname "$0")" && pwd)/smoke_check.py"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

"${PYTHON:-python3}" -m venv "$work/venv"
"$work/venv/bin/pip" install --quiet "$wheel"

# A throwaway config and data directory, and a working directory away from
# the source tree, so only the installed package is exercised.
export WRAP_CONFIG_DIR="$work/config" WRAP_DATA_DIR="$work/data"
cd "$work"

"$work/venv/bin/wrap" --version
"$work/venv/bin/wrap" config
"$work/venv/bin/wrap" config --effective > /dev/null
"$work/venv/bin/python" -I "$check"
echo "smoke test passed"
