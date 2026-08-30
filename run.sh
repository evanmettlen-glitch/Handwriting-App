#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY="./.venv/bin/python"
if [[ ! -x "$PY" ]]; then
    echo "No virtualenv found. Run ./install.sh first." >&2
    exit 1
fi

exec "$PY" -m handwriting_app "$@"
