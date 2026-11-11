#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
"$PYTHON" tools/pseudo_label_mining.py "$@"
