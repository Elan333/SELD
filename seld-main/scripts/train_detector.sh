#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
"$PYTHON" -m models.SELD.self_training "$@"
