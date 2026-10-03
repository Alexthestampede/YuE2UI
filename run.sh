#!/usr/bin/env bash
# Launch the YuE2UI server.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

# ROCm/MIOpen: skip the per-shape kernel search (~40-90s per new shape on
# first use). FAST picks a good kernel instantly; output is identical.
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"

exec python server.py