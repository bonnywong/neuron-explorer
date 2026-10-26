#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python3 setup_runtime.py "$@"
export HF_HOME="${HF_HOME:-$PWD/.model-cache}"
printf '%s\n' 'Open http://localhost:8791 — weights load only when you choose Start Qwen.'
exec .venv/bin/python explorer/server.py
