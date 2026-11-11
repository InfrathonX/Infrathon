#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONDONTWRITEBYTECODE=1

exec python3 /workspace/public_tests/public_eval.py "$@"
