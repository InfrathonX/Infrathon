#!/usr/bin/env bash
# Capture complete v3 real-RULER performance fixtures. Use --help for options.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec "${PYTHON_BIN:-python3}" "${TASK_ROOT}/scripts/capture_performance_v3.py" "$@"
