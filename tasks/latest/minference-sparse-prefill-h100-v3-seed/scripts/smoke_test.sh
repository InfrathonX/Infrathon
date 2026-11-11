#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON_BIN:-python3}" "${TASK_ROOT}/scripts/validate_layout.py" --structural-only
