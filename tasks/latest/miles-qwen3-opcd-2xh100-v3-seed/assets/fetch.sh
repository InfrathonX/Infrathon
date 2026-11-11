#!/usr/bin/env bash
# Fetch this task's Infrathon assets into ${INFRATHON_ASSETS_ROOT} and verify them
# (scripts/infrathon/assets.py).  Extra arguments are passed through.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository_root="$(cd "${task_root}/../../.." && pwd)"
exec "${PYTHON_BIN:-${repository_root}/.venv/bin/python}" \
  "${repository_root}/scripts/infrathon/assets.py" fetch "$(basename "${task_root}")" "$@"
