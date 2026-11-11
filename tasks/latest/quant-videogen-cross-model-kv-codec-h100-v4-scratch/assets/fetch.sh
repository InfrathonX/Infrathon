#!/usr/bin/env bash
# Fetch this task's Infrathon assets into ${INFRATHON_ASSETS_ROOT} and verify them
# (scripts/infrathon/assets.py; extra arguments are passed through), then assemble
# QVG_ARTIFACT_ROOT from them (scripts/assemble_artifact_root.py).
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repository_root="$(cd "${task_root}/../../.." && pwd)"
python="${PYTHON_BIN:-${repository_root}/.venv/bin/python}"
"${python}" "${repository_root}/scripts/infrathon/assets.py" fetch "$(basename "${task_root}")" "$@"
exec "${python}" "${task_root}/scripts/assemble_artifact_root.py"
