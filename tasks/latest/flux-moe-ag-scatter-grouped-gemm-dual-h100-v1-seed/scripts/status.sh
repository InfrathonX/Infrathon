#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${SCRIPT_DIR}/lib.sh"
run_id="${1:-}"; [[ -n "${run_id}" ]] || die "usage: $0 RUN_ID"
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe RUN_ID"
root="${LOGS_DIR}/runs/${run_id}"; echo "run id: ${run_id}"; echo "root: ${root}"
find "${root}" -maxdepth 2 -type f \( -name 'final_result.json' -o -name 'completion.json' -o -name 'launcher.exit' \) -print 2>/dev/null || true
