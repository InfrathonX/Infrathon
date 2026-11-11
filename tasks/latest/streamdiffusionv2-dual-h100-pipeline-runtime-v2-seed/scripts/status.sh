#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${SCRIPT_DIR}/lib.sh"
find "${LOGS_DIR}" -maxdepth 3 -type f \( -name 'final_result.json' -o -name 'launcher.exit' -o -name 'run_config.json' \) -print 2>/dev/null | sort | tail -30
