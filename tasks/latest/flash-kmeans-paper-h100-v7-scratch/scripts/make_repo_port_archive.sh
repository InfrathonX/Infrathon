#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

# Compatibility entry point retained for older local automation. v7-scratch has no
# repository starter; its smoke/calibration submission is built from the exact
# upstream torch_fallback.py kept outside both Docker build contexts.
"${MIGRATION_ROOT}/scripts/make_host_calibration_archives.sh" >/dev/null
echo "${ARTIFACTS_DIR}/torch-fallback-baseline.tar.gz"
