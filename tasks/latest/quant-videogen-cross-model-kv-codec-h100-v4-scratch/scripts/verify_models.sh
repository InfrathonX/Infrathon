#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_artifact_root
require_command python3
verification_dir="${LOGS_DIR}/verification"
mkdir -p "${verification_dir}"
# verify_models.py writes only into a directory others cannot write (a umask of 002 would
# leave it group-writable), as scripts/run_agent.sh does for its run directory.
chmod 0755 "${verification_dir}"
PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/verify_models.py" \
  --artifact-root "${ARTIFACT_ROOT}" \
  --output "${verification_dir}/model-lock.json"
