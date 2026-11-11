#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
mkdir -p "${ARTIFACTS_DIR}"
output="${1:-${ARTIFACTS_DIR}/minference-prefill-v3-oracle.tar.gz}"
[[ ! -e "${output}" ]] || die "refusing to replace archive: ${output}"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  -C "${MIGRATION_ROOT}/oracle" -czf "${output}" FINAL_VERSION notes.md candidates
sha256_file "${output}"
