#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${SCRIPT_DIR}/lib.sh"
oracle="${MIGRATION_ROOT}/oracle"
[[ -f "${oracle}/FINAL_VERSION" && -d "${oracle}/candidates/runtime_v001" ]] || die "oracle seed is incomplete"
mkdir -p "${ARTIFACTS_DIR}"
tar -C "${oracle}" --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner -cf - candidates FINAL_VERSION notes.md | gzip -n > "${ARTIFACTS_DIR}/oracle.tar.gz"
sha256sum "${ARTIFACTS_DIR}/oracle.tar.gz"
