#!/usr/bin/env bash
# Build the pre-Hopper compatibility variant of the official adapter archive.
# See host_reference/official_oracle_ampere_compat/README.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
VENDOR_ROOT="${FLASH_KMEANS_UPSTREAM:-${VENDOR_ROOT:-}}"
if [[ -z "${VENDOR_ROOT}" ]]; then
  echo "ERROR: set FLASH_KMEANS_UPSTREAM to an upstream flash-kmeans checkout" \
       "pinned at 12ed49b2eddc207dfc5ed27dba244af2365bd9c6." >&2
  exit 1
fi
[[ -f "${VENDOR_ROOT}/flash_kmeans/assign_euclid_triton.py" ]] \
  || die "${VENDOR_ROOT} is not a flash-kmeans checkout"
require_command patch

scratch="$(mktemp -d)"
trap 'rm -rf "${scratch}"' EXIT
cp -r "${VENDOR_ROOT}/flash_kmeans" "${scratch}/flash_kmeans"
if [[ -f "${VENDOR_ROOT}/UPSTREAM_COMMIT" ]]; then
  cp "${VENDOR_ROOT}/UPSTREAM_COMMIT" "${scratch}/UPSTREAM_COMMIT"
else
  git -C "${VENDOR_ROOT}" rev-parse HEAD > "${scratch}/UPSTREAM_COMMIT"
fi
patch -p1 -d "${scratch}" \
  < "${MIGRATION_ROOT}/host_reference/official_oracle_ampere_compat/upstream.patch"

mkdir -p "${MIGRATION_ROOT}/artifacts/ampere-compat"
python3 "${SCRIPT_DIR}/build_official_repo_head_archive.py" \
  --vendor-root "${scratch}" \
  --device-smem-budget \
  --source-output "${MIGRATION_ROOT}/artifacts/ampere-compat/generated_task_adapter.py" \
  --output "${MIGRATION_ROOT}/artifacts/official-oracle-ampere-compat.tar.gz"
