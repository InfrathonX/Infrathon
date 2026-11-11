#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
# The official-oracle archive is built from an upstream Flash-KMeans checkout
# that this repository deliberately does not vendor.  Provide it explicitly:
#
#   git clone https://github.com/svg-project/flash-kmeans /path/to/flash-kmeans
#   git -C /path/to/flash-kmeans checkout 12ed49b2eddc207dfc5ed27dba244af2365bd9c6
#   FLASH_KMEANS_UPSTREAM=/path/to/flash-kmeans scripts/make_host_calibration_archives.sh
#
# Only flash_kmeans/assign_euclid_triton.py and
# flash_kmeans/centroid_update_triton.py are read.
VENDOR_ROOT="${FLASH_KMEANS_UPSTREAM:-${VENDOR_ROOT:-}}"
if [[ -z "${VENDOR_ROOT}" ]]; then
  echo "ERROR: set FLASH_KMEANS_UPSTREAM to an upstream flash-kmeans checkout" \
       "pinned at 12ed49b2eddc207dfc5ed27dba244af2365bd9c6." >&2
  exit 1
fi
if [[ ! -f "${VENDOR_ROOT}/flash_kmeans/assign_euclid_triton.py" ]]; then
  echo "ERROR: ${VENDOR_ROOT} is not a flash-kmeans checkout" \
       "(flash_kmeans/assign_euclid_triton.py is missing)." >&2
  exit 1
fi

python3 "${SCRIPT_DIR}/build_torch_fallback_archive.py" \
  --source "${MIGRATION_ROOT}/host_reference/baseline/torch_fallback.py" \
  --source-output \
    "${MIGRATION_ROOT}/host_reference/baseline/generated_task_adapter.py" \
  --output "${MIGRATION_ROOT}/artifacts/torch-fallback-baseline.tar.gz"

python3 "${SCRIPT_DIR}/build_fast_pytorch_kmeans_archive.py" \
  --upstream-root \
    "${MIGRATION_ROOT}/host_reference/fast_pytorch_kmeans" \
  --source-output \
    "${MIGRATION_ROOT}/host_reference/fast_pytorch_kmeans/generated_task_adapter.py" \
  --output "${MIGRATION_ROOT}/artifacts/fast-pytorch-kmeans-baseline.tar.gz"

python3 "${SCRIPT_DIR}/build_fastkmeans_archive.py" \
  --upstream-root \
    "${MIGRATION_ROOT}/host_reference/fastkmeans" \
  --source-output \
    "${MIGRATION_ROOT}/host_reference/fastkmeans/generated_task_adapter.py" \
  --output "${MIGRATION_ROOT}/artifacts/fastkmeans-baseline.tar.gz"

python3 "${MIGRATION_ROOT}/host_reference/standard_pytorch_kmeans/build_archive.py" \
  --source \
    "${MIGRATION_ROOT}/host_reference/standard_pytorch_kmeans/standard_kmeans.py" \
  --source-output \
    "${MIGRATION_ROOT}/host_reference/standard_pytorch_kmeans/generated_task_adapter.py" \
  --output \
    "${MIGRATION_ROOT}/artifacts/standard-pytorch-kmeans-baseline.tar.gz"

python3 "${SCRIPT_DIR}/build_official_repo_head_archive.py" \
  --vendor-root "${VENDOR_ROOT}" \
  --source-output \
    "${MIGRATION_ROOT}/host_reference/official_oracle/generated_task_adapter.py" \
  --output "${MIGRATION_ROOT}/artifacts/official-oracle.tar.gz"

# Strongest admissible upstream-derived oracle (official HEAD adapter + the
# upstream fa3-branch epilogue + an H100 NVL retune); packaged from the
# committed adapter, see host_reference/official_oracle_fa3epi_h100/README.md.
"${SCRIPT_DIR}/make_fa3epi_h100_oracle_archive.sh" >/dev/null

echo "Host-only calibration archives created."
