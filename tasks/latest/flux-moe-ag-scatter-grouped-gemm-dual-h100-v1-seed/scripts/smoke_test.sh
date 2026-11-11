#!/usr/bin/env bash
set -euo pipefail
SFORGE_ENTRY_SCRIPT="$0"; SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${SCRIPT_DIR}/lib.sh"
require_sforge; "${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}"; PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" -m pytest -p no:cacheprovider -q "${MIGRATION_ROOT}/tests"
# oracle/FINAL_VERSION selects the strongest oracle that runs on the suite (a forced,
# non-admissible paper candidate when one exists); anything run through the normal
# admission-enforcing Judge uses the
# best admissible one (see oracle/notes.md).
ORACLE_ADMISSIBLE_VERSION="${ORACLE_ADMISSIBLE_VERSION:-1}"
if [[ "${1:-}" == "--oracle" ]]; then
  ensure_docker_group "$@"; require_command docker; require_command flock; require_command nvidia-smi; assert_built_images_current
  resolve_gpu_pair "${FLUXMOE_GPU_INDICES:-}"; export FLUXMOE_GPU_INDICES="${GPU_PAIR_INDICES}" FLUXMOE_GPU_UUIDS="${GPU_PAIR_UUIDS}"
  lock_gpu_group /tmp/mlsysforge-gpu
  python3 "${MIGRATION_ROOT}/scripts/preflight.py" --gpu-indices "${GPU_PAIR_INDICES}" --expected-gpu-uuids "${GPU_PAIR_UUIDS}" >/dev/null
  docker run --rm --gpus "\"device=${GPU_PAIR_UUIDS}\"" -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 -v "${MIGRATION_ROOT}/oracle:/oracle:ro" --entrypoint python "${WORK_IMAGE}" /workspace/benchmark/public_runner.py --lab-root /oracle --suite full --candidate-version "${ORACLE_ADMISSIBLE_VERSION}"
  for fd in "${GROUP_LOCK_FDS[@]}"; do flock -u "${fd}"; done
  oracle_archive="$(ORACLE_VERSION="${ORACLE_ADMISSIBLE_VERSION}" "${MIGRATION_ROOT}/scripts/make_oracle_archive.sh" | sed -n '1p')"
  FLUXMOE_GPU_INDICES="${GPU_PAIR_INDICES}" "${MIGRATION_ROOT}/scripts/eval_archive.sh" "${oracle_archive}"
elif [[ $# -ne 0 ]]; then die "usage: $0 [--oracle]"; fi
echo "Focused task smoke: PASS"
