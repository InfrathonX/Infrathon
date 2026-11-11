#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${SCRIPT_DIR}/lib.sh"
python_bin="${PYTHON_BIN}"
[[ -x "${python_bin}" ]] || python_bin=python3
"${python_bin}" "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}"
PYTHONDONTWRITEBYTECODE=1 "${python_bin}" -m pytest -p no:cacheprovider -q "${MIGRATION_ROOT}/tests"
if [[ "${1:-}" == "--image" ]]; then
  SFORGE_ENTRY_SCRIPT="$0"; export SFORGE_ENTRY_SCRIPT; ensure_docker_group "$@"
  resolve_gpu_pair "${SDV2_GPU_INDICES:-}"
  physical_gpu="${GPU_PAIR_INDICES%%,*}"
  docker run --rm --network none --gpus "device=${physical_gpu}" -e CUDA_VISIBLE_DEVICES=0 --entrypoint python3 "${WORK_IMAGE}" /opt/sdv2_upstream/smoke_load.py
elif [[ $# -ne 0 ]]; then die "usage: $0 [--image]"; fi
echo "Focused task smoke: PASS"
