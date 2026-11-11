#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; source "${SCRIPT_DIR}/lib.sh"
if [[ "${1:-}" == "--replay-core" ]]; then
  shift; exec "${PYTHON_BIN}" "${TASK_ROOT}/scripts/sforge_replay_entry.py" "$@"
fi
archive="${1:-}"; [[ -f "${archive}" && ! -L "${archive}" && -s "${archive}" ]] || die "usage: $0 ARCHIVE.tar.gz [RUN_ID]"
archive="$(cd "$(dirname "${archive}")" && pwd)/$(basename "${archive}")"; run_id="${2:-flux-moe-ag-final-$(date -u +%Y%m%d-%H%M%S)}"
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe RUN_ID"
SFORGE_PYTHON="${SFORGE_PYTHON:-${REPOSITORY_ROOT}/.venv/bin/python}"; [[ -x "${SFORGE_PYTHON}" ]] || die "missing SForge Python"
require_command docker; require_command nvidia-smi
resolve_gpu_pair "${FLUXMOE_GPU_INDICES:-}"; export FLUXMOE_GPU_INDICES="${GPU_PAIR_INDICES}" FLUXMOE_GPU_UUIDS="${GPU_PAIR_UUIDS}"
if ! docker info >/dev/null 2>&1; then
  [[ "${FLUXMOE_DOCKER_GROUP_REEXEC:-0}" != 1 ]] || die "Docker unavailable after group re-exec"
  printf -v reexec '%q ' env FLUXMOE_DOCKER_GROUP_REEXEC=1 REPOSITORY_ROOT="${REPOSITORY_ROOT}" SFORGE_PYTHON="${SFORGE_PYTHON}" FLUXMOE_GPU_INDICES="${GPU_PAIR_INDICES}" FLUXMOE_GPU_UUIDS="${GPU_PAIR_UUIDS}" bash "$0" "$@"
  exec sg docker -c "${reexec}"
fi
docker image inspect "${WORK_IMAGE}" "${JUDGE_IMAGE}" >/dev/null || die "Work/Judge images are missing"
python3 "${TASK_ROOT}/scripts/preflight.py" --gpu-indices "${GPU_PAIR_INDICES}" --expected-gpu-uuids "${GPU_PAIR_UUIDS}" >/dev/null
mkdir -p "${LOGS_DIR}"; export SFORGE_TASKS_DIR="${TASK_ROOT}/tasks" SFORGE_LOG_DIR="${LOGS_DIR}"
exec "${SFORGE_PYTHON}" "${TASK_ROOT}/scripts/sforge_dual_h100_entry.py" eval --backend docker --task "${TASK_ID}" --archive "${archive}" --run-id "${run_id}" --timeout 7200 --judge-cpu-limit 32 --judge-mem-limit 128g --json
