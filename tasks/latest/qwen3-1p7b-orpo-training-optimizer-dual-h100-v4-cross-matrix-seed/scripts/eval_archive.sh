#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

archive="${1:-}"
[[ -f "${archive}" && ! -L "${archive}" && -s "${archive}" ]] \
  || die "usage: $0 ARCHIVE.tar.gz [RUN_ID]"
archive="$(cd "$(dirname "${archive}")" && pwd)/$(basename "${archive}")"
run_id="${2:-qwen3-orpo-final-$(date -u +%Y%m%d-%H%M%S)}"
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
  || die "unsafe RUN_ID: ${run_id}"

find_repository_root() {
  local candidate="${TASK_ROOT}"
  while [[ "${candidate}" != "/" ]]; do
    if [[ -f "${candidate}/pyproject.toml" && -d "${candidate}/sforge" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    candidate="$(dirname "${candidate}")"
  done
  die "cannot locate the MLSys Forge/SForge repository root"
}

REPOSITORY_ROOT="${REPOSITORY_ROOT:-$(find_repository_root)}"
SFORGE_PYTHON="${SFORGE_PYTHON:-${REPOSITORY_ROOT}/.venv/bin/python}"
[[ -x "${SFORGE_PYTHON}" ]] || die "missing SForge Python: ${SFORGE_PYTHON}"
require_command docker
require_command nvidia-smi

# Docker access may require refreshing supplementary groups in a long-lived
# shell. Re-execute exactly once without forwarding unrelated environment.
if ! docker info >/dev/null 2>&1; then
  [[ "${ORPO_DOCKER_GROUP_REEXEC:-0}" != "1" ]] \
    || die "Docker is still unavailable after docker-group re-exec"
  getent group docker | awk -F: -v user="$(id -un)" '
    $1 == "docker" { n=split($4,m,","); for(i=1;i<=n;i++) if(m[i]==user) ok=1 }
    END { exit(ok ? 0 : 1) }
  ' || die "Docker is unavailable and the current user is not in the docker group"
  printf -v reexec '%q ' env \
    ORPO_DOCKER_GROUP_REEXEC=1 \
    REPOSITORY_ROOT="${REPOSITORY_ROOT}" \
    SFORGE_PYTHON="${SFORGE_PYTHON}" \
    ORPO_GPU_UUIDS="${ORPO_GPU_UUIDS:?set ORPO_GPU_UUIDS to the two physical GPU UUIDs of the NVLink pair this run owns, comma separated (nvidia-smi --query-gpu=index,uuid --format=csv,noheader)}" \
    ORPO_MODEL_HOST_PATH="${ORPO_MODEL_HOST_PATH:-}" \
    ORPO_DATA_HOST_ROOT="${ORPO_DATA_HOST_ROOT:-}" \
    ORPO_PRIVATE_DATA_HOST_ROOT="${ORPO_PRIVATE_DATA_HOST_ROOT:-}" \
    SFORGE_AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}" \
    SFORGE_VERSION_STORE_ROOT="${SFORGE_VERSION_STORE_ROOT:-}" \
    bash "$0" "$@"
  exec sg docker -c "${reexec}"
fi

docker image inspect "${WORK_IMAGE}" "${JUDGE_IMAGE}" >/dev/null \
  || die "final Work/Judge images are missing; run scripts/build_images.sh"

model_repo="${ORPO_MODEL_HOST_PATH:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--Qwen--Qwen3-1.7B}"
# Data defaults: the task's Infrathon assets (assets/fetch.sh) when
# INFRATHON_ASSETS_ROOT is set, else the host artifact directory.
# Unset: the default root of assets/fetch.sh, when it exists.
[[ -n "${INFRATHON_ASSETS_ROOT:-}" || ! -d "${HOME}/.cache/infrathon/assets" ]] \
  || INFRATHON_ASSETS_ROOT="${HOME}/.cache/infrathon/assets"
if [[ -n "${INFRATHON_ASSETS_ROOT:-}" ]]; then
  default_data_root="${INFRATHON_ASSETS_ROOT}/qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed/public/tldr_preference_v1"
  default_private_data_root="${INFRATHON_ASSETS_ROOT}/qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed/judge/tldr_preference_cross_v4"
else
  default_data_root="${HOME}/mlsysforge-artifacts/qwen3-orpo/tldr_preference_v1"
  default_private_data_root="${HOME}/mlsysforge-artifacts/qwen3-orpo/tldr_preference_cross_v4"
fi
data_root="${ORPO_DATA_HOST_ROOT:-${default_data_root}}"
private_data_root="${ORPO_PRIVATE_DATA_HOST_ROOT:-${default_private_data_root}}"
gpu_indices="${ORPO_GPU_INDICES:?set ORPO_GPU_INDICES to the two physical GPU indices of the NVLink pair this run owns, comma separated (nvidia-smi --query-gpu=index,uuid --format=csv,noheader)}"
export ORPO_GPU_INDICES="${gpu_indices}"
python3 "${TASK_ROOT}/scripts/preflight.py" \
  --gpu-indices "${gpu_indices}" \
  --expected-gpu-uuids "${ORPO_GPU_UUIDS:?set ORPO_GPU_UUIDS to the two physical GPU UUIDs of the NVLink pair this run owns, comma separated (nvidia-smi --query-gpu=index,uuid --format=csv,noheader)}" \
  --model-path "${model_repo}/snapshots/${MODEL_REVISION}" \
  --data-root "${data_root}" \
  --private-data-root "${private_data_root}" >/dev/null

mkdir -p "${LOGS_DIR}"
export SFORGE_TASKS_DIR="${TASK_ROOT}/tasks"
export SFORGE_LOG_DIR="${LOGS_DIR}"
export ORPO_MODEL_HOST_PATH="${model_repo}"
export ORPO_DATA_HOST_ROOT="${data_root}"
export ORPO_PRIVATE_DATA_HOST_ROOT="${private_data_root}"

exec "${SFORGE_PYTHON}" "${TASK_ROOT}/scripts/sforge_dual_h100_entry.py" eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${archive}" \
  --run-id "${run_id}" \
  --timeout 12000 \
  --judge-cpu-limit 16 \
  --judge-mem-limit 96g \
  --json
