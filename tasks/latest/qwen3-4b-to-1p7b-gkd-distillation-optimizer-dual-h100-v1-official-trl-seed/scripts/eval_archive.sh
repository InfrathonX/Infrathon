#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

archive="${1:-}"
[[ -f "${archive}" && ! -L "${archive}" && -s "${archive}" ]] \
  || die "usage: $0 ARCHIVE.tar.gz [RUN_ID]"
archive="$(cd "$(dirname "${archive}")" && pwd)/$(basename "${archive}")"
run_id="${2:-qwen3-gkd-final-$(date -u +%Y%m%d-%H%M%S)}"
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

# The task pins no GPU pair: the operator names it, and it is resolved to
# UUIDs once so the re-exec below carries the same physical devices.
gpu_indices="${GKD_GPU_INDICES:-}"
gpu_uuids="${GKD_GPU_UUIDS:-}"
if [[ -z "${gpu_uuids}" ]]; then
  resolve_gpu_pair "${gpu_indices}"
  gpu_indices="${GPU_PAIR_INDICES}"
  gpu_uuids="${GPU_PAIR_UUIDS}"
fi
[[ -n "${gpu_indices}" ]] \
  || die "set GKD_GPU_INDICES to the two physical GPU indices for this evaluation"
export GKD_GPU_UUIDS="${gpu_uuids}"
export GKD_GPU_INDICES="${gpu_indices}"

# Docker access may require refreshing supplementary groups in a long-lived
# shell. Re-execute exactly once without forwarding unrelated environment.
if ! docker info >/dev/null 2>&1; then
  [[ "${GKD_DOCKER_GROUP_REEXEC:-0}" != "1" ]] \
    || die "Docker is still unavailable after docker-group re-exec"
  getent group docker | awk -F: -v user="$(id -un)" '
    $1 == "docker" { n=split($4,m,","); for(i=1;i<=n;i++) if(m[i]==user) ok=1 }
    END { exit(ok ? 0 : 1) }
  ' || die "Docker is unavailable and the current user is not in the docker group"
  printf -v reexec '%q ' env \
    GKD_DOCKER_GROUP_REEXEC=1 \
    REPOSITORY_ROOT="${REPOSITORY_ROOT}" \
    SFORGE_PYTHON="${SFORGE_PYTHON}" \
    GKD_GPU_UUIDS="${gpu_uuids}" \
    GKD_GPU_INDICES="${gpu_indices}" \
    GKD_STUDENT_MODEL_HOST_PATH="${GKD_STUDENT_MODEL_HOST_PATH:-}" \
    GKD_TEACHER_MODEL_HOST_PATH="${GKD_TEACHER_MODEL_HOST_PATH:-}" \
    GKD_DATA_HOST_ROOT="${GKD_DATA_HOST_ROOT:-}" \
    GKD_PRIVATE_DATA_HOST_ROOT="${GKD_PRIVATE_DATA_HOST_ROOT:-}" \
    SFORGE_AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}" \
    SFORGE_VERSION_STORE_ROOT="${SFORGE_VERSION_STORE_ROOT:-}" \
    bash "$0" "$@"
  exec sg docker -c "${reexec}"
fi

docker image inspect "${WORK_IMAGE}" "${JUDGE_IMAGE}" >/dev/null \
  || die "final Work/Judge images are missing; run scripts/build_images.sh"

model_repo="${GKD_STUDENT_MODEL_HOST_PATH:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--Qwen--Qwen3-1.7B}"
teacher_model_repo="${GKD_TEACHER_MODEL_HOST_PATH:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--Qwen--Qwen3-4B}"
# Data defaults: the task's Infrathon assets (assets/fetch.sh) when
# INFRATHON_ASSETS_ROOT is set, else the host artifact directory.
# Unset: the default root of assets/fetch.sh, when it exists.
[[ -n "${INFRATHON_ASSETS_ROOT:-}" || ! -d "${HOME}/.cache/infrathon/assets" ]] \
  || INFRATHON_ASSETS_ROOT="${HOME}/.cache/infrathon/assets"
if [[ -n "${INFRATHON_ASSETS_ROOT:-}" ]]; then
  default_data_root="${INFRATHON_ASSETS_ROOT}/qwen3-4b-to-1p7b-gkd-distillation-optimizer-dual-h100-v1-official-trl-seed/judge/tldr_gkd_v1"
  default_private_data_root="${INFRATHON_ASSETS_ROOT}/qwen3-4b-to-1p7b-gkd-distillation-optimizer-dual-h100-v1-official-trl-seed/judge/tldr_gkd_v1"
else
  default_data_root="${HOME}/mlsysforge-artifacts/qwen3-gkd/tldr_gkd_v1"
  default_private_data_root="${HOME}/mlsysforge-artifacts/qwen3-gkd/tldr_gkd_v1"
fi
data_root="${GKD_DATA_HOST_ROOT:-${default_data_root}}"
private_data_root="${GKD_PRIVATE_DATA_HOST_ROOT:-${default_private_data_root}}"
python3 "${TASK_ROOT}/scripts/preflight.py" \
  --gpu-indices "${gpu_indices}" \
  --expected-gpu-uuids "${gpu_uuids}" \
  --model-path "${model_repo}/snapshots/${STUDENT_MODEL_REVISION}" \
  --teacher-model-path "${teacher_model_repo}/snapshots/${TEACHER_MODEL_REVISION}" \
  --data-root "${data_root}" \
  --private-data-root "${private_data_root}" >/dev/null

mkdir -p "${LOGS_DIR}"
export SFORGE_TASKS_DIR="${TASK_ROOT}/tasks"
export SFORGE_LOG_DIR="${LOGS_DIR}"
export GKD_STUDENT_MODEL_HOST_PATH="${model_repo}"
export GKD_TEACHER_MODEL_HOST_PATH="${teacher_model_repo}"
export GKD_DATA_HOST_ROOT="${data_root}"
export GKD_PRIVATE_DATA_HOST_ROOT="${private_data_root}"

exec "${SFORGE_PYTHON}" "${TASK_ROOT}/scripts/sforge_dual_h100_entry.py" eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${archive}" \
  --run-id "${run_id}" \
  --timeout 12000 \
  --judge-cpu-limit 16 \
  --judge-mem-limit 96g \
  --json
