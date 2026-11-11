#!/usr/bin/env bash
set -euo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_sforge
require_command flock
require_command jq
require_command nvidia-smi

archive="${1:-}"
[[ -f "${archive}" ]] || die "usage: $0 ARCHIVE.tar.gz [RUN_ID]"
run_id="${2:-manual-eval-$(date -u +%Y%m%d-%H%M%S)}"
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
  || die "unsafe RUN_ID: ${run_id}"
[[ ! -e "${LOGS_DIR}/runs/${run_id}" ]] \
  || die "run directory already exists: ${run_id}"
assert_built_images_current
judge_gpu="$(canonical_gpu_index "${JUDGE_GPU:-2}")"
judge_uuid="$(gpu_uuid "${judge_gpu}")"

exec 9>"$(gpu_lock_path "${judge_uuid}")"
flock -n 9 || die "judge GPU ${judge_gpu} lock is already held"
assert_gpu_idle "${judge_gpu}" "judge"
assert_container_gpu "${JUDGE_IMAGE}" "${judge_uuid}" "judge"

export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${judge_uuid},CUDA_VISIBLE_DEVICES=0,FLASH_KMEANS_TARGET=H100"

"${SFORGE_BIN}" \
  eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${archive}" \
  --run-id "${run_id}" \
  --judge-cpu-limit 8 \
  --judge-mem-limit 32g \
  --json
