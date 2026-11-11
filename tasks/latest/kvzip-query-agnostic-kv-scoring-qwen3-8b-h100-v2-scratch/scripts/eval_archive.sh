#!/usr/bin/env bash
set -euo pipefail

# Evaluate one submission archive on the complete frozen final suite through
# `sforge eval` (round manual-1), with the Judge spread over JUDGE_GPUS.
#   JUDGE_GPUS=1,2,3 scripts/eval_archive.sh ARCHIVE.tar.gz [RUN_ID]

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_sforge
require_command flock
require_command jq
require_command nvidia-smi

archive="${1:-}"
[[ -f "${archive}" && ! -L "${archive}" ]] || die "usage: JUDGE_GPUS=1,2,3 $0 ARCHIVE.tar.gz [RUN_ID]"
run_id="${2:-kvzip-kv-scoring-scratch-v2-final-eval-$(date -u +%Y%m%d-%H%M%S)}"
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe RUN_ID: ${run_id}"
[[ ! -e "${LOGS_DIR}/runs/${run_id}" ]] || die "run directory already exists: ${run_id}"

assert_built_images_current
mapfile -t judge_uuids < <(judge_gpu_uuids "${JUDGE_GPUS:-}")
for uuid in "${judge_uuids[@]}"; do
  lock_gpu "${uuid}" "judge"
  assert_gpu_idle "${uuid}" "judge"
done
n_gpus="${#judge_uuids[@]}"

validate_model_repo "${KV_SCORING_MODEL_REPO}" >/dev/null
export KV_SCORING_MODEL_REPO
export KV_SCORING_JUDGE_GPU_UUIDS="${judge_uuids[*]}"
read -r t_candidate t_total t_judge t_eval < <(judge_timeouts "${n_gpus}")
export SFORGE_JUDGE_EXTRA_ENV="KV_SCORING_CANDIDATE_TIMEOUT=${KV_SCORING_CANDIDATE_TIMEOUT:-${t_candidate}},KV_SCORING_TOTAL_TIMEOUT=${KV_SCORING_TOTAL_TIMEOUT:-${t_total}},KV_SCORING_JUDGE_TIMEOUT=${KV_SCORING_JUDGE_TIMEOUT:-${t_judge}}"
# KV_SCORING_HOST_ORACLE=1 (host oracle qualification only) lets the Judge
# admit the exact oracle transport asset past the released-weights gate.
[[ "${KV_SCORING_HOST_ORACLE:-}" != 1 ]] || SFORGE_JUDGE_EXTRA_ENV+=",KV_SCORING_HOST_ORACLE=1"
"${SFORGE_PYTHON}" "${TASK_ENTRY}" eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${archive}" \
  --run-id "${run_id}" \
  --timeout "${FINAL_EVAL_TIMEOUT:-${t_eval}}" \
  --judge-cpu-limit "$((8 * n_gpus))" \
  --judge-mem-limit "${JUDGE_MEM_LIMIT:-$((32 + 8 * n_gpus))g}" \
  --json
for uuid in "${judge_uuids[@]}"; do assert_gpu_released "${uuid}"; done
