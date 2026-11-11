#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate one submission archive on the complete hidden v4 suite through
# `sforge eval` (round manual-1), with the Judge spread over JUDGE_GPUS:
#   JUDGE_GPUS=0,1,2,3,4,5,6,7 scripts/eval_archive.sh \
#     [--skip-integrity-review] ARCHIVE.tar.gz [RUN_ID]
# Agent runs happen on PyroMind; this is the local Judge-only path for finals,
# controls and (through run_oracle.sh) the trusted oracle.  Both host lock
# names of every Judge GPU are held for the whole evaluation.  The Judge gets
# its CPU and memory share per GPU from the task JSON and a timeout that never
# undercuts the controller's own list-scheduling bound (lib.sh judge_plan).
# --skip-integrity-review is for trusted oracle and control archives only,
# never an agent's final archive.

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

usage="usage: JUDGE_GPUS=0,1,... $0 [--skip-integrity-review] ARCHIVE.tar.gz [RUN_ID]"
review_args=()
if [[ "${1:-}" == --skip-integrity-review ]]; then
  review_args=(--skip-integrity-review)
  shift
fi
(( $# >= 1 && $# <= 2 )) || die "${usage}"
archive="$1"
[[ -f "${archive}" && ! -L "${archive}" ]] || die "${usage}"
run_id="${2:-qvg-h100-scratch-v4-eval-$(date -u +%Y%m%d-%H%M%S)}"
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$ ]] || die "unsafe RUN_ID: ${run_id}"

ensure_docker_group ${review_args[@]+"${review_args[@]}"} "${archive}" "${run_id}"
for command in docker flock jq nvidia-smi python3 stat; do
  require_command "${command}"
done
require_sforge
require_artifact_root
[[ ! -e "${LOGS_DIR}/runs/${run_id}" ]] || die "run directory already exists: ${run_id}"

assert_built_images_current
judge_uuid_lines="$(judge_gpu_uuids "${JUDGE_GPUS:-}")"
mapfile -t judge_uuids <<<"${judge_uuid_lines}"
for uuid in "${judge_uuids[@]}"; do
  lock_gpu "${uuid}" judge
done
for uuid in "${judge_uuids[@]}"; do
  assert_gpu_idle "${uuid}" judge
  assert_no_compute_process "${uuid}" judge
done
assert_identical_gpus judge "${judge_uuids[@]}"
n_gpus="${#judge_uuids[@]}"
plan="$(judge_plan "${n_gpus}")"
read -r cpu_limit mem_limit controller_seconds eval_seconds <<<"${plan}"
eval_timeout="${FINAL_EVAL_TIMEOUT:-${eval_seconds}}"
[[ "${eval_timeout}" =~ ^[1-9][0-9]*$ ]] \
  && (( eval_timeout > controller_seconds )) \
  || die "FINAL_EVAL_TIMEOUT must exceed the controller bound of ${controller_seconds} s for ${n_gpus} GPU(s)"

# The controller reads its budget from the baked release and needs no extra
# environment; a CUDA_VISIBLE_DEVICES mask would hide all but one Judge GPU.
unset SFORGE_JUDGE_EXTRA_ENV
export QVG_JUDGE_GPU_UUIDS="${judge_uuids[*]}"
echo "Judge plan: ${n_gpus} GPU(s), ${JUDGE_CPU_LIMIT:-${cpu_limit}} CPUs, ${JUDGE_MEM_LIMIT:-${mem_limit}}, controller bound ${controller_seconds} s, eval timeout ${eval_timeout} s"
"${SFORGE_PYTHON}" "${RUNTIME_ENTRY}" eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${archive}" \
  --run-id "${run_id}" \
  --timeout "${eval_timeout}" \
  --judge-cpu-limit "${JUDGE_CPU_LIMIT:-${cpu_limit}}" \
  --judge-mem-limit "${JUDGE_MEM_LIMIT:-${mem_limit}}" \
  --json \
  ${review_args[@]+"${review_args[@]}"}
for uuid in "${judge_uuids[@]}"; do
  assert_gpu_released "${uuid}" judge
done
echo "report: ${LOGS_DIR}/runs/${run_id}/${TASK_ID}/submissions/manual-1/report.json"
