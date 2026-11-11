#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

run_id="${1:-}"
if [[ -z "${run_id}" ]]; then
  # Formal runs only: single-allocation (execution-profile) and legacy dual-GPU
  # run IDs; the newest directory wins.
  mapfile -t candidates < <(
    ls -1td "${LOGS_DIR}"/runs/standalone-kgrouped-wgrad-v2-single-cuda-* \
      "${LOGS_DIR}"/runs/standalone-kgrouped-wgrad-v2-h100-* 2>/dev/null || true
  )
  ((${#candidates[@]} > 0)) || die "no standalone K-grouped wgrad v2 runs below ${LOGS_DIR}/runs"
  run_root="${candidates[0]}"
  run_id="$(basename "${run_root}")"
else
  [[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
    || die "unsafe RUN_ID: ${run_id}"
  run_root="${LOGS_DIR}/runs/${run_id}"
fi

echo "run id: ${run_id}"
echo "root:   ${run_root}"
launcher_exit="${LOGS_DIR}/launch-${run_id}.exit"
if [[ -s "${launcher_exit}" ]]; then
  echo "launcher exit: $(tr -d '[:space:]' <"${launcher_exit}")"
else
  echo "launcher exit: pending"
fi

completion="${run_root}/completion.json"
if [[ -s "${completion}" ]]; then
  jq '{status, finished_at, task_id, work_gpu_uuid, judge_gpu_uuid, final_archive, final_report}' "${completion}"
  exit 0
fi

task_root="${run_root}/${TASK_ID}"
if [[ -s "${task_root}/final_result.json" ]]; then
  jq '{timed_out, runtime_seconds, archive_size_bytes, agent_submissions, auto_submissions}' \
    "${task_root}/final_result.json"
else
  echo "final result: pending"
fi
