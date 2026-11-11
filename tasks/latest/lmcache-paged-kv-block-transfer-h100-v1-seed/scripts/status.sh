#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

run_id="${1:-}"
if [[ -z "${run_id}" ]]; then
  shopt -s nullglob
  candidates=("${LOGS_DIR}"/runs/lmcache-kv-transfer-v1-h100-*)
  shopt -u nullglob
  ((${#candidates[@]} > 0)) || die "no LMCache KV-transfer runs below ${LOGS_DIR}/runs"
  run_root="${candidates[${#candidates[@]}-1]}"
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
