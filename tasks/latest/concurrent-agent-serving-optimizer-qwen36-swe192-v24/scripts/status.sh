#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

echo "Concurrent Agent Serving Optimizer MLSys Forge status"
echo "task: ${TASK_ID}"
echo

echo "tmux sessions:"
if command -v tmux >/dev/null 2>&1; then
  tmux -L mlsysforge-kvbench list-sessions 2>/dev/null \
    | grep -E 'kvbench|qwen36' \
    || echo "  no matching session"
else
  echo "  tmux unavailable"
fi

echo
echo "task and model containers:"
container_snapshot=""
if command -v docker >/dev/null 2>&1; then
  container_snapshot="$(docker ps --format '{{.Names}} {{.Image}} {{.Status}}' 2>/dev/null || true)"
  if [[ -z "${container_snapshot}" ]] && user_in_docker_group "$(id -un)"; then
    container_snapshot="$(sg docker -c \
      "docker ps --format '{{.Names}} {{.Image}} {{.Status}}'" 2>/dev/null || true)"
  fi
fi
if [[ -n "${container_snapshot}" ]]; then
  printf '%s\n' "${container_snapshot}" \
    | grep -E 'kvbench|qwen36|sforge' \
    || echo "  no matching running container"
else
  echo "  Docker unavailable or no running containers"
fi

echo
echo "GPU snapshot:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu \
    --format=csv,noheader 2>/dev/null \
    || echo "  GPU query failed"
else
  echo "  nvidia-smi unavailable"
fi

echo
echo "configured vLLM health:"
vllm_base_url="${KVBENCH_VLLM_BASE_URL:-}"
if command -v curl >/dev/null 2>&1 \
   && [[ -n "${vllm_base_url}" ]] \
   && curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS \
      "${vllm_base_url%/}/health" \
      >/dev/null 2>&1; then
  echo "  ready"
else
  echo "  unavailable, not ready, or KVBENCH_VLLM_BASE_URL is unset"
fi

echo
echo "build manifest:"
manifest="${LOGS_DIR}/build/image-manifest.json"
if [[ -s "${manifest}" && ! -L "${manifest}" ]] && command -v jq >/dev/null 2>&1; then
  jq '{created_at, source_digest, work, judge}' "${manifest}"
else
  echo "  not present"
fi

run_id="${1:-}"
launch_metadata=""
if [[ -z "${run_id}" && -d "${LOGS_DIR}/launch" ]] \
    && command -v jq >/dev/null 2>&1; then
  # Prefer the latest formal launcher identity.  A final evaluation creates a
  # newer sibling run directory named <RUN_ID>-final-120m-aN, so selecting the
  # newest logs/runs directory can otherwise make status jump away from the
  # actual budgeted run during its terminal evaluation phase.
  launch_metadata="$(find "${LOGS_DIR}/launch" -maxdepth 1 -type f -name '*.json' \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
  if [[ -n "${launch_metadata}" ]]; then
    run_id="$(jq -er '.run_id | select(type == "string")' \
      "${launch_metadata}" 2>/dev/null || true)"
  fi
fi
if [[ -z "${run_id}" ]]; then
  latest="$(find "${LOGS_DIR}/runs" -mindepth 1 -maxdepth 1 -type d \
    -printf '%T@ %f\n' 2>/dev/null | sort -nr | head -1 || true)"
  run_id="${latest#* }"
fi

echo
echo "latest run:"
if [[ -z "${run_id}" ]]; then
  echo "  no run directory"
  exit 0
fi
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
  || die "unsafe run id: ${run_id}"
run_root="${LOGS_DIR}/runs/${run_id}"
echo "  ${run_root}"
if [[ ! -d "${run_root}" ]]; then
  echo "  run directory is absent"
  exit 0
fi

echo
echo "formal launcher state:"
launch_json="${LOGS_DIR}/launch/${run_id}.json"
launch_exit="${LOGS_DIR}/launch/${run_id}.exit"
if [[ -s "${launch_json}" && ! -L "${launch_json}" ]] \
    && command -v jq >/dev/null 2>&1; then
  jq '{run_id,session,socket,launched_at,log,exit_file}' "${launch_json}" \
    2>/dev/null || echo "  launcher metadata is not valid JSON"
else
  echo "  no launcher metadata (this may be a smoke or standalone eval run)"
fi
if [[ -s "${launch_exit}" && ! -L "${launch_exit}" ]]; then
  echo "  launcher exit: $(tr -d '\r\n' <"${launch_exit}")"
else
  echo "  launcher exit: still running or unavailable"
fi

echo
echo "completion manifest:"
completion_path="${run_root}/completion.json"
if [[ -s "${completion_path}" && ! -L "${completion_path}" ]] \
    && command -v jq >/dev/null 2>&1; then
  jq '{
        status,
        stage,
        failure_reason: .failure.reason,
        started_at,
        finished_at,
        exit_code: .failure.exit_code,
        run_rc: .run.rc,
        final_rc: .final_evaluation.rc,
        final_classification: .final_evaluation.classification,
        final_valid: .final_evaluation.valid,
        final_score: .final_evaluation.score,
        selected_round: .selected.round,
        selected_score: .selected.score,
        cleanup_status: .cleanup.status
      }' \
    "${completion_path}" 2>/dev/null || echo "  completion manifest is not valid JSON"
else
  echo "  not present"
fi

echo
echo "recent reports:"
mapfile -t reports < <(
  find "${run_root}" -type f -name report.json -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr | head -8 | cut -d' ' -f2-
)
if (( ${#reports[@]} == 0 )); then
  echo "  no report.json yet"
else
  printf '  %s\n' "${reports[@]}"
  if command -v jq >/dev/null 2>&1; then
    echo
    echo "newest report summary:"
    jq '{valid, score, score_0_100, pass_rate, passed, total_tests, runtime_seconds}' \
      "${reports[0]}" 2>/dev/null || echo "  report is not valid JSON"
  fi
fi

echo
echo "recent launcher output:"
launcher_log="${LOGS_DIR}/launch/${run_id}.log"
if [[ ! -f "${launcher_log}" || -L "${launcher_log}" ]]; then
  launcher_log="$(find "${LOGS_DIR}" -maxdepth 2 -type f \
    \( -name "*${run_id}*.log" -o -name 'launch-*.log' \) \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)"
fi
if [[ -n "${launcher_log}" ]]; then
  echo "  ${launcher_log}"
  tail -n 30 "${launcher_log}" 2>/dev/null || true
else
  echo "  no launcher log"
fi
