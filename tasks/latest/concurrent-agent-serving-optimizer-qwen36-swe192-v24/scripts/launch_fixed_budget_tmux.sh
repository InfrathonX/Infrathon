#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
SOCKET="${TMUX_SOCKET:-mlsysforge-kvbench}"
MODEL_VALUE="${MODEL:-}"
REASONING_VALUE="${REASONING_EFFORT:-max}"
TIMEOUT_VALUE="${TASK_TIMEOUT:-21600}"
[[ -n "${MODEL_VALUE}" ]] || { echo "MODEL must name the agent's model ID" >&2; exit 1; }
[[ "${TIMEOUT_VALUE}" =~ ^[1-9][0-9]*$ ]] || { echo "TASK_TIMEOUT must be a positive integer" >&2; exit 1; }
# The same run id run_fixed_budget.sh derives: model (without gpt-/claude-), effort, budget.
model_slug="${MODEL_VALUE#gpt-}"
model_slug="${model_slug#claude-}"
RUN_ID="${RUN_ID:-kvbench-qwen36-swe192-v24-${model_slug}-${REASONING_VALUE}-$(( (TIMEOUT_VALUE + 3599) / 3600 ))h-${RUN_STAMP}}"
SESSION="${TMUX_SESSION:-kvbench-${RUN_STAMP}}"
COOLDOWN_VALUE="${SUBMISSION_COOLDOWN:-300}"
WEB_SEARCH_VALUE="${CODEX_WEB_SEARCH_MODE:-live}"
LAUNCH_DIR="${LOGS_DIR}/launch"
LOG_FILE="${LAUNCH_DIR}/${RUN_ID}.log"
EXIT_FILE="${LAUNCH_DIR}/${RUN_ID}.exit"

command -v tmux >/dev/null 2>&1 || { echo "missing tmux" >&2; exit 1; }
[[ -n "${CODEX_AUTH_JSON_PATH:-}" ]] || {
  echo "CODEX_AUTH_JSON_PATH must point to a host-only auth.json file" >&2
  exit 1
}
[[ -n "${SFORGE_KVBENCH_NONCE_SOURCE:-}" ]] || {
  echo "SFORGE_KVBENCH_NONCE_SOURCE must point to the external nonce source" >&2
  exit 1
}
for required_name in \
  KVBENCH_VLLM_BASE_URL \
  KVBENCH_HIDDEN_BASELINE_PATH \
  KVBENCH_PUBLIC_BASELINE_PATH; do
  [[ -n "${!required_name:-}" ]] || {
    echo "${required_name} must be configured" >&2
    exit 1
  }
done
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
  echo "unsafe RUN_ID" >&2
  exit 1
}
[[ ! -e "${LOG_FILE}" && ! -e "${EXIT_FILE}" ]] || {
  echo "launch artifacts already exist for ${RUN_ID}" >&2
  exit 1
}
if tmux -L "${SOCKET}" has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
mkdir -p "${LAUNCH_DIR}"

run_command=(
  env
  "REPOSITORY_ROOT=${REPOSITORY_ROOT}"
  "SFORGE_ROOT=${SFORGE_ROOT}"
  "SFORGE_BIN=${SFORGE_BIN}"
  "PYTHON_BIN=${PYTHON_BIN}"
  "TASKS_DIR=${TASKS_DIR}"
  "LOGS_DIR=${LOGS_DIR}"
  "ARTIFACTS_DIR=${ARTIFACTS_DIR}"
  "RUN_ID=${RUN_ID}"
  "RUN_STAMP=${RUN_STAMP}"
  "MODEL=${MODEL_VALUE}"
  "REASONING_EFFORT=${REASONING_VALUE}"
  "TASK_TIMEOUT=${TIMEOUT_VALUE}"
  "SUBMISSION_COOLDOWN=${COOLDOWN_VALUE}"
  "CODEX_WEB_SEARCH_MODE=${WEB_SEARCH_VALUE}"
  "CODEX_CLI_VERSION=${CODEX_CLI_VERSION:-0.148.0}"
  "CODEX_AUTH_JSON_PATH=${CODEX_AUTH_JSON_PATH}"
  "SFORGE_KVBENCH_NONCE_SOURCE=${SFORGE_KVBENCH_NONCE_SOURCE}"
  "KVBENCH_VLLM_BASE_URL=${KVBENCH_VLLM_BASE_URL}"
  "KVBENCH_HIDDEN_BASELINE_PATH=${KVBENCH_HIDDEN_BASELINE_PATH}"
  "KVBENCH_PUBLIC_BASELINE_PATH=${KVBENCH_PUBLIC_BASELINE_PATH}"
)
for forwarded_name in SFORGE_AGENT_RUNTIME_DIR SFORGE_VERSION_STORE_ROOT; do
  if [[ -n "${!forwarded_name:-}" ]]; then
    run_command+=("${forwarded_name}=${!forwarded_name}")
  fi
done
run_command+=("${SCRIPT_DIR}/run_fixed_budget.sh")
printf -v quoted_command '%q ' "${run_command[@]}"
printf -v quoted_log '%q' "${LOG_FILE}"
printf -v quoted_exit '%q' "${EXIT_FILE}"
payload="${quoted_command}>${quoted_log} 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" >${quoted_exit}; exit \"\$rc\""

tmux -L "${SOCKET}" new-session -d -s "${SESSION}" "${payload}"
jq -n \
  --arg run_id "${RUN_ID}" \
  --arg session "${SESSION}" \
  --arg socket "${SOCKET}" \
  --arg model "${MODEL_VALUE}" \
  --arg reasoning_effort "${REASONING_VALUE}" \
  --argjson timeout_seconds "${TIMEOUT_VALUE}" \
  --argjson submission_cooldown_seconds "${COOLDOWN_VALUE}" \
  --arg codex_web_search "${WEB_SEARCH_VALUE}" \
  --arg log "${LOG_FILE}" \
  --arg exit_file "${EXIT_FILE}" \
  --arg launched_at "$(date -u +%FT%TZ)" \
  '{run_id:$run_id,session:$session,socket:$socket,
    model:$model,reasoning_effort:$reasoning_effort,
    timeout_seconds:$timeout_seconds,
    submission_cooldown_seconds:$submission_cooldown_seconds,
    codex_web_search:$codex_web_search,
    log:$log,exit_file:$exit_file,launched_at:$launched_at}' \
  >"${LAUNCH_DIR}/${RUN_ID}.json"

echo "Launched ${RUN_ID}"
echo "tmux: tmux -L ${SOCKET} attach -t ${SESSION}"
echo "log: ${LOG_FILE}"
echo "exit: ${EXIT_FILE}"
