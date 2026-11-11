#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
# The docker-group re-exec below forwards a fixed allowlist that does not carry
# the Claude Code selection (MLSYSFORGE_AGENT, CLAUDE_OAUTH_TOKEN_FILE), so a
# claude-code launch must already reach Docker (launch/concurrent-serving/run.sh
# starts it under `sg docker` when needed).
if [[ "${MLSYSFORGE_AGENT:-codex}" != codex ]] && ! docker info >/dev/null 2>&1; then
  die "MLSYSFORGE_AGENT=${MLSYSFORGE_AGENT} needs Docker access without the docker-group re-exec (run it under sg docker)"
fi
ensure_docker_group "$@"

# MLSYSFORGE_AGENT (launch/concurrent-serving/run.sh --agent): codex (default)
# or claude-code (sforge/harness/agent/claude_local.py: OAuth token file, pinned
# runtime, one Agent without subagents, WebSearch/WebFetch when online).
AGENT="${MLSYSFORGE_AGENT:-codex}"
MODEL="${MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-0.148.0}"
TASK_TIMEOUT="${TASK_TIMEOUT:-21600}"
SUBMISSION_COOLDOWN="${SUBMISSION_COOLDOWN:-300}"
CODEX_WEB_SEARCH_MODE="${CODEX_WEB_SEARCH_MODE:-live}"
JUDGE_PORT="${JUDGE_PORT:-18188}"
CODEX_AUTH_JSON_PATH="${CODEX_AUTH_JSON_PATH:-}"
CLAUDE_OAUTH_TOKEN_FILE="${CLAUDE_OAUTH_TOKEN_FILE:-}"
# The Claude Code runtime a claude-code run mounts: the version
# sforge/harness/agent/claude_local.py binds (CLAUDE_RUNTIME_VERSION) and the
# launch/common.sh CLAUDE_CODE_VERSION_PIN.
CLAUDE_CODE_VERSION="2.1.287"
# The nonce source is a Judge-only input (judge/private/nonce_source.txt, else the
# asset-root copy from assets/fetch.sh: default_nonce_source in lib.sh);
# SFORGE_KVBENCH_NONCE_SOURCE only overrides it.
ORIGINAL_NONCE_SOURCE="${SFORGE_KVBENCH_NONCE_SOURCE:-$(default_nonce_source)}"
VLLM_BASE_URL="${KVBENCH_VLLM_BASE_URL:-}"
HIDDEN_BASELINE_PATH="${KVBENCH_HIDDEN_BASELINE_PATH:-}"
PUBLIC_BASELINE_PATH="${KVBENCH_PUBLIC_BASELINE_PATH:-}"
case "${AGENT}" in
  codex)
    [[ -n "${CODEX_AUTH_JSON_PATH}" ]] \
      || die "CODEX_AUTH_JSON_PATH must point to a host-only auth.json file"
    ;;
  claude-code)
    [[ -n "${CLAUDE_OAUTH_TOKEN_FILE}" ]] \
      || die "CLAUDE_OAUTH_TOKEN_FILE must point to a host-only Claude OAuth token file"
    # No Codex CLI runs in a Claude Code run; the completion record says so.
    CODEX_CLI_VERSION="none"
    ;;
  *) die "MLSYSFORGE_AGENT must be codex or claude-code" ;;
esac
[[ -n "${MODEL}" ]] \
  || die "MODEL is required (the agent's model ID; launch/concurrent-serving/run.sh --model)"
[[ -n "${ORIGINAL_NONCE_SOURCE}" ]] \
  || die "SFORGE_KVBENCH_NONCE_SOURCE must point to the external nonce source"
[[ -n "${VLLM_BASE_URL}" ]] || die "KVBENCH_VLLM_BASE_URL is required"
[[ -n "${HIDDEN_BASELINE_PATH}" ]] || die "KVBENCH_HIDDEN_BASELINE_PATH is required"
[[ -n "${PUBLIC_BASELINE_PATH}" ]] || die "KVBENCH_PUBLIC_BASELINE_PATH is required"
unset SFORGE_KVBENCH_NONCE_SOURCE
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
[[ "${TASK_TIMEOUT}" =~ ^[0-9]+$ ]] \
  || die "TASK_TIMEOUT must be a positive integer of seconds"
model_slug="${MODEL#gpt-}"
model_slug="${model_slug#claude-}"
budget_slug="$(( (10#${TASK_TIMEOUT} + 3599) / 3600 ))h"
RUN_ID="${RUN_ID:-kvbench-qwen36-swe192-v24-${model_slug}-${REASONING_EFFORT}-${budget_slug}-${RUN_STAMP}}"
RUN_LOG_ROOT="${LOGS_DIR}/runs/${RUN_ID}"
TASK_RUN_ROOT="${RUN_LOG_ROOT}/${TASK_ID}"
SFORGE_PYTHON="$(dirname -- "${SFORGE_BIN}")/python"
EXPECTED_MLSYSFORGE_ROOT="${REPOSITORY_ROOT}"
RUNTIME_ENTRY="${MIGRATION_ROOT}/scripts/sforge_runtime_entry.py"
PROXY_SOURCE="${MIGRATION_ROOT}/scripts/restricted_connect_proxy.py"
HOST_PROXY_BRIDGE_SOURCE="${MIGRATION_ROOT}/scripts/host_proxy_bridge.py"
HOST_PROXY_HOST="127.0.0.1"
HOST_PROXY_PORT="18080"
DOCKER_PROXY_PORT="19080"
NONCE_HELPER="${MIGRATION_ROOT}/scripts/prepare_nonce_source.py"
PREFLIGHT_HELPER="${MIGRATION_ROOT}/scripts/preflight.py"
CLEANUP_HELPER="${MIGRATION_ROOT}/scripts/cleanup_formal_runtime.py"
FINAL_VALIDATOR="${MIGRATION_ROOT}/scripts/validate_final_evaluation.py"
COMPLETION_WRITER="${MIGRATION_ROOT}/scripts/write_formal_completion.py"
TASK_SPEC="${TASKS_DIR}/${TASK_ID}.json"
LOCK_ROOT="/run/user/$(id -u)/mlsysforge-kvbench-qwen36-swe192-locks"
MEASUREMENT_LOCK="${LOCK_ROOT}/benchmark-measurement.lock"
FORMAL_RUN_LOCK="${LOCK_ROOT}/mlsysforge-v24-formal-run.lock"
EVAL_ROOT="${HOME}/.cache/mlsysforge-kvbench-qwen36-swe192-evals"
NONCE_RUNTIME_PARENT="/run/user/$(id -u)"
NONCE_RUNTIME_DIR=""
DEDICATED_NONCE_SOURCE=""
COMPLETION_PATH="${RUN_LOG_ROOT}/completion.json"
PREFLIGHT_PATH="${RUN_LOG_ROOT}/preflight.json"
EXPERIMENT_MANIFEST_PATH="${RUN_LOG_ROOT}/experiment-manifest.json"
BASELINE_SNAPSHOT_ROOT="${RUN_LOG_ROOT}/baseline-snapshots"
SOURCE_VERIFICATION_PATH="${TASK_RUN_ROOT}/trusted-source-final-verification.json"
FINAL_ARCHIVE="${TASK_RUN_ROOT}/final_archive.tar.gz"
FINAL_SNAPSHOT="${TASK_RUN_ROOT}/final_snapshot.json"
RUN_HISTORY="${TASK_RUN_ROOT}/run_history.json"
SELECTION_ROOT="${TASK_RUN_ROOT}/selection"
SELECTION_PATH="${SELECTION_ROOT}/best_selection.json"
SELECTED_VALIDATION="${TASK_RUN_ROOT}/selected-evaluation-validation.json"
MAX_FINAL_ATTEMPTS="${MAX_FINAL_ATTEMPTS:-3}"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
  || { echo "ERROR: unsafe RUN_ID" >&2; exit 1; }
[[ ! -e "${RUN_LOG_ROOT}" ]] \
  || { echo "ERROR: run directory already exists: ${RUN_LOG_ROOT}" >&2; exit 1; }
mkdir -p "${RUN_LOG_ROOT}"
chmod 0700 "${RUN_LOG_ROOT}"

run_started_at="$(date -u +%FT%TZ)"
stage="initialized"
failure_reason="formal run is still in progress"
completion_requested="failed"
no_valid_submission_reason=""
cleanup_status="not_started"
run_rc=""
final_eval_rc=""
final_attempt=""
final_eval_id=""
final_validation=""
judge_pid=""
active_pid=""
active_pgid=""
isolated_network=""
proxy_name=""
host_proxy_bridge_pid=""
measurement_lock_held=0
formal_lock_held=0
exit_started=0

die() {
  failure_reason="$*"
  echo "ERROR: $*" >&2
  exit 1
}

write_completion() {
  local requested_status="$1"
  local exit_code="$2"
  local cleanup_value="$3"
  local -a command=(
    python3 "${COMPLETION_WRITER}"
    --output "${COMPLETION_PATH}"
    --requested-status "${requested_status}"
    --run-id "${RUN_ID}"
    --task-id "${TASK_ID}"
    --started-at "${run_started_at}"
    --finished-at "$(date -u +%FT%TZ)"
    --model "${MODEL}"
    --reasoning-effort "${REASONING_EFFORT}"
    --codex-cli-version "${CODEX_CLI_VERSION}"
    --task-timeout "${TASK_TIMEOUT}"
    --submission-cooldown "${SUBMISSION_COOLDOWN}"
    --stage "${stage}"
    --reason "${failure_reason}"
    --exit-code "${exit_code}"
    --run-rc "${run_rc}"
    --final-rc "${final_eval_rc}"
    --cleanup-status "${cleanup_value}"
    --launcher "${SCRIPT_DIR}/run_fixed_budget.sh"
    --task-spec "${TASK_SPEC}"
    --preflight "${PREFLIGHT_PATH}"
    --experiment-manifest "${EXPERIMENT_MANIFEST_PATH}"
    --trusted-source-verification "${SOURCE_VERIFICATION_PATH}"
    --final-archive "${FINAL_ARCHIVE}"
    --final-snapshot "${FINAL_SNAPSHOT}"
    --final-validation "${final_validation:-${TASK_RUN_ROOT}/final-evaluation-validation.unavailable.json}"
    --selection "${SELECTION_PATH}"
    --selected-validation "${SELECTED_VALIDATION}"
  )
  if [[ -n "${DEDICATED_NONCE_SOURCE}" ]]; then
    command+=(--dedicated-nonce-source "${DEDICATED_NONCE_SOURCE}")
  fi
  if [[ -n "${final_attempt}" ]]; then
    command+=(--final-attempt "${final_attempt}")
  fi
  if [[ -n "${final_eval_id}" ]]; then
    command+=(--final-eval-id "${final_eval_id}")
  fi
  "${command[@]}"
}

wait_for_group_exit() {
  local pgid="$1"
  local seconds="$2"
  local deadline=$((SECONDS + seconds))
  while kill -0 -- "-${pgid}" 2>/dev/null; do
    (( SECONDS < deadline )) || return 1
    sleep 1
  done
  return 0
}

terminate_active_group() {
  [[ -n "${active_pgid}" ]] || return 0
  if kill -0 -- "-${active_pgid}" 2>/dev/null; then
    kill -INT -- "-${active_pgid}" 2>/dev/null || true
    if ! wait_for_group_exit "${active_pgid}" 360; then
      kill -TERM -- "-${active_pgid}" 2>/dev/null || true
      if ! wait_for_group_exit "${active_pgid}" 30; then
        kill -KILL -- "-${active_pgid}" 2>/dev/null || true
        wait_for_group_exit "${active_pgid}" 10 || true
      fi
    fi
  fi
  if [[ -n "${active_pid}" ]]; then
    wait "${active_pid}" 2>/dev/null || true
  fi
  active_pid=""
  active_pgid=""
}

terminate_pid_bounded() {
  local pid="$1"
  local seconds="$2"
  [[ -n "${pid}" ]] || return 0
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM "${pid}" 2>/dev/null || true
    local deadline=$((SECONDS + seconds))
    while kill -0 "${pid}" 2>/dev/null; do
      if (( SECONDS >= deadline )); then
        kill -KILL "${pid}" 2>/dev/null || true
        break
      fi
      sleep 1
    done
  fi
  wait "${pid}" 2>/dev/null || true
}

stop_judge_server() {
  if [[ -n "${judge_pid}" ]]; then
    terminate_pid_bounded "${judge_pid}" 30
    judge_pid=""
  fi
}

stop_host_proxy_bridge() {
  if [[ -n "${host_proxy_bridge_pid}" ]]; then
    terminate_pid_bounded "${host_proxy_bridge_pid}" 30
    host_proxy_bridge_pid=""
  fi
}

cleanup_proxy_and_network() {
  local rc=0
  if [[ -n "${proxy_name}" ]] && docker container inspect "${proxy_name}" >/dev/null 2>&1; then
    local proxy_owner
    proxy_owner="$(docker inspect --format '{{index .Config.Labels "mlsysforge.run_id"}}' "${proxy_name}" 2>/dev/null || true)"
    if [[ "${proxy_owner}" == "${RUN_ID}" ]]; then
      docker rm --force "${proxy_name}" >/dev/null 2>&1 || rc=1
    else
      rc=1
    fi
  fi
  if [[ -n "${isolated_network}" ]] && docker network inspect "${isolated_network}" >/dev/null 2>&1; then
    local network_owner
    network_owner="$(docker network inspect --format '{{index .Labels "mlsysforge.run_id"}}' "${isolated_network}" 2>/dev/null || true)"
    if [[ "${network_owner}" == "${RUN_ID}" ]]; then
      docker network rm "${isolated_network}" >/dev/null 2>&1 || rc=1
    else
      rc=1
    fi
  fi
  return "${rc}"
}

cleanup_owned_runtime() {
  local output="$1"
  [[ -n "${DEDICATED_NONCE_SOURCE}" && -e "${DEDICATED_NONCE_SOURCE}" ]] \
    || return 0
  python3 "${CLEANUP_HELPER}" cleanup \
    --nonce-source "${DEDICATED_NONCE_SOURCE}" \
    --migration-root "${MIGRATION_ROOT}" \
    --eval-root "${EVAL_ROOT}" \
    --judge-image "${JUDGE_IMAGE}" \
    --runner-grace-seconds 360 \
    --stack-timeout-seconds 420 \
    --stop-timeout-seconds 30 \
    --output "${output}"
}

remove_dedicated_nonce() {
  if [[ -n "${DEDICATED_NONCE_SOURCE}" ]] \
      && [[ -e "${DEDICATED_NONCE_SOURCE}" || -L "${DEDICATED_NONCE_SOURCE}" ]]; then
    python3 "${NONCE_HELPER}" delete \
      --path "${DEDICATED_NONCE_SOURCE}" \
      --run-root "${NONCE_RUNTIME_DIR}" || return 1
  fi
  [[ -n "${NONCE_RUNTIME_DIR}" ]] || return 0
  case "${NONCE_RUNTIME_DIR}" in
    "${NONCE_RUNTIME_PARENT}"/mlsysforge-v24-nonce.*) ;;
    *) return 1 ;;
  esac
  [[ ! -L "${NONCE_RUNTIME_DIR}" && -d "${NONCE_RUNTIME_DIR}" ]] || return 1
  [[ "$(stat -c '%u:%a' -- "${NONCE_RUNTIME_DIR}")" == "$(id -u):700" ]] \
    || return 1
  rmdir -- "${NONCE_RUNTIME_DIR}"
}

on_exit() {
  local original_rc="$?"
  (( exit_started == 0 )) || return
  exit_started=1
  trap - EXIT INT TERM
  set +e
  local requested="failed"
  # Two legitimate terminal outcomes: a selected valid submission (`passed`)
  # and a run whose every hidden submission was authoritatively invalid
  # (`completed_no_valid_submission`). Both require rc 0 and exact cleanup.
  case "${completion_requested}" in
    passed|completed_no_valid_submission)
      [[ "${original_rc}" == "0" ]] && requested="${completion_requested}"
      ;;
  esac
  stage="cleanup"
  write_completion failed "${original_rc}" in_progress >/dev/null 2>&1 || true
  terminate_active_group
  stop_judge_server
  stop_host_proxy_bridge
  local cleanup_rc=0
  cleanup_owned_runtime "${RUN_LOG_ROOT}/cleanup-runtime.json" || cleanup_rc=1
  cleanup_proxy_and_network || cleanup_rc=1
  remove_dedicated_nonce || cleanup_rc=1
  if (( measurement_lock_held == 1 )); then
    flock -u 7 >/dev/null 2>&1 || cleanup_rc=1
    measurement_lock_held=0
  fi
  exec 7>&- 2>/dev/null || true
  # Keep the formal-run lease through the final atomic completion write. The
  # descriptor is closed by process exit, preventing another formal run from
  # entering this run's terminal cleanup/commit window.
  if (( cleanup_rc == 0 )); then
    cleanup_status="passed"
  else
    cleanup_status="failed"
    requested="failed"
    failure_reason="${failure_reason}; exact runtime cleanup failed"
  fi
  case "${requested}" in
    passed)
      stage="complete"
      failure_reason=""
      ;;
    completed_no_valid_submission)
      stage="complete"
      failure_reason="${no_valid_submission_reason}"
      ;;
    *)
      stage="failed"
      ;;
  esac
  write_completion "${requested}" "${original_rc}" "${cleanup_status}"
  local completion_rc="$?"
  if [[ "${requested}" != "failed" && "${completion_rc}" == "0" ]]; then
    exit 0
  fi
  (( original_rc != 0 )) && exit "${original_rc}"
  exit 1
}

handle_signal() {
  local signal_name="$1"
  stage="signal_${signal_name}"
  failure_reason="received ${signal_name}; bounded cleanup requested"
  if [[ "${signal_name}" == "INT" ]]; then
    exit 130
  fi
  exit 143
}

start_logged() {
  local log_file="$1"
  shift
  setsid --wait "$@" > >(tee "${log_file}") 2>&1 &
  active_pid="$!"
  active_pgid="${active_pid}"
}

wait_logged() {
  local rc=0
  [[ -n "${active_pid}" ]] || return 125
  wait "${active_pid}" || rc="$?"
  active_pid=""
  active_pgid=""
  return "${rc}"
}

run_logged() {
  local log_file="$1"
  shift
  start_logged "${log_file}" "$@"
  wait_logged
}

trap on_exit EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM
write_completion failed 0 not_started >/dev/null

stage="static_validation"
for command_name in curl docker flock install jq python3 setsid sha256sum ss stat; do
  require_command "${command_name}"
done
require_sforge
# Model, effort, and budget are validated independently. Binding them into a
# combined whitelist forced a task rebuild for every new duration and blocked
# short validation runs entirely; the run manifest already attests the exact
# triple that was used.
# Codex runs any model ID; claude-opus-5-5 is the Claude Code model, only with
# the claude-code agent.
[[ "${MODEL}" =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]] \
  || die "unsupported formal model: ${MODEL}"
case "${MODEL}" in
  claude-opus-5-5)
    [[ "${AGENT}" == claude-code ]] || die "${MODEL} runs only with the claude-code agent" ;;
  claude-*) die "unsupported formal model: ${MODEL}" ;;
  *) [[ "${AGENT}" == codex ]] || die "${MODEL} runs only with the codex agent" ;;
esac
if [[ "${AGENT}" == claude-code ]]; then
  case "${REASONING_EFFORT}" in
    low|medium|high|xhigh|max) ;;
    *) die "unsupported Claude Code effort: ${REASONING_EFFORT} (low|medium|high|xhigh|max)" ;;
  esac
else
  case "${REASONING_EFFORT}" in
    high|max|xhigh|ultra) ;;
    *) die "unsupported reasoning effort: ${REASONING_EFFORT}" ;;
  esac
fi
[[ "${TASK_TIMEOUT}" =~ ^[0-9]+$ ]] \
  || die "TASK_TIMEOUT must be a positive integer of seconds"
TASK_TIMEOUT="$((10#${TASK_TIMEOUT}))"
(( TASK_TIMEOUT >= 3600 && TASK_TIMEOUT % 60 == 0 )) \
  || die "TASK_TIMEOUT must be minute-aligned and at least 3600 seconds"
case "${CODEX_WEB_SEARCH_MODE}" in
  disabled|live) ;;
  *) die "CODEX_WEB_SEARCH_MODE must be exactly disabled or live" ;;
esac
# The Codex version is chosen by the launcher (launch/common.sh pin or an
# explicit CODEX_CLI_VERSION) and must match the mounted runtime below; the
# agent-free Work image does not constrain it.
[[ "${SUBMISSION_COOLDOWN}" == "300" ]] || die "formal run requires 300-second cooldown"
[[ "${MLSYSFORGE_ROOT}" == "${EXPECTED_MLSYSFORGE_ROOT}" ]] \
  || die "formal run requires the fixed MLSys Forge checkout"
[[ "${SFORGE_BIN}" == "${EXPECTED_MLSYSFORGE_ROOT}/.venv/bin/sforge" ]] \
  || die "formal run requires the fixed SForge entrypoint"
[[ "${PYTHON_BIN}" == "${EXPECTED_MLSYSFORGE_ROOT}/.venv/bin/python" \
    && "${SFORGE_PYTHON}" == "${PYTHON_BIN}" ]] \
  || die "formal run requires the fixed MLSys Forge Python interpreter"
[[ "${MAX_FINAL_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || die "invalid final attempt bound"
(( MAX_FINAL_ATTEMPTS <= 5 )) || die "final attempt bound exceeds fail-closed maximum"
[[ "${JUDGE_PORT}" =~ ^[0-9]+$ ]] || die "invalid Judge port"
(( 10#${JUDGE_PORT} >= 1024 && 10#${JUDGE_PORT} <= 65535 )) || die "invalid Judge port"
[[ -x "${SFORGE_PYTHON}" ]] || die "missing SForge Python"
for required_file in \
  "${RUNTIME_ENTRY}" "${PROXY_SOURCE}" "${NONCE_HELPER}" "${PREFLIGHT_HELPER}" \
  "${HOST_PROXY_BRIDGE_SOURCE}" \
  "${CLEANUP_HELPER}" "${FINAL_VALIDATOR}" "${COMPLETION_WRITER}" \
  "${TASK_SPEC}"; do
  [[ -f "${required_file}" && ! -L "${required_file}" ]] \
    || die "missing or unsafe formal runner dependency: ${required_file}"
done
[[ -f "${ORIGINAL_NONCE_SOURCE}" && ! -L "${ORIGINAL_NONCE_SOURCE}" ]] \
  || die "missing original nonce source"
# The task JSON's "internet" field (default true, as in SForge) selects the
# sidecar policy: true = the model-host allowlist plus any public host on
# 80/443 (every resolved address must be public); false = allowlist only.
TASK_INTERNET="$(
  jq -r 'if has("internet") then .internet else true end
    | if type == "boolean" then . else error("internet must be a boolean") end' \
    "${TASK_SPEC}"
)" || die "task JSON internet field must be a boolean"
proxy_egress_args=()
if [[ "${TASK_INTERNET}" == "true" ]]; then
  proxy_egress_args=(--allow-public-egress)
  WORK_NETWORK_POLICY="internal_with_public_egress_connect_proxy"
else
  WORK_NETWORK_POLICY="internal_with_restricted_connect_proxy"
fi
# The model API each agent talks to; any HTTP status proves the route.
MODEL_API_PROBE_URL="https://api.openai.com/v1/models"
if [[ "${AGENT}" == claude-code ]]; then
  MODEL_API_PROBE_URL="https://api.anthropic.com/v1/models"
  # The sidecar's allowlist-only mode tunnels just the OpenAI hosts; Claude
  # Code reaches api.anthropic.com through public egress.
  [[ "${TASK_INTERNET}" == "true" ]] \
    || die "claude-code needs the task JSON internet: true (the sidecar allowlist names only OpenAI hosts)"
fi
host_proxy_code="$(
  curl --noproxy '' --proxy "http://${HOST_PROXY_HOST}:${HOST_PROXY_PORT}" \
    --connect-timeout 5 --max-time 20 -sS -o /dev/null -w '%{http_code}' \
    "${MODEL_API_PROBE_URL}"
)" || die "fixed host proxy cannot reach the model API"
[[ "${host_proxy_code}" =~ ^[234][0-9][0-9]$ ]] \
  || die "fixed host proxy model API preflight returned ${host_proxy_code:-000}"
if [[ "${AGENT}" == claude-code ]]; then
  [[ -f "${CLAUDE_OAUTH_TOKEN_FILE}" && ! -L "${CLAUDE_OAUTH_TOKEN_FILE}" && -s "${CLAUDE_OAUTH_TOKEN_FILE}" ]] \
    || die "Claude OAuth token file must be a nonempty regular, non-symlink file"
  [[ "$(stat -c '%a' "${CLAUDE_OAUTH_TOKEN_FILE}")" == "600" ]] \
    || die "Claude OAuth token file must have mode 600"
  [[ "$(stat -c '%u' "${CLAUDE_OAUTH_TOKEN_FILE}")" == "$(id -u)" ]] \
    || die "Claude OAuth token file must be owned by the launching user"
else
  assert_private_auth_file "${CODEX_AUTH_JSON_PATH}"
fi

# The Work image is agent-free; a host-provided runtime supplies the CLI.
AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}"
[[ -n "${AGENT_RUNTIME_DIR}" ]] \
  || die "set SFORGE_AGENT_RUNTIME_DIR (build one with launch/build-agent-runtime.sh)"
runtime_marker="${AGENT_RUNTIME_DIR}/.sforge-agent-runtime.json"
[[ -d "${AGENT_RUNTIME_DIR}" && ! -L "${AGENT_RUNTIME_DIR}" && -f "${runtime_marker}" ]] \
  || die "agent runtime directory is missing or unmarked: ${AGENT_RUNTIME_DIR}"
runtime_kind="$(jq -er '.kind' "${runtime_marker}")"
runtime_agent="$(jq -er '.agent' "${runtime_marker}")"
runtime_version="$(jq -er '.version' "${runtime_marker}")"
runtime_tree_sha256="$(jq -er '.tree_sha256' "${runtime_marker}")"
[[ "${runtime_kind}" == "sforge-agent-runtime" ]] \
  || die "invalid agent runtime marker: ${runtime_marker}"
if [[ "${AGENT}" == claude-code ]]; then
  [[ "${runtime_agent}" == "claude-code" ]] \
    || die "a claude-code run requires a claude-code runtime, found: ${runtime_agent}"
  [[ "${runtime_version}" == "${CLAUDE_CODE_VERSION}" ]] \
    || die "agent runtime version ${runtime_version} differs from the pinned Claude Code ${CLAUDE_CODE_VERSION}"
else
  [[ "${runtime_agent}" == "codex" ]] \
    || die "formal run requires a codex runtime, found: ${runtime_agent}"
  [[ "${runtime_version}" == "${CODEX_CLI_VERSION}" ]] \
    || die "agent runtime version ${runtime_version} differs from the pinned ${CODEX_CLI_VERSION}"
fi
export SFORGE_AGENT_RUNTIME_DIR="${AGENT_RUNTIME_DIR}"

assert_built_images_current
if ss -ltn | awk '{print $4}' | grep -Eq ":${JUDGE_PORT}$"; then
  die "Judge port is already in use: ${JUDGE_PORT}"
fi

install -d -m 0700 "${LOCK_ROOT}"
[[ ! -L "${LOCK_ROOT}" && "$(stat -c '%u:%a' "${LOCK_ROOT}")" == "$(id -u):700" ]] \
  || die "formal lock directory must be current-uid owner mode 0700"
# Create the measurement lock as the launching user before any Judge runs:
# the nested runner opens it O_CREAT inside the Judge container as uid 0, and
# a root-owned 0600 leftover makes the host-side manual-drain step fail with
# EACCES after the whole agent budget has been spent.
[[ -e "${MEASUREMENT_LOCK}" ]] || install -m 0600 /dev/null "${MEASUREMENT_LOCK}"
[[ ! -L "${MEASUREMENT_LOCK}" && "$(stat -c '%u' "${MEASUREMENT_LOCK}")" == "$(id -u)" ]] \
  || die "measurement lock must be a current-uid regular file: ${MEASUREMENT_LOCK}"
exec 8>"${FORMAL_RUN_LOCK}"
flock -n 8 || die "another v2.4 formal launcher owns the formal-run lease"
formal_lock_held=1

stage="preflight"
python3 "${PREFLIGHT_HELPER}" \
  --nonce-source "${ORIGINAL_NONCE_SOURCE}" \
  --vllm-base-url "${VLLM_BASE_URL}" \
  --hidden-baseline "${HIDDEN_BASELINE_PATH}" \
  --public-baseline "${PUBLIC_BASELINE_PATH}" \
  --mlsysforge-root "${MLSYSFORGE_ROOT}" \
  --output "${PREFLIGHT_PATH}" \
  >"${RUN_LOG_ROOT}/preflight.log"

# Freeze both operator-selected baseline files immediately after the trusted
# preflight. Every later Judge/validator process consumes these run-scoped,
# read-only bytes, closing the path-selection TOCTOU window.
PREFLIGHT_SHA256="$(sha256sum -- "${PREFLIGHT_PATH}" | awk '{print $1}')"
HIDDEN_BASELINE_IDENTITY="$({
  jq -ce '{
    file_sha256: .baseline_file_sha256,
    valid_steps_per_min: .baseline_valid_steps_per_min,
    backend_serving_contract_sha256: .backend_serving_contract_sha256,
    workload_nonce_sha256: .nonce_commitment_sha256
  }
  | select(
      (.file_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
      and (.valid_steps_per_min | type == "number" and . > 0)
      and (.backend_serving_contract_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
      and (.workload_nonce_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
    )' "${PREFLIGHT_PATH}"
})" || die "preflight hidden baseline identity is invalid"
expected_hidden_sha256="$(jq -er '.file_sha256' <<<"${HIDDEN_BASELINE_IDENTITY}")"
expected_public_sha256="$(jq -er '.public_baseline_file_sha256' "${PREFLIGHT_PATH}")"
install -d -m 0700 -- "${BASELINE_SNAPSHOT_ROOT}"
hidden_baseline_snapshot="${BASELINE_SNAPSHOT_ROOT}/hidden.json"
public_baseline_snapshot="${BASELINE_SNAPSHOT_ROOT}/public.json"
install -m 0400 -- "${HIDDEN_BASELINE_PATH}" "${hidden_baseline_snapshot}"
install -m 0400 -- "${PUBLIC_BASELINE_PATH}" "${public_baseline_snapshot}"
for snapshot in "${hidden_baseline_snapshot}" "${public_baseline_snapshot}"; do
  [[ -f "${snapshot}" && ! -L "${snapshot}" ]] \
    || die "run-scoped baseline snapshot is missing or unsafe"
  [[ "$(stat -c '%u:%a' -- "${snapshot}")" == "$(id -u):400" ]] \
    || die "run-scoped baseline snapshot must be current-uid owner mode 0400"
done
[[ "$(sha256sum -- "${hidden_baseline_snapshot}" | awk '{print $1}')" == "${expected_hidden_sha256}" ]] \
  || die "hidden baseline changed between preflight and snapshot"
[[ "$(sha256sum -- "${public_baseline_snapshot}" | awk '{print $1}')" == "${expected_public_sha256}" ]] \
  || die "public baseline changed between preflight and snapshot"
HIDDEN_BASELINE_PATH="${hidden_baseline_snapshot}"
PUBLIC_BASELINE_PATH="${public_baseline_snapshot}"
export KVBENCH_HIDDEN_BASELINE_PATH="${HIDDEN_BASELINE_PATH}"
export KVBENCH_PUBLIC_BASELINE_PATH="${PUBLIC_BASELINE_PATH}"
[[ -d "${NONCE_RUNTIME_PARENT}" && ! -L "${NONCE_RUNTIME_PARENT}" ]] \
  || die "dedicated nonce runtime parent is missing or unsafe"
[[ "$(stat -c '%u:%a' -- "${NONCE_RUNTIME_PARENT}")" == "$(id -u):700" ]] \
  || die "dedicated nonce runtime parent must be current-uid owner mode 0700"
NONCE_RUNTIME_DIR="$(mktemp -d -- "${NONCE_RUNTIME_PARENT}/mlsysforge-v24-nonce.XXXXXX")"
chmod 0700 -- "${NONCE_RUNTIME_DIR}"
[[ ! -L "${NONCE_RUNTIME_DIR}" \
    && "$(stat -c '%u:%a' -- "${NONCE_RUNTIME_DIR}")" == "$(id -u):700" ]] \
  || die "dedicated nonce runtime directory ownership or mode is unsafe"
DEDICATED_NONCE_SOURCE="${NONCE_RUNTIME_DIR}/nonce-source"
python3 "${NONCE_HELPER}" create \
  --source "${ORIGINAL_NONCE_SOURCE}" \
  --output "${DEDICATED_NONCE_SOURCE}" \
  --run-root "${NONCE_RUNTIME_DIR}"
python3 "${NONCE_HELPER}" validate \
  --path "${DEDICATED_NONCE_SOURCE}" \
  --run-root "${NONCE_RUNTIME_DIR}"
export SFORGE_KVBENCH_NONCE_SOURCE="${DEDICATED_NONCE_SOURCE}"

stage="judge_server"
export SFORGE_TASKS_DIR="${TASKS_DIR}"
export SFORGE_LOG_DIR="${LOGS_DIR}"
export SFORGE_KVBENCH_MIGRATION_ROOT="${MIGRATION_ROOT}"
export SFORGE_KVBENCH_EVAL_ROOT="${EVAL_ROOT}"
export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=void"
export SFORGE_ADMIN_SECRET
SFORGE_ADMIN_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"

"${SFORGE_PYTHON}" "${RUNTIME_ENTRY}" serve --port "${JUDGE_PORT}" \
  >"${RUN_LOG_ROOT}/judge-server.log" 2>&1 &
judge_pid="$!"
wait_for_http "http://127.0.0.1:${JUDGE_PORT}/openapi.json" 90

isolated_network="${RUN_ID}-internal"
proxy_name="${RUN_ID}-restricted-proxy"
docker_bridge_gateway="$(
  docker network inspect bridge \
    | jq -er '.[0].IPAM.Config[0].Gateway
      | select(type == "string" and length > 0)'
)"
if ! bridge_port_listeners="$(
  ss -H -ltn "sport = :${DOCKER_PROXY_PORT}" 2>/dev/null
)"; then
  die "failed to inspect Docker proxy bridge port"
fi
[[ -z "${bridge_port_listeners}" ]] \
  || die "Docker proxy bridge port ${DOCKER_PROXY_PORT} is already in use"
docker network create \
  --driver bridge \
  --internal \
  --label "mlsysforge.run_id=${RUN_ID}" \
  "${isolated_network}" >/dev/null
docker run --detach --rm \
  --name "${proxy_name}" \
  --label "mlsysforge.run_id=${RUN_ID}" \
  --label "mlsysforge.role=restricted-proxy" \
  --network bridge \
  --add-host host.docker.internal:host-gateway \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --volume "${PROXY_SOURCE}:/opt/mlsysforge/restricted_proxy.py:ro" \
  --entrypoint python3 \
  "${WORK_IMAGE}" \
  /opt/mlsysforge/restricted_proxy.py \
  --judge-port "${JUDGE_PORT}" \
  --judge-forward-port "${JUDGE_PORT}" \
  --upstream-proxy-host "${docker_bridge_gateway}" \
  --upstream-proxy-port "${DOCKER_PROXY_PORT}" \
  ${proxy_egress_args[@]+"${proxy_egress_args[@]}"} >/dev/null
proxy_bridge_ip="$(
  docker inspect "${proxy_name}" \
    | jq -er '.[0].NetworkSettings.Networks.bridge.IPAddress
      | select(type == "string" and length > 0)'
)"
python3 "${HOST_PROXY_BRIDGE_SOURCE}" \
  --listen-host "${docker_bridge_gateway}" \
  --listen-port "${DOCKER_PROXY_PORT}" \
  --allowed-peer "${proxy_bridge_ip}" \
  >"${RUN_LOG_ROOT}/host-proxy-bridge.log" 2>&1 &
host_proxy_bridge_pid="$!"
for _ in $(seq 1 30); do
  if bridge_code="$(
    docker exec "${proxy_name}" curl --noproxy '' \
      --proxy "http://${docker_bridge_gateway}:${DOCKER_PROXY_PORT}" \
      --connect-timeout 2 --max-time 8 -sS -o /dev/null -w '%{http_code}' \
      "${MODEL_API_PROBE_URL}" 2>/dev/null
  )" && [[ "${bridge_code}" =~ ^[234][0-9][0-9]$ ]]; then
    break
  fi
  bridge_code=""
  sleep 1
done
[[ "${bridge_code}" =~ ^[234][0-9][0-9]$ ]] \
  || die "source-restricted Docker proxy bridge cannot reach the model API"
kill -0 "${host_proxy_bridge_pid}" 2>/dev/null \
  || die "host proxy bridge exited after health check"
docker network connect --alias runtime-proxy "${isolated_network}" "${proxy_name}"
proxy_ip="$(
  docker inspect "${proxy_name}" \
    | jq -er --arg network "${isolated_network}" \
      '.[0].NetworkSettings.Networks[$network].IPAddress
       | select(type == "string" and length > 0)'
)"

for _ in $(seq 1 30); do
  if docker run --rm --network "${isolated_network}" \
      --env NVIDIA_VISIBLE_DEVICES=void --entrypoint curl "${WORK_IMAGE}" \
      --connect-timeout 2 --max-time 3 -fsS \
      "http://${proxy_ip}:3128/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker run --rm --network "${isolated_network}" \
  --env NVIDIA_VISIBLE_DEVICES=void --entrypoint curl "${WORK_IMAGE}" \
  --connect-timeout 2 --max-time 5 -fsS \
  "http://${proxy_ip}:${JUDGE_PORT}/openapi.json" >/dev/null
if docker run --rm --network "${isolated_network}" \
    --env NVIDIA_VISIBLE_DEVICES=void --entrypoint curl "${WORK_IMAGE}" \
    --connect-timeout 2 --max-time 4 -fsS https://example.com >/dev/null 2>&1; then
  die "Work network unexpectedly has direct public egress"
fi
# CONNECT policy: AUTHORITY EXPECTED-STATUS pairs. Public egress mode must
# tunnel a public host and still refuse non-public destinations (the sidecar's
# own loopback, the Docker bridge gateway and its host proxy bridge port, the
# host alias, link-local metadata, RFC 1918) and non-web ports; allowlist mode
# refuses any host outside the model allowlist.
if [[ "${TASK_INTERNET}" == "true" ]]; then
  connect_policy_probes=(
    "github.com:443" 200
    "github.com:22" 403
    "127.0.0.1:443" 403
    "${docker_bridge_gateway}:443" 403
    "${docker_bridge_gateway}:${DOCKER_PROXY_PORT}" 403
    "host.docker.internal:443" 403
    "169.254.169.254:80" 403
    "10.0.0.1:443" 403
  )
else
  connect_policy_probes=("github.com:443" 403)
fi
docker run --rm --network "${isolated_network}" \
  --env NVIDIA_VISIBLE_DEVICES=void --entrypoint python3 "${WORK_IMAGE}" -c \
  'import socket,sys
host=sys.argv[1]
probes=sys.argv[2:]
for authority,expected in zip(probes[0::2],probes[1::2]):
    with socket.create_connection((host,3128),20) as stream:
        stream.sendall(f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n".encode())
        status=stream.recv(128).split(b"\r\n",1)[0].split()
    if len(status)<2 or status[1]!=expected.encode():
        raise SystemExit(f"proxy CONNECT {authority} returned {status!r}, expected {expected}")' \
  "${proxy_ip}" "${connect_policy_probes[@]}" \
  || die "restricted Work proxy CONNECT policy preflight failed (${WORK_NETWORK_POLICY})"
sidecar_api_code="$(
  docker run --rm --network "${isolated_network}" \
    --env NVIDIA_VISIBLE_DEVICES=void --entrypoint curl "${WORK_IMAGE}" \
    --noproxy '' --proxy "http://${proxy_ip}:3128" \
    --connect-timeout 5 --max-time 20 -sS -o /dev/null -w '%{http_code}' \
    "${MODEL_API_PROBE_URL}"
)" || die "restricted Work proxy cannot reach the model API"
[[ "${sidecar_api_code}" =~ ^[234][0-9][0-9]$ ]] \
  || die "restricted Work proxy model API preflight returned ${sidecar_api_code:-000}"
if [[ "${AGENT}" == codex ]]; then
  sidecar_chatgpt_code="$(
    docker run --rm --network "${isolated_network}" \
      --env NVIDIA_VISIBLE_DEVICES=void --entrypoint curl "${WORK_IMAGE}" \
      --noproxy '' --proxy "http://${proxy_ip}:3128" \
      --connect-timeout 5 --max-time 20 -sS -o /dev/null -w '%{http_code}' \
      https://chatgpt.com/backend-api/models
  )" || die "restricted Work proxy cannot reach the Codex API"
  [[ "${sidecar_chatgpt_code}" =~ ^[234][0-9][0-9]$ ]] \
    || die "restricted Work proxy Codex API preflight returned ${sidecar_chatgpt_code:-000}"
fi
if [[ "${TASK_INTERNET}" == "true" ]]; then
  sidecar_public_code="$(
    docker run --rm --network "${isolated_network}" \
      --env NVIDIA_VISIBLE_DEVICES=void --entrypoint curl "${WORK_IMAGE}" \
      --noproxy '' --proxy "http://${proxy_ip}:3128" \
      --connect-timeout 5 --max-time 20 -sS -o /dev/null -w '%{http_code}' \
      https://pypi.org/simple/pip/
  )" || die "public egress Work proxy cannot reach a public HTTPS host"
  [[ "${sidecar_public_code}" =~ ^[234][0-9][0-9]$ ]] \
    || die "public egress Work proxy preflight returned ${sidecar_public_code:-000}"
fi
unauthorized_bridge_code="$(
  docker run --rm --network bridge --entrypoint curl "${WORK_IMAGE}" \
    --noproxy '' --proxy "http://${docker_bridge_gateway}:${DOCKER_PROXY_PORT}" \
    --connect-timeout 2 --max-time 5 -sS -o /dev/null -w '%{http_code}' \
    https://api.openai.com/v1/models 2>/dev/null || true
)"
[[ "${unauthorized_bridge_code}" == "000" ]] \
  || die "host proxy bridge accepted an unrelated Docker bridge source"

unset \
  SFORGE_AGENT_API_KEY SFORGE_AGENT_API_BASE_URL SFORGE_AGENT_MODEL \
  CODEX_API_KEY OPENAI_API_KEY OPENAI_BASE_URL
if [[ "${AGENT}" == claude-code ]]; then
  # sforge/harness/agent/claude_local.py install_from_environment reads these; the
  # token is installed into Work as an Agent-owned 0600 file, never into an
  # environment, image or log.
  unset SFORGE_CODEX_AUTH_JSON_PATH SFORGE_CODEX_CLI_VERSION \
    ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL CLAUDE_CODE_OAUTH_TOKEN
  export MLSYSFORGE_CLAUDE_OAUTH_TOKEN_FILE="${CLAUDE_OAUTH_TOKEN_FILE}"
  export MLSYSFORGE_AGENT_MODEL="${MODEL}" MLSYSFORGE_AGENT_EFFORT="${REASONING_EFFORT}"
  export MLSYSFORGE_AGENT_BUDGET_SECONDS="${TASK_TIMEOUT}" MLSYSFORGE_RUN_LOG_ROOT="${RUN_LOG_ROOT}"
else
  export SFORGE_CODEX_AUTH_JSON_PATH="${CODEX_AUTH_JSON_PATH}"
  export SFORGE_CODEX_REASONING_EFFORT="${REASONING_EFFORT}"
  export SFORGE_CODEX_CLI_VERSION="${CODEX_CLI_VERSION}"
fi
export MLSYSFORGE_AGENT="${AGENT}"
export CODEX_WEB_SEARCH_MODE
export SFORGE_AGENT_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=void"
export SFORGE_ISOLATED_DOCKER_NETWORK="${isolated_network}"
export SFORGE_HTTP_PROXY="http://${proxy_ip}:3128"
export SFORGE_HTTPS_PROXY="http://${proxy_ip}:3128"
# Work shells keep these proxy variables (nothing unsets them in Work); the
# sidecar address, which also serves the Judge forward, stays direct.
export SFORGE_NO_PROXY="${proxy_ip},127.0.0.1,localhost"
# Read by scripts/sforge_runtime_entry.py: "false" forces the offline Codex
# flags and prompt, "true" keeps them online (web search per the knob above).
export MLSYSFORGE_WORK_INTERNET="${TASK_INTERNET}"

jq -n \
  --arg run_id "${RUN_ID}" \
  --arg started_at "${run_started_at}" \
  --arg agent "${AGENT}" \
  --arg model "${MODEL}" \
  --arg effort "${REASONING_EFFORT}" \
  --arg cli "${CODEX_CLI_VERSION}" \
  --arg runtime_dir "${AGENT_RUNTIME_DIR}" \
  --arg runtime_agent "${runtime_agent}" \
  --arg runtime_version "${runtime_version}" \
  --arg runtime_tree_sha256 "${runtime_tree_sha256}" \
  --arg preflight_sha256 "${PREFLIGHT_SHA256}" \
  --argjson hidden_baseline_identity "${HIDDEN_BASELINE_IDENTITY}" \
  --arg public_nonce_commitment "$(jq -er '.public_nonce_commitment_sha256' "${PREFLIGHT_PATH}")" \
  --arg public_baseline_sha256 "$(jq -er '.public_baseline_file_sha256' "${PREFLIGHT_PATH}")" \
  --argjson public_direct_rate "$(jq -er '.public_baseline_valid_steps_per_min' "${PREFLIGHT_PATH}")" \
  --arg runtime_sources "$(jq -er '.trusted_runtime_sources.tree_sha256' "${PREFLIGHT_PATH}")" \
  --arg execution_sources "$(jq -er '.trusted_execution_sources.tree_sha256' "${PREFLIGHT_PATH}")" \
  --arg source_algorithm "$(jq -er '.trusted_runtime_sources.algorithm' "${PREFLIGHT_PATH}")" \
  --argjson runtime_file_count "$(jq -er '.trusted_runtime_sources.file_count' "${PREFLIGHT_PATH}")" \
  --argjson runtime_total_bytes "$(jq -er '.trusted_runtime_sources.total_bytes' "${PREFLIGHT_PATH}")" \
  --argjson execution_file_count "$(jq -er '.trusted_execution_sources.file_count' "${PREFLIGHT_PATH}")" \
  --argjson execution_total_bytes "$(jq -er '.trusted_execution_sources.total_bytes' "${PREFLIGHT_PATH}")" \
  --arg mlsysforge_root "$(jq -er '.mlsysforge_root' "${PREFLIGHT_PATH}")" \
  --arg mlsysforge_sources "$(jq -er '.mlsysforge_harness_sources.tree_sha256' "${PREFLIGHT_PATH}")" \
  --arg mlsysforge_algorithm "$(jq -er '.mlsysforge_harness_sources.algorithm' "${PREFLIGHT_PATH}")" \
  --argjson mlsysforge_file_count "$(jq -er '.mlsysforge_harness_sources.file_count' "${PREFLIGHT_PATH}")" \
  --argjson mlsysforge_total_bytes "$(jq -er '.mlsysforge_harness_sources.total_bytes' "${PREFLIGHT_PATH}")" \
  --argjson mlsysforge_exclusions "$(jq -ce '.mlsysforge_generated_exclusions' "${PREFLIGHT_PATH}")" \
  --argjson mlsysforge_import_origins "$(jq -ce '.mlsysforge_import_origins' "${PREFLIGHT_PATH}")" \
  --argjson timeout "${TASK_TIMEOUT}" \
  --argjson cooldown "${SUBMISSION_COOLDOWN}" \
  --arg web_search "${CODEX_WEB_SEARCH_MODE}" \
  --arg network_policy "${WORK_NETWORK_POLICY}" \
  --argjson internet "${TASK_INTERNET}" \
  '{
    schema_version: 2,
    run_id: $run_id,
    started_at: $started_at,
    agent: $agent,
    model: $model,
    reasoning_effort: $effort,
    codex_cli_version: $cli,
    agent_runtime: {
      dir: $runtime_dir,
      agent: $runtime_agent,
      version: $runtime_version,
      tree_sha256: $runtime_tree_sha256
    },
    timeout_seconds: $timeout,
    preflight_sha256: $preflight_sha256,
    hidden_baseline_identity: $hidden_baseline_identity,
    auto_evaluation: false,
    manual_hidden_submission_limit: null,
    manual_submission_cooldown_seconds: $cooldown,
    public_diagnostic: {
      profile: "public-swe192-distinct",
      workflows: 192,
      concurrency: 192,
      max_steps: 100,
      measurement_warmup_seconds: 300,
      measurement_window_seconds: 600,
      frozen_direct_valid_steps_per_min: $public_direct_rate,
      frozen_direct_file_sha256: $public_baseline_sha256,
      workload_nonce_commitment_sha256: $public_nonce_commitment,
      candidate_only: true,
      hidden_cooldown_consumed: false,
      participates_in_hidden_selection: false
    },
    trusted_runtime_sources: {
      algorithm: $source_algorithm,
      tree_sha256: $runtime_sources,
      file_count: $runtime_file_count,
      total_bytes: $runtime_total_bytes
    },
    trusted_execution_sources: {
      algorithm: $source_algorithm,
      tree_sha256: $execution_sources,
      file_count: $execution_file_count,
      total_bytes: $execution_total_bytes
    },
    mlsysforge_root: $mlsysforge_root,
    mlsysforge_generated_exclusions: $mlsysforge_exclusions,
    mlsysforge_import_origins: $mlsysforge_import_origins,
    mlsysforge_harness_sources: {
      algorithm: $mlsysforge_algorithm,
      tree_sha256: $mlsysforge_sources,
      file_count: $mlsysforge_file_count,
      total_bytes: $mlsysforge_total_bytes
    },
    network: $network_policy,
    internet: $internet,
    codex_web_search: $web_search
  }' >"${EXPERIMENT_MANIFEST_PATH}"

# The host harness talks directly to the task-local Judge proxy.  Standard
# proxy variables belong to the operator shell and must not redirect SForge's
# registration requests.  Agent containers still receive the explicitly
# attested SFORGE_* restricted proxy values above.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy \
  NO_PROXY no_proxy

stage="agent_run"
echo "Starting ${RUN_ID}: ${AGENT} ${MODEL}/${REASONING_EFFORT}, exact ${TASK_TIMEOUT}s budget"
echo "Codex web search mode: ${CODEX_WEB_SEARCH_MODE}"
echo "Work network: ${WORK_NETWORK_POLICY}"
echo "Manual hidden Judge submissions are unlimited with ${SUBMISSION_COOLDOWN}s cooldown"
set +e
run_logged "${RUN_LOG_ROOT}/sforge-run.log" \
  "${SFORGE_PYTHON}" "${RUNTIME_ENTRY}" run \
  --backend docker \
  --task "${TASK_ID}" \
  --agent "${AGENT}" \
  --model "${MODEL}" \
  --timeout "${TASK_TIMEOUT}" \
  --disable-auto-eval \
  --submission-cooldown "${SUBMISSION_COOLDOWN}" \
  --judge-url "http://${proxy_ip}:${JUDGE_PORT}" \
  --enable-internet \
  --work-cpu-limit 16 \
  --work-mem-limit 64g \
  --judge-cpu-limit 4 \
  --judge-mem-limit 8g \
  --run-id "${RUN_ID}"
run_rc="$?"
set -e
(( run_rc == 0 )) || die "SForge agent run failed with exit ${run_rc}"

for required in "${FINAL_ARCHIVE}" "${FINAL_SNAPSHOT}" "${RUN_HISTORY}"; do
  [[ -s "${required}" && ! -L "${required}" ]] \
    || die "missing final run artifact: ${required}"
done
python3 "${MIGRATION_ROOT}/scripts/check_final_snapshot.py" \
  "${FINAL_SNAPSHOT}" "${FINAL_ARCHIVE}" "${TASK_TIMEOUT}" \
  >"${TASK_RUN_ROOT}/final-snapshot-check.log"

# Stop ingress before waiting for the measurement lock. This removes daemon
# submission queues; an already-running outer Judge remains independent and is
# allowed to finish or is reconciled precisely after the lock is acquired.
stage="manual_drain"
stop_judge_server
exec 7>"${MEASUREMENT_LOCK}"
flock -w 3600 7 || die "timed out waiting for the last manual hidden evaluation"
measurement_lock_held=1
cleanup_owned_runtime "${TASK_RUN_ROOT}/handoff-cleanup.json" \
  || die "manual Judge runtime did not reconcile cleanly"
python3 "${CLEANUP_HELPER}" verify-quiescent \
  --nonce-source "${DEDICATED_NONCE_SOURCE}" \
  --migration-root "${MIGRATION_ROOT}" \
  --eval-root "${EVAL_ROOT}" \
  --judge-image "${JUDGE_IMAGE}" \
  --output "${TASK_RUN_ROOT}/handoff-quiescent.json"

stage="final_evaluation"
for attempt in $(seq 1 "${MAX_FINAL_ATTEMPTS}"); do
  final_attempt="${attempt}"
  final_eval_id="${RUN_ID}-final-120m-a${attempt}"
  final_task_root="${LOGS_DIR}/runs/${final_eval_id}/${TASK_ID}"
  final_report="${final_task_root}/submissions/manual-1/report.json"
  final_private_artifact="${final_task_root}/submissions/manual-1/private_judge_artifacts.tar.gz"
  final_evaluated_archive="${final_task_root}/submissions/manual-1/submission.tar.gz"
  final_validation="${TASK_RUN_ROOT}/final-evaluation-attempt-${attempt}.validation.json"
  start_logged "${TASK_RUN_ROOT}/final-120m-eval-attempt-${attempt}.log" \
    "${SFORGE_PYTHON}" "${RUNTIME_ENTRY}" eval \
    --backend docker \
    --task "${TASK_ID}" \
    --archive "${FINAL_ARCHIVE}" \
    --run-id "${final_eval_id}" \
    --timeout 3000 \
    --judge-cpu-limit 4 \
    --judge-mem-limit 8g \
    --json
  # The evaluator process now exists while the launcher still owns the flock.
  # Release immediately; a self-race is fail-closed and retryable only when the
  # private diagnostic is the frozen runner's unique LOCK_NB exit-4 signature.
  flock -u 7
  measurement_lock_held=0
  set +e
  wait_logged
  final_eval_rc="$?"
  set -e

  set +e
  python3 "${FINAL_VALIDATOR}" \
    --report "${final_report}" \
    --private-artifact "${final_private_artifact}" \
    --evaluated-archive "${final_evaluated_archive}" \
    --expected-archive "${FINAL_ARCHIVE}" \
    --expected-preflight "${PREFLIGHT_PATH}" \
    --expected-submission-id "${final_eval_id}" \
    --eval-rc "${final_eval_rc}" \
    --max-runtime-seconds 3060 \
    --output "${final_validation}"
  validation_rc="$?"
  set -e
  if (( validation_rc == 0 )); then
    break
  fi
  if (( validation_rc != 75 || attempt == MAX_FINAL_ATTEMPTS )); then
    die "final evaluation was not authoritative (${final_validation})"
  fi
  stage="final_lock_handoff_retry"
  flock -w 120 7 || die "could not reacquire measurement lock after handoff contention"
  measurement_lock_held=1
  cleanup_owned_runtime "${TASK_RUN_ROOT}/handoff-retry-${attempt}-cleanup.json" \
    || die "handoff retry runtime cleanup failed"
  stage="final_evaluation"
done

[[ -s "${final_validation}" ]] || die "final validation result is missing"
jq -e '
  .schema_version == 1
  and .authoritative == true
  and .retryable == false
  and (.classification == "candidate_valid" or .classification == "candidate_invalid")
  and .eval_rc == 0
' "${final_validation}" >/dev/null \
  || die "final evaluation did not satisfy the authoritative terminal contract"

stage="selection"
set +e
python3 "${MIGRATION_ROOT}/scripts/select_best.py" \
  --task-id "${TASK_ID}" \
  --source-root "${TASK_RUN_ROOT}" \
  --source-root "${final_task_root}" \
  --validator "${FINAL_VALIDATOR}" \
  --expected-preflight "${PREFLIGHT_PATH}" \
  --output-dir "${SELECTION_ROOT}" \
  >"${TASK_RUN_ROOT}/selection.log" \
  2>"${TASK_RUN_ROOT}/selection.stderr.log"
selection_rc="$?"
set -e
if (( selection_rc != 0 )); then
  # Zero-valid is the only selector failure that is a legitimate terminal
  # outcome: every discovered hidden submission was authoritatively rejected.
  # Any other selector error (preflight drift, unsafe evidence, partial
  # publication) stays a hard failure. The selector raises before it creates
  # its output directory, so no selection evidence may exist on this path.
  selection_message="$(tail -n 1 "${TASK_RUN_ROOT}/selection.stderr.log" || true)"
  [[ "${selection_message}" == "no authoritatively eligible hidden submission was found"* ]] \
    || die "selection failed (rc ${selection_rc}): ${selection_message:-no message}"
  [[ ! -e "${SELECTION_PATH}" && ! -e "${SELECTED_VALIDATION}" ]] \
    || die "selector reported zero valid submissions but left selection evidence"
  no_valid_submission_reason="${selection_message}"
  echo "Selection: ${selection_message}; continuing to final quiescence and trusted-source verification" >&2
else
  selected_report="$(jq -er '.preserved_report' "${SELECTION_PATH}")"
  selected_submission_id="$(jq -er '.submission_id' "${selected_report}")"
  selected_private_artifact="$(jq -er '.preserved_private_artifact' "${SELECTION_PATH}")"
  selected_evaluated_archive="$(jq -er '.preserved_archive' "${SELECTION_PATH}")"
  selected_expected_archive="$(jq -er '.preserved_archive' "${SELECTION_PATH}")"
  python3 "${FINAL_VALIDATOR}" \
    --report "${selected_report}" \
    --private-artifact "${selected_private_artifact}" \
    --evaluated-archive "${selected_evaluated_archive}" \
    --expected-archive "${selected_expected_archive}" \
    --expected-preflight "${PREFLIGHT_PATH}" \
    --expected-submission-id "${selected_submission_id}" \
    --eval-rc 0 \
    --max-runtime-seconds 3060 \
    --output "${SELECTED_VALIDATION}"
  jq -e --slurpfile selection "${SELECTION_PATH}" '
    .schema_version == 1
    and .classification == "candidate_valid"
    and .authoritative == true
    and .retryable == false
    and .valid == true
    and (.score | type == "number")
    and .hidden_baseline_identity == $selection[0].hidden_baseline_identity
  ' "${SELECTED_VALIDATION}" >/dev/null \
    || die "selected best report is not an authoritative valid measurement"
fi

# The final evaluator normally cleans itself. Reacquire the lock and require a
# fully empty task runtime before completion can ever become passed.
stage="final_quiescence"
flock -w 120 7 || die "could not reacquire measurement lock after final evaluation"
measurement_lock_held=1
cleanup_owned_runtime "${TASK_RUN_ROOT}/final-cleanup.json" \
  || die "final evaluator runtime cleanup failed"
python3 "${CLEANUP_HELPER}" verify-quiescent \
  --nonce-source "${DEDICATED_NONCE_SOURCE}" \
  --migration-root "${MIGRATION_ROOT}" \
  --eval-root "${EVAL_ROOT}" \
  --judge-image "${JUDGE_IMAGE}" \
  --output "${TASK_RUN_ROOT}/final-quiescent.json"
python3 "${PREFLIGHT_HELPER}" \
  --verify-trusted-sources "${PREFLIGHT_PATH}" \
  --mlsysforge-root "${MLSYSFORGE_ROOT}" \
  --output "${SOURCE_VERIFICATION_PATH}" \
  >"${TASK_RUN_ROOT}/trusted-source-final-verification.log"
flock -u 7
measurement_lock_held=0

if [[ -n "${no_valid_submission_reason}" ]]; then
  completion_requested="completed_no_valid_submission"
  failure_reason="${no_valid_submission_reason}"
else
  completion_requested="passed"
  failure_reason=""
fi
stage="ready_for_completion"
echo "Completed ${RUN_ID}; strict completion (${completion_requested}) will be committed after cleanup"
echo "Final boundary: classification=$(jq -r '.classification' "${final_validation}"), valid=$(jq -r '.valid' "${final_validation}"), score=$(jq -r '.score' "${final_validation}")"
if [[ -n "${no_valid_submission_reason}" ]]; then
  echo "Best measured: none (${no_valid_submission_reason})"
else
  echo "Best measured: round=$(jq -r '.best.round' "${SELECTION_PATH}"), score=$(jq -r '.best.score' "${SELECTION_PATH}")"
fi
echo "Completion: ${COMPLETION_PATH}"
