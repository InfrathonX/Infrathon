#!/usr/bin/env bash
# Local Docker run of MInference v3 on two H100s of this host: Work on WORK_GPU,
# every Judge (the agent's submissions and the independent final) on JUDGE_GPU.
# launch/minference-v3/run.sh is the user entry; this script is the task side.
#
#   agent run:  Codex (or Claude Code, MLSYSFORGE_AGENT=claude-code) in the Work
#               container for TASK_TIMEOUT, manual submissions
#               to a host Judge server (cooldown EVAL_INTERVAL), then the final
#               Judge of the final archive
#   EVAL_ARCHIVE=FILE: no agent; FILE goes through the final Judge path only
#
# Images: scripts/build_images.sh or scripts/infrathon/pull_images.sh (checked against
# their build record).  Assets: assets/fetch.sh into INFRATHON_ASSETS_ROOT.  The
# Judge and Work container setup (mounts, exact GPU, bootstraps) is
# scripts/sforge_local_entry_v3.py; the network is the kernel launchers'
# internal network plus restricted CONNECT sidecar (scripts/restricted_proxy.py).
set -euo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_run_lib.sh"
ensure_docker_group "$@"
require_sforge
require_command curl
require_command docker
require_command flock
require_command jq
require_command nvidia-smi
require_command sha256sum
require_command ss

ENTRY="${SCRIPT_DIR}/sforge_local_entry_v3.py"
RESTRICTED_PROXY_SOURCE="${SCRIPT_DIR}/restricted_proxy.py"
EVAL_ARCHIVE="${EVAL_ARCHIVE:-}"
SKIP_INTEGRITY_REVIEW="${SKIP_INTEGRITY_REVIEW:-0}"
AUTH_JSON="${CODEX_AUTH_JSON_PATH:-}"
# MLSYSFORGE_AGENT=claude-code (launch/minference-v3/run.sh --agent claude-code)
# drives the pinned Claude Code runtime with a 0600 file holding one
# `claude setup-token` OAuth token (CLAUDE_OAUTH_TOKEN_FILE); the local entry
# binds it through sforge/harness/agent/claude_local.py.
AGENT_NAME="${MLSYSFORGE_AGENT:-codex}"
CLAUDE_TOKEN_FILE="${CLAUDE_OAUTH_TOKEN_FILE:-}"
# Optional OpenAI-compatible model API, relayed as raw TCP by the sidecar (see
# the kernel launchers); the key travels as a file path, never on a command line.
MODEL_API_BASE_URL="${MODEL_API_BASE_URL:-}"
MODEL_API_KEY_FILE="${MODEL_API_KEY_FILE:-}"
MODEL_FORWARD_PORT=3131
MODEL="${MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
# launch/common.sh CODEX_CLI_VERSION_PIN, which launch/minference-v3/run.sh passes.
CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-0.155.1}"
TASK_TIMEOUT="${TASK_TIMEOUT:-7200}"
# The Judge's own eval_timeout is 7200 s.
EVAL_INTERVAL="${EVAL_INTERVAL:-300}"
FINAL_EVAL_TIMEOUT="${FINAL_EVAL_TIMEOUT:-7200}"
WORK_GPU="${WORK_GPU:-0}"
JUDGE_GPU="${JUDGE_GPU:-1}"
JUDGE_PORT="${JUDGE_PORT:-18497}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
model_slug="$(sed -E 's/^gpt-[0-9.]+-//' <<<"${MODEL}" | tr 'A-Z' 'a-z' | sed -E 's/[^a-z0-9.]+/-/g; s/^-+//; s/-+$//')"

for numeric_name in TASK_TIMEOUT EVAL_INTERVAL FINAL_EVAL_TIMEOUT JUDGE_PORT; do
  numeric_value="${!numeric_name}"
  [[ "${numeric_value}" =~ ^[0-9]+$ && "$((10#${numeric_value}))" -gt 0 ]] \
    || die "${numeric_name} must be a positive integer"
done
TASK_TIMEOUT="$((10#${TASK_TIMEOUT}))"
EVAL_INTERVAL="$((10#${EVAL_INTERVAL}))"
FINAL_EVAL_TIMEOUT="$((10#${FINAL_EVAL_TIMEOUT}))"
JUDGE_PORT="$((10#${JUDGE_PORT}))"
(( JUDGE_PORT >= 1024 && JUDGE_PORT <= 65535 )) || die "invalid unprivileged JUDGE_PORT: ${JUDGE_PORT}"
[[ "${JUDGE_PORT}" != 3128 && "${JUDGE_PORT}" != 3130 && "${JUDGE_PORT}" != "${MODEL_FORWARD_PORT}" ]] \
  || die "judge port conflicts with a restricted-proxy port (3128, 3130, ${MODEL_FORWARD_PORT})"
[[ "${SKIP_INTEGRITY_REVIEW}" == 0 || "${SKIP_INTEGRITY_REVIEW}" == 1 ]] \
  || die "SKIP_INTEGRITY_REVIEW must be 0 or 1"

if [[ -n "${EVAL_ARCHIVE}" ]]; then
  [[ -f "${EVAL_ARCHIVE}" && ! -L "${EVAL_ARCHIVE}" && -s "${EVAL_ARCHIVE}" ]] \
    || die "EVAL_ARCHIVE must be a nonempty regular, non-symlink file: ${EVAL_ARCHIVE}"
  EVAL_ARCHIVE="$(cd -- "$(dirname -- "${EVAL_ARCHIVE}")" && pwd -P)/$(basename -- "${EVAL_ARCHIVE}")"
  RUN_ID="${RUN_ID:-minference-v3-eval-${RUN_STAMP}}"
else
  (( SKIP_INTEGRITY_REVIEW == 0 )) || die "SKIP_INTEGRITY_REVIEW applies to EVAL_ARCHIVE only"
  [[ -n "${MODEL}" ]] || die "set MODEL to the agent model id (launch/minference-v3/run.sh --model)"
  [[ "${MODEL}" =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]] || die "unsafe MODEL: ${MODEL}"
  case "${REASONING_EFFORT}" in
    high|max|xhigh|ultra) ;;
    *) die "unsupported reasoning effort: ${REASONING_EFFORT}" ;;
  esac
  case "${AGENT_NAME}" in
    codex) ;;
    claude-code)
      [[ -z "${MODEL_API_BASE_URL}" ]] || die "Claude Code uses its OAuth token; unset MODEL_API_BASE_URL"
      [[ "${REASONING_EFFORT}" != ultra ]] || die "Claude Code efforts are high|xhigh|max"
      ;;
    *) die "MLSYSFORGE_AGENT must be codex or claude-code" ;;
  esac
  (( TASK_TIMEOUT >= 90 && TASK_TIMEOUT > EVAL_INTERVAL && TASK_TIMEOUT % 60 == 0 && EVAL_INTERVAL % 60 == 0 )) \
    || die "timeout and evaluation interval must be minute-aligned, the timeout at least 90 s and longer than the interval"
  RUN_ID="${RUN_ID:-minference-v3-h100-${model_slug}-${REASONING_EFFORT}-$(((TASK_TIMEOUT + 59) / 60))m-${RUN_STAMP}}"
fi
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe RUN_ID: ${RUN_ID}"
RUN_LOG_ROOT="${LOGS_DIR}/runs/${RUN_ID}"
TASK_RUN_ROOT="${RUN_LOG_ROOT}/${TASK_ID}"
RUN_ASSETS="${LOGS_DIR}/run-assets/${RUN_ID}"
WORK_CONTAINER_NAME="sforge.run.${TASK_ID}.${RUN_ID}"

MODEL_API_KEY=""; model_api_host=""; model_api_port=""; model_api_path=""
if [[ -z "${EVAL_ARCHIVE}" ]]; then
  for source in "${RESTRICTED_PROXY_SOURCE}" "${ENTRY}"; do
    [[ -f "${source}" && ! -L "${source}" && -s "${source}" ]] || die "missing launcher source: ${source}"
  done
  if [[ -n "${MODEL_API_BASE_URL}" ]]; then
    [[ "${MODEL_API_BASE_URL}" =~ ^http://([^/:[:space:]]+):([0-9]+)(/.*)?$ ]] \
      || die "MODEL_API_BASE_URL must be http://HOST:PORT[/path] (relayed as raw TCP by the sidecar)"
    model_api_host="${BASH_REMATCH[1]}"; model_api_port="${BASH_REMATCH[2]}"; model_api_path="${BASH_REMATCH[3]:-}"
    [[ -n "${MODEL_API_KEY_FILE}" ]] || die "set MODEL_API_KEY_FILE with MODEL_API_BASE_URL"
    assert_private_auth_file "${MODEL_API_KEY_FILE}"
    MODEL_API_KEY="$(tr -d '[:space:]' < "${MODEL_API_KEY_FILE}")"
    [[ -n "${MODEL_API_KEY}" ]] || die "MODEL_API_KEY_FILE is empty"
  elif [[ "${AGENT_NAME}" == "claude-code" ]]; then
    [[ -n "${CLAUDE_TOKEN_FILE}" ]] || die "set CLAUDE_OAUTH_TOKEN_FILE"
    assert_private_auth_file "${CLAUDE_TOKEN_FILE}"
  else
    [[ -n "${AUTH_JSON}" ]] || die "set CODEX_AUTH_JSON_PATH"
    assert_private_auth_file "${AUTH_JSON}"
  fi
  # The Work image is agent-free; a host-provided runtime supplies the CLI.
  AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}"
  [[ -n "${AGENT_RUNTIME_DIR}" ]] || die "set SFORGE_AGENT_RUNTIME_DIR (build one with launch/build-agent-runtime.sh)"
  runtime_marker="${AGENT_RUNTIME_DIR}/.sforge-agent-runtime.json"
  [[ -d "${AGENT_RUNTIME_DIR}" && ! -L "${AGENT_RUNTIME_DIR}" && -f "${runtime_marker}" ]] \
    || die "agent runtime directory is missing or unmarked: ${AGENT_RUNTIME_DIR}"
  [[ "$(jq -er '.kind' "${runtime_marker}")" == sforge-agent-runtime \
     && "$(jq -er '.agent' "${runtime_marker}")" == "${AGENT_NAME}" ]] \
    || die "SFORGE_AGENT_RUNTIME_DIR must be a ${AGENT_NAME} runtime: ${AGENT_RUNTIME_DIR}"
  runtime_version="$(jq -er '.version' "${runtime_marker}")"
  if [[ "${AGENT_NAME}" == "codex" ]]; then
    [[ "${runtime_version}" == "${CODEX_CLI_VERSION}" ]] \
      || die "agent runtime version ${runtime_version} differs from the pinned ${CODEX_CLI_VERSION}"
  else
    # No Codex CLI in a Claude Code run; the runtime marker pins Claude Code.
    CODEX_CLI_VERSION=""
  fi
  export SFORGE_AGENT_RUNTIME_DIR="${AGENT_RUNTIME_DIR}"
  task_internet="$(jq -r '.internet' "${TASKS_DIR}/${TASK_ID}.json")"
  case "${task_internet}" in
    true) sidecar_public_egress="--allow-public-egress" ;;
    false) sidecar_public_egress="" ;;
    *) die "task JSON internet must be true or false, got: ${task_internet}" ;;
  esac
  export MLSYSFORGE_WORK_INTERNET="${task_internet}"
fi

# Agent submissions and every final are integrity-reviewed (docs/LLM_JUDGE.md);
# only an EVAL_ARCHIVE qualification may skip it.
if (( ! SKIP_INTEGRITY_REVIEW )); then
  [[ "${SFORGE_LLM_JUDGE_MODE:-}" == "enforce" ]] \
    || die "set SFORGE_LLM_JUDGE_MODE=enforce and the SFORGE_LLM_JUDGE_* reviewer settings"
  review_summary="$(
    cd -- "${REPOSITORY_ROOT}" && "${PYTHON_BIN}" -B - <<'PY' 2>&1
from sforge.harness.llm_judge.config import load_llm_judge_config
config = load_llm_judge_config()
config.check_ready()
print(f"{config.mode} {config.engine} {config.agent} {config.model} effort={config.reasoning_effort} "
      f"runtime={config.agent_runtime_dir}")
PY
  )" || die "integrity review is not ready: ${review_summary##*$'\n'}"
  echo "Integrity review: ${review_summary}"
fi

assert_images_current
"${PYTHON_BIN}" "${REPOSITORY_ROOT}/scripts/infrathon/assets.py" check "${TASK_DIRECTORY}" \
  || die "MInference assets are missing under ${INFRATHON_ASSETS_ROOT}; run ${MIGRATION_ROOT}/assets/fetch.sh"

WORK_GPU="$(canonical_gpu_index "${WORK_GPU}")"
JUDGE_GPU="$(canonical_gpu_index "${JUDGE_GPU}")"
work_gpu_uuid="$(gpu_uuid "${WORK_GPU}")"
judge_gpu_uuid="$(gpu_uuid "${JUDGE_GPU}")"
[[ -n "${EVAL_ARCHIVE}" || "${work_gpu_uuid}" != "${judge_gpu_uuid}" ]] \
  || die "Work and Judge must use different physical GPUs"
[[ ! -e "${RUN_LOG_ROOT}" ]] || die "run directory already exists: ${RUN_LOG_ROOT}"
if [[ -z "${EVAL_ARCHIVE}" ]] && ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${JUDGE_PORT}$"; then
  die "judge port ${JUDGE_PORT} is already in use"
fi

if [[ "${SFORGE_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run (no lock, container or GPU work): ${RUN_ID}"
  if [[ -n "${EVAL_ARCHIVE}" ]]; then
    echo "  mode:     final Judge of ${EVAL_ARCHIVE} (integrity review $([[ ${SKIP_INTEGRITY_REVIEW} == 1 ]] && echo skipped || echo on))"
  else
    echo "  agent:    ${AGENT_NAME} ${MODEL} effort=${REASONING_EFFORT} runtime ${AGENT_RUNTIME_DIR} (${runtime_version})"
    if [[ -n "${MODEL_API_BASE_URL}" ]]; then
      echo "  credential: model API ${MODEL_API_BASE_URL} (key file ${MODEL_API_KEY_FILE})"
    elif [[ "${AGENT_NAME}" == "claude-code" ]]; then
      echo "  credential: Claude OAuth token file ${CLAUDE_TOKEN_FILE}"
    else
      echo "  credential: Codex auth ${AUTH_JSON}"
    fi
    echo "  budget:   ${TASK_TIMEOUT}s; manual submissions, cooldown ${EVAL_INTERVAL}s; final Judge timeout ${FINAL_EVAL_TIMEOUT}s"
    echo "  Work GPU: ${WORK_GPU} (${work_gpu_uuid}); judge port ${JUDGE_PORT}; internet=${task_internet}"
  fi
  echo "  Judge GPU: ${JUDGE_GPU} (${judge_gpu_uuid})"
  echo "  assets:   ${TASK_ASSETS}"
  echo "  logs:     ${RUN_LOG_ROOT}"
  exit 0
fi

exec 9>"$(gpu_lock_path "${judge_gpu_uuid}")"
flock -n 9 || die "Judge GPU ${JUDGE_GPU} lock is already held"
assert_gpu_idle "${JUDGE_GPU}" "Judge"
assert_container_accelerator "${JUDGE_IMAGE}" "${judge_gpu_uuid}" "judge"
if [[ -z "${EVAL_ARCHIVE}" ]]; then
  exec 8>"$(gpu_lock_path "${work_gpu_uuid}")"
  flock -n 8 || die "Work GPU ${WORK_GPU} lock is already held"
  assert_gpu_idle "${WORK_GPU}" "Work"
  assert_container_accelerator "${WORK_IMAGE}" "${work_gpu_uuid}" "work"
fi

mkdir -p "${RUN_LOG_ROOT}"
judge_pid=""
active_pid=""
isolated_network=""
restricted_proxy_name=""

stop_active_process() {
  if [[ -z "${active_pid}" ]] || ! kill -0 "${active_pid}" 2>/dev/null; then
    active_pid=""
    return
  fi
  kill -TERM "${active_pid}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "${active_pid}" 2>/dev/null || break
    sleep 1
  done
  kill -0 "${active_pid}" 2>/dev/null && kill -KILL "${active_pid}" 2>/dev/null || true
  wait "${active_pid}" 2>/dev/null || true
  active_pid=""
}

stop_judge_server() {
  if [[ -n "${judge_pid}" ]] && kill -0 "${judge_pid}" 2>/dev/null; then
    kill "${judge_pid}" 2>/dev/null || true
    wait "${judge_pid}" 2>/dev/null || true
  fi
  judge_pid=""
}

cleanup() {
  stop_active_process
  stop_judge_server
  docker rm -f "${WORK_CONTAINER_NAME}" >/dev/null 2>&1 || true
  # Judges are named <benchmark>.judge.<task_id>.<submission>; SForge removes
  # them.  This catches a final Judge (named after the run) that a killed
  # process left behind; an interactive one is named by its submission hash,
  # which cannot be told apart from another run's.
  docker ps -aq --filter "name=^${BENCHMARK_NAME}.judge.${TASK_ID}.${RUN_ID}" | xargs -r docker rm -f >/dev/null 2>&1 || true
  [[ -z "${restricted_proxy_name}" ]] || docker rm -f "${restricted_proxy_name}" >/dev/null 2>&1 || true
  [[ -z "${isolated_network}" ]] || docker network rm "${isolated_network}" >/dev/null 2>&1 || true
  remove_run_assets "${RUN_ASSETS}"
}

handle_signal() {
  local exit_code=143
  [[ "$1" == "INT" ]] && exit_code=130
  trap - INT TERM
  cleanup
  exit "${exit_code}"
}

run_logged_command() {
  local log_file="$1"
  shift
  "$@" > >(tee "${log_file}") 2>&1 &
  active_pid="$!"
  local command_rc=0
  wait "${active_pid}" || command_rc="$?"
  active_pid=""
  return "${command_rc}"
}

trap cleanup EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

prepare_run_assets "${RUN_ASSETS}"
export MINFERENCE_MODEL_DIR="${MODEL_ASSET}"
export MINFERENCE_WORK_MODEL_DIR="${WORK_MODEL_ASSET}"
export MINFERENCE_PUBLIC_PROMPTS_DIR="${RUN_ASSETS}/public-prompts"
export MINFERENCE_PRIVATE_PROMPTS_DIR="${RUN_ASSETS}/kernel-prompts"
export MINFERENCE_JUDGE_GPU_UUID="${judge_gpu_uuid}"
export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${judge_gpu_uuid},CUDA_VISIBLE_DEVICES=0,PYTHONDONTWRITEBYTECODE=1"
export SFORGE_ADMIN_SECRET
SFORGE_ADMIN_SECRET="$("${PYTHON_BIN}" -c 'import secrets; print(secrets.token_urlsafe(48))')"
work_image_id="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")"
judge_image_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"

final_eval() {  # $1 = archive, $2 = eval run id
  local review_args=()
  (( ! SKIP_INTEGRITY_REVIEW )) || review_args=(--skip-integrity-review)
  [[ ! -e "${LOGS_DIR}/runs/$2" || "$2" == "${RUN_ID}" ]] || die "final eval directory already exists: $2"
  # Phase final, round final-1 (scripts/sforge_local_entry_v3.py); the last step of either mode.
  export MINFERENCE_FINAL_EVAL=1
  run_logged_command "${RUN_LOG_ROOT}/final-eval.log" \
    "${PYTHON_BIN}" "${ENTRY}" eval \
    --backend docker \
    --task "${TASK_ID}" \
    --archive "$1" \
    --run-id "$2" \
    --timeout "${FINAL_EVAL_TIMEOUT}" \
    --judge-cpu-limit 8 \
    --judge-mem-limit 48g \
    "${review_args[@]}" \
    --json
}

write_completion() {  # $1 = archive, $2 = final report, $3 = status
  local score valid
  score="$(jq -c '.score // null' "$2" 2>/dev/null || echo null)"
  valid="$(jq -c '.valid // false' "$2" 2>/dev/null || echo false)"
  jq -n \
    --arg status "$3" \
    --arg finished_at "$(date -u +%FT%TZ)" \
    --arg run_id "${RUN_ID}" \
    --arg task_id "${TASK_ID}" \
    --arg mode "$([[ -n "${EVAL_ARCHIVE}" ]] && echo eval-archive || echo agent-run)" \
    --arg model "${MODEL}" \
    --arg effort "${REASONING_EFFORT}" \
    --arg work_gpu_uuid "${work_gpu_uuid}" \
    --arg judge_gpu_uuid "${judge_gpu_uuid}" \
    --arg work_image_id "${work_image_id}" \
    --arg judge_image_id "${judge_image_id}" \
    --arg archive "$1" \
    --arg archive_sha256 "$(sha256sum "$1" | awk '{print $1}')" \
    --arg final_report "$2" \
    --argjson score "${score}" \
    --argjson valid "${valid}" \
    --argjson timeout "${TASK_TIMEOUT}" \
    --argjson eval_interval "${EVAL_INTERVAL}" \
    '{schema_version: 1, status: $status, finished_at: $finished_at, run_id: $run_id, task_id: $task_id,
      mode: $mode, model: (if $mode == "agent-run" then $model else null end),
      reasoning_effort: (if $mode == "agent-run" then $effort else null end),
      timeout_seconds: $timeout, submission_cooldown_seconds: $eval_interval,
      work_gpu_uuid: (if $mode == "agent-run" then $work_gpu_uuid else null end), judge_gpu_uuid: $judge_gpu_uuid,
      images: {work: $work_image_id, judge: $judge_image_id},
      final_archive: {path: $archive, sha256: $archive_sha256},
      final_report: $final_report, valid: $valid, score: $score}' >"${RUN_LOG_ROOT}/completion.json"
}

if [[ -n "${EVAL_ARCHIVE}" ]]; then
  echo "Final Judge of ${EVAL_ARCHIVE}"
  echo "  Judge GPU: ${JUDGE_GPU} (${judge_gpu_uuid}); image ${judge_image_id:7:12}"
  echo "  logs: ${RUN_LOG_ROOT}"
  eval_rc=0
  final_eval "${EVAL_ARCHIVE}" "${RUN_ID}" || eval_rc="$?"
  report="$(ls -1d "${LOGS_DIR}/runs/${RUN_ID}/${TASK_ID}/submissions"/manual-*/report.json 2>/dev/null | tail -n 1)"
  [[ -n "${report}" && -s "${report}" ]] || die "the final Judge wrote no report (exit ${eval_rc})"
  write_completion "${EVAL_ARCHIVE}" "${report}" "$( (( eval_rc == 0 )) && echo passed || echo failed)"
  assert_gpu_released "${judge_gpu_uuid}"
  echo "Completed ${RUN_ID}: $(jq -c '{valid, score}' "${RUN_LOG_ROOT}/completion.json")"
  exit "${eval_rc}"
fi

# Judge server for the agent's submissions on the Docker bridge gateway (the
# sidecar relays Work to it) and loopback; no public interface is bound.
bridge_gateway="$(docker network inspect bridge | jq -er '.[0].IPAM.Config[0].Gateway')"
[[ "${bridge_gateway}" =~ ^[0-9]+(\.[0-9]+){3}$ ]] || die "could not resolve the Docker bridge gateway"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${bridge_gateway},127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
"${PYTHON_BIN}" "${ENTRY}" serve --host "${bridge_gateway}" --host 127.0.0.1 --port "${JUDGE_PORT}" \
  >"${RUN_LOG_ROOT}/judge-server.log" 2>&1 &
judge_pid="$!"
wait_for_http "http://${bridge_gateway}:${JUDGE_PORT}/openapi.json" 90
wait_for_http "http://127.0.0.1:${JUDGE_PORT}/openapi.json" 30

# Work has only an internal Docker network; it reaches the outside solely
# through the CONNECT sidecar (model auth, plus public hosts on 80/443 with
# internet=true; never an internal address) and the Judge through its relay.
isolated_network="${RUN_ID}-internal"
restricted_proxy_name="${RUN_ID}-restricted-proxy"
docker network create --driver bridge --internal --label "mlsysforge.run_id=${RUN_ID}" "${isolated_network}" >/dev/null
docker run --detach --rm --pull=never \
  --name "${restricted_proxy_name}" \
  --network bridge \
  --add-host host.docker.internal:host-gateway \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --volume "${RESTRICTED_PROXY_SOURCE}:/opt/mlsysforge/restricted_proxy.py:ro" \
  --entrypoint python3 \
  "${WORK_IMAGE}" \
  /opt/mlsysforge/restricted_proxy.py \
  ${sidecar_public_egress:+"${sidecar_public_egress}"} \
  --evaluation-upstream-host "${bridge_gateway}" \
  --evaluation-upstream-port "${JUDGE_PORT}" \
  ${model_api_host:+--model-forward "${MODEL_FORWARD_PORT}:${model_api_host}:${model_api_port}"} >/dev/null
docker network connect --alias runtime-proxy "${isolated_network}" "${restricted_proxy_name}"
restricted_proxy_ip="$(
  docker inspect "${restricted_proxy_name}" \
    | jq -er --arg network "${isolated_network}" \
      '.[0].NetworkSettings.Networks[$network].IPAddress | select(type == "string" and length > 0)'
)"
export NO_PROXY="${restricted_proxy_ip},${NO_PROXY}"
export no_proxy="${NO_PROXY}"
probe() {  # run one network check from inside the internal network
  docker run --rm --pull=never --network "${isolated_network}" --env NVIDIA_VISIBLE_DEVICES=void "$@"
}
for _ in $(seq 1 30); do
  probe --entrypoint curl "${WORK_IMAGE}" --connect-timeout 2 --max-time 3 -fsS \
    "http://${restricted_proxy_ip}:3128/health" >/dev/null 2>&1 && break
  sleep 1
done
probe --entrypoint curl "${WORK_IMAGE}" --connect-timeout 2 --max-time 5 -fsS \
  "http://${restricted_proxy_ip}:3130/openapi.json" >/dev/null \
  || die "Judge relay on the restricted proxy does not reach the Judge server"
if probe --entrypoint curl "${WORK_IMAGE}" --connect-timeout 2 --max-time 4 -fsS https://example.com >/dev/null 2>&1; then
  die "internal work network unexpectedly has direct public egress"
fi
probe --entrypoint python3 "${WORK_IMAGE}" -c '
import socket,sys
p=sys.argv[1]
def check(host, expected):
    with socket.create_connection((p,3128),5) as s:
        s.sendall(f"CONNECT {host}:443 HTTP/1.1\r\nHost: {host}:443\r\n\r\n".encode("ascii"))
        line=s.recv(256).split(b"\r\n",1)[0]
        if expected not in line:
            raise SystemExit(f"{host}: expected {expected!r}, got {line!r}")
check("chatgpt.com", b" 200 ")
if sys.argv[2] == "true":
    check("github.com", b" 200 ")
    check(sys.argv[3], b" 403 ")
else:
    check("github.com", b" 403 ")' "${restricted_proxy_ip}" "${task_internet}" "${bridge_gateway}" \
  || die "restricted proxy allowlist preflight failed"
echo "Restricted network: PASS (${isolated_network}, proxy ${restricted_proxy_ip}, public egress ${task_internet})"

unset SFORGE_AGENT_API_KEY SFORGE_AGENT_API_BASE_URL SFORGE_AGENT_MODEL SFORGE_AGENT_TIMEOUT \
  CODEX_API_KEY OPENAI_API_KEY OPENAI_BASE_URL
if [[ -n "${MODEL_API_BASE_URL}" ]]; then
  export SFORGE_AGENT_API_BASE_URL="http://${restricted_proxy_ip}:${MODEL_FORWARD_PORT}${model_api_path}"
  export SFORGE_AGENT_API_KEY="${MODEL_API_KEY}"
elif [[ "${AGENT_NAME}" == "claude-code" ]]; then
  unset SFORGE_CODEX_AUTH_JSON_PATH
  export MLSYSFORGE_CLAUDE_OAUTH_TOKEN_FILE="${CLAUDE_TOKEN_FILE}"
else
  export SFORGE_CODEX_AUTH_JSON_PATH="${AUTH_JSON}"
fi
# Read by sforge_local_entry_v3.py (claude_local.install_from_environment).
export MLSYSFORGE_AGENT="${AGENT_NAME}" MLSYSFORGE_AGENT_MODEL="${MODEL}"
export MLSYSFORGE_AGENT_EFFORT="${REASONING_EFFORT}" MLSYSFORGE_AGENT_BUDGET_SECONDS="${TASK_TIMEOUT}"
export MLSYSFORGE_RUN_LOG_ROOT="${RUN_LOG_ROOT}"
export SFORGE_ISOLATED_DOCKER_NETWORK="${isolated_network}"
export SFORGE_HTTP_PROXY="http://${restricted_proxy_ip}:3128"
export SFORGE_HTTPS_PROXY="http://${restricted_proxy_ip}:3128"
export SFORGE_NO_PROXY="${restricted_proxy_ip},127.0.0.1,localhost"
export SFORGE_CODEX_REASONING_EFFORT="${REASONING_EFFORT}"
export SFORGE_CODEX_CLI_VERSION="${CODEX_CLI_VERSION}"
export SFORGE_AGENT_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${work_gpu_uuid},CUDA_VISIBLE_DEVICES=0,PYTHONDONTWRITEBYTECODE=1"

jq -n \
  --arg run_id "${RUN_ID}" --arg started_at "$(date -u +%FT%TZ)" --arg task_id "${TASK_ID}" \
  --arg agent "${AGENT_NAME}" \
  --arg model "${MODEL}" --arg effort "${REASONING_EFFORT}" --arg cli_version "${CODEX_CLI_VERSION}" \
  --arg work_gpu_uuid "${work_gpu_uuid}" --arg judge_gpu_uuid "${judge_gpu_uuid}" \
  --arg work_image_id "${work_image_id}" --arg judge_image_id "${judge_image_id}" \
  --arg entry_sha256 "$(sha256sum "${ENTRY}" | awk '{print $1}')" \
  --arg proxy_sha256 "$(sha256sum "${RESTRICTED_PROXY_SOURCE}" | awk '{print $1}')" \
  --arg model_asset "${MODEL_ASSET}" \
  --argjson timeout "${TASK_TIMEOUT}" --argjson eval_interval "${EVAL_INTERVAL}" --argjson internet "${task_internet}" \
  '{schema_version: 1, run_id: $run_id, task_id: $task_id, started_at: $started_at,
    agent: $agent, model: $model, reasoning_effort: $effort, codex_cli_version: $cli_version,
    timeout_seconds: $timeout, scheduled_auto_evaluation: false, agent_submission_cooldown_seconds: $eval_interval,
    final_evaluation: "independent final Judge of the final archive (phase final, round final-1)",
    work_gpu_uuid: $work_gpu_uuid, judge_gpu_uuid: $judge_gpu_uuid,
    images: {work: $work_image_id, judge: $judge_image_id},
    assets: {model: $model_asset, prompts: "run-scoped single-link copies; final prompts root-only"},
    network: {mode: "docker_internal_restricted_connect_proxy", public_connect_egress: $internet,
              entry_sha256: $entry_sha256, restricted_proxy_sha256: $proxy_sha256}}' \
  >"${RUN_LOG_ROOT}/experiment-manifest.json"

echo "Starting ${RUN_ID}"
echo "  model: ${MODEL} (${REASONING_EFFORT}); budget ${TASK_TIMEOUT}s; manual submissions, cooldown ${EVAL_INTERVAL}s"
echo "  Work GPU: ${WORK_GPU} (${work_gpu_uuid}); Judge GPU: ${JUDGE_GPU} (${judge_gpu_uuid})"
echo "  logs: ${RUN_LOG_ROOT}"
run_rc=0
run_logged_command "${RUN_LOG_ROOT}/sforge-run.log" \
  "${PYTHON_BIN}" "${ENTRY}" run \
  --backend docker \
  --task "${TASK_ID}" \
  --agent "${AGENT_NAME}" \
  --model "${MODEL}" \
  --timeout "${TASK_TIMEOUT}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --disable-auto-eval \
  --submission-cooldown "${EVAL_INTERVAL}" \
  --judge-url "http://${restricted_proxy_ip}:3130" \
  --enable-internet \
  --work-cpu-limit 8 \
  --work-mem-limit 48g \
  --judge-cpu-limit 8 \
  --judge-mem-limit 48g \
  --run-id "${RUN_ID}" || run_rc="$?"
(( run_rc == 0 )) || die "SForge run failed with exit ${run_rc}"

final_archive="${TASK_RUN_ROOT}/final_archive.tar.gz"
[[ -s "${final_archive}" ]] || die "SForge run ended without a final archive: ${final_archive}"
stop_judge_server
if docker container inspect "${WORK_CONTAINER_NAME}" >/dev/null 2>&1; then
  die "Work container still exists at the final-evaluation barrier"
fi
assert_gpu_released "${work_gpu_uuid}"
assert_gpu_released "${judge_gpu_uuid}"
assert_gpu_idle "${JUDGE_GPU}" "final Judge"

final_eval_id="${RUN_ID}-final"
final_rc=0
final_eval "${final_archive}" "${final_eval_id}" || final_rc="$?"
final_report="${LOGS_DIR}/runs/${final_eval_id}/${TASK_ID}/submissions/manual-1/report.json"
[[ -s "${final_report}" ]] || die "the final Judge wrote no report (exit ${final_rc})"
write_completion "${final_archive}" "${final_report}" "$( (( final_rc == 0 )) && echo passed || echo failed)"
assert_gpu_released "${judge_gpu_uuid}"
echo "Completed ${RUN_ID}: $(jq -c '{valid, score}' "${RUN_LOG_ROOT}/completion.json")"
echo "  final report: ${final_report}"
exit "${final_rc}"
