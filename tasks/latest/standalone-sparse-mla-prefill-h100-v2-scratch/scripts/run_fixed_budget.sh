#!/usr/bin/env bash
set -euo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_sforge
require_command curl
require_command docker
require_command flock
require_command jq
require_command nvidia-smi
require_command sha256sum
require_command ss

AUTH_JSON="${CODEX_AUTH_JSON_PATH:-}"
# MLSYSFORGE_AGENT=claude-code (launch/standalone-sparse-mla-v2/run.sh --agent
# claude-code) drives the pinned Claude Code runtime with a 0600 file holding
# one `claude setup-token` OAuth token (CLAUDE_OAUTH_TOKEN_FILE); the isolated
# entry binds it through sforge/harness/agent/claude_local.py.
AGENT_NAME="${MLSYSFORGE_AGENT:-codex}"
CLAUDE_TOKEN_FILE="${CLAUDE_OAUTH_TOKEN_FILE:-}"
# Optional OpenAI-compatible model API (e.g. a self-hosted DeepSeek behind
# sglang/vLLM).  With MODEL_API_BASE_URL set, Codex authenticates with the key
# in MODEL_API_KEY_FILE against that endpoint instead of the ChatGPT auth.json
# session, and MODEL may be any id the endpoint serves.  The key travels as a
# file path, never on a command line (the docker-group re-exec is visible in ps).
MODEL_API_BASE_URL="${MODEL_API_BASE_URL:-}"
MODEL_API_KEY_FILE="${MODEL_API_KEY_FILE:-}"
# Work lives on an internal Docker network; a custom model API is reachable
# only through one raw TCP relay on the restricted sidecar (this port).
MODEL_FORWARD_PORT=3131
# Restricted-proxy sidecar pattern (eagle3 / llada / flash-kmeans): the Work
# container is attached to an --internal Docker network whose only exit is a
# host-controlled CONNECT sidecar that admits the three Codex auth/model
# hosts and relays the Judge port. With the task JSON's internet=true it
# also runs --allow-public-egress (CONNECT to any globally routable host on
# 80/443; internal addresses refused). Both sources are task-local copies.
ISOLATED_ENTRY="${SCRIPT_DIR}/sforge_isolated_entry.py"
RESTRICTED_PROXY_SOURCE="${SCRIPT_DIR}/restricted_proxy.py"
[[ -f "${ISOLATED_ENTRY}" && ! -L "${ISOLATED_ENTRY}" ]] \
  || die "missing isolated SForge entry: ${ISOLATED_ENTRY}"
[[ -f "${RESTRICTED_PROXY_SOURCE}" && ! -L "${RESTRICTED_PROXY_SOURCE}" ]] \
  || die "missing restricted proxy source: ${RESTRICTED_PROXY_SOURCE}"
MODEL="${MODEL:-}"
[[ -n "${MODEL}" ]] || die "set MODEL to the agent model id (launch/<task>/run.sh --model)"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-0.148.0}"
TASK_TIMEOUT="${TASK_TIMEOUT:-7200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-300}"
FINAL_EVAL_TIMEOUT="${FINAL_EVAL_TIMEOUT:-5400}"
# SFORGE_SINGLE_GPU_HANDOFF=1 is exported only by run_single_gpu_fixed_budget.sh
# after its fail-closed production preflight.  The historical two-GPU path is
# selected by launch/<task>/run.sh --legacy-dual-gpu and never by fallback.
SINGLE_GPU_HANDOFF="${SFORGE_SINGLE_GPU_HANDOFF:-0}"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  [[ ! -v WORK_GPU && ! -v JUDGE_GPU ]] \
    || die "single GPU handoff rejects WORK_GPU/JUDGE_GPU"
  work_gpu_uuid="${SFORGE_ACCELERATOR_DEVICE_UUID:-}"
  judge_gpu_uuid="${work_gpu_uuid}"
  WORK_GPU="scheduler"
  JUDGE_GPU="scheduler"
else
  [[ "${SINGLE_GPU_HANDOFF}" == "0" ]] \
    || die "SFORGE_SINGLE_GPU_HANDOFF must be 0 or 1"
  WORK_GPU="${WORK_GPU:-0}"
  JUDGE_GPU="${JUDGE_GPU:-1}"
fi
JUDGE_PORT="${JUDGE_PORT:-18096}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
# Label-safe: ZHIPU/GLM-5.3-Flash -> zhipu-glm-5.3-flash.
model_slug="$(tr 'A-Z' 'a-z' <<<"${MODEL}" | sed -E 's/[^a-z0-9.]+/-/g; s/^-+//; s/-+$//')"
budget_minutes="$(((TASK_TIMEOUT + 59) / 60))"

# Any model id: a ChatGPT-auth Codex model or whatever MODEL_API_BASE_URL serves.
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
for numeric_name in TASK_TIMEOUT EVAL_INTERVAL FINAL_EVAL_TIMEOUT; do
  numeric_value="${!numeric_name}"
  [[ "${numeric_value}" =~ ^[0-9]+$ && "$((10#${numeric_value}))" -gt 0 ]] \
    || die "${numeric_name} must be a positive integer"
done
TASK_TIMEOUT="$((10#${TASK_TIMEOUT}))"
EVAL_INTERVAL="$((10#${EVAL_INTERVAL}))"
FINAL_EVAL_TIMEOUT="$((10#${FINAL_EVAL_TIMEOUT}))"
(( TASK_TIMEOUT >= 90 )) || die "TASK_TIMEOUT must be at least 90 seconds"
(( TASK_TIMEOUT > EVAL_INTERVAL && TASK_TIMEOUT % 60 == 0 && EVAL_INTERVAL % 60 == 0 )) \
  || die "timeout and evaluation interval must be minute-aligned with at least one pre-boundary tick"
final_minutes="$(((TASK_TIMEOUT + 59) / 60))"
interval_minutes="$((EVAL_INTERVAL / 60))"

[[ "${JUDGE_PORT}" =~ ^[0-9]+$ ]] || die "invalid JUDGE_PORT: ${JUDGE_PORT}"
JUDGE_PORT="$((10#${JUDGE_PORT}))"
(( JUDGE_PORT >= 1024 && JUDGE_PORT <= 65535 )) \
  || die "invalid unprivileged JUDGE_PORT: ${JUDGE_PORT}"
[[ "${JUDGE_PORT}" != "3128" && "${JUDGE_PORT}" != "3130" && "${JUDGE_PORT}" != "${MODEL_FORWARD_PORT}" ]] \
  || die "judge port conflicts with a reserved sidecar port"
MODEL_API_KEY=""; model_api_host=""; model_api_port=""; model_api_path=""
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

# The selected portable execution profile supplies the capability envelope
# and the software-ABI / calibration identities of the measurement profile.
readarray -t portable_identity < <(
  "${PYTHON_BIN}" - "${TASKS_DIR}/${TASK_ID}.json" <<'PY'
import json
import sys
from pathlib import Path
from sforge.harness.portable_spec import parse_portable_spec

raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
portable = parse_portable_spec(raw["portable"], task_id=raw["task_id"])
profile = portable.execution_profiles[portable.defaults.execution_profile_id]
metric = portable.metrics[portable.defaults.metric_id]
print(portable.evaluation_contract_sha256)
print(portable.defaults.execution_profile_id)
print(profile.software_abi.spec_sha256)
print(profile.calibration_protocol.spec_sha256)
print(profile.vendor)
print(profile.count)
print(profile.min_memory_bytes)
print(profile.min_compute_capability)
print(metric.unit)
PY
)
(( ${#portable_identity[@]} == 9 )) \
  || die "could not resolve portable contract identity and capability profile"
evaluation_contract_sha256="${portable_identity[0]}"
execution_profile_id="${portable_identity[1]}"
software_abi_sha256="${portable_identity[2]}"
calibration_protocol_sha256="${portable_identity[3]}"
profile_vendor="${portable_identity[4]}"
profile_count="${portable_identity[5]}"
profile_min_memory_bytes="${portable_identity[6]}"
profile_min_compute_capability="${portable_identity[7]}"
metric_unit="${portable_identity[8]}"

# Network policy follows the task JSON.  internet=true: the sidecar also runs
# with --allow-public-egress (CONNECT to any globally routable host on ports
# 80/443; loopback, private, Docker-bridge, link-local and other non-global
# addresses refused) and Codex runs with live web search.  internet=false keeps
# the model-auth CONNECT allowlist only, and the entry (via
# MLSYSFORGE_WORK_INTERNET) restores the offline Codex flags and prompt.  Work
# never has a direct route out.
task_internet="$(jq -r '.internet' "${TASKS_DIR}/${TASK_ID}.json")"
case "${task_internet}" in
  true) sidecar_public_egress="--allow-public-egress"; codex_web_search="live" ;;
  false) sidecar_public_egress=""; codex_web_search="disabled" ;;
  *) die "task JSON internet must be true or false, got: ${task_internet}" ;;
esac
# Read by scripts/sforge_isolated_entry.py when it builds the Codex command
# and prompt.
export MLSYSFORGE_WORK_INTERNET="${task_internet}"

if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  RUN_ID="${RUN_ID:-standalone-sparse-mla-v2-${execution_profile_id}-${model_slug}-${REASONING_EFFORT}-${budget_minutes}m-${RUN_STAMP}}"
else
  RUN_ID="${RUN_ID:-standalone-sparse-mla-v2-h100-${model_slug}-${REASONING_EFFORT}-${budget_minutes}m-${RUN_STAMP}}"
fi
RUN_LOG_ROOT="${LOGS_DIR}/runs/${RUN_ID}"
TASK_RUN_ROOT="${RUN_LOG_ROOT}/${TASK_ID}"
WORK_CONTAINER_NAME="sforge.run.${TASK_ID}.${RUN_ID}"

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
[[ "${runtime_agent}" == "${AGENT_NAME}" ]] \
  || die "formal ${AGENT_NAME} run requires a ${AGENT_NAME} runtime, found: ${runtime_agent}"
if [[ "${AGENT_NAME}" == "codex" ]]; then
  [[ "${runtime_version}" == "${CODEX_CLI_VERSION}" ]] \
    || die "agent runtime version ${runtime_version} differs from the pinned ${CODEX_CLI_VERSION}"
else
  # No Codex CLI in a Claude Code run; the runtime marker pins Claude Code.
  CODEX_CLI_VERSION=""
fi
export SFORGE_AGENT_RUNTIME_DIR="${AGENT_RUNTIME_DIR}"

assert_built_images_current

# The Judge server runs the mandatory LLM integrity review (docs/LLM_JUDGE.md)
# from the SFORGE_LLM_JUDGE_* environment; refuse anything but a ready,
# enforcing configuration before any lock, container or GPU work.
[[ "${SFORGE_LLM_JUDGE_MODE:-}" == "enforce" ]] \
  || die "set SFORGE_LLM_JUDGE_MODE=enforce and the SFORGE_LLM_JUDGE_* reviewer settings"
review_summary="$(
  cd -- "${REPOSITORY_ROOT}" && "${PYTHON_BIN}" -B - <<'PY' 2>&1
from sforge.harness.llm_judge.config import load_llm_judge_config
config = load_llm_judge_config()
config.check_ready()
print(f"{config.mode} {config.engine} {config.agent} {config.model} "
      f"effort={config.reasoning_effort} runtime={config.agent_runtime_dir}")
PY
)" || die "integrity review is not ready: ${review_summary##*$'\n'}"
echo "Integrity review: ${review_summary}"

# SFORGE_LAUNCH_DRY_RUN=1 (launch/<task>/run.sh --dry-run): print the resolved
# plan and stop here -- no GPU lock, container, network or Judge server.
if [[ "${SFORGE_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  dry_failures=()
  [[ ! -e "${RUN_LOG_ROOT}" ]] || dry_failures+=("run directory already exists: ${RUN_LOG_ROOT}")
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${JUDGE_PORT}$"; then
    dry_failures+=("judge port ${JUDGE_PORT} is already in use")
  fi
  if [[ -n "${MODEL_API_BASE_URL}" ]]; then
    dry_credential="model API ${MODEL_API_BASE_URL} (key file ${MODEL_API_KEY_FILE})"
  elif [[ "${AGENT_NAME}" == "claude-code" ]]; then
    dry_credential="Claude OAuth token file ${CLAUDE_TOKEN_FILE}"
  else
    dry_credential="Codex auth ${AUTH_JSON}"
  fi
  dry_gpu="${WORK_GPU}/${JUDGE_GPU}"
  if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
    dry_gpu="${work_gpu_uuid} (Work and Judge, one lease; lock $(gpu_lock_path "${work_gpu_uuid}"))"
    dry_used="$(nvidia-smi --id="${work_gpu_uuid}" --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | xargs)" \
      || dry_failures+=("nvidia-smi cannot see ${work_gpu_uuid}")
    dry_gpu+=" [index,used: ${dry_used:-?}]"
  fi
  echo "Dry run (no lock, container or GPU work): ${RUN_ID}"
  echo "  agent:        ${AGENT_NAME} ${MODEL} effort=${REASONING_EFFORT} runtime ${AGENT_RUNTIME_DIR} (${runtime_version})"
  echo "  credential:   ${dry_credential}"
  echo "  budget:       ${TASK_TIMEOUT}s; auto-eval off; manual submit cooldown ${EVAL_INTERVAL}s; final Judge timeout ${FINAL_EVAL_TIMEOUT}s"
  echo "  GPU:          ${dry_gpu}"
  echo "  judge port:   ${JUDGE_PORT}"
  echo "  network:      internet=${task_internet} web_search=${codex_web_search}"
  echo "  version store: ${SFORGE_VERSION_STORE_ROOT:-<unset>}"
  echo "  cpuset:       work=${SFORGE_WORK_CPUSET:-${SFORGE_CPUSET:-off}} judge=${SFORGE_JUDGE_CPUSET:-${SFORGE_CPUSET:-off}}"
  echo "  logs:         ${RUN_LOG_ROOT}"
  if (( ${#dry_failures[@]} )); then
    printf '  CHECK FAILED: %s\n' "${dry_failures[@]}"
    exit 1
  fi
  echo "  checks:       PASS"
  exit 0
fi

if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  # One whole-GPU lease for the entire run: Work, every interactive Judge and
  # the standalone final Judge serialize on this UUID-keyed flock.
  [[ "${work_gpu_uuid}" =~ ^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]] \
    || die "SFORGE_ACCELERATOR_DEVICE_UUID must be a qualified canonical whole-GPU UUID"
  exec 8>"$(gpu_lock_path "${work_gpu_uuid}")"
  flock -n 8 || die "single GPU allocation ${work_gpu_uuid} lock is already held"
  if ! "${PYTHON_BIN}" - "${work_gpu_uuid}" <<'PY'
import json
import sys

from sforge.harness.accelerator_observer import NvmlAcceleratorObserver
from sforge.harness.backend import AcceleratorAllocation

allocation = AcceleratorAllocation(vendor="nvidia", device_ids=(sys.argv[1],))
attestation = NvmlAcceleratorObserver().attest_idle(
    allocation,
    phase="before_work_create",
)
print("Initial accelerator idle attestation: " + json.dumps(
    attestation.to_dict(), sort_keys=True, separators=(",", ":")
))
PY
  then
    die "production NVML observer could not attest the single GPU idle before Work creation"
  fi
  assert_portable_gpu_idle \
    "${work_gpu_uuid}" \
    "single GPU" \
    "${profile_vendor}" \
    "${profile_count}" \
    "${profile_min_compute_capability}" \
    "${profile_min_memory_bytes}"
  assert_container_accelerator "${WORK_IMAGE}" "${work_gpu_uuid}" "work"
  assert_container_accelerator "${JUDGE_IMAGE}" "${judge_gpu_uuid}" "judge"
else
  [[ "$(docker info --format '{{.DefaultRuntime}}')" == "nvidia" ]] \
    || die "the legacy local GPU adapter requires Docker default runtime=nvidia"
  WORK_GPU="$(canonical_gpu_index "${WORK_GPU}")"
  JUDGE_GPU="$(canonical_gpu_index "${JUDGE_GPU}")"
  work_gpu_uuid="$(gpu_uuid "${WORK_GPU}")"
  judge_gpu_uuid="$(gpu_uuid "${JUDGE_GPU}")"
  [[ "${work_gpu_uuid}" != "${judge_gpu_uuid}" ]] \
    || die "Work and Judge must use different physical GPUs"
  exec 8>"$(gpu_lock_path "${work_gpu_uuid}")"
  exec 9>"$(gpu_lock_path "${judge_gpu_uuid}")"
  flock -n 8 || die "Work GPU ${WORK_GPU} lock is already held"
  flock -n 9 || die "Judge GPU ${JUDGE_GPU} lock is already held"
  assert_gpu_idle "${WORK_GPU}" "Work"
  assert_gpu_idle "${JUDGE_GPU}" "Judge"
  assert_container_gpu "${WORK_IMAGE}" "${work_gpu_uuid}" "work"
  assert_container_gpu "${JUDGE_IMAGE}" "${judge_gpu_uuid}" "judge"
fi

if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  handoff_mode="single_allocation_serial_v1"
  scheduling_mode="single_gpu_lease_shared_final_v3_handoff"
  allocation_key="${SFORGE_ACCELERATOR_ALLOCATION_KEY}"
  trusted_lease_id="${SFORGE_TRUSTED_LEASE_ID}"
else
  handoff_mode="legacy_dual_gpu"
  scheduling_mode="dual_h100_shared_final_v3"
  allocation_key=""
  trusted_lease_id=""
fi

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
  || die "unsafe RUN_ID: ${RUN_ID}"
[[ ! -e "${RUN_LOG_ROOT}" ]] \
  || die "run directory already exists: ${RUN_LOG_ROOT}"
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${JUDGE_PORT}$"; then
  die "judge port ${JUDGE_PORT} is already in use"
fi

mkdir -p "${RUN_LOG_ROOT}"
judge_pid=""
active_pid=""
final_judge_container_name=""
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
  if kill -0 "${active_pid}" 2>/dev/null; then
    kill -KILL "${active_pid}" 2>/dev/null || true
  fi
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
  if [[ -n "${final_judge_container_name}" ]]; then
    docker rm -f "${final_judge_container_name}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${restricted_proxy_name}" ]]; then
    docker rm -f "${restricted_proxy_name}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${isolated_network}" ]]; then
    docker network rm "${isolated_network}" >/dev/null 2>&1 || true
  fi
}

handle_signal() {
  local signal_name="$1"
  local exit_code=143
  [[ "${signal_name}" == "INT" ]] && exit_code=130
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

write_command_json_atomically() {
  local output_file="$1"
  shift
  local temporary_file="${output_file}.tmp.$$"
  local command_rc=0
  if "$@" >"${temporary_file}"; then
    mv -f -- "${temporary_file}" "${output_file}"
    return 0
  else
    command_rc="$?"
  fi
  rm -f -- "${temporary_file}"
  return "${command_rc}"
}

trap cleanup EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

export SFORGE_TASKS_DIR="${TASKS_DIR}"
export SFORGE_LOG_DIR="${LOGS_DIR}"
export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${judge_gpu_uuid},CUDA_VISIBLE_DEVICES=0,PYTHONDONTWRITEBYTECODE=1"
export SFORGE_ADMIN_SECRET
SFORGE_ADMIN_SECRET="$("${PYTHON_BIN}" -c 'import secrets; print(secrets.token_urlsafe(48))')"

# In single-allocation mode the coordinator inherits SFORGE_SINGLE_GPU_HANDOFF
# and the allocation/lease identities exported by the wrapper; it owns the
# Work pause -> NVML idle attestation -> Judge -> destroy -> attest -> resume
# cycle for every interactive submission and the standalone final Judge.
# Host-local Judge control must bypass any ambient outbound proxy.
bridge_gateway="$(docker network inspect bridge | jq -er '.[0].IPAM.Config[0].Gateway')"
[[ "${bridge_gateway}" =~ ^[0-9]+(\.[0-9]+){3}$ ]] \
  || die "could not resolve the Docker bridge gateway"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${bridge_gateway},127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
# The coordinator binds the Docker bridge gateway for the Work sidecar's
# Judge relay and loopback for the standalone-final handoff client; no
# public interface is ever bound.
"${SFORGE_BIN}" serve --host "${bridge_gateway}" --host 127.0.0.1 --port "${JUDGE_PORT}" \
  >"${RUN_LOG_ROOT}/judge-server.log" 2>&1 &
judge_pid="$!"
wait_for_http "http://${bridge_gateway}:${JUDGE_PORT}/openapi.json" 90
wait_for_http "http://127.0.0.1:${JUDGE_PORT}/openapi.json" 30

# SForge's stock Docker isolation needs passwordless host iptables. This
# host deliberately does not grant it, so use a rootless alternative: the
# Work container has only an internal Docker network and reaches the outside
# solely through a CONNECT sidecar -- the Codex auth/model allowlist, plus
# (internet=true) any globally routable host on 80/443, never an internal
# address. Judge traffic uses a separate fixed-target TCP relay (port 3130)
# on that sidecar.
isolated_network="${RUN_ID}-internal"
restricted_proxy_name="${RUN_ID}-restricted-proxy"
docker network create \
  --driver bridge \
  --internal \
  --label "mlsysforge.run_id=${RUN_ID}" \
  "${isolated_network}" >/dev/null
docker run --detach --rm \
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
docker network connect \
  --alias runtime-proxy \
  "${isolated_network}" \
  "${restricted_proxy_name}"
restricted_proxy_ip="$(
  docker inspect "${restricted_proxy_name}" \
    | jq -er --arg network "${isolated_network}" \
      '.[0].NetworkSettings.Networks[$network].IPAddress
       | select(type == "string" and length > 0)'
)"
# SForge's host-side session registration posts to the restricted proxy IP.
export NO_PROXY="${restricted_proxy_ip},${NO_PROXY}"
export no_proxy="${NO_PROXY}"

# Fail-closed network preflight from inside the internal network: the
# sidecar answers, the Judge relay reaches the coordinator, there is no
# direct public egress, an allowlisted CONNECT succeeds, and either a
# non-allowlisted one is refused with 403 or (--allow-public-egress) a public
# host is tunnelled while the Docker bridge gateway is refused with 403.
for _ in $(seq 1 30); do
  if docker run --rm \
      --network "${isolated_network}" \
      --env NVIDIA_VISIBLE_DEVICES=void \
      --entrypoint curl \
      "${WORK_IMAGE}" \
      --connect-timeout 2 \
      --max-time 3 \
      -fsS "http://${restricted_proxy_ip}:3128/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker run --rm \
  --network "${isolated_network}" \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint curl \
  "${WORK_IMAGE}" \
  --connect-timeout 2 \
  --max-time 5 \
  -fsS "http://${restricted_proxy_ip}:3128/health" >/dev/null \
  || die "restricted proxy sidecar is not answering on the internal network"
docker run --rm \
  --network "${isolated_network}" \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint curl \
  "${WORK_IMAGE}" \
  --connect-timeout 2 \
  --max-time 5 \
  -fsS "http://${restricted_proxy_ip}:3130/openapi.json" >/dev/null \
  || die "Judge relay on the restricted proxy does not reach the coordinator"
if docker run --rm \
    --network "${isolated_network}" \
    --env NVIDIA_VISIBLE_DEVICES=void \
    --entrypoint curl \
    "${WORK_IMAGE}" \
    --connect-timeout 2 \
    --max-time 4 \
    -fsS https://example.com >/dev/null 2>&1; then
  die "internal work network unexpectedly has direct public egress"
fi
docker run --rm \
  --network "${isolated_network}" \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint python3 \
  "${WORK_IMAGE}" \
  -c \
  'import socket,sys
p=sys.argv[1]
def probe(host, expected):
    with socket.create_connection((p,3128),5) as s:
        request=f"CONNECT {host}:443 HTTP/1.1\r\nHost: {host}:443\r\n\r\n"
        s.sendall(request.encode("ascii"))
        line=s.recv(256).split(b"\r\n",1)[0]
        if expected not in line:
            raise SystemExit(f"{host}: expected {expected!r}, got {line!r}")
probe("chatgpt.com", b" 200 ")
if sys.argv[2] == "true":
    probe("github.com", b" 200 ")
    probe(sys.argv[3], b" 403 ")
else:
    probe("github.com", b" 403 ")' \
  "${restricted_proxy_ip}" "${task_internet}" "${bridge_gateway}" \
  || die "restricted proxy allowlist preflight failed"
if [[ "${task_internet}" == "true" ]]; then
  echo "Public-egress network: PASS (${isolated_network}, proxy ${restricted_proxy_ip}; no direct egress; CONNECT to public hosts on 80/443; internal addresses refused)"
else
  echo "Restricted network: PASS (${isolated_network}, proxy ${restricted_proxy_ip})"
fi

unset \
  SFORGE_AGENT_API_KEY \
  SFORGE_AGENT_API_BASE_URL \
  SFORGE_AGENT_MODEL \
  SFORGE_AGENT_TIMEOUT \
  CODEX_API_KEY \
  OPENAI_API_KEY \
  OPENAI_BASE_URL
if [[ -n "${MODEL_API_BASE_URL}" ]]; then
  export SFORGE_AGENT_API_BASE_URL="http://${restricted_proxy_ip}:${MODEL_FORWARD_PORT}${model_api_path}"
  export SFORGE_AGENT_API_KEY="${MODEL_API_KEY}"
elif [[ "${AGENT_NAME}" == "claude-code" ]]; then
  unset SFORGE_CODEX_AUTH_JSON_PATH
  export MLSYSFORGE_CLAUDE_OAUTH_TOKEN_FILE="${CLAUDE_TOKEN_FILE}"
else
  export SFORGE_CODEX_AUTH_JSON_PATH="${AUTH_JSON}"
fi
# Read by sforge_isolated_entry.py (claude_local.install_from_environment).
export MLSYSFORGE_AGENT="${AGENT_NAME}" MLSYSFORGE_AGENT_MODEL="${MODEL}"
export MLSYSFORGE_AGENT_EFFORT="${REASONING_EFFORT}" MLSYSFORGE_AGENT_BUDGET_SECONDS="${TASK_TIMEOUT}"
export MLSYSFORGE_RUN_LOG_ROOT="${RUN_LOG_ROOT}"
# The isolated entry refuses any Work network that is not this internal one;
# the Work container receives only the restricted sidecar as its proxy.
export SFORGE_ISOLATED_DOCKER_NETWORK="${isolated_network}"
export SFORGE_HTTP_PROXY="http://${restricted_proxy_ip}:3128"
export SFORGE_HTTPS_PROXY="http://${restricted_proxy_ip}:3128"
export SFORGE_NO_PROXY="${restricted_proxy_ip},127.0.0.1,localhost"
export SFORGE_CODEX_REASONING_EFFORT="${REASONING_EFFORT}"
export SFORGE_CODEX_CLI_VERSION="${CODEX_CLI_VERSION}"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  # sforge-submit runs the root-owned cooperative drain hook before archive
  # and HTTP admission; the host coordinator still pauses Work and attests.
  export SFORGE_AGENT_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${work_gpu_uuid},CUDA_VISIBLE_DEVICES=0,PYTHONDONTWRITEBYTECODE=1,SFORGE_PRE_SUBMIT_HOOK=/app/workdir/sparse_mla_prefill_lab/.sforge/prepare_submit.sh"
else
  export SFORGE_AGENT_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${work_gpu_uuid},CUDA_VISIBLE_DEVICES=0,PYTHONDONTWRITEBYTECODE=1"
fi

isolated_entry_sha256="$(sha256sum "${ISOLATED_ENTRY}" | awk '{print $1}')"
restricted_proxy_sha256="$(sha256sum "${RESTRICTED_PROXY_SOURCE}" | awk '{print $1}')"
work_image_id="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")"
judge_image_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
image_digest="$(image_source_digest)"
image_manifest="${LOGS_DIR}/build/image-manifest.json"
final_cases_sha256="$(jq -er '.final_cases.sha256' "${image_manifest}")"
final_cases_suite_id="$(jq -er '.final_cases.suite_id' "${image_manifest}")"
final_cases_count="$(jq -er '.final_cases.case_count' "${image_manifest}")"
edge_cases_sha256="$(jq -er '.edge_cases.sha256' "${image_manifest}")"
edge_cases_suite_id="$(jq -er '.edge_cases.suite_id' "${image_manifest}")"
edge_cases_count="$(jq -er '.edge_cases.case_count' "${image_manifest}")"

# Two deliberately separate identities.  deployment_attestation records the
# physical UUID and observed clocks for audit only.  measurement_profile hashes
# the stable comparability key: model / compute capability / memory / driver /
# exact image IDs / software ABI / calibration protocol / campaign epoch.  The
# UUID and observed clocks never enter that key.
measurement_gpu_name="$(gpu_name "${judge_gpu_uuid}")"
measurement_driver_version="$(
  nvidia-smi --id="${judge_gpu_uuid}" \
    --query-gpu=driver_version --format=csv,noheader | xargs
)"
measurement_compute_capability="$(
  nvidia-smi --id="${judge_gpu_uuid}" \
    --query-gpu=compute_cap --format=csv,noheader | xargs
)"
measurement_memory_mib="$(
  nvidia-smi --id="${judge_gpu_uuid}" \
    --query-gpu=memory.total --format=csv,noheader,nounits | xargs
)"
measurement_sm_clock_mhz="$(
  nvidia-smi --id="${judge_gpu_uuid}" \
    --query-gpu=clocks.current.sm --format=csv,noheader,nounits | xargs
)"
measurement_memory_clock_mhz="$(
  nvidia-smi --id="${judge_gpu_uuid}" \
    --query-gpu=clocks.current.memory --format=csv,noheader,nounits | xargs
)"
calibration_epoch="${SFORGE_CALIBRATION_EPOCH:-}"
[[ -z "${calibration_epoch}" \
   || "${calibration_epoch}" =~ ^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$ ]] \
  || die "SFORGE_CALIBRATION_EPOCH must be a label-safe campaign identity"
if [[ -n "${calibration_epoch}" ]]; then
  qualification_status="qualified_campaign_epoch"
else
  qualification_status="unqualified_missing_calibration_epoch"
fi
deployment_attestation_json="$(
  jq -cn \
    --arg gpu_uuid "${judge_gpu_uuid}" \
    --arg gpu_name "${measurement_gpu_name}" \
    --arg sm_clock_mhz "${measurement_sm_clock_mhz}" \
    --arg memory_clock_mhz "${measurement_memory_clock_mhz}" \
    '{
      accelerator_uuid: $gpu_uuid,
      accelerator_name: $gpu_name,
      observed_sm_clock_mhz: $sm_clock_mhz,
      observed_memory_clock_mhz: $memory_clock_mhz
    }'
)"
measurement_profile_json="$(
  jq -cn \
    --arg gpu_name "${measurement_gpu_name}" \
    --arg driver_version "${measurement_driver_version}" \
    --arg compute_capability "${measurement_compute_capability}" \
    --arg memory_mib "${measurement_memory_mib}" \
    --arg work_image_id "${work_image_id}" \
    --arg judge_image_id "${judge_image_id}" \
    --arg software_abi_sha256 "${software_abi_sha256}" \
    --arg calibration_protocol_sha256 "${calibration_protocol_sha256}" \
    --arg calibration_epoch "${calibration_epoch}" \
    --arg qualification_status "${qualification_status}" \
    '{
      accelerator: {
        model: $gpu_name,
        driver_version: $driver_version,
        compute_capability: $compute_capability,
        memory_mib: $memory_mib
      },
      images: {work: $work_image_id, judge: $judge_image_id},
      software_abi_sha256: $software_abi_sha256,
      calibration_protocol_sha256: $calibration_protocol_sha256,
      calibration_epoch: $calibration_epoch,
      qualification_status: $qualification_status
    }'
)"
measurement_profile_sha256="sha256:$(
  printf '%s' "${measurement_profile_json}" | sha256sum | awk '{print $1}'
)"
raw_score_comparison_identity_json="$(
  jq -cn \
    --arg evaluation_contract_sha256 "${evaluation_contract_sha256}" \
    --arg measurement_profile_sha256 "${measurement_profile_sha256}" \
    '{
      evaluation_contract_sha256: $evaluation_contract_sha256,
      measurement_profile_sha256: $measurement_profile_sha256
    }'
)"
raw_score_comparison_identity_sha256="sha256:$(
  printf '%s' "${raw_score_comparison_identity_json}" \
    | sha256sum | awk '{print $1}'
)"

experiment_manifest="${RUN_LOG_ROOT}/experiment-manifest.json"
jq -n \
  --arg run_id "${RUN_ID}" \
  --arg started_at "$(date -u +%FT%TZ)" \
  --arg task_id "${TASK_ID}" \
  --arg agent "${AGENT_NAME}" \
  --arg model "${MODEL}" \
  --arg effort "${REASONING_EFFORT}" \
  --arg cli_version "${CODEX_CLI_VERSION}" \
  --arg runtime_dir "${AGENT_RUNTIME_DIR}" \
  --arg runtime_agent "${runtime_agent}" \
  --arg runtime_version "${runtime_version}" \
  --arg runtime_tree_sha256 "${runtime_tree_sha256}" \
  --arg work_gpu_index "${WORK_GPU}" \
  --arg work_gpu_uuid "${work_gpu_uuid}" \
  --arg judge_gpu_index "${JUDGE_GPU}" \
  --arg judge_gpu_uuid "${judge_gpu_uuid}" \
  --arg network_mode "docker_internal_restricted_connect_proxy" \
  --arg isolated_entry_sha256 "${isolated_entry_sha256}" \
  --arg restricted_proxy_sha256 "${restricted_proxy_sha256}" \
  --arg codex_web_search "${codex_web_search}" \
  --argjson public_connect_egress "${task_internet}" \
  --arg work_image_id "${work_image_id}" \
  --arg judge_image_id "${judge_image_id}" \
  --arg image_source_digest "${image_digest}" \
  --arg final_cases_sha256 "${final_cases_sha256}" \
  --arg final_cases_suite_id "${final_cases_suite_id}" \
  --arg edge_cases_sha256 "${edge_cases_sha256}" \
  --arg edge_cases_suite_id "${edge_cases_suite_id}" \
  --arg scheduling_mode "${scheduling_mode}" \
  --arg handoff_mode "${handoff_mode}" \
  --arg allocation_key "${allocation_key}" \
  --arg trusted_lease_id "${trusted_lease_id}" \
  --arg evaluation_contract_sha256 "${evaluation_contract_sha256}" \
  --arg execution_profile_id "${execution_profile_id}" \
  --arg measurement_profile_sha256 "${measurement_profile_sha256}" \
  --arg raw_score_comparison_identity_sha256 "${raw_score_comparison_identity_sha256}" \
  --argjson raw_score_comparison_identity "${raw_score_comparison_identity_json}" \
  --arg qualification_status "${qualification_status}" \
  --arg metric_unit "${metric_unit}" \
  --argjson measurement_profile "${measurement_profile_json}" \
  --argjson deployment_attestation "${deployment_attestation_json}" \
  --argjson timeout "${TASK_TIMEOUT}" \
  --argjson eval_interval "${EVAL_INTERVAL}" \
  --argjson final_cases_count "${final_cases_count}" \
  --argjson edge_cases_count "${edge_cases_count}" \
  '{
    schema_version: 3,
    run_id: $run_id,
    task_id: $task_id,
    started_at: $started_at,
    agent: $agent,
    model: $model,
    reasoning_effort: $effort,
    codex_cli_version: $cli_version,
    agent_runtime: {
      dir: $runtime_dir,
      agent: $runtime_agent,
      version: $runtime_version,
      tree_sha256: $runtime_tree_sha256
    },
    codex_web_search: $codex_web_search,
    timeout_seconds: $timeout,
    eval_interval_seconds: $eval_interval,
    scheduled_auto_evaluation: false,
    agent_submission_cooldown_seconds: $eval_interval,
    manual_evaluation_suite: "final-holdout-v4",
    final_evaluation: "independent_timeout_boundary_shared_final",
    evaluation_policy: "full-final-shared-v3",
    selection_protocol_id: "sparse-mla-shared-final-v3",
    final_holdout: {
      suite_id: $final_cases_suite_id,
      manifest_sha256: $final_cases_sha256,
      case_count: $final_cases_count
    },
    edge_holdout: {
      suite_id: $edge_cases_suite_id,
      manifest_sha256: $edge_cases_sha256,
      case_count: $edge_cases_count
    },
    score_unit: "effective_tflops",
    scheduling: $scheduling_mode,
    resource_handoff: {
      mode: $handoff_mode,
      allocation_key: $allocation_key,
      trusted_lease_id: $trusted_lease_id,
      evaluation_contract_sha256: $evaluation_contract_sha256,
      execution_profile_id: $execution_profile_id
    },
    deployment_attestation: $deployment_attestation,
    measurement_profile: ($measurement_profile + {
      comparability_key_sha256: $measurement_profile_sha256,
      raw_score_comparability: "requires_qualified_pair_identity"
    }),
    raw_score_comparison: ($raw_score_comparison_identity + {
      identity_sha256: $raw_score_comparison_identity_sha256,
      metric_unit: $metric_unit,
      policy: "same_evaluation_contract_and_measurement_profile_only",
      qualification_status: $qualification_status
    }),
    network: {
      mode: $network_mode,
      direct_public_egress: false,
      public_connect_egress: $public_connect_egress,
      connect_allowlist: [
        "api.openai.com:443",
        "auth.openai.com:443",
        "chatgpt.com:443"
      ],
      isolated_entry_sha256: $isolated_entry_sha256,
      restricted_proxy_sha256: $restricted_proxy_sha256
    },
    work_gpu: {index: $work_gpu_index, uuid: $work_gpu_uuid},
    judge_gpu: {index: $judge_gpu_index, uuid: $judge_gpu_uuid},
    images: {
      work_id: $work_image_id,
      judge_id: $judge_image_id,
      source_digest: $image_source_digest
    }
  }' >"${RUN_LOG_ROOT}/experiment-manifest.json"
[[ -s "${experiment_manifest}" ]] \
  || die "experiment manifest was not written"
experiment_manifest_sha256="sha256:$(
  sha256sum "${experiment_manifest}" | awk '{print $1}'
)"

echo "Starting ${RUN_ID}"
echo "  agent: ${AGENT_NAME}; model: ${MODEL} (${REASONING_EFFORT})"
echo "  Work budget: ${TASK_TIMEOUT}s; autojudge disabled; manual Judge cooldown ${EVAL_INTERVAL}s"
echo "  scheduling: ${scheduling_mode}"
echo "  Work GPU: ${WORK_GPU} (${work_gpu_uuid})"
echo "  Judge GPU: ${JUDGE_GPU} (${judge_gpu_uuid})"
echo "  Judge mode: manual submissions only"
echo "  network: docker_internal_restricted_connect_proxy (${isolated_network} via ${restricted_proxy_ip}; public CONNECT egress ${task_internet}; web search ${codex_web_search})"
echo "  final Judge: ${final_cases_suite_id} (${final_cases_sha256:0:12}) + ${edge_cases_suite_id} (${edge_cases_sha256:0:12})"
echo "  measurement profile: ${measurement_profile_sha256:7:12} (${qualification_status})"
echo "  logs: ${RUN_LOG_ROOT}"

# The isolated entry places Work on the internal network above; Codex web
# search and the evolve prompt follow MLSYSFORGE_WORK_INTERNET (exported above).
# --enable-internet only bypasses SForge's host-iptables implementation
# (unavailable here); the network itself has no direct egress.
set +e
run_logged_command "${RUN_LOG_ROOT}/sforge-run.log" \
  "${PYTHON_BIN}" "${ISOLATED_ENTRY}" run \
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
  --work-mem-limit 32g \
  --judge-cpu-limit 8 \
  --judge-mem-limit 32g \
  --run-id "${RUN_ID}"
run_rc="$?"
set -e
(( run_rc == 0 )) || die "SForge run failed with exit ${run_rc}"

final_archive="${TASK_RUN_ROOT}/final_archive.tar.gz"
final_result="${TASK_RUN_ROOT}/final_result.json"
final_snapshot="${TASK_RUN_ROOT}/final_snapshot.json"
run_history="${TASK_RUN_ROOT}/run_history.json"
for required in \
  "${final_archive}" \
  "${final_result}" \
  "${final_snapshot}" \
  "${run_history}"; do
  [[ -s "${required}" ]] \
    || die "SForge run ended without nonempty artifact: ${required}"
done

# The Work session is complete when SForge returns.  Prove Work is gone and
# the locked allocation is idle before the final evaluation.  In legacy mode
# the control server is removed here; in single-allocation mode it stays up
# because the standalone final Judge is performed by that same coordinator
# under the still-held lease.
if [[ "${SINGLE_GPU_HANDOFF}" != "1" ]]; then
  stop_judge_server
fi
if docker container inspect "${WORK_CONTAINER_NAME}" >/dev/null 2>&1; then
  die "Work container still exists at the final-evaluation barrier"
fi
assert_gpu_released "${work_gpu_uuid}"
assert_gpu_released "${judge_gpu_uuid}"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  assert_portable_gpu_idle \
    "${judge_gpu_uuid}" \
    "final Judge" \
    "${profile_vendor}" \
    "${profile_count}" \
    "${profile_min_compute_capability}" \
    "${profile_min_memory_bytes}"
else
  assert_gpu_idle "${JUDGE_GPU}" "final Judge"
fi

# The experiment is recorded at run scope; stage its exact, digest-bound bytes
# inside the task root before the validator reads any per-round attestations.
"${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_formal_artifacts.py" stage \
  --task-root "${TASK_RUN_ROOT}" \
  --source-manifest "${experiment_manifest}" \
  --expected-sha256 "${experiment_manifest_sha256}"
formal_precheck="${TASK_RUN_ROOT}/formal-precheck.json"
write_command_json_atomically \
  "${formal_precheck}" \
  "${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_formal_artifacts.py" pre \
  --task-root "${TASK_RUN_ROOT}" \
  --run-id "${RUN_ID}" \
  --task-id "${TASK_ID}" \
  --agent "${AGENT_NAME}" \
  --model "${MODEL}" \
  --budget-seconds "${TASK_TIMEOUT}" \
  --interval-seconds "${EVAL_INTERVAL}" \
  --evaluation-mode "${scheduling_mode}"
final_archive_sha256="$(jq -er '.final_archive.sha256' "${formal_precheck}")"
auto_evaluations="$(jq -er '.active_run_hidden_evaluations' "${formal_precheck}")"
echo "Formal precheck: PASS (archive sha256=${final_archive_sha256})"

final_eval_id="${RUN_ID}-final-${final_minutes}m"
final_judge_container_name="${BENCHMARK_NAME}.judge.${TASK_ID}.${final_eval_id}"
[[ ! -e "${LOGS_DIR}/runs/${final_eval_id}" ]] \
  || die "final eval directory already exists: ${final_eval_id}"

final_handoff_args=()
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  # This explicit trusted CLI capability is required: global handoff config
  # alone must never reinterpret an ordinary replay/eval as the final owner.
  final_handoff_args+=(
    --standalone-final-handoff
    --handoff-source-run-id "${RUN_ID}"
    --judge-url "http://127.0.0.1:${JUDGE_PORT}"
  )
fi
set +e
run_logged_command "${TASK_RUN_ROOT}/final-${final_minutes}m-eval.log" \
  "${SFORGE_BIN}" eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${final_archive}" \
  --run-id "${final_eval_id}" \
  --timeout "${FINAL_EVAL_TIMEOUT}" \
  --judge-cpu-limit 8 \
  --judge-mem-limit 32g \
  "${final_handoff_args[@]}" \
  --json
final_eval_rc="$?"
set -e
(( final_eval_rc == 0 )) \
  || die "independent final evaluation failed with exit ${final_eval_rc}"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  stop_judge_server
fi
assert_gpu_released "${judge_gpu_uuid}"

final_task_root="${LOGS_DIR}/runs/${final_eval_id}/${TASK_ID}"
formal_postcheck="${TASK_RUN_ROOT}/formal-postcheck.json"
completion="${RUN_LOG_ROOT}/completion.json"

# Persist a durable record of a rejected final.  The validator
# still fails closed (non-zero exit); this captures its `invalid_final` record
# into formal-postcheck.json and writes a `failed` completion.json so the
# reason outlives the launcher's stdout, then aborts the launcher exactly as
# before.  Neither record carries evaluated_archive/final_report/private-Judge
# bindings, so nothing downstream can read them as a pass.
record_invalid_final() {
  local postcheck_rc="$1"
  local stdout_file="$2"
  local stderr_file="$3"
  local reason
  reason="$(sed -n 's/^formal artifact validation failed: //p' "${stderr_file}" | head -n 1 || true)"
  [[ -n "${reason}" ]] || reason="$(tail -n 1 "${stderr_file}" || true)"
  [[ -n "${reason}" ]] || reason="formal postcheck exited ${postcheck_rc} without a reason"
  if ! jq -e '.status == "invalid_final" and (.reason | type == "string")' \
      "${stdout_file}" >/dev/null 2>&1; then
    # The validator died before it could emit a record (crash, bad CLI, ...):
    # synthesise the same shape from its stderr.
    jq -n \
      --arg evaluation_mode "${scheduling_mode}" \
      --arg final_eval_id "${final_eval_id}" \
      --arg task_id "${TASK_ID}" \
      --arg expected_archive_sha256 "${final_archive_sha256}" \
      --arg final_task_root "${final_task_root}" \
      --arg reason "${reason}" \
      --argjson exit_code "${postcheck_rc}" \
      '{schema_version:1,evaluation_mode:$evaluation_mode,status:"invalid_final",
        phase:"post",final_eval_id:$final_eval_id,task_id:$task_id,
        expected_archive_sha256:$expected_archive_sha256,final_task_root:$final_task_root,
        reason:$reason,error_type:"launcher_postcheck_failure",validator_exit_code:$exit_code}' \
      >"${stdout_file}"
  fi
  jq --arg stderr "$(cat "${stderr_file}")" --argjson exit_code "${postcheck_rc}" \
    '. + {validator_exit_code:$exit_code, validator_stderr:$stderr}' \
    "${stdout_file}" >"${formal_postcheck}.invalid"
  mv -f -- "${formal_postcheck}.invalid" "${formal_postcheck}"
  rm -f -- "${stdout_file}" "${stderr_file}"
  write_command_json_atomically \
    "${completion}" \
    jq -n \
    --arg finished_at "$(date -u +%FT%TZ)" \
    --arg run_id "${RUN_ID}" \
    --arg task_id "${TASK_ID}" \
    --arg work_gpu_uuid "${work_gpu_uuid}" \
    --arg judge_gpu_uuid "${judge_gpu_uuid}" \
    --arg scheduling_mode "${scheduling_mode}" \
    --arg reason "${reason}" \
    --arg experiment_manifest "${experiment_manifest}" \
    --arg experiment_manifest_sha256 "${experiment_manifest_sha256}" \
    --arg final_archive "${final_archive}" \
    --arg final_archive_sha256 "${final_archive_sha256}" \
    --arg formal_precheck "${formal_precheck}" \
    --arg formal_postcheck "${formal_postcheck}" \
    --argjson sforge_run_exit "${run_rc}" \
    --argjson final_eval_exit "${final_eval_rc}" \
    --argjson formal_postcheck_exit "${postcheck_rc}" \
    --argjson auto_evaluations "${auto_evaluations}" \
    '{
      schema_version: 2,
      status: "failed",
      failure_stage: "formal_postcheck",
      reason: $reason,
      finished_at: $finished_at,
      run_id: $run_id,
      task_id: $task_id,
      work_gpu_uuid: $work_gpu_uuid,
      judge_gpu_uuid: $judge_gpu_uuid,
      scheduling: $scheduling_mode,
      hidden_evaluations_during_work: $auto_evaluations,
      sforge_run_exit: $sforge_run_exit,
      final_eval_exit: $final_eval_exit,
      formal_postcheck_exit: $formal_postcheck_exit,
      experiment_manifest: {
        path: $experiment_manifest,
        sha256: $experiment_manifest_sha256
      },
      final_archive: {path: $final_archive, sha256: $final_archive_sha256},
      validation: {precheck: $formal_precheck, postcheck: $formal_postcheck}
    }' || true
  die "independent final evaluation is invalid (recorded in ${formal_postcheck}): ${reason}"
}

postcheck_stdout="${formal_postcheck}.tmp.$$"
postcheck_stderr="${formal_postcheck}.stderr.$$"
postcheck_rc=0
"${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_formal_artifacts.py" post \
  --task-root "${TASK_RUN_ROOT}" \
  --final-task-root "${final_task_root}" \
  --final-eval-id "${final_eval_id}" \
  --task-id "${TASK_ID}" \
  --expected-archive-sha256 "${final_archive_sha256}" \
  --expected-manifest-sha256 "${final_cases_sha256}" \
  --expected-case-count "${final_cases_count}" \
  --evaluation-mode "${scheduling_mode}" \
  >"${postcheck_stdout}" 2>"${postcheck_stderr}" || postcheck_rc="$?"
if (( postcheck_rc != 0 )); then
  cat -- "${postcheck_stderr}" >&2
  record_invalid_final "${postcheck_rc}" "${postcheck_stdout}" "${postcheck_stderr}"
fi
cat -- "${postcheck_stderr}" >&2
rm -f -- "${postcheck_stderr}"
jq -e '.phase == "post" and .status != "invalid_final"
       and (.final_report.sha256 | type == "string")
       and (.evaluated_archive.sha256 | type == "string")' \
    "${postcheck_stdout}" >/dev/null \
  || die "formal postcheck exited 0 without a bound PASS record"
mv -f -- "${postcheck_stdout}" "${formal_postcheck}"

final_submission_root="${final_task_root}/submissions/manual-1"
final_report="${final_submission_root}/report.json"
final_evaluated_archive="${final_submission_root}/submission.tar.gz"
final_report_sha256="$(jq -er '.final_report.sha256' "${formal_postcheck}")"
evaluated_archive_sha256="$(jq -er '.evaluated_archive.sha256' "${formal_postcheck}")"
private_judge_source="$(jq -er '.private_judge_artifact.path' "${formal_postcheck}")"
private_judge_sha256="$(jq -er '.private_judge_artifact.sha256' "${formal_postcheck}")"
private_judge_suite_id="$(jq -er '.private_judge_artifact.suite_id' "${formal_postcheck}")"
private_judge_manifest_sha256="$(jq -er '.private_judge_artifact.manifest_sha256' "${formal_postcheck}")"
private_judge_case_count="$(jq -er '.private_judge_artifact.case_count' "${formal_postcheck}")"
final_private_judge_artifact="${TASK_RUN_ROOT}/final-${final_minutes}m-private-judge-artifacts.tar.gz"
cp -- "${private_judge_source}" "${final_private_judge_artifact}"
chmod 600 "${final_private_judge_artifact}"
[[ "$(sha256sum "${final_private_judge_artifact}" | awk '{print $1}')" == "${private_judge_sha256}" ]] \
  || die "copied private Judge artifact digest mismatch"

# Freeze the completion attestation only after post-run validation passed.
write_command_json_atomically \
  "${completion}" \
  jq -n \
  --arg finished_at "$(date -u +%FT%TZ)" \
  --arg run_id "${RUN_ID}" \
  --arg task_id "${TASK_ID}" \
  --arg work_gpu_uuid "${work_gpu_uuid}" \
  --arg judge_gpu_uuid "${judge_gpu_uuid}" \
  --arg scheduling_mode "${scheduling_mode}" \
  --arg evaluation_contract_sha256 "${evaluation_contract_sha256}" \
  --arg execution_profile_id "${execution_profile_id}" \
  --arg measurement_profile_sha256 "${measurement_profile_sha256}" \
  --arg raw_score_comparison_identity_sha256 "${raw_score_comparison_identity_sha256}" \
  --arg qualification_status "${qualification_status}" \
  --arg experiment_manifest "${experiment_manifest}" \
  --arg experiment_manifest_sha256 "${experiment_manifest_sha256}" \
  --arg final_archive "${final_archive}" \
  --arg final_archive_sha256 "${final_archive_sha256}" \
  --arg final_report "${final_report}" \
  --arg final_report_sha256 "${final_report_sha256}" \
  --arg evaluated_archive "${final_evaluated_archive}" \
  --arg evaluated_archive_sha256 "${evaluated_archive_sha256}" \
  --arg private_judge_artifact "${final_private_judge_artifact}" \
  --arg private_judge_sha256 "${private_judge_sha256}" \
  --arg private_judge_suite_id "${private_judge_suite_id}" \
  --arg private_judge_manifest_sha256 "${private_judge_manifest_sha256}" \
  --arg formal_precheck "${formal_precheck}" \
  --arg formal_postcheck "${formal_postcheck}" \
  --argjson sforge_run_exit "${run_rc}" \
  --argjson final_eval_exit "${final_eval_rc}" \
  --argjson auto_evaluations "${auto_evaluations}" \
  --argjson private_judge_case_count "${private_judge_case_count}" \
  '{
    schema_version: 2,
    status: "passed",
    finished_at: $finished_at,
    run_id: $run_id,
    task_id: $task_id,
    work_gpu_uuid: $work_gpu_uuid,
    judge_gpu_uuid: $judge_gpu_uuid,
    scheduling: $scheduling_mode,
    hidden_evaluations_during_work: $auto_evaluations,
    sforge_run_exit: $sforge_run_exit,
    final_eval_exit: $final_eval_exit,
    portable_contract: {
      evaluation_contract_sha256: $evaluation_contract_sha256,
      execution_profile_id: $execution_profile_id
    },
    measurement_profile: {
      comparability_key_sha256: $measurement_profile_sha256,
      qualification_status: $qualification_status
    },
    raw_score_comparison: {
      evaluation_contract_sha256: $evaluation_contract_sha256,
      measurement_profile_sha256: $measurement_profile_sha256,
      identity_sha256: $raw_score_comparison_identity_sha256,
      policy: "same_evaluation_contract_and_measurement_profile_only",
      qualification_status: $qualification_status
    },
    experiment_manifest: {
      path: $experiment_manifest,
      sha256: $experiment_manifest_sha256
    },
    final_archive: {path: $final_archive, sha256: $final_archive_sha256},
    final_report: {path: $final_report, sha256: $final_report_sha256},
    evaluated_archive: {
      path: $evaluated_archive,
      sha256: $evaluated_archive_sha256
    },
    private_judge_artifact: {
      path: $private_judge_artifact,
      sha256: $private_judge_sha256,
      suite_id: $private_judge_suite_id,
      manifest_sha256: $private_judge_manifest_sha256,
      case_count: $private_judge_case_count
    },
    validation: {precheck: $formal_precheck, postcheck: $formal_postcheck}
  }'

echo "Completed ${RUN_ID}"
echo "  final report: ${final_report}"
echo "  private Judge artifact: ${final_private_judge_artifact}"
echo "  archive SHA-256: ${final_archive_sha256}"

(( run_rc == 0 && final_eval_rc == 0 ))
