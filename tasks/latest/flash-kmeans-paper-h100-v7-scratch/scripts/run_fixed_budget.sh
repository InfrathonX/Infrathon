#!/usr/bin/env bash
set -euo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_sforge
require_command awk
require_command curl
require_command flock
require_command jq
require_command nvidia-smi
require_command python3
require_command realpath
require_command sha256sum
require_command ss

AUTH_JSON="${CODEX_AUTH_JSON_PATH:-}"
# Optional OpenAI-compatible model API (e.g. a self-hosted DeepSeek).  With
# MODEL_API_BASE_URL set, the restricted sidecar gains one raw TCP relay to
# that endpoint, Codex talks to it in API-key mode (key from the mode-0600
# MODEL_API_KEY_FILE) and MODEL may be any id the endpoint serves.
MODEL_API_BASE_URL="${MODEL_API_BASE_URL:-}"
MODEL_API_KEY_FILE="${MODEL_API_KEY_FILE:-}"
MODEL_FORWARD_PORT=3131
SFORGE_PYTHON="$(dirname "${SFORGE_BIN}")/python"
ISOLATED_ENTRY="${MIGRATION_ROOT}/scripts/sforge_isolated_entry.py"
RESTRICTED_PROXY_SOURCE="${MIGRATION_ROOT}/scripts/restricted_connect_proxy.py"
JUDGE_SERVER_SOURCE="${MLSYSFORGE_ROOT}/sforge/harness/judge_server.py"
MODEL="${MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-0.148.0}"
TASK_TIMEOUT="${TASK_TIMEOUT:-7200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1800}"
SUBMISSION_COOLDOWN="${SUBMISSION_COOLDOWN:-300}"

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
  WORK_GPU="${WORK_GPU:-1}"
  JUDGE_GPU="${JUDGE_GPU:-2}"
fi
JUDGE_PORT="${JUDGE_PORT:-18284}"

if [[ "${SINGLE_GPU_HANDOFF}" == "0" ]]; then
  [[ "${WORK_GPU}" != "${JUDGE_GPU}" ]] \
    || die "work and judge must use different GPUs"
fi
# Any model ID (launch/flash-kmeans/run.sh --model); the run manifest records it.
[[ -n "${MODEL}" ]] || die "fixed run requires MODEL (the agent's model ID)"
[[ "${MODEL}" =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]] || die "invalid MODEL id: ${MODEL}"
case "${REASONING_EFFORT}" in
  high|max|xhigh) ;;
  *) die "fixed run requires REASONING_EFFORT=high, max, or xhigh" ;;
esac
# The Codex version is chosen by the launcher (launch/common.sh pin or an
# explicit CODEX_CLI_VERSION) and must match the mounted runtime below; the
# agent-free Work image does not constrain it.

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
[[ "${runtime_agent}" == "codex" ]] \
  || die "formal run requires a codex runtime, found: ${runtime_agent}"
[[ "${runtime_version}" == "${CODEX_CLI_VERSION}" ]] \
  || die "agent runtime version ${runtime_version} differs from the pinned ${CODEX_CLI_VERSION}"
export SFORGE_AGENT_RUNTIME_DIR="${AGENT_RUNTIME_DIR}"

[[ "${TASK_TIMEOUT}" =~ ^[0-9]+$ \
   && "${EVAL_INTERVAL}" =~ ^[0-9]+$ \
   && "${SUBMISSION_COOLDOWN}" =~ ^[0-9]+$ ]] \
  || die "TASK_TIMEOUT, EVAL_INTERVAL, and SUBMISSION_COOLDOWN must be positive integers"
(( TASK_TIMEOUT > 0 && EVAL_INTERVAL > 0 && SUBMISSION_COOLDOWN > 0 )) \
  || die "TASK_TIMEOUT, EVAL_INTERVAL, and SUBMISSION_COOLDOWN must be positive"
(( TASK_TIMEOUT > EVAL_INTERVAL && TASK_TIMEOUT % 60 == 0 && EVAL_INTERVAL % 60 == 0 )) \
  || die "fixed run requires minute-aligned timeout with at least one pre-boundary tick"
final_minutes="$((TASK_TIMEOUT / 60))"
interval_minutes="$((EVAL_INTERVAL / 60))"
[[ "${JUDGE_PORT}" =~ ^[0-9]+$ ]] || die "invalid JUDGE_PORT: ${JUDGE_PORT}"
JUDGE_PORT="$((10#${JUDGE_PORT}))"
(( JUDGE_PORT >= 1024 && JUDGE_PORT <= 65535 )) \
  || die "invalid unprivileged JUDGE_PORT: ${JUDGE_PORT}"
[[ "${JUDGE_PORT}" != "3128" ]] \
  || die "judge port conflicts with the restricted CONNECT proxy"
MODEL_API_KEY=""; model_api_host=""; model_api_port=""; model_api_path=""
if [[ -n "${MODEL_API_BASE_URL}" ]]; then
  [[ "${MODEL_API_BASE_URL}" =~ ^http://([^/:[:space:]]+):([0-9]+)(/.*)?$ ]] \
    || die "MODEL_API_BASE_URL must be http://HOST:PORT[/path] (relayed as raw TCP by the sidecar)"
  model_api_host="${BASH_REMATCH[1]}"; model_api_port="${BASH_REMATCH[2]}"; model_api_path="${BASH_REMATCH[3]:-}"
  [[ -n "${MODEL_API_KEY_FILE}" ]] || die "set MODEL_API_KEY_FILE with MODEL_API_BASE_URL"
  assert_private_auth_file "${MODEL_API_KEY_FILE}"
  MODEL_API_KEY="$(tr -d '[:space:]' < "${MODEL_API_KEY_FILE}")"
  [[ -n "${MODEL_API_KEY}" ]] || die "MODEL_API_KEY_FILE is empty"
else
  [[ -n "${AUTH_JSON}" ]] \
    || die "set CODEX_AUTH_JSON_PATH to the host-only Codex auth file"
  assert_private_auth_file "${AUTH_JSON}"
fi
[[ -x "${SFORGE_PYTHON}" ]] \
  || die "missing SForge virtualenv Python: ${SFORGE_PYTHON}"
"${SFORGE_PYTHON}" -c \
  'from sforge.harness.judge_server import _add_family_band_trends' \
  || die "SForge lacks persistent manual family-band trend support"
readarray -t portable_identity < <(
  "${SFORGE_PYTHON}" - "${TASKS_DIR}/${TASK_ID}.json" <<'PY'
import json
import sys
from pathlib import Path
from sforge.harness.portable_spec import parse_portable_spec

raw = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
portable = parse_portable_spec(raw["portable"], task_id=raw["task_id"])
profile = portable.execution_profiles[portable.defaults.execution_profile_id]
print(portable.evaluation_contract_sha256)
print(portable.defaults.execution_profile_id)
print(profile.software_abi.spec_sha256)
print(profile.calibration_protocol.spec_sha256)
print(profile.vendor)
print(profile.count)
print(profile.min_memory_bytes)
print(profile.min_compute_capability)
PY
)
(( ${#portable_identity[@]} == 8 )) \
  || die "could not resolve portable contract identity and capability profile"
evaluation_contract_sha256="${portable_identity[0]}"
execution_profile_id="${portable_identity[1]}"
software_abi_sha256="${portable_identity[2]}"
calibration_protocol_sha256="${portable_identity[3]}"
profile_vendor="${portable_identity[4]}"
profile_count="${portable_identity[5]}"
profile_min_memory_bytes="${portable_identity[6]}"
profile_min_compute_capability="${portable_identity[7]}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
model_slug="$(sed -E 's/^gpt-[0-9.]+-//' <<<"${MODEL}" | tr 'A-Z' 'a-z' | sed -E 's/[^a-z0-9.]+/-/g; s/^-+//; s/-+$//')"
budget_minutes="$((TASK_TIMEOUT / 60))"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  RUN_ID="${RUN_ID:-flash-kmeans-v7-scratch-${execution_profile_id}-${model_slug}-${REASONING_EFFORT}-${budget_minutes}m-${RUN_STAMP}}"
else
  RUN_ID="${RUN_ID:-flash-kmeans-v7-scratch-h100-${model_slug}-${REASONING_EFFORT}-${budget_minutes}m-${RUN_STAMP}}"
fi
RUN_LOG_ROOT="${LOGS_DIR}/runs/${RUN_ID}"
TASK_RUN_ROOT="${RUN_LOG_ROOT}/${TASK_ID}"
[[ -f "${ISOLATED_ENTRY}" && ! -L "${ISOLATED_ENTRY}" ]] \
  || die "missing isolated SForge entry: ${ISOLATED_ENTRY}"
[[ -f "${RESTRICTED_PROXY_SOURCE}" && ! -L "${RESTRICTED_PROXY_SOURCE}" ]] \
  || die "missing restricted proxy source: ${RESTRICTED_PROXY_SOURCE}"
[[ -f "${JUDGE_SERVER_SOURCE}" && ! -L "${JUDGE_SERVER_SOURCE}" ]] \
  || die "missing persistent Judge server source: ${JUDGE_SERVER_SOURCE}"
assert_built_images_current

if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  [[ "${work_gpu_uuid}" =~ ^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]] \
    || die "SFORGE_ACCELERATOR_DEVICE_UUID must be a qualified canonical whole-GPU UUID"
  exec 8>"$(gpu_lock_path "${work_gpu_uuid}")"
  flock -n 8 || die "single GPU allocation ${work_gpu_uuid} lock is already held"
  if ! "${SFORGE_PYTHON}" - "${work_gpu_uuid}" <<'PY'
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
    || die "legacy local GPU pinning requires Docker default runtime=nvidia"
  WORK_GPU="$(canonical_gpu_index "${WORK_GPU}")"
  JUDGE_GPU="$(canonical_gpu_index "${JUDGE_GPU}")"
  work_gpu_uuid="$(gpu_uuid "${WORK_GPU}")"
  judge_gpu_uuid="$(gpu_uuid "${JUDGE_GPU}")"
  [[ "${work_gpu_uuid}" != "${judge_gpu_uuid}" ]] \
    || die "work and judge resolve to the same physical GPU: ${work_gpu_uuid}"

  exec 8>"$(gpu_lock_path "${work_gpu_uuid}")"
  exec 9>"$(gpu_lock_path "${judge_gpu_uuid}")"
  flock -n 8 || die "work GPU ${WORK_GPU} lock is already held"
  flock -n 9 || die "judge GPU ${JUDGE_GPU} lock is already held"
  assert_gpu_idle "${WORK_GPU}" "work"
  assert_gpu_idle "${JUDGE_GPU}" "judge"
  assert_container_gpu "${WORK_IMAGE}" "${work_gpu_uuid}" "work"
  assert_container_gpu "${JUDGE_IMAGE}" "${judge_gpu_uuid}" "judge"
fi

if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  handoff_mode="single_allocation_serial_v1"
  allocation_key="${SFORGE_ACCELERATOR_ALLOCATION_KEY}"
  trusted_lease_id="${SFORGE_TRUSTED_LEASE_ID}"
else
  handoff_mode="legacy_dual_gpu"
  allocation_key=""
  trusted_lease_id=""
fi

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
  || die "unsafe RUN_ID: ${RUN_ID}"
[[ ! -e "${RUN_LOG_ROOT}" ]] || die "run directory already exists: ${RUN_LOG_ROOT}"

if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${JUDGE_PORT}$"; then
  die "judge port ${JUDGE_PORT} is already in use"
fi

mkdir -p "${RUN_LOG_ROOT}"
judge_pid=""
active_pid=""
isolated_network=""
restricted_proxy_name=""
cleanup() {
  if [[ -n "${judge_pid}" ]] && kill -0 "${judge_pid}" 2>/dev/null; then
    kill "${judge_pid}" 2>/dev/null || true
    wait "${judge_pid}" 2>/dev/null || true
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
  if [[ -n "${active_pid}" ]] && kill -0 "${active_pid}" 2>/dev/null; then
    kill "-${signal_name}" "${active_pid}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${active_pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${active_pid}" 2>/dev/null; then
      kill -KILL "${active_pid}" 2>/dev/null || true
    fi
    wait "${active_pid}" 2>/dev/null || true
  fi
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
    return
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
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${judge_gpu_uuid},CUDA_VISIBLE_DEVICES=0"
else
  export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${judge_gpu_uuid},CUDA_VISIBLE_DEVICES=0,FLASH_KMEANS_TARGET=H100"
fi
export SFORGE_ADMIN_SECRET
SFORGE_ADMIN_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"

"${SFORGE_BIN}" \
  serve --port "${JUDGE_PORT}" \
  >"${RUN_LOG_ROOT}/judge-server.log" 2>&1 &
judge_pid="$!"
wait_for_http "http://127.0.0.1:${JUDGE_PORT}/openapi.json" 90

# SForge's stock Docker isolation needs passwordless host iptables. This
# host deliberately does not grant it, so use a rootless alternative: the
# work container has only an internal Docker network and reaches the outside
# solely through a CONNECT sidecar -- the model/auth allowlist, plus
# (internet=true) any globally routable host on 80/443, never an internal
# address. Judge traffic uses a separate fixed-target TCP forwarder on that
# same sidecar.
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
  --judge-port "${JUDGE_PORT}" \
  --judge-forward-port "${JUDGE_PORT}" \
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

# A non-global address the public-egress mode must still refuse.
bridge_gateway="$(docker network inspect bridge | jq -er '.[0].IPAM.Config[0].Gateway')"
[[ "${bridge_gateway}" =~ ^[0-9]+(\.[0-9]+){3}$ ]] \
  || die "could not resolve the Docker bridge gateway"

# Preflight from inside the internal network: the sidecar answers, the Judge
# relay reaches the coordinator, direct public egress is impossible, and the
# CONNECT policy holds: chatgpt.com 200; github.com 403 in allowlist mode, or
# with --allow-public-egress github.com 200 and the Docker bridge gateway (a
# non-global address) 403.
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
  -fsS "http://${restricted_proxy_ip}:${JUDGE_PORT}/openapi.json" >/dev/null
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
  "${restricted_proxy_ip}" "${task_internet}" "${bridge_gateway}"
if [[ "${task_internet}" == "true" ]]; then
  echo "Public-egress network: PASS (${isolated_network}, proxy ${restricted_proxy_ip}; no direct egress; CONNECT to public hosts on 80/443; internal addresses refused)"
else
  echo "Restricted network: PASS (${isolated_network}, proxy ${restricted_proxy_ip})"
fi

# Force file-based ChatGPT authentication. No credential material is copied
# into this repository or written to the run manifest.
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
else
  export SFORGE_CODEX_AUTH_JSON_PATH="${AUTH_JSON}"
fi
export SFORGE_CODEX_REASONING_EFFORT="${REASONING_EFFORT}"
export SFORGE_CODEX_CLI_VERSION="${CODEX_CLI_VERSION}"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  export SFORGE_AGENT_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${work_gpu_uuid},CUDA_VISIBLE_DEVICES=0,SFORGE_PRE_SUBMIT_HOOK=/app/workdir/flash_kmeans_lab/.sforge/prepare_submit.sh"
else
  export SFORGE_AGENT_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${work_gpu_uuid},CUDA_VISIBLE_DEVICES=0,FLASH_KMEANS_TARGET=H100"
fi
export SFORGE_ISOLATED_DOCKER_NETWORK="${isolated_network}"
export SFORGE_HTTP_PROXY="http://${restricted_proxy_ip}:3128"
export SFORGE_HTTPS_PROXY="http://${restricted_proxy_ip}:3128"
export SFORGE_NO_PROXY="${restricted_proxy_ip},127.0.0.1,localhost"

# Keep host-side Judge registration and final evaluation off any inherited
# workstation proxy. The agent container still receives only the restricted
# sidecar through the explicit SFORGE_* proxy settings above.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${restricted_proxy_ip},127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"

isolated_entry_sha256="$(sha256sum "${ISOLATED_ENTRY}" | awk '{print $1}')"
restricted_proxy_sha256="$(
  sha256sum "${RESTRICTED_PROXY_SOURCE}" | awk '{print $1}'
)"
judge_server_sha256="$(
  sha256sum "${JUDGE_SERVER_SOURCE}" | awk '{print $1}'
)"
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
work_image_id="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")"
judge_image_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
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
raw_latency_comparison_identity_json="$(
  jq -cn \
    --arg evaluation_contract_sha256 "${evaluation_contract_sha256}" \
    --arg measurement_profile_sha256 "${measurement_profile_sha256}" \
    '{
      evaluation_contract_sha256: $evaluation_contract_sha256,
      measurement_profile_sha256: $measurement_profile_sha256
    }'
)"
raw_latency_comparison_identity_sha256="sha256:$(
  printf '%s' "${raw_latency_comparison_identity_json}" \
    | sha256sum | awk '{print $1}'
)"

experiment_manifest="${RUN_LOG_ROOT}/experiment-manifest.json"
jq -n \
  --arg run_id "${RUN_ID}" \
  --arg started_at "$(date -u +%FT%TZ)" \
  --arg model "${MODEL}" \
  --arg effort "${REASONING_EFFORT}" \
  --arg cli_version "${CODEX_CLI_VERSION}" \
  --arg runtime_dir "${AGENT_RUNTIME_DIR}" \
  --arg runtime_agent "${runtime_agent}" \
  --arg runtime_version "${runtime_version}" \
  --arg runtime_tree_sha256 "${runtime_tree_sha256}" \
  --arg work_gpu "${WORK_GPU}" \
  --arg work_gpu_uuid "${work_gpu_uuid}" \
  --arg judge_gpu "${JUDGE_GPU}" \
  --arg judge_gpu_uuid "${judge_gpu_uuid}" \
  --arg network_mode "docker_internal_restricted_connect_proxy" \
  --arg isolated_entry_sha256 "${isolated_entry_sha256}" \
  --arg restricted_proxy_sha256 "${restricted_proxy_sha256}" \
  --arg judge_server_sha256 "${judge_server_sha256}" \
  --arg handoff_mode "${handoff_mode}" \
  --arg allocation_key "${allocation_key}" \
  --arg trusted_lease_id "${trusted_lease_id}" \
  --arg evaluation_contract_sha256 "${evaluation_contract_sha256}" \
  --arg execution_profile_id "${execution_profile_id}" \
  --arg measurement_profile_sha256 "${measurement_profile_sha256}" \
  --arg raw_latency_comparison_identity_sha256 "${raw_latency_comparison_identity_sha256}" \
  --argjson raw_latency_comparison_identity "${raw_latency_comparison_identity_json}" \
  --arg qualification_status "${qualification_status}" \
  --argjson measurement_profile "${measurement_profile_json}" \
  --argjson deployment_attestation "${deployment_attestation_json}" \
  --arg codex_web_search "${codex_web_search}" \
  --argjson public_connect_egress "${task_internet}" \
  --argjson timeout "${TASK_TIMEOUT}" \
  --argjson eval_interval "${EVAL_INTERVAL}" \
  --argjson submission_cooldown "${SUBMISSION_COOLDOWN}" \
  '{
    run_id: $run_id,
    started_at: $started_at,
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
    manual_hidden_submission_limit: null,
    manual_submission_cooldown_seconds: $submission_cooldown,
    judge_server_sha256: $judge_server_sha256,
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
      raw_latency_comparability: "requires_qualified_pair_identity"
    }),
    raw_latency_comparison: ($raw_latency_comparison_identity + {
      identity_sha256: $raw_latency_comparison_identity_sha256,
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
    work_gpu: {index: $work_gpu, uuid: $work_gpu_uuid},
    judge_gpu: {index: $judge_gpu, uuid: $judge_gpu_uuid}
  }' > "${experiment_manifest}"
[[ -s "${experiment_manifest}" ]] \
  || die "experiment manifest was not written"
experiment_manifest_sha256="sha256:$(
  sha256sum "${experiment_manifest}" | awk '{print $1}'
)"

echo "Starting ${RUN_ID}"
echo "  model: ${MODEL} (${REASONING_EFFORT})"
echo "  budget: ${TASK_TIMEOUT}s, autojudge disabled; manual Judge cooldown ${EVAL_INTERVAL}s"
echo "  manual Judge submissions: unlimited, ${SUBMISSION_COOLDOWN}s cooldown"
echo "  Codex web search: ${codex_web_search}; public CONNECT egress: ${task_internet}"
echo "  work GPU: ${WORK_GPU}; judge GPU: ${JUDGE_GPU}"
echo "  logs: ${RUN_LOG_ROOT}"

# Codex web search and the evolve prompt follow MLSYSFORGE_WORK_INTERNET
# (exported above from the task JSON). The --enable-internet flag only
# bypasses SForge's unavailable host-iptables implementation; the Docker
# network above has no direct egress.
set +e
run_logged_command "${RUN_LOG_ROOT}/sforge-run.log" \
  "${SFORGE_PYTHON}" \
  "${ISOLATED_ENTRY}" \
  run \
  --backend docker \
  --task "${TASK_ID}" \
  --agent codex \
  --model "${MODEL}" \
  --timeout "${TASK_TIMEOUT}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --disable-auto-eval \
  --submission-cooldown "${SUBMISSION_COOLDOWN}" \
  --judge-url "http://${restricted_proxy_ip}:${JUDGE_PORT}" \
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

precheck_mode_args=()
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  precheck_mode_args=(--evaluation-mode single_gpu_lease_manual_judge_then_standalone_final_handoff)
fi
formal_precheck="${TASK_RUN_ROOT}/formal-precheck.json"
write_command_json_atomically \
  "${formal_precheck}" \
  python3 "${MIGRATION_ROOT}/scripts/validate_formal_artifacts.py" pre \
  --task-root "${TASK_RUN_ROOT}" \
  --run-id "${RUN_ID}" \
  --task-id "${TASK_ID}" \
  --model "${MODEL}" \
  --budget-seconds "${TASK_TIMEOUT}" \
  --interval-seconds "${EVAL_INTERVAL}" \
  "${precheck_mode_args[@]}"
final_archive_sha256="$(
  jq -er \
    '.final_archive.sha256
     | select(type == "string" and test("^[0-9a-f]{64}$"))' \
    "${formal_precheck}"
)"
echo "Formal precheck: PASS (archive sha256=${final_archive_sha256})"

# The hardened scheduler emits only strict pre-timeout ticks. Evaluate the
# immutable timeout-boundary archive synchronously for the guaranteed final
# fixed-budget point.
final_eval_id="${RUN_ID}-final-${final_minutes}m"
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
  "${SFORGE_BIN}" \
  eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${final_archive}" \
  --run-id "${final_eval_id}" \
  --timeout 3600 \
  --judge-cpu-limit 8 \
  --judge-mem-limit 32g \
  "${final_handoff_args[@]}" \
  --json
final_eval_rc="$?"
set -e
(( final_eval_rc == 0 )) \
  || die "independent final evaluation failed with exit ${final_eval_rc}"

final_task_root="${LOGS_DIR}/runs/${final_eval_id}/${TASK_ID}"
final_submission_root="${final_task_root}/submissions/manual-1"
final_report="${final_submission_root}/report.json"
final_evaluated_archive="${final_submission_root}/submission.tar.gz"
[[ -s "${final_report}" ]] \
  || die "independent final evaluation did not produce a nonempty report"
[[ -s "${final_evaluated_archive}" ]] \
  || die "independent final evaluation did not persist its submission archive"

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
  local reason evaluation_mode
  reason="$(sed -n 's/^ERROR: //p' "${stderr_file}" | head -n 1 || true)"
  [[ -n "${reason}" ]] || reason="$(tail -n 1 "${stderr_file}" || true)"
  [[ -n "${reason}" ]] || reason="formal postcheck exited ${postcheck_rc} without a reason"
  evaluation_mode="dual_gpu_independent_final"
  if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
    evaluation_mode="single_gpu_lease_manual_judge_then_standalone_final_handoff"
  fi
  if ! jq -e '.status == "invalid_final" and (.reason | type == "string")' \
      "${stdout_file}" >/dev/null 2>&1; then
    # The validator died before it could emit a record (crash, bad CLI, ...):
    # synthesise the same shape from its stderr.
    jq -n \
      --arg evaluation_mode "${evaluation_mode}" \
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
    --arg reason "${reason}" \
    --arg evaluation_mode "${evaluation_mode}" \
    --arg evaluation_contract_sha256 "${evaluation_contract_sha256}" \
    --arg execution_profile_id "${execution_profile_id}" \
    --arg measurement_profile_sha256 "${measurement_profile_sha256}" \
    --arg qualification_status "${qualification_status}" \
    --arg experiment_manifest "${experiment_manifest}" \
    --arg experiment_manifest_sha256 "${experiment_manifest_sha256}" \
    --arg final_archive "${final_archive}" \
    --arg final_archive_sha256 "${final_archive_sha256}" \
    --arg formal_precheck "${formal_precheck}" \
    --arg formal_postcheck "${formal_postcheck}" \
    --argjson sforge_run_exit "${run_rc}" \
    --argjson final_eval_exit "${final_eval_rc}" \
    --argjson formal_postcheck_exit "${postcheck_rc}" \
    '{
      schema_version: 1,
      status: "failed",
      failure_stage: "formal_postcheck",
      reason: $reason,
      finished_at: $finished_at,
      run_id: $run_id,
      task_id: $task_id,
      evaluation_mode: $evaluation_mode,
      sforge_run_exit: $sforge_run_exit,
      final_eval_exit: $final_eval_exit,
      formal_postcheck_exit: $formal_postcheck_exit,
      portable_contract: {
        evaluation_contract_sha256: $evaluation_contract_sha256,
        execution_profile_id: $execution_profile_id
      },
      measurement_profile: {
        comparability_key_sha256: $measurement_profile_sha256,
        qualification_status: $qualification_status
      },
      experiment_manifest: {
        path: $experiment_manifest,
        sha256: $experiment_manifest_sha256
      },
      final_archive: {
        path: $final_archive,
        sha256: $final_archive_sha256
      },
      validation: {
        precheck: $formal_precheck,
        postcheck: $formal_postcheck
      }
    }' || true
  die "independent final evaluation is invalid (recorded in ${formal_postcheck}): ${reason}"
}

postcheck_stdout="${formal_postcheck}.tmp.$$"
postcheck_stderr="${formal_postcheck}.stderr.$$"
postcheck_rc=0
python3 "${MIGRATION_ROOT}/scripts/validate_formal_artifacts.py" post \
  --task-root "${TASK_RUN_ROOT}" \
  --final-task-root "${final_task_root}" \
  --final-eval-id "${final_eval_id}" \
  --task-id "${TASK_ID}" \
  --expected-archive-sha256 "${final_archive_sha256}" \
  "${precheck_mode_args[@]}" \
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
final_report_sha256="$(
  jq -er \
    '.final_report.sha256
     | select(type == "string" and test("^[0-9a-f]{64}$"))' \
    "${formal_postcheck}"
)"
evaluated_archive_sha256="$(
  jq -er \
    '.evaluated_archive.sha256
     | select(type == "string" and test("^[0-9a-f]{64}$"))' \
    "${formal_postcheck}"
)"
private_judge_source="$(
  jq -er '.private_judge_artifact.path | select(type == "string" and length > 0)' \
    "${formal_postcheck}"
)"
private_judge_sha256="$(
  jq -er \
    '.private_judge_artifact.sha256
     | select(type == "string" and test("^[0-9a-f]{64}$"))' \
    "${formal_postcheck}"
)"
private_judge_member_sha256="$(
  jq -er \
    '.private_judge_artifact.member_sha256
     | select(type == "string" and test("^[0-9a-f]{64}$"))' \
    "${formal_postcheck}"
)"
final_private_judge_artifact="$(
  realpath -m \
    "${TASK_RUN_ROOT}/final-${final_minutes}m-private-judge-artifacts.tar.gz"
)"
cp -- "${private_judge_source}" "${final_private_judge_artifact}"
chmod 600 "${final_private_judge_artifact}"
[[ "$(sha256sum "${final_private_judge_artifact}" | awk '{print $1}')" \
    == "${private_judge_sha256}" ]] \
  || die "copied private Judge artifact digest mismatch"

final_curve_report="${TASK_RUN_ROOT}/final-${final_minutes}m-report.json"
write_command_json_atomically \
  "${final_curve_report}" \
  jq -n \
  --arg final_archive_sha256 "${final_archive_sha256}" \
  --arg evaluated_archive_sha256 "${evaluated_archive_sha256}" \
  --arg final_report_sha256 "${final_report_sha256}" \
  --arg private_judge_artifact "${final_private_judge_artifact}" \
  --arg private_judge_sha256 "${private_judge_sha256}" \
  --arg private_judge_member_sha256 "${private_judge_member_sha256}" \
  --slurpfile precheck "${formal_precheck}" \
  --slurpfile postcheck "${formal_postcheck}" \
  '{
    snapshot_elapsed_seconds: $precheck[0].snapshot_elapsed_seconds,
    snapshot_captured_at: $precheck[0].snapshot_captured_at,
    seal_requested_elapsed_seconds: $precheck[0].seal_requested_elapsed_seconds,
    final_archive_sha256: $final_archive_sha256,
    evaluated_archive_sha256: $evaluated_archive_sha256,
    final_report_sha256: $final_report_sha256,
    private_judge_artifact: {
      path: $private_judge_artifact,
      sha256: $private_judge_sha256,
      member_sha256: $private_judge_member_sha256
    },
    report: $postcheck[0].validated_report
  }'

# This launcher runs SForge with --disable-auto-eval: only explicit manual
# submissions and the independent timeout-boundary final are judged. If the
# version store contains checkpoints, replay_candidates.sh may evaluate them
# offline without returning hidden-Judge feedback to the agent.
python3 "${MIGRATION_ROOT}/scripts/collect_curve.py" \
  "${TASK_RUN_ROOT}" \
  --final-eval "${final_curve_report}" \
  --no-auto-snapshots \
  --final-minutes "${final_minutes}" \
  --output-dir "${TASK_RUN_ROOT}/curve"

curve_json="${TASK_RUN_ROOT}/curve/curve.json"
curve_markdown="${TASK_RUN_ROOT}/curve/curve.md"
[[ -s "${curve_json}" && -s "${curve_markdown}" ]] \
  || die "curve collection did not produce both nonempty outputs"

# Freeze the completion attestation only after post-run validation passed.
write_command_json_atomically \
  "${completion}" \
  jq -n \
  --arg finished_at "$(date -u +%FT%TZ)" \
  --arg run_id "${RUN_ID}" \
  --arg task_id "${TASK_ID}" \
  --arg evaluation_contract_sha256 "${evaluation_contract_sha256}" \
  --arg execution_profile_id "${execution_profile_id}" \
  --arg measurement_profile_sha256 "${measurement_profile_sha256}" \
  --arg raw_latency_comparison_identity_sha256 "${raw_latency_comparison_identity_sha256}" \
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
  --arg private_judge_member_sha256 "${private_judge_member_sha256}" \
  --arg curve_json "${curve_json}" \
  --arg curve_markdown "${curve_markdown}" \
  --arg formal_precheck "${formal_precheck}" \
  --arg formal_postcheck "${formal_postcheck}" \
  --argjson sforge_run_exit "${run_rc}" \
  --argjson final_eval_exit "${final_eval_rc}" \
  '{
    schema_version: 1,
    status: "passed",
    finished_at: $finished_at,
    run_id: $run_id,
    task_id: $task_id,
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
    raw_latency_comparison: {
      evaluation_contract_sha256: $evaluation_contract_sha256,
      measurement_profile_sha256: $measurement_profile_sha256,
      identity_sha256: $raw_latency_comparison_identity_sha256,
      policy: "same_evaluation_contract_and_measurement_profile_only",
      qualification_status: $qualification_status
    },
    experiment_manifest: {
      path: $experiment_manifest,
      sha256: $experiment_manifest_sha256
    },
    final_archive: {
      path: $final_archive,
      sha256: $final_archive_sha256
    },
    final_report: {
      path: $final_report,
      sha256: $final_report_sha256
    },
    evaluated_archive: {
      path: $evaluated_archive,
      sha256: $evaluated_archive_sha256
    },
    private_judge_artifact: {
      path: $private_judge_artifact,
      sha256: $private_judge_sha256,
      member_sha256: $private_judge_member_sha256
    },
    curve: {
      json: $curve_json,
      markdown: $curve_markdown
    },
    validation: {
      precheck: $formal_precheck,
      postcheck: $formal_postcheck
    }
  }'

echo "Completed ${RUN_ID}"
echo "  curve: ${TASK_RUN_ROOT}/curve/curve.md"
echo "  private Judge rows: ${final_private_judge_artifact}"
echo "  SForge exit: ${run_rc}; final eval exit: ${final_eval_rc}"

(( run_rc == 0 && final_eval_rc == 0 ))
