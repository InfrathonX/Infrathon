#!/usr/bin/env bash
set -Eeuo pipefail

# Wait for the ORPO_GPU_INDICES pair, then run one fixed-budget Codex or Claude Code session.
# With ORPO_JUDGE_GPU_INDICES=C,D (the default of launch/qwen3-orpo/run.sh) the
# Judge gets that second pair and Work is never paused for a Judge round
# (separate mode, all four GPUs leased for the whole run).
# ORPO_JUDGE_GPU_INDICES=shared instead runs the Judge on the Work pair while
# the host pauses Work.  A caller that leaves it unset is refused.
# Auto-evaluation is deliberately disabled. The Agent can still call
# sforge-submit an unlimited number of times, subject to a 300-second cooldown.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

# The checkout this script lives in is the checkout the run executes: code,
# venv, and the borrowed llada proxy all resolve from here, so launching from
# a worktree runs that worktree's branch instead of silently using another
# checkout. REPO can still be overridden for recovery scenarios.
REPO="${REPO:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
[[ -f "${REPO}/sforge/harness/run_agent.py" ]] \
  || die "REPO does not look like an MLSys Forge checkout: ${REPO}"
[[ -x "${REPO}/.venv/bin/python" ]] \
  || die "missing SForge virtualenv at ${REPO}/.venv (create it before launching)"
TASK_ROOT="${REPO}/tasks/latest/qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed"
TASK_ID="qwen3_1p7b_orpo_training_optimizer_dual_h100_v4"
BENCHMARK="qwen3_1p7b_orpo_dual_h100"
SFORGE_PYTHON="${REPO}/.venv/bin/python"
ENTRY="${TASK_ROOT}/scripts/sforge_dual_h100_entry.py"
WORK_IMAGE="${BENCHMARK}.work.${TASK_ID}:local"
JUDGE_IMAGE="${BENCHMARK}.judge.${TASK_ID}:local"
PROXY_SOURCE="${REPO}/tasks/latest/llada-h100-decoder-optimization-v2-vanilla-seed/scripts/restricted_connect_proxy.py"

AUTH_JSON="${CODEX_AUTH_JSON_PATH:-${HOME}/.codex/auth.json}"
# MLSYSFORGE_AGENT=claude-code drives the pinned Claude Code runtime with a
# 0600 file holding one `claude setup-token` OAuth token (CLAUDE_OAUTH_TOKEN_FILE).
AGENT_NAME="${MLSYSFORGE_AGENT:-codex}"
CLAUDE_TOKEN_FILE="${CLAUDE_OAUTH_TOKEN_FILE:-}"
# Host locations of the three inputs. The defaults are the reference layout;
# another host points at its own copies with the same variables that
# eval_archive.sh and replay_candidates.sh already read (or a launch config).
MODEL_REPO="${ORPO_MODEL_HOST_PATH:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--Qwen--Qwen3-1.7B}"
MODEL_REV="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
# Data defaults: the task's Infrathon assets (assets/fetch.sh) when
# INFRATHON_ASSETS_ROOT is set, else the host artifact directory.
# Unset: the default root of assets/fetch.sh, when it exists.
[[ -n "${INFRATHON_ASSETS_ROOT:-}" || ! -d "${HOME}/.cache/infrathon/assets" ]] \
  || INFRATHON_ASSETS_ROOT="${HOME}/.cache/infrathon/assets"
if [[ -n "${INFRATHON_ASSETS_ROOT:-}" ]]; then
  default_data_root="${INFRATHON_ASSETS_ROOT}/qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed/public/tldr_preference_v1"
  default_private_data_root="${INFRATHON_ASSETS_ROOT}/qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed/judge/tldr_preference_cross_v4"
else
  default_data_root="${HOME}/mlsysforge-artifacts/qwen3-orpo/tldr_preference_v1"
  default_private_data_root="${HOME}/mlsysforge-artifacts/qwen3-orpo/tldr_preference_cross_v4"
fi
PUBLIC_DATA="${ORPO_DATA_HOST_ROOT:-${default_data_root}}"
PRIVATE_DATA="${ORPO_PRIVATE_DATA_HOST_ROOT:-${default_private_data_root}}"
GPU_UUIDS="${ORPO_GPU_UUIDS:?set ORPO_GPU_UUIDS to the two physical GPU UUIDs of the NVLink pair this run owns, comma separated (nvidia-smi --query-gpu=index,uuid --format=csv,noheader)}"
GPU_INDICES="${ORPO_GPU_INDICES:?set ORPO_GPU_INDICES to the two physical GPU indices of the NVLink pair this run owns, comma separated (nvidia-smi --query-gpu=index,uuid --format=csv,noheader)}"
GPU6_UUID="${GPU_UUIDS%%,*}"
GPU7_UUID="${GPU_UUIDS##*,}"

AGENT_MODEL="${AGENT_MODEL:?set AGENT_MODEL to the agent model id (launch/qwen3-orpo/run.sh --model)}"
# Optional OpenAI-compatible model API (e.g. a self-hosted DeepSeek).  With
# MODEL_API_BASE_URL set, the restricted sidecar gains one raw TCP relay to
# that endpoint, Codex talks to it in API-key mode (key from the mode-0600
# MODEL_API_KEY_FILE) and AGENT_MODEL may be any id the endpoint serves.
MODEL_API_BASE_URL="${MODEL_API_BASE_URL:-}"
MODEL_API_KEY_FILE="${MODEL_API_KEY_FILE:-}"
MODEL_FORWARD_PORT=3131
REASONING_EFFORT="${REASONING_EFFORT:-ultra}"
# Default run label: the model id without a gpt-N.M- prefix, plus the effort.
default_run_label="$(sed -E 's/^gpt-[0-9.]+-//' <<<"${AGENT_MODEL}" | tr 'A-Z' 'a-z' | sed -E 's/[^a-z0-9.]+/-/g; s/^-+//; s/-+$//')-${REASONING_EFFORT}"
RUN_LABEL="${RUN_LABEL:-${default_run_label}}"
MIN_DELAY_SECONDS="${MIN_DELAY_SECONDS:-1800}"
POLL_SECONDS="${POLL_SECONDS:-30}"
GPU_MAX_USED_MIB="${GPU_MAX_USED_MIB:-256}"
TASK_TIMEOUT="${TASK_TIMEOUT:-21600}"
SUBMISSION_COOLDOWN=300
JUDGE_PORT="${JUDGE_PORT:-18476}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ID="${RUN_ID:-qwen3-orpo-v4-${RUN_LABEL}-$(( (TASK_TIMEOUT + 59) / 60 ))m-${RUN_STAMP}}"
LOG_ROOT="${TASK_ROOT}/logs/runs/${RUN_ID}"
NETWORK="${RUN_ID}-internal"
PROXY="${RUN_ID}-restricted-proxy"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

for value_name in MIN_DELAY_SECONDS POLL_SECONDS GPU_MAX_USED_MIB TASK_TIMEOUT JUDGE_PORT; do
  value="${!value_name}"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${value_name} must be a non-negative integer"
done
(( POLL_SECONDS > 0 )) || die "POLL_SECONDS must be greater than zero"
(( TASK_TIMEOUT > 0 )) || die "TASK_TIMEOUT must be greater than zero"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe RUN_ID: ${RUN_ID}"
case "${REASONING_EFFORT}" in
  max|ultra|xhigh|high|medium|low) ;;
  *) die "unsupported reasoning effort: ${REASONING_EFFORT}" ;;
esac
case "${AGENT_NAME}" in
  codex) ;;
  claude-code)
    [[ -z "${MODEL_API_BASE_URL}" ]] || die "Claude Code uses its OAuth token; unset MODEL_API_BASE_URL"
    [[ "${REASONING_EFFORT}" != ultra ]] || die "Claude Code efforts are low|medium|high|xhigh|max"
    ;;
  *) die "MLSYSFORGE_AGENT must be codex or claude-code" ;;
esac

command -v docker >/dev/null || die "docker is not installed"
command -v flock >/dev/null || die "flock is not installed"
command -v jq >/dev/null || die "jq is not installed"
command -v nvidia-smi >/dev/null || die "nvidia-smi is not installed"

# GPU mode, named by ORPO_JUDGE_GPU_INDICES.  A pair C,D: separate, the Judge
# UUIDs come from ORPO_JUDGE_GPU_UUIDS or nvidia-smi and preflight.py checks
# both against each other.  `shared`: the Judge uses the Work pair.  Unset (an
# older queue script or a direct call) is refused instead of silently
# running shared and pausing Work.
JUDGE_GPU_INDICES="${ORPO_JUDGE_GPU_INDICES:-}"
JUDGE_GPU_UUIDS="${ORPO_JUDGE_GPU_UUIDS:-}"
case "${JUDGE_GPU_INDICES}" in
  "")
    die "set ORPO_JUDGE_GPU_INDICES: C,D for a separate Judge pair (Work never paused; launch/qwen3-orpo/run.sh --judge-gpus) or shared to run the Judge on the Work pair and pause Work for each Judge round (--shared-gpus)"
    ;;
  shared)
    [[ -z "${JUDGE_GPU_UUIDS}" ]] \
      || die "ORPO_JUDGE_GPU_INDICES=shared runs the Judge on the Work pair; unset ORPO_JUDGE_GPU_UUIDS"
    ;;
esac
if [[ "${JUDGE_GPU_INDICES}" != shared ]]; then
  GPU_MODE="separate"
  [[ "${GPU_INDICES}" =~ ^([0-9]+),([0-9]+)$ && "${BASH_REMATCH[1]}" != "${BASH_REMATCH[2]}" ]] \
    || die "ORPO_GPU_INDICES must be two distinct comma-separated physical GPU indices"
  work_index_a="${BASH_REMATCH[1]}"; work_index_b="${BASH_REMATCH[2]}"
  [[ "${JUDGE_GPU_INDICES}" =~ ^([0-9]+),([0-9]+)$ && "${BASH_REMATCH[1]}" != "${BASH_REMATCH[2]}" ]] \
    || die "ORPO_JUDGE_GPU_INDICES must be two distinct comma-separated physical GPU indices (or 'shared')"
  judge_index_a="${BASH_REMATCH[1]}"; judge_index_b="${BASH_REMATCH[2]}"
  for judge_index in "${judge_index_a}" "${judge_index_b}"; do
    [[ "${judge_index}" != "${work_index_a}" && "${judge_index}" != "${work_index_b}" ]] \
      || die "the Judge pair ${JUDGE_GPU_INDICES} overlaps the Work pair ${GPU_INDICES}; pass --judge-gpus with two other GPUs, or --shared-gpus to run both on one pair"
  done
  if [[ -z "${JUDGE_GPU_UUIDS}" ]]; then
    gpu_listing="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)" \
      || die "nvidia-smi could not enumerate this host's GPUs"
    judge_uuid_a="$(awk -F', *' -v want="${judge_index_a}" '$1 == want { print $2 }' <<<"${gpu_listing}")"
    judge_uuid_b="$(awk -F', *' -v want="${judge_index_b}" '$1 == want { print $2 }' <<<"${gpu_listing}")"
    [[ -n "${judge_uuid_a}" && -n "${judge_uuid_b}" ]] \
      || die "this host does not expose the Judge GPU indices ${JUDGE_GPU_INDICES}"
    JUDGE_GPU_UUIDS="${judge_uuid_a},${judge_uuid_b}"
  fi
  [[ "${JUDGE_GPU_UUIDS}" =~ ^GPU-[^,[:space:]]+,GPU-[^,[:space:]]+$ ]] \
    || die "ORPO_JUDGE_GPU_UUIDS must be the two GPU UUIDs of the Judge pair, comma separated"
  JUDGE_GPU_A_UUID="${JUDGE_GPU_UUIDS%%,*}"
  JUDGE_GPU_B_UUID="${JUDGE_GPU_UUIDS##*,}"
  (( $(printf '%s\n' "${GPU6_UUID}" "${GPU7_UUID}" "${JUDGE_GPU_A_UUID}" "${JUDGE_GPU_B_UUID}" \
        | tr 'A-Z' 'a-z' | sort -u | wc -l) == 4 )) \
    || die "the Work pair ${GPU_UUIDS} and the Judge pair ${JUDGE_GPU_UUIDS} must be four distinct GPUs"
  GPU_PLAN="${GPU_INDICES} (Work) and ${JUDGE_GPU_INDICES} (Judge)"
  SERVE_ENTRY="${TASK_ROOT}/scripts/separate_judge_entry.py"
  [[ -f "${SERVE_ENTRY}" ]] || die "missing separate-GPU Judge entry: ${SERVE_ENTRY}"
else
  GPU_MODE="shared"
  JUDGE_GPU_INDICES="${GPU_INDICES}"
  JUDGE_GPU_UUIDS="${GPU_UUIDS}"
  JUDGE_GPU_A_UUID="${GPU6_UUID}"
  JUDGE_GPU_B_UUID="${GPU7_UUID}"
  GPU_PLAN="${GPU_INDICES}"
  SERVE_ENTRY="${ENTRY}"
fi
separate_gpus() { [[ "${GPU_MODE}" == "separate" ]]; }

# The task JSON's `internet` flag decides Work's egress
# (docs/INTERNET_AND_WEB_SEARCH.md).  true: the restricted sidecar also
# tunnels to public internet hosts on 443/80 (private, loopback and link-local
# addresses stay refused) and Codex runs with live web search.  false: the
# historical allowlist-only sidecar and offline Codex.  Work stays on the
# internal network either way.
TASK_INTERNET="$(
  jq -er '.internet | if type == "boolean" then tostring
    else error("internet must be a boolean") end' "${TASK_ROOT}/tasks/${TASK_ID}.json"
)" || die "task JSON lacks a boolean internet flag"
if [[ "${TASK_INTERNET}" == "true" ]]; then
  PROXY_EGRESS_ARGS=(--allow-public-egress)
  CODEX_WEB_SEARCH="live"
else
  PROXY_EGRESS_ARGS=()
  CODEX_WEB_SEARCH="disabled"
fi
[[ -x "${SFORGE_PYTHON}" ]] || die "missing SForge Python: ${SFORGE_PYTHON}"
MODEL_API_KEY=""; model_api_host=""; model_api_port=""; model_api_path=""
if [[ -n "${MODEL_API_BASE_URL}" ]]; then
  [[ "${MODEL_API_BASE_URL}" =~ ^http://([^/:[:space:]]+):([0-9]+)(/.*)?$ ]] \
    || die "MODEL_API_BASE_URL must be http://HOST:PORT[/path] (relayed as raw TCP by the sidecar)"
  model_api_host="${BASH_REMATCH[1]}"; model_api_port="${BASH_REMATCH[2]}"; model_api_path="${BASH_REMATCH[3]:-}"
  [[ -n "${MODEL_API_KEY_FILE}" && -r "${MODEL_API_KEY_FILE}" && ! -L "${MODEL_API_KEY_FILE}" ]] \
    || die "set MODEL_API_KEY_FILE to a readable regular file"
  [[ "$(stat -c '%a' "${MODEL_API_KEY_FILE}")" == "600" ]] || die "MODEL_API_KEY_FILE must have mode 600"
  MODEL_API_KEY="$(tr -d '[:space:]' < "${MODEL_API_KEY_FILE}")"
  [[ -n "${MODEL_API_KEY}" ]] || die "MODEL_API_KEY_FILE is empty"
elif [[ "${AGENT_NAME}" == claude-code ]]; then
  [[ -n "${CLAUDE_TOKEN_FILE}" && -f "${CLAUDE_TOKEN_FILE}" && ! -L "${CLAUDE_TOKEN_FILE}" ]] \
    || die "set CLAUDE_OAUTH_TOKEN_FILE to the Claude Code OAuth token file"
  [[ "$(stat -c '%a' "${CLAUDE_TOKEN_FILE}")" == "600" ]] || die "CLAUDE_OAUTH_TOKEN_FILE must have mode 600"
else
  [[ -r "${AUTH_JSON}" ]] || die "missing Codex auth file: ${AUTH_JSON}"
fi

# The Work image is agent-free; a host-provided runtime supplies the pinned
# Codex CLI, bind-mounted read-only at /opt/agent-runtime by SForge.
if [[ "${AGENT_NAME}" == claude-code ]]; then
  AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}"
  marker="${AGENT_RUNTIME_DIR}/.sforge-agent-runtime.json"
  [[ -d "${AGENT_RUNTIME_DIR}" && ! -L "${AGENT_RUNTIME_DIR}" && -f "${marker}" ]] \
    || die "Claude Code agent runtime is missing or unmarked: ${AGENT_RUNTIME_DIR}"
  AGENT_RUNTIME_AGENT="$(jq -er '.agent' "${marker}")"
  AGENT_RUNTIME_VERSION="$(jq -er '.version' "${marker}")"
  AGENT_RUNTIME_TREE_SHA256="$(jq -er '.tree_sha256' "${marker}")"
  [[ "${AGENT_RUNTIME_AGENT}" == claude-code ]] || die "SFORGE_AGENT_RUNTIME_DIR is not a claude-code runtime"
  export CODEX_CLI_VERSION=""
else
  export CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-0.148.0}"
  require_codex_agent_runtime "${CODEX_CLI_VERSION}"
fi

[[ -r "${MODEL_REPO}/snapshots/${MODEL_REV}/config.json" ]] || die "missing pinned model snapshot"
[[ -d "${PUBLIC_DATA}" && -d "${PRIVATE_DATA}" ]] || die "missing frozen TLDR artifacts"

# Fail early instead of waiting for 30 minutes in a shell that cannot launch Docker.
docker version >/dev/null 2>&1 || die "this shell cannot access the Docker daemon; run from a Docker-authorized login/session"
docker image inspect "${WORK_IMAGE}" "${JUDGE_IMAGE}" >/dev/null \
  || die "the pinned Work/Judge images are missing"

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/launcher.log") 2>&1

echo "run_id=${RUN_ID}"
echo "agent=${AGENT_NAME} model=${AGENT_MODEL} reasoning=${REASONING_EFFORT} timeout=${TASK_TIMEOUT}s"
echo "autojudge=disabled manual_submissions=unlimited cooldown=${SUBMISSION_COOLDOWN}s"
echo "minimum_delay=${MIN_DELAY_SECONDS}s gpu_pair=${GPU_INDICES}"
echo "gpu_mode=${GPU_MODE} work_gpu_pair=${GPU_INDICES} judge_gpu_pair=${JUDGE_GPU_INDICES}"
echo "agent_runtime=${AGENT_RUNTIME_DIR} (${AGENT_RUNTIME_AGENT} ${AGENT_RUNTIME_VERSION})"
echo "internet=${TASK_INTERNET} codex_web_search=${CODEX_WEB_SEARCH} (Work on an internal network; egress only via the sidecar)"

jq -n \
  --arg run_id "${RUN_ID}" \
  --arg agent "${AGENT_NAME}" \
  --arg model "${AGENT_MODEL}" \
  --arg effort "${REASONING_EFFORT}" \
  --arg cli_version "${CODEX_CLI_VERSION}" \
  --arg runtime_dir "${AGENT_RUNTIME_DIR}" \
  --arg runtime_agent "${AGENT_RUNTIME_AGENT}" \
  --arg runtime_version "${AGENT_RUNTIME_VERSION}" \
  --arg runtime_tree_sha256 "${AGENT_RUNTIME_TREE_SHA256}" \
  --argjson timeout "${TASK_TIMEOUT}" \
  --argjson cooldown "${SUBMISSION_COOLDOWN}" \
  --argjson internet "${TASK_INTERNET}" \
  --arg codex_web_search "${CODEX_WEB_SEARCH}" \
  --arg gpu_mode "${GPU_MODE}" \
  --arg gpu_indices "${GPU_INDICES}" \
  --arg gpu_uuids "${GPU_UUIDS}" \
  --arg judge_gpu_indices "${JUDGE_GPU_INDICES}" \
  --arg judge_gpu_uuids "${JUDGE_GPU_UUIDS}" \
  '{
    run_id: $run_id,
    gpu_mode: $gpu_mode,
    gpu_pair: {
      indices: ($gpu_indices | split(",")),
      uuids: ($gpu_uuids | split(",")),
      container_visible_devices: "0,1"
    },
    work_gpu_pair: {
      indices: ($gpu_indices | split(",")),
      uuids: ($gpu_uuids | split(",")),
      container_visible_devices: "0,1"
    },
    judge_gpu_pair: {
      indices: ($judge_gpu_indices | split(",")),
      uuids: ($judge_gpu_uuids | split(",")),
      container_visible_devices: "0,1"
    },
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
    timeout_seconds: $timeout,
    submission_cooldown_seconds: $cooldown,
    scheduled_auto_evaluation: false,
    internet: $internet,
    codex_web_search: $codex_web_search
  }' >"${LOG_ROOT}/launch-manifest.json"

judge_pid=""
proxy_log_pid=""
locks_held=0
SINGLE_LOCK_FDS=()

release_locks() {
  local index
  if (( locks_held )); then
    for (( index=${#SINGLE_LOCK_FDS[@]}-1; index>=0; index-- )); do
      flock -u "${SINGLE_LOCK_FDS[index]}" 2>/dev/null || true
    done
    if separate_gpus; then
      flock -u 7 2>/dev/null || true
      flock -u 6 2>/dev/null || true
    fi
    flock -u 9 2>/dev/null || true
    flock -u 8 2>/dev/null || true
    locks_held=0
  fi
}

cleanup() {
  set +e
  [[ -z "${proxy_log_pid}" ]] || kill "${proxy_log_pid}" 2>/dev/null
  [[ -z "${judge_pid}" ]] || kill "${judge_pid}" 2>/dev/null
  docker rm -f "${PROXY}" >/dev/null 2>&1
  docker network rm "${NETWORK}" >/dev/null 2>&1
  release_locks
}
trap cleanup EXIT INT TERM

exec 8>"/tmp/mlsysforge-gpu-${GPU6_UUID}.lock"
exec 9>"/tmp/mlsysforge-gpu-${GPU7_UUID}.lock"
# Separate mode leases the Judge pair the same way, on fds 6/7.
IDLE_UUIDS=("${GPU6_UUID}" "${GPU7_UUID}")
if separate_gpus; then
  exec 6>"/tmp/mlsysforge-gpu-${JUDGE_GPU_A_UUID}.lock"
  exec 7>"/tmp/mlsysforge-gpu-${JUDGE_GPU_B_UUID}.lock"
  IDLE_UUIDS+=("${JUDGE_GPU_A_UUID}" "${JUDGE_GPU_B_UUID}")
  # The Judge pair sits idle between Judge rounds, so separate mode also
  # holds the single-GPU family's lock (/tmp/sforge-gpu-<uuid>.lock) on all
  # four GPUs: a single-GPU launcher then refuses them instead of landing on
  # the idle Judge pair.
  for single_uuid in "${IDLE_UUIDS[@]}"; do
    exec {single_lock_fd}>"/tmp/sforge-gpu-${single_uuid}.lock"
    SINGLE_LOCK_FDS+=("${single_lock_fd}")
  done
fi

# Probe every GPU in IDLE_UUIDS (the leased pair or pairs).
gpu_pair_idle() {
  local query owners uuid used target seen=" "
  query="$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits 2>/dev/null)" || return 1
  while IFS=',' read -r uuid used; do
    uuid="${uuid//[[:space:]]/}"
    used="${used//[[:space:]]/}"
    for target in "${IDLE_UUIDS[@]}"; do
      if [[ "${uuid}" == "${target}" ]]; then
        seen+="${uuid} "
        (( used <= GPU_MAX_USED_MIB )) || return 1
      fi
    done
  done <<<"${query}"
  for target in "${IDLE_UUIDS[@]}"; do
    [[ "${seen}" == *" ${target} "* ]] || return 1
  done

  owners="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null || true)"
  for target in "${IDLE_UUIDS[@]}"; do
    ! grep -Fxq -- "${target}" <<<"${owners}" || return 1
  done

  # The single-GPU tasks lock /tmp/sforge-gpu-<uuid>.lock for their whole run
  # (Work, interactive Judges and the final Judge). Their agent is idle
  # between benchmarks, so memory and process checks alone would let this
  # launcher land on a card that is still leased and break that run's next
  # handoff. Treat a held single-GPU lock as busy. A separate-mode run that
  # already holds those locks itself has proven them free by taking them.
  if separate_gpus && (( locks_held )); then
    return 0
  fi
  local single_lock
  for target in "${IDLE_UUIDS[@]}"; do
    single_lock="/tmp/sforge-gpu-${target}.lock"
    [[ -e "${single_lock}" ]] || continue
    flock -n "${single_lock}" true 2>/dev/null || return 1
  done
}

try_pair_locks() {
  flock -n 8 || return 1
  if ! flock -n 9; then
    flock -u 8
    return 1
  fi
  if separate_gpus; then
    if ! flock -n 6; then
      flock -u 9; flock -u 8
      return 1
    fi
    if ! flock -n 7; then
      flock -u 6; flock -u 9; flock -u 8
      return 1
    fi
    local fd taken=()
    for fd in "${SINGLE_LOCK_FDS[@]}"; do
      if ! flock -n "${fd}"; then
        for fd in "${taken[@]}"; do flock -u "${fd}"; done
        flock -u 7; flock -u 6; flock -u 9; flock -u 8
        return 1
      fi
      taken+=("${fd}")
    done
  fi
  locks_held=1
}

deadline=$(( $(date +%s) + MIN_DELAY_SECONDS ))
while (( $(date +%s) < deadline )); do
  remaining=$(( deadline - $(date +%s) ))
  echo "[$(date -u +%FT%TZ)] waiting minimum delay: ${remaining}s"
  sleep "$(( remaining < POLL_SECONDS ? remaining : POLL_SECONDS ))"
done

echo "[$(date -u +%FT%TZ)] minimum delay elapsed; waiting for the idle GPU pair ${GPU_PLAN} and its locks"
while true; do
  if gpu_pair_idle && try_pair_locks; then
    # Close the race between the read-only probe and lock acquisition. Processes
    # that do not honor our lock are still caught by this second probe.
    if gpu_pair_idle; then
      break
    fi
    release_locks
  fi
  echo "[$(date -u +%FT%TZ)] GPU pair ${GPU_PLAN} busy or reserved; retrying in ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

echo "[$(date -u +%FT%TZ)] GPU pair ${GPU_PLAN} idle and locked; running frozen preflight"
# The frozen preflight checks one pair (identity, envelope, idleness, NVLink),
# so separate mode runs it once per pair.
run_pair_preflight() {
  python3 "${TASK_ROOT}/scripts/preflight.py" \
    --gpu-indices "$1" \
    --expected-gpu-uuids "$2" \
    --max-used-mib "${GPU_MAX_USED_MIB}" \
    --model-path "${MODEL_REPO}/snapshots/${MODEL_REV}" \
    --data-root "${PUBLIC_DATA}" \
    --private-data-root "${PRIVATE_DATA}"
}
run_pair_preflight "${GPU_INDICES}" "${GPU_UUIDS}"
if separate_gpus; then
  run_pair_preflight "${JUDGE_GPU_INDICES}" "${JUDGE_GPU_UUIDS}"
fi

export SFORGE_TASKS_DIR="${TASK_ROOT}/tasks"
export SFORGE_LOG_DIR="${TASK_ROOT}/logs"
export SFORGE_ADMIN_SECRET
SFORGE_ADMIN_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
export ORPO_GPU_UUIDS="${GPU_UUIDS}"
if separate_gpus; then
  # Read only by scripts/separate_judge_entry.py (the Judge server below).
  export ORPO_JUDGE_GPU_INDICES="${JUDGE_GPU_INDICES}" ORPO_JUDGE_GPU_UUIDS="${JUDGE_GPU_UUIDS}"
else
  unset ORPO_JUDGE_GPU_INDICES ORPO_JUDGE_GPU_UUIDS
fi
export ORPO_MODEL_HOST_PATH="${MODEL_REPO}"
export ORPO_DATA_HOST_ROOT="${PUBLIC_DATA}"
export ORPO_PRIVATE_DATA_HOST_ROOT="${PRIVATE_DATA}"

bridge_gateway="$(docker network inspect bridge | jq -er '.[0].IPAM.Config[0].Gateway')"
# Fail closed on a Judge port another run already owns. The readiness probe
# below only proves that *some* server answers on the port; a collision would
# otherwise register this run against a foreign Judge and fail with 403.
if ss -H -ltn "sport = :${JUDGE_PORT}" 2>/dev/null | grep -q .; then
  die "JUDGE_PORT ${JUDGE_PORT} is already bound on this host; each concurrent run needs its own port"
fi
# Separate mode serves through separate_judge_entry.py: Judge containers on
# the Judge pair, no Work pause.  Shared mode serves through the entry itself.
"${SFORGE_PYTHON}" "${SERVE_ENTRY}" serve \
  --host "${bridge_gateway}" --port "${JUDGE_PORT}" \
  >"${LOG_ROOT}/judge-server.log" 2>&1 &
judge_pid=$!
for _ in $(seq 1 90); do
  curl --noproxy '*' -fsS "http://${bridge_gateway}:${JUDGE_PORT}/openapi.json" >/dev/null 2>&1 && break
  sleep 1
done
curl --noproxy '*' -fsS "http://${bridge_gateway}:${JUDGE_PORT}/openapi.json" >/dev/null \
  || die "Judge server did not become ready"

docker network create --driver bridge --internal \
  --label "mlsysforge.run_id=${RUN_ID}" "${NETWORK}" >/dev/null
docker run --detach --rm \
  --name "${PROXY}" \
  --network bridge \
  --add-host host.docker.internal:host-gateway \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --volume "${PROXY_SOURCE}:/opt/mlsysforge/restricted_proxy.py:ro" \
  --entrypoint python3 \
  "${WORK_IMAGE}" \
  /opt/mlsysforge/restricted_proxy.py \
  --evaluation-upstream-host "${bridge_gateway}" \
  --evaluation-upstream-port "${JUDGE_PORT}" \
  "${PROXY_EGRESS_ARGS[@]}" \
  ${model_api_host:+--model-forward "${MODEL_FORWARD_PORT}:${model_api_host}:${model_api_port}"} >/dev/null
docker logs -f "${PROXY}" >"${LOG_ROOT}/host-proxy-bridge.log" 2>&1 &
proxy_log_pid=$!
docker network connect --alias runtime-proxy "${NETWORK}" "${PROXY}"
proxy_ip="$(docker inspect "${PROXY}" | jq -er --arg network "${NETWORK}" '.[0].NetworkSettings.Networks[$network].IPAddress')"
export NO_PROXY="${NO_PROXY:-},${bridge_gateway},${proxy_ip}"
export no_proxy="${NO_PROXY}"

for _ in $(seq 1 30); do
  if docker run --rm --network "${NETWORK}" --env NVIDIA_VISIBLE_DEVICES=void \
      --entrypoint curl "${WORK_IMAGE}" --connect-timeout 2 --max-time 3 \
      -fsS "http://${proxy_ip}:3128/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker run --rm --network "${NETWORK}" --env NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint curl "${WORK_IMAGE}" --connect-timeout 2 --max-time 5 \
  -fsS "http://${proxy_ip}:3128/health" >/dev/null \
  || die "restricted proxy did not become ready"
docker run --rm --network "${NETWORK}" --env NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint curl "${WORK_IMAGE}" --connect-timeout 2 --max-time 5 \
  -fsS "http://${proxy_ip}:3130/openapi.json" >/dev/null \
  || die "Judge is not reachable from the isolated Work network"

if [[ -n "${MODEL_API_BASE_URL}" ]]; then
  export SFORGE_AGENT_API_BASE_URL="http://${proxy_ip}:${MODEL_FORWARD_PORT}${model_api_path}"
  export SFORGE_AGENT_API_KEY="${MODEL_API_KEY}"
  unset SFORGE_CODEX_AUTH_JSON_PATH CODEX_API_KEY OPENAI_API_KEY OPENAI_BASE_URL
elif [[ "${AGENT_NAME}" == claude-code ]]; then
  unset SFORGE_CODEX_AUTH_JSON_PATH CODEX_API_KEY OPENAI_API_KEY OPENAI_BASE_URL
  export MLSYSFORGE_CLAUDE_OAUTH_TOKEN_FILE="${CLAUDE_TOKEN_FILE}"
else
  export SFORGE_CODEX_AUTH_JSON_PATH="${AUTH_JSON}"
fi
export SFORGE_CODEX_REASONING_EFFORT="${REASONING_EFFORT}"
export MLSYSFORGE_AGENT="${AGENT_NAME}" MLSYSFORGE_AGENT_MODEL="${AGENT_MODEL}"
export MLSYSFORGE_AGENT_EFFORT="${REASONING_EFFORT}" MLSYSFORGE_AGENT_BUDGET_SECONDS="${TASK_TIMEOUT}"
export MLSYSFORGE_RUN_LOG_ROOT="${LOG_ROOT}"
export SFORGE_ISOLATED_DOCKER_NETWORK="${NETWORK}"
# --enable-internet below only bypasses SForge's sudo-iptables isolation; the
# internal network and the sidecar are the enforcement boundary.  The entry
# reads the task's internet flag from here: true keeps Codex web search live,
# false restores the offline Codex flags and prompt.
export MLSYSFORGE_WORK_INTERNET="${TASK_INTERNET}"
export SFORGE_HTTP_PROXY="http://${proxy_ip}:3128"
export SFORGE_HTTPS_PROXY="http://${proxy_ip}:3128"
export SFORGE_NO_PROXY="${proxy_ip},127.0.0.1,localhost"

echo "[$(date -u +%FT%TZ)] starting managed run"
echo "autojudge is OFF; only Agent-triggered sforge-submit evaluations can run"
set +e
"${SFORGE_PYTHON}" "${ENTRY}" run \
  --backend docker \
  --task "${TASK_ID}" \
  --agent "${AGENT_NAME}" \
  --model "${AGENT_MODEL}" \
  --timeout "${TASK_TIMEOUT}" \
  --disable-auto-eval \
  --submission-cooldown "${SUBMISSION_COOLDOWN}" \
  --judge-url "http://${proxy_ip}:3130" \
  --enable-internet \
  --work-cpu-limit 16 \
  --work-mem-limit 96g \
  --judge-cpu-limit 16 \
  --judge-mem-limit 96g \
  --run-id "${RUN_ID}"
rc=$?
set -e
printf '%s\n' "${rc}" >"${LOG_ROOT}/launcher.exit"
echo "[$(date -u +%FT%TZ)] managed run finished rc=${rc}"

# The authoritative score is a fresh Judge of the final snapshot;
# run it here on the Judge pair (the Work pair in shared mode) while this
# launcher still holds every pair lock.
final_archive="${LOG_ROOT}/${TASK_ID}/final_archive.tar.gz"
if [[ "${FINAL_EVAL:-1}" == 1 && -s "${final_archive}" ]]; then
  echo "[$(date -u +%FT%TZ)] final Judge of ${final_archive} on pair ${JUDGE_GPU_INDICES}"
  set +e
  ORPO_GPU_INDICES="${JUDGE_GPU_INDICES}" ORPO_GPU_UUIDS="${JUDGE_GPU_UUIDS}" LOGS_DIR="${LOG_ROOT}/final-eval" \
    "${TASK_ROOT}/scripts/eval_archive.sh" "${final_archive}" "${RUN_ID}-final" \
    >"${LOG_ROOT}/final-eval.log" 2>&1
  final_rc=$?
  set -e
  printf '%s\n' "${final_rc}" >"${LOG_ROOT}/final-eval.exit"
  echo "[$(date -u +%FT%TZ)] final Judge finished rc=${final_rc} (report under ${LOG_ROOT}/final-eval)"
fi
exit "${rc}"
