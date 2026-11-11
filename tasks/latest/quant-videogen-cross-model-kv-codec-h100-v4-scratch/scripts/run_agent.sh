#!/usr/bin/env bash
set -Eeuo pipefail
umask 022  # private outputs (model lock, Judge reports) refuse group-writable parents

# Host (native Docker) fixed-budget agent run: one Work GPU, 1-8 Judge GPUs.
#
#   WORK_GPU=<index|uuid> JUDGE_GPUS=<i,j,...> JUDGE_PORT=<port> MODEL=<id> \
#     scripts/run_agent.sh            # QVG_DRY_RUN=1: print the plan, start nothing
#
# launch/quant-videogen/run.sh sets these from its flags.  Work gets exactly
# one GPU; every interactive Judge and the independent final Judge run the
# complete 17-case hidden suite spread over the Judge GPUs (QVG_JUDGE_GPU_UUIDS).
# Both host lock names of every GPU are held for the whole run, final included.
# The Judge's CPU/memory share and its timeout come from lib.sh judge_plan for
# the Judge GPU count (the task JSON's eval_timeout is sized for eight GPUs);
# scripts/sforge_agent_entry.py carries that timeout into the Judge server and
# the Work-side sforge-submit wait.  Work sits on an internal Docker network
# whose only way out is a restricted CONNECT sidecar (task JSON internet: true,
# so public hosts on 80/443 are allowed, internal addresses are not).
#
# Agents: MLSYSFORGE_AGENT=codex (default; CODEX_AUTH_JSON_PATH, or an
# OpenAI-compatible MODEL_API_BASE_URL + mode-0600 MODEL_API_KEY_FILE relayed by
# the sidecar) or claude-code (CLAUDE_OAUTH_TOKEN_FILE).  The SFORGE_LLM_JUDGE_*
# integrity review must be configured and enforcing; the run refuses otherwise.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

AGENT_ENTRY="${SCRIPT_DIR}/sforge_agent_entry.py"
TASK_JSON="${TASKS_DIR}/${TASK_ID}.json"
# The repository's shared Work sidecar (public egress + model relay), as the
# Flux launcher uses it.
PROXY_SOURCE="${REPOSITORY_ROOT}/tasks/latest/llada-h100-decoder-optimization-v2-vanilla-seed/scripts/restricted_connect_proxy.py"
EVALUATION_FORWARD_PORT=3130
MODEL_FORWARD_PORT=3131
SUBMISSION_COOLDOWN=300

DRY_RUN="${QVG_DRY_RUN:-0}"
AGENT_NAME="${MLSYSFORGE_AGENT:-codex}"
MODEL="${MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
TASK_TIMEOUT="${TASK_TIMEOUT:-86400}"
WORK_GPU_SELECTOR="${WORK_GPU:-}"
JUDGE_GPU_SELECTORS="${JUDGE_GPUS:-}"
JUDGE_PORT="${JUDGE_PORT:-}"
AUTH_JSON="${CODEX_AUTH_JSON_PATH:-${HOME}/.codex/auth.json}"
CLAUDE_TOKEN_FILE="${CLAUDE_OAUTH_TOKEN_FILE:-}"
MODEL_API_BASE_URL="${MODEL_API_BASE_URL:-}"
MODEL_API_KEY_FILE="${MODEL_API_KEY_FILE:-}"
CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-0.148.0}"
FINAL_EVAL="${FINAL_EVAL:-1}"
SKIP_MODEL_VERIFY="${QVG_SKIP_MODEL_VERIFY:-0}"

[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || die "QVG_DRY_RUN must be 0 or 1"
[[ "${FINAL_EVAL}" == 0 || "${FINAL_EVAL}" == 1 ]] || die "FINAL_EVAL must be 0 or 1"
[[ "${SKIP_MODEL_VERIFY}" == 0 || "${SKIP_MODEL_VERIFY}" == 1 ]] \
  || die "QVG_SKIP_MODEL_VERIFY must be 0 or 1"
[[ -n "${MODEL}" ]] || die "set MODEL (launch/quant-videogen/run.sh --model)"
[[ "${TASK_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || die "TASK_TIMEOUT must be a positive integer"
[[ -n "${WORK_GPU_SELECTOR}" ]] || die "set WORK_GPU to the Work GPU (index or UUID)"
[[ -n "${JUDGE_GPU_SELECTORS}" ]] || die "set JUDGE_GPUS to the Judge GPUs, e.g. JUDGE_GPUS=1,2,3"
[[ "${JUDGE_PORT}" =~ ^[0-9]+$ ]] || die "set JUDGE_PORT to a run-unique TCP port"
JUDGE_PORT="$((10#${JUDGE_PORT}))"
(( JUDGE_PORT >= 1024 && JUDGE_PORT <= 65535 && JUDGE_PORT != 3128 \
   && JUDGE_PORT != EVALUATION_FORWARD_PORT && JUDGE_PORT != MODEL_FORWARD_PORT )) \
  || die "JUDGE_PORT must be 1024..65535 and not a sidecar port"
case "${AGENT_NAME}" in
  codex)
    case "${REASONING_EFFORT}" in max|ultra|xhigh|high|medium|low) ;;
      *) die "unsupported Codex reasoning effort: ${REASONING_EFFORT}" ;; esac
    ;;
  claude-code)
    [[ -z "${MODEL_API_BASE_URL}" ]] || die "Claude Code uses its OAuth token; unset MODEL_API_BASE_URL"
    case "${REASONING_EFFORT}" in max|xhigh|high|medium|low) ;;
      *) die "Claude Code efforts are low|medium|high|xhigh|max" ;; esac
    ;;
  *) die "MLSYSFORGE_AGENT must be codex or claude-code" ;;
esac
[[ "$(jq -r '.internet' "${TASK_JSON}")" == true ]] \
  || die "this launcher assumes the task JSON's internet: true"

# A label-safe short model name: Example-Model -> example-model, ZHIPU/GLM-5.3-Flash -> zhipu-glm-5.3-flash.
model_short="$(tr 'A-Z' 'a-z' <<<"${MODEL}" | sed -E 's/[^a-z0-9.]+/-/g; s/^-+//; s/-+$//')"
budget_label="$(( TASK_TIMEOUT / 3600 ))h"
(( TASK_TIMEOUT % 3600 == 0 )) || budget_label="${TASK_TIMEOUT}s"
RUN_LABEL="${RUN_LABEL:-${model_short}-${REASONING_EFFORT}-${budget_label}}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ID="${RUN_ID:-qvg-v4-${RUN_LABEL}-${RUN_STAMP}}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ && ${#RUN_ID} -le 120 ]] \
  || die "unsafe or oversized RUN_ID: ${RUN_ID}"
RUN_ROOT="${LOGS_DIR}/runs/${RUN_ID}"

# --- GPUs and the Judge plan ------------------------------------------------
check_failures=()
fail_check() { check_failures+=("$*"); }

require_command nvidia-smi
judge_uuids=(); judge_indices=()
if nvidia-smi -L >/dev/null 2>&1; then
  work_index="$(gpu_index "${WORK_GPU_SELECTOR}")"
  work_uuid="$(gpu_uuid "${work_index}")"
  [[ "${work_uuid}" =~ ^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]] \
    || die "Work GPU ${work_index} has no whole-device UUID"
  judge_uuid_lines="$(judge_gpu_uuids "${JUDGE_GPU_SELECTORS}")"
  mapfile -t judge_uuids <<<"${judge_uuid_lines}"
  for uuid in "${judge_uuids[@]}"; do
    [[ "${uuid}" != "${work_uuid}" ]] || die "the Work GPU ${work_index} is also a Judge GPU"
    judge_indices+=("$(gpu_index "${uuid}")")
  done
else
  # Only a dry run may continue: it shows the plan with unresolved UUIDs.
  (( DRY_RUN )) || die "nvidia-smi cannot enumerate the GPUs"
  fail_check "nvidia-smi cannot enumerate the GPUs; UUIDs, idle state and locks are unresolved"
  work_index="${WORK_GPU_SELECTOR}"; work_uuid="(unresolved)"
  IFS=, read -r -a judge_indices <<<"${JUDGE_GPU_SELECTORS}"
  for selector in "${judge_indices[@]}"; do
    [[ "${selector}" != "${WORK_GPU_SELECTOR}" ]] || die "the Work GPU ${selector} is also a Judge GPU"
    judge_uuids+=("(unresolved)")
  done
  (( ${#judge_uuids[@]} >= 1 && ${#judge_uuids[@]} <= 8 )) || die "JUDGE_GPUS must name 1 to 8 GPUs"
fi
n_judge="${#judge_uuids[@]}"
plan="$(judge_plan "${n_judge}")"
read -r plan_cpu plan_mem controller_seconds eval_seconds <<<"${plan}"
judge_cpu="${JUDGE_CPU_LIMIT:-${plan_cpu}}"
judge_mem="${JUDGE_MEM_LIMIT:-${plan_mem}}"
eval_timeout="${FINAL_EVAL_TIMEOUT:-${eval_seconds}}"
[[ "${eval_timeout}" =~ ^[1-9][0-9]*$ ]] && (( eval_timeout > controller_seconds )) \
  || die "the Judge timeout must exceed the controller bound of ${controller_seconds} s for ${n_judge} GPU(s)"
work_cpu="$(jq -er '.work.cpu_limit' "${TASK_JSON}")"
work_mem="$(jq -er '.work.mem_limit' "${TASK_JSON}")"

# --- Agent credentials and runtime -----------------------------------------

model_api_host=""; model_api_port=""; model_api_path=""
if [[ -n "${MODEL_API_BASE_URL}" ]]; then
  if [[ "${MODEL_API_BASE_URL}" =~ ^http://([^/:[:space:]]+):([0-9]+)(/.*)?$ ]]; then
    model_api_host="${BASH_REMATCH[1]}"; model_api_port="${BASH_REMATCH[2]}"
    model_api_path="${BASH_REMATCH[3]:-}"
  else
    fail_check "MODEL_API_BASE_URL must be http://HOST:PORT[/path] (the sidecar relays raw TCP)"
  fi
  if [[ -z "${MODEL_API_KEY_FILE}" ]]; then
    fail_check "set MODEL_API_KEY_FILE with MODEL_API_BASE_URL"
  else
    ( assert_private_auth_file "${MODEL_API_KEY_FILE}" ) 2>/dev/null \
      || fail_check "MODEL_API_KEY_FILE must be a nonempty mode-600 file owned by this user"
  fi
  credential="model API ${MODEL_API_BASE_URL} (key file ${MODEL_API_KEY_FILE})"
elif [[ "${AGENT_NAME}" == claude-code ]]; then
  [[ -n "${CLAUDE_TOKEN_FILE}" && -f "${CLAUDE_TOKEN_FILE}" && ! -L "${CLAUDE_TOKEN_FILE}" \
     && "$(stat -c '%a' "${CLAUDE_TOKEN_FILE}")" == 600 ]] \
    || fail_check "set CLAUDE_OAUTH_TOKEN_FILE to a mode-600 regular token file"
  credential="Claude OAuth token file ${CLAUDE_TOKEN_FILE}"
else
  ( assert_private_auth_file "${AUTH_JSON}" ) 2>/dev/null \
    || fail_check "Codex auth file must be a nonempty mode-600 file owned by this user: ${AUTH_JSON}"
  credential="Codex auth ${AUTH_JSON}"
fi

AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}"
runtime_marker="${AGENT_RUNTIME_DIR}/.sforge-agent-runtime.json"
runtime_agent=""; runtime_version=""; runtime_tree_sha256=""
if [[ -n "${AGENT_RUNTIME_DIR}" && -d "${AGENT_RUNTIME_DIR}" && ! -L "${AGENT_RUNTIME_DIR}" \
      && -f "${runtime_marker}" ]] \
   && [[ "$(jq -r '.kind' "${runtime_marker}")" == sforge-agent-runtime ]]; then
  runtime_agent="$(jq -r '.agent' "${runtime_marker}")"
  runtime_version="$(jq -r '.version' "${runtime_marker}")"
  runtime_tree_sha256="$(jq -r '.tree_sha256' "${runtime_marker}")"
  [[ "${runtime_agent}" == "${AGENT_NAME}" ]] \
    || fail_check "SFORGE_AGENT_RUNTIME_DIR is a ${runtime_agent} runtime, not ${AGENT_NAME}"
  if [[ "${AGENT_NAME}" == codex && "${runtime_version}" != "${CODEX_CLI_VERSION}" ]]; then
    fail_check "Codex runtime ${runtime_version} differs from CODEX_CLI_VERSION ${CODEX_CLI_VERSION}"
  fi
else
  fail_check "SFORGE_AGENT_RUNTIME_DIR is missing or unmarked: ${AGENT_RUNTIME_DIR:-<unset>}"
fi

# The integrity review is mandatory and must enforce (docs/LLM_JUDGE.md).
[[ "${SFORGE_LLM_JUDGE_MODE:-}" == enforce ]] \
  || fail_check "set SFORGE_LLM_JUDGE_MODE=enforce and the SFORGE_LLM_JUDGE_* reviewer settings"
review_summary=""
if [[ -x "${SFORGE_PYTHON}" ]]; then
  if ! review_summary="$(
    cd -- "${REPOSITORY_ROOT}" && "${SFORGE_PYTHON}" -B - <<'PY' 2>&1
from sforge.harness.llm_judge.config import load_llm_judge_config
config = load_llm_judge_config()
config.check_ready()
print(f"{config.mode} {config.engine} {config.agent} {config.model} "
      f"effort={config.reasoning_effort} runtime={config.agent_runtime_dir}")
PY
  )"; then
    fail_check "integrity review is not ready: ${review_summary##*$'\n'}"
    review_summary=""
  fi
else
  fail_check "missing SForge Python: ${SFORGE_PYTHON}"
fi

[[ -d "${ARTIFACT_ROOT}" ]] || fail_check "QVG_ARTIFACT_ROOT is not a directory: ${ARTIFACT_ROOT:-<unset>} (run assets/fetch.sh, or set QVG_ARTIFACT_ROOT)"
[[ -f "${PROXY_SOURCE}" && ! -L "${PROXY_SOURCE}" ]] || fail_check "missing Work sidecar: ${PROXY_SOURCE}"
[[ -n "${SFORGE_VERSION_STORE_ROOT:-}" ]] \
  || fail_check "set SFORGE_VERSION_STORE_ROOT (launch/quant-videogen/run.sh resolves it)"
if ss -H -ltn "sport = :${JUDGE_PORT}" 2>/dev/null | grep -q .; then
  fail_check "JUDGE_PORT ${JUDGE_PORT} is already bound on this host"
fi
[[ ! -e "${RUN_ROOT}" ]] || fail_check "run directory already exists: ${RUN_ROOT}"

lock_state() {
  local uuid="$1" path state=free
  [[ "${uuid}" == GPU-* ]] || { echo unknown; return; }
  for path in "$(gpu_lock_path "${uuid}")" "${GPU_LOCK_DIR}/mlsysforge-gpu-${uuid}.lock"; do
    [[ -e "${path}" ]] || continue
    flock -n "${path}" true 2>/dev/null || state="held (${path##*/})"
  done
  echo "${state}"
}

if (( DRY_RUN )); then
  image_state="current"
  ( assert_built_images_current ) >/dev/null 2>&1 \
    || image_state="NOT current (scripts/build_images.sh)"
  echo "Quant-VideoGen v4 agent run plan (dry run: no lock, container or GPU work)"
  echo "  run id:        ${RUN_ID}"
  echo "  run root:      ${RUN_ROOT}"
  echo "  agent:         ${AGENT_NAME} model=${MODEL} effort=${REASONING_EFFORT}"
  echo "  credential:    ${credential}"
  echo "  runtime:       ${AGENT_RUNTIME_DIR:-<unset>} (${runtime_agent:-?} ${runtime_version:-?})"
  echo "  budget:        ${TASK_TIMEOUT} s, submission cooldown ${SUBMISSION_COOLDOWN} s, auto-eval off"
  echo "  work GPU:      ${work_index} ${work_uuid} [lock $(lock_state "${work_uuid}")]"
  for i in "${!judge_uuids[@]}"; do
    echo "  judge GPU:     ${judge_indices[$i]} ${judge_uuids[$i]} [lock $(lock_state "${judge_uuids[$i]}")]"
  done
  echo "  work limits:   ${work_cpu} CPU, ${work_mem}"
  echo "  judge plan:    ${n_judge} GPU(s), ${judge_cpu} CPU, ${judge_mem}, controller bound ${controller_seconds} s, Judge timeout ${eval_timeout} s (interactive and final)"
  echo "  judge port:    ${JUDGE_PORT} (Docker bridge gateway only)"
  echo "  network:       internal Work network + restricted sidecar, public egress 80/443${model_api_host:+, model relay ${MODEL_FORWARD_PORT} -> ${model_api_host}:${model_api_port}}"
  echo "  review:        ${review_summary:-<not ready>}"
  echo "  images:        ${image_state}"
  echo "  artifact root: ${ARTIFACT_ROOT:-<unset>}$([[ "${SKIP_MODEL_VERIFY}" == 1 ]] && echo ' (model verification skipped)')"
  echo "  version store: ${SFORGE_VERSION_STORE_ROOT:-<unset>}"
  echo "  final Judge:   $([[ "${FINAL_EVAL}" == 1 ]] && echo "on the same Judge GPUs under the held locks -> ${RUN_ROOT}/final-eval" || echo off)"
  echo "  cpuset:        work=${SFORGE_WORK_CPUSET:-${SFORGE_CPUSET:-off}} judge=${SFORGE_JUDGE_CPUSET:-${SFORGE_CPUSET:-off}}"
  if (( ${#check_failures[@]} )); then
    printf '  CHECK FAILED:  %s\n' "${check_failures[@]}"
    exit 1
  fi
  echo "  checks:        PASS"
  exit 0
fi
if (( ${#check_failures[@]} )); then
  printf 'ERROR: %s\n' "${check_failures[@]}" >&2
  exit 1
fi

# --- Live run ----------------------------------------------------------------
for command in curl docker flock jq python3 sha256sum ss stat; do
  require_command "${command}"
done
docker info >/dev/null 2>&1 \
  || die "this shell cannot reach the Docker daemon; launch from a docker-group session"
require_sforge
require_artifact_root
assert_built_images_current

for uuid in "${work_uuid}" "${judge_uuids[@]}"; do
  lock_gpu "${uuid}" "$([[ "${uuid}" == "${work_uuid}" ]] && echo work || echo judge)"
done
assert_gpu_idle "${work_uuid}" work
assert_no_compute_process "${work_uuid}" work
for uuid in "${judge_uuids[@]}"; do
  assert_gpu_idle "${uuid}" judge
  assert_no_compute_process "${uuid}" judge
done
assert_identical_gpus judge "${judge_uuids[@]}"

# verify_models.py and the private Judge outputs refuse a group/world-writable
# parent, so the run root must not inherit a permissive umask (e.g. 0002).
mkdir -p "${RUN_ROOT}"
chmod 0755 "${RUN_ROOT}"
exec > >(tee -a "${RUN_ROOT}/launcher.log") 2>&1
echo "[$(date -u +%FT%TZ)] ${RUN_ID}: GPUs locked (work ${work_index}, judge ${judge_indices[*]})"

model_lock_sha256="skipped"
if [[ "${SKIP_MODEL_VERIFY}" != 1 ]]; then
  echo "[$(date -u +%FT%TZ)] verifying the external models and assets"
  python3 "${SCRIPT_DIR}/verify_models.py" --artifact-root "${ARTIFACT_ROOT}" \
    --output "${RUN_ROOT}/model-lock.json" >/dev/null
  model_lock_sha256="$(sha256sum "${RUN_ROOT}/model-lock.json" | awk '{print $1}')"
fi

MODEL_API_KEY=""
if [[ -n "${MODEL_API_BASE_URL}" ]]; then
  MODEL_API_KEY="$(tr -d '[:space:]' <"${MODEL_API_KEY_FILE}")"
  [[ -n "${MODEL_API_KEY}" ]] || die "MODEL_API_KEY_FILE is empty"
fi

network_name="${RUN_ID}-internal"
proxy_name="${RUN_ID}-proxy"
judge_pid=""
proxy_log_pid=""
cleanup() {
  set +e
  [[ -z "${proxy_log_pid}" ]] || kill "${proxy_log_pid}" 2>/dev/null
  if [[ -n "${judge_pid}" ]] && kill -0 "${judge_pid}" 2>/dev/null; then
    kill "${judge_pid}" 2>/dev/null
    wait "${judge_pid}" 2>/dev/null
  fi
  docker rm -f "${proxy_name}" >/dev/null 2>&1
  docker network rm "${network_name}" >/dev/null 2>&1
}
trap cleanup EXIT
trap 'exit 143' HUP INT TERM

export SFORGE_ADMIN_SECRET
SFORGE_ADMIN_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
# The Judge takes every listed GPU from QVG_JUDGE_GPU_UUIDS; a single-device
# extra-env mask would hide all but one.  The Work adapter refuses a Work GPU
# that is also a Judge GPU.
unset SFORGE_JUDGE_EXTRA_ENV
export QVG_JUDGE_GPU_UUIDS="${judge_uuids[*]}"
export QVG_JUDGE_EVAL_TIMEOUT="${eval_timeout}"

bridge_gateway="$(docker network inspect bridge | jq -er '.[0].IPAM.Config[0].Gateway')"
[[ "${bridge_gateway}" =~ ^[0-9]+(\.[0-9]+){3}$ ]] || die "could not resolve the Docker bridge gateway"
if ss -H -ltn "sport = :${JUDGE_PORT}" 2>/dev/null | grep -q .; then
  die "JUDGE_PORT ${JUDGE_PORT} is already bound on this host"
fi
"${SFORGE_PYTHON}" "${AGENT_ENTRY}" serve --host "${bridge_gateway}" --port "${JUDGE_PORT}" \
  >"${RUN_ROOT}/judge-server.log" 2>&1 &
judge_pid=$!
for _ in $(seq 1 90); do
  curl --noproxy '*' -fsS "http://${bridge_gateway}:${JUDGE_PORT}/openapi.json" >/dev/null 2>&1 && break
  kill -0 "${judge_pid}" 2>/dev/null || die "Judge server exited; see ${RUN_ROOT}/judge-server.log"
  sleep 1
done
curl --noproxy '*' -fsS "http://${bridge_gateway}:${JUDGE_PORT}/openapi.json" >/dev/null \
  || die "Judge server did not become ready"

docker network create --driver bridge --internal \
  --label "mlsysforge.run_id=${RUN_ID}" "${network_name}" >/dev/null
docker run --detach --rm --name "${proxy_name}" --network bridge \
  --add-host host.docker.internal:host-gateway --env NVIDIA_VISIBLE_DEVICES=void \
  --volume "${PROXY_SOURCE}:/opt/mlsysforge/restricted_proxy.py:ro" \
  --entrypoint python3 "${WORK_IMAGE}" /opt/mlsysforge/restricted_proxy.py \
  --evaluation-upstream-host "${bridge_gateway}" \
  --evaluation-upstream-port "${JUDGE_PORT}" \
  --allow-public-egress \
  ${model_api_host:+--model-forward "${MODEL_FORWARD_PORT}:${model_api_host}:${model_api_port}"} >/dev/null
docker logs -f "${proxy_name}" >"${RUN_ROOT}/sidecar.log" 2>&1 &
proxy_log_pid=$!
docker network connect --alias runtime-proxy "${network_name}" "${proxy_name}"
proxy_ip="$(docker inspect "${proxy_name}" | jq -er --arg n "${network_name}" '.[0].NetworkSettings.Networks[$n].IPAddress | select(length > 0)')"

in_work_network() {
  docker run --rm --network "${network_name}" --env NVIDIA_VISIBLE_DEVICES=void \
    --entrypoint curl "${WORK_IMAGE}" --noproxy '*' --connect-timeout 2 "$@"
}
for _ in $(seq 1 30); do
  in_work_network --max-time 3 -fsS "http://${proxy_ip}:3128/health" >/dev/null 2>&1 && break
  sleep 1
done
in_work_network --max-time 5 -fsS "http://${proxy_ip}:3128/health" >/dev/null \
  || die "restricted sidecar did not become ready"
in_work_network --max-time 5 -fsS "http://${proxy_ip}:${EVALUATION_FORWARD_PORT}/openapi.json" >/dev/null \
  || die "Judge is not reachable from the internal Work network"
if in_work_network --max-time 4 -fsS https://example.com >/dev/null 2>&1; then
  die "internal Work network unexpectedly has direct egress"
fi
if [[ -n "${model_api_host}" ]]; then
  in_work_network --max-time 10 -sS -o /dev/null \
    "http://${proxy_ip}:${MODEL_FORWARD_PORT}${model_api_path}/models" \
    || die "the model API relay ${MODEL_API_BASE_URL} is unreachable through the sidecar"
fi

# Host-side registration and polling address the Judge directly.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${bridge_gateway},${proxy_ip},127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
unset SFORGE_AGENT_API_KEY SFORGE_AGENT_API_BASE_URL SFORGE_CODEX_AUTH_JSON_PATH \
  CODEX_API_KEY OPENAI_API_KEY OPENAI_BASE_URL
if [[ -n "${MODEL_API_BASE_URL}" ]]; then
  export SFORGE_AGENT_API_BASE_URL="http://${proxy_ip}:${MODEL_FORWARD_PORT}${model_api_path}"
  export SFORGE_AGENT_API_KEY="${MODEL_API_KEY}"
elif [[ "${AGENT_NAME}" == claude-code ]]; then
  export MLSYSFORGE_CLAUDE_OAUTH_TOKEN_FILE="${CLAUDE_TOKEN_FILE}"
else
  export SFORGE_CODEX_AUTH_JSON_PATH="${AUTH_JSON}"
fi
export SFORGE_CODEX_REASONING_EFFORT="${REASONING_EFFORT}"
export SFORGE_CODEX_CLI_VERSION="${CODEX_CLI_VERSION}"
export MLSYSFORGE_AGENT="${AGENT_NAME}" MLSYSFORGE_AGENT_MODEL="${MODEL}"
export MLSYSFORGE_AGENT_EFFORT="${REASONING_EFFORT}" MLSYSFORGE_AGENT_BUDGET_SECONDS="${TASK_TIMEOUT}"
export MLSYSFORGE_RUN_LOG_ROOT="${RUN_ROOT}"
export SFORGE_AGENT_RUNTIME_DIR="${AGENT_RUNTIME_DIR}"
export SFORGE_ISOLATED_DOCKER_NETWORK="${network_name}"
export SFORGE_HTTP_PROXY="http://${proxy_ip}:3128"
export SFORGE_HTTPS_PROXY="http://${proxy_ip}:3128"
export SFORGE_NO_PROXY="${proxy_ip},127.0.0.1,localhost"
export SFORGE_AGENT_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${work_uuid},CUDA_VISIBLE_DEVICES=0"

work_image_id="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")"
judge_image_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
jq -n \
  --arg run_id "${RUN_ID}" --arg task_id "${TASK_ID}" \
  --arg agent "${AGENT_NAME}" --arg model "${MODEL}" --arg effort "${REASONING_EFFORT}" \
  --arg started_at "$(date -u +%FT%TZ)" \
  --arg work_index "${work_index}" --arg work_uuid "${work_uuid}" \
  --arg judge_indices "${judge_indices[*]}" --arg judge_uuids "${judge_uuids[*]}" \
  --arg judge_mem "${judge_mem}" --arg work_mem "${work_mem}" \
  --arg runtime_dir "${AGENT_RUNTIME_DIR}" --arg runtime_agent "${runtime_agent}" \
  --arg runtime_version "${runtime_version}" --arg runtime_tree_sha256 "${runtime_tree_sha256}" \
  --arg work_image_id "${work_image_id}" --arg judge_image_id "${judge_image_id}" \
  --arg release_sha256 "$(sha256sum "${RELEASE_PATH}" | awk '{print $1}')" \
  --arg model_lock_sha256 "${model_lock_sha256}" \
  --arg model_api "${MODEL_API_BASE_URL}" \
  --argjson timeout "${TASK_TIMEOUT}" --argjson cooldown "${SUBMISSION_COOLDOWN}" \
  --argjson judge_cpu "${judge_cpu}" --argjson work_cpu "${work_cpu}" \
  --argjson controller_seconds "${controller_seconds}" --argjson eval_timeout "${eval_timeout}" \
  '{schema_version: 1, run_id: $run_id, task_id: $task_id, started_at: $started_at,
    agent: $agent, model: $model, reasoning_effort: $effort,
    model_api_base_url: (if $model_api == "" then null else $model_api end),
    agent_runtime: {dir: $runtime_dir, agent: $runtime_agent, version: $runtime_version,
      tree_sha256: $runtime_tree_sha256},
    timeout_seconds: $timeout, submission_cooldown_seconds: $cooldown,
    scheduled_auto_evaluation: false, internet: true,
    work: {gpu: {index: $work_index, uuid: $work_uuid}, cpu_limit: $work_cpu, mem_limit: $work_mem},
    judge: {gpu_indices: ($judge_indices | split(" ")), gpu_uuids: ($judge_uuids | split(" ")),
      cpu_limit: $judge_cpu, mem_limit: $judge_mem,
      controller_bound_seconds: $controller_seconds, eval_timeout_seconds: $eval_timeout},
    integrity_review: "enforce",
    images: {work: $work_image_id, judge: $judge_image_id},
    release_sha256: $release_sha256, model_lock_sha256: $model_lock_sha256,
    final_evaluation: "independent full hidden suite on the same Judge GPUs under the held locks"}' \
  >"${RUN_ROOT}/launch-manifest.json"

echo "[$(date -u +%FT%TZ)] starting ${RUN_ID}: ${AGENT_NAME} ${MODEL}/${REASONING_EFFORT}, ${TASK_TIMEOUT} s"
echo "  work GPU ${work_index}; judge GPUs ${judge_indices[*]} (${judge_cpu} CPU, ${judge_mem}, timeout ${eval_timeout} s)"
set +e
"${SFORGE_PYTHON}" "${AGENT_ENTRY}" run \
  --backend docker \
  --task "${TASK_ID}" \
  --agent "${AGENT_NAME}" \
  --model "${MODEL}" \
  --timeout "${TASK_TIMEOUT}" \
  --disable-auto-eval \
  --submission-cooldown "${SUBMISSION_COOLDOWN}" \
  --judge-url "http://${proxy_ip}:${EVALUATION_FORWARD_PORT}" \
  --enable-internet \
  --work-cpu-limit "${work_cpu}" \
  --work-mem-limit "${work_mem}" \
  --judge-cpu-limit "${judge_cpu}" \
  --judge-mem-limit "${judge_mem}" \
  --run-id "${RUN_ID}"
run_rc=$?
set -e
printf '%s\n' "${run_rc}" >"${RUN_ROOT}/launcher.exit"
echo "[$(date -u +%FT%TZ)] managed run finished rc=${run_rc}"

# The authoritative score is an independent Judge of the final archive, run
# here on the same Judge GPUs while this shell still holds every lock.  It is
# the scripts/eval_archive.sh invocation without that script's own locking.
final_archive="${RUN_ROOT}/${TASK_ID}/final_archive.tar.gz"
if [[ "${FINAL_EVAL}" == 1 && -s "${final_archive}" ]]; then
  # The final runs its own in-process Judge; the interactive server is done.
  kill "${judge_pid}" 2>/dev/null || true
  wait "${judge_pid}" 2>/dev/null || true
  judge_pid=""
  set +e
  (
    set -e
    assert_gpu_released "${work_uuid}" Work
    for uuid in "${judge_uuids[@]}"; do
      assert_gpu_released "${uuid}" judge
      assert_gpu_idle "${uuid}" judge
    done
    echo "Judge plan: ${n_judge} GPU(s), ${judge_cpu} CPUs, ${judge_mem}, controller bound ${controller_seconds} s, eval timeout ${eval_timeout} s"
    SFORGE_LOG_DIR="${RUN_ROOT}/final-eval" "${SFORGE_PYTHON}" "${RUNTIME_ENTRY}" eval \
      --backend docker \
      --task "${TASK_ID}" \
      --archive "${final_archive}" \
      --run-id "${RUN_ID}-final" \
      --timeout "${eval_timeout}" \
      --judge-cpu-limit "${judge_cpu}" \
      --judge-mem-limit "${judge_mem}" \
      --json
    for uuid in "${judge_uuids[@]}"; do
      assert_gpu_released "${uuid}" judge
    done
    echo "report: ${RUN_ROOT}/final-eval/runs/${RUN_ID}-final/${TASK_ID}/submissions/manual-1/report.json"
  ) >"${RUN_ROOT}/final-eval.log" 2>&1
  final_rc=$?
  set -e
  printf '%s\n' "${final_rc}" >"${RUN_ROOT}/final-eval.exit"
  echo "[$(date -u +%FT%TZ)] final Judge finished rc=${final_rc} (${RUN_ROOT}/final-eval.log)"
elif [[ "${FINAL_EVAL}" == 1 ]]; then
  echo "[$(date -u +%FT%TZ)] no final archive at ${final_archive}; final Judge skipped"
fi
exit "${run_rc}"
