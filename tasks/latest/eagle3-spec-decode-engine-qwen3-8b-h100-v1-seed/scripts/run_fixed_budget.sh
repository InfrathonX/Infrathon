#!/usr/bin/env bash
# Formal fixed-budget lifecycle for EAGLE-3 speculative decoding: single
# whole-GPU lease by default (SFORGE_SINGLE_GPU_HANDOFF=1, exported only by
# run_single_gpu_fixed_budget.sh after its fail-closed preflight), or the
# historical dual-GPU path selected by launch --legacy-dual-gpu.
# Lineage: LLaDA v2 run_fixed_budget.sh. Task-specific workload-policy,
# quality-control, and row-rotation branches are documented as dropped in the
# port ledger because this task has one tiered oracle policy and six fixed cells.
set -Eeuo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

ensure_docker_group "$@"
require_sforge
for required_command in curl docker flock jq nvidia-smi python3 realpath seq sha256sum ss; do
  require_command "${required_command}"
done

MODEL="${MODEL:-}"
[[ -n "${MODEL}" ]] || die "set MODEL to the agent model id (launch/eagle3-spec-decode/run.sh --model)"
# Optional OpenAI-compatible model API (e.g. a self-hosted DeepSeek).  With
# MODEL_API_BASE_URL set, the restricted sidecar gains one raw TCP relay to
# that endpoint, Codex talks to it in API-key mode (key from the mode-0600
# MODEL_API_KEY_FILE) and MODEL may be any id the endpoint serves.
MODEL_API_BASE_URL="${MODEL_API_BASE_URL:-}"
MODEL_API_KEY_FILE="${MODEL_API_KEY_FILE:-}"
MODEL_FORWARD_PORT=3131
REASONING_EFFORT="${REASONING_EFFORT:-max}"
# MLSYSFORGE_AGENT=claude-code (launch/eagle3-spec-decode/run.sh --agent
# claude-code) drives the pinned Claude Code runtime with a 0600 file holding
# one `claude setup-token` OAuth token (CLAUDE_OAUTH_TOKEN_FILE); the entry
# binds it through sforge/harness/agent/claude_local.py.
AGENT_NAME="${MLSYSFORGE_AGENT:-codex}"
CLAUDE_TOKEN_FILE="${CLAUDE_OAUTH_TOKEN_FILE:-}"
# The repository-wide Codex pin is the default; REQUIRED_CODEX_CLI_VERSION (for
# a model that only a newer CLI serves) raises it below, and an explicit
# operator value always wins.
OPERATOR_CLI_PIN="${CODEX_CLI_VERSION-}"
CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-0.148.0}"
TASK_TIMEOUT="${TASK_TIMEOUT:-7200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1800}"
FINAL_EVAL_TIMEOUT="${FINAL_EVAL_TIMEOUT:-5400}"
JUDGE_PORT="${JUDGE_PORT:-18386}"

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
# Read by scripts/sforge_h100_entry.py when it builds the Codex command
# and prompt.
export MLSYSFORGE_WORK_INTERNET="${task_internet}"

# Any model id: a ChatGPT-auth Codex model or whatever MODEL_API_BASE_URL serves.
# REQUIRED_CODEX_CLI_VERSION names the oldest Codex CLI that serves MODEL
# (default 0.148.0; set it for a model that newer CLIs alone accept). A newer
# CLI from the launcher is accepted; the agent-free Work image does not
# constrain it.
[[ "${MODEL}" =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]] || die "unsafe MODEL: ${MODEL}"
REQUIRED_CODEX_CLI_VERSION="${REQUIRED_CODEX_CLI_VERSION:-0.148.0}"
[[ "${REQUIRED_CODEX_CLI_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
  || die "unsafe REQUIRED_CODEX_CLI_VERSION: ${REQUIRED_CODEX_CLI_VERSION}"
case "${AGENT_NAME}" in
  codex) ;;
  claude-code)
    [[ -z "${MODEL_API_BASE_URL}" ]] || die "Claude Code uses its OAuth token; unset MODEL_API_BASE_URL"
    [[ "${REASONING_EFFORT}" != ultra ]] || die "Claude Code efforts are high|xhigh|max"
    ;;
  *) die "MLSYSFORGE_AGENT must be codex or claude-code" ;;
esac
if [[ -z "${OPERATOR_CLI_PIN}" ]]; then
  CODEX_CLI_VERSION="${REQUIRED_CODEX_CLI_VERSION}"
fi
codex_version_at_least() {
  [[ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" == "$2" ]]
}
case "${REASONING_EFFORT}" in
  high|max|xhigh|ultra)
    ;;
  *)
    die "formal-profile reasoning effort is not allowed: ${REASONING_EFFORT}"
    ;;
esac
[[ "${AGENT_NAME}" != "codex" ]] \
  || codex_version_at_least "${CODEX_CLI_VERSION}" "${REQUIRED_CODEX_CLI_VERSION}" \
  || die "formal-profile requires CODEX_CLI_VERSION>=${REQUIRED_CODEX_CLI_VERSION} for ${MODEL}"
for numeric_name in TASK_TIMEOUT EVAL_INTERVAL FINAL_EVAL_TIMEOUT JUDGE_PORT; do
  numeric_value="${!numeric_name}"
  [[ "${numeric_value}" =~ ^[1-9][0-9]*$ ]] \
    || die "${numeric_name} must be a positive integer"
done
(( TASK_TIMEOUT >= 90 )) || die "TASK_TIMEOUT must be at least 90 seconds"
(( EVAL_INTERVAL < TASK_TIMEOUT )) \
  || die "EVAL_INTERVAL must be shorter than TASK_TIMEOUT"
(( JUDGE_PORT >= 1024 && JUDGE_PORT <= 65535 && JUDGE_PORT != 3128 && JUDGE_PORT != 3130 )) \
  || die "JUDGE_PORT is invalid or reserved"

AUTH_JSON="${CODEX_AUTH_JSON_PATH:-}"
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

# Pin the host-provided agent runtime and refuse a moved CLI tree.
AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:?set SFORGE_AGENT_RUNTIME_DIR}"
runtime_marker="${AGENT_RUNTIME_DIR}/.sforge-agent-runtime.json"
[[ -d "${AGENT_RUNTIME_DIR}" && ! -L "${AGENT_RUNTIME_DIR}" ]] \
  || die "agent runtime directory is missing or unsafe"
[[ -f "${runtime_marker}" && ! -L "${runtime_marker}" ]] \
  || die "agent runtime marker is missing or unsafe"
[[ "$(jq -er '.kind' "${runtime_marker}")" == "sforge-agent-runtime" ]] \
  || die "agent runtime marker has the wrong kind"
[[ "$(jq -er '.agent' "${runtime_marker}")" == "${AGENT_NAME}" ]] \
  || die "formal-profile ${AGENT_NAME} run requires a ${AGENT_NAME} runtime"
if [[ "${AGENT_NAME}" == "codex" ]]; then
  [[ "$(jq -er '.version' "${runtime_marker}")" == "${CODEX_CLI_VERSION}" ]] \
    || die "agent runtime CLI version differs from the pin"
else
  # No Codex CLI in a Claude Code run; the runtime marker pins Claude Code.
  CODEX_CLI_VERSION=""
fi
runtime_tree_sha256="$(jq -er '.tree_sha256' "${runtime_marker}")"
[[ "${runtime_tree_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "agent runtime marker lacks a valid tree digest"
export SFORGE_AGENT_RUNTIME_DIR="${AGENT_RUNTIME_DIR}"

# Formal scoring remains fail-closed until helper D changes only the status.
policy_path="${EAGLE3_PRIVATE_DATA}/policy.json"
policy_status="$(jq -er '.status' "${policy_path}")"
[[ "${policy_status}" == "QUALIFIED" ]] \
  || die "formal-profile requires QUALIFIED policy; found ${policy_status}"
[[ "$(jq -er '.tiers | length' "${policy_path}")" == "8" ]] \
  || die "formal-profile requires all eight frozen A2 tiers"

assert_built_images_current
build_manifest="${BUILD_DIR}/image-manifest.json"
recorded_source_digest="$(jq -er '.source_digest' "${build_manifest}")"
recorded_work_image_id="$(jq -er '.work.id' "${build_manifest}")"
recorded_judge_image_id="$(jq -er '.judge.id' "${build_manifest}")"
[[ "${recorded_source_digest}" == "$(image_source_digest)" ]] \
  || die "image sources differ from the immutable build record"

SINGLE_GPU_HANDOFF="${SFORGE_SINGLE_GPU_HANDOFF:-0}"
[[ "${SINGLE_GPU_HANDOFF}" == "0" || "${SINGLE_GPU_HANDOFF}" == "1" ]] \
  || die "SFORGE_SINGLE_GPU_HANDOFF must be 0 or 1"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
# Label-safe: gpt-N.M-<name> -> <name>, ZHIPU/GLM-5.3-Flash -> zhipu-glm-5.3-flash.
model_slug="$(sed -E 's/^gpt-[0-9.]+-//' <<<"${MODEL}")"
model_slug="$(tr 'A-Z' 'a-z' <<<"${model_slug}" | sed -E 's/[^a-z0-9.]+/-/g; s/^-+//; s/-+$//')"
budget_minutes="$(((TASK_TIMEOUT + 59) / 60))"
RUN_ID="${RUN_ID:-eagle3-spec-${model_slug}-${REASONING_EFFORT}-${budget_minutes}m-${RUN_STAMP}}"
RUN_ROOT="${LOGS_DIR}/runs/${RUN_ID}"

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
  [[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || dry_failures+=("unsafe RUN_ID: ${RUN_ID}")
  [[ ! -e "${RUN_ROOT}" ]] || dry_failures+=("run directory already exists: ${RUN_ROOT}")
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
  if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
    dry_uuid="${SFORGE_ACCELERATOR_DEVICE_UUID:-}"
    dry_gpu="${dry_uuid} (Work and Judge, one lease; lock $(gpu_lock_path "${dry_uuid}"))"
    dry_used="$(nvidia-smi --id="${dry_uuid}" --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | xargs)" \
      || dry_failures+=("nvidia-smi cannot see ${dry_uuid}")
    dry_gpu+=" [index,used: ${dry_used:-?}]"
  else
    dry_gpu="Work ${WORK_GPU:-<unset>} / Judge ${JUDGE_GPU:-<unset>} (legacy dual GPU)"
  fi
  echo "Dry run (no lock, container or GPU work): ${RUN_ID}"
  echo "  agent:        ${AGENT_NAME} ${MODEL} effort=${REASONING_EFFORT} runtime ${AGENT_RUNTIME_DIR} ($(jq -er '.version' "${runtime_marker}"))"
  echo "  credential:   ${dry_credential}"
  echo "  budget:       ${TASK_TIMEOUT}s; auto-eval off; unlimited manual submits, cooldown ${EVAL_INTERVAL}s"
  echo "  final Judge:  standalone after the deadline, same GPU under the held lock; timeout ${FINAL_EVAL_TIMEOUT}s"
  echo "  GPU:          ${dry_gpu}"
  echo "  judge port:   ${JUDGE_PORT}"
  echo "  network:      internet=${task_internet} web_search=${codex_web_search}"
  echo "  version store: ${SFORGE_VERSION_STORE_ROOT:-<unset>}"
  echo "  cpuset:       work=${SFORGE_WORK_CPUSET:-${SFORGE_CPUSET:-off}} judge=${SFORGE_JUDGE_CPUSET:-${SFORGE_CPUSET:-off}}"
  echo "  logs:         ${RUN_ROOT}"
  if (( ${#dry_failures[@]} )); then
    printf '  CHECK FAILED: %s\n' "${dry_failures[@]}"
    exit 1
  fi
  echo "  checks:       PASS"
  exit 0
fi

if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  # One scheduler-issued whole GPU shared serially by Work and Judge.
  [[ ! -v WORK_GPU && ! -v JUDGE_GPU ]] \
    || die "single GPU handoff rejects WORK_GPU/JUDGE_GPU"
  [[ "${SFORGE_ACCELERATOR_DEVICE_UUID:-}" =~ ^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]] \
    || die "SFORGE_ACCELERATOR_DEVICE_UUID must be a qualified canonical whole-GPU UUID"
  work_uuid="${SFORGE_ACCELERATOR_DEVICE_UUID}"
  judge_uuid="${SFORGE_ACCELERATOR_DEVICE_UUID}"
  WORK_GPU="$(gpu_index_of_uuid "${work_uuid}")"
  JUDGE_GPU="${WORK_GPU}"

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

  acquire_single_gpu_lock "${work_uuid}"
  if ! "${PYTHON_BIN}" - "${work_uuid}" <<'PY'
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
    "${work_uuid}" \
    "single GPU" \
    "${profile_vendor}" \
    "${profile_count}" \
    "${profile_min_compute_capability}" \
    "${profile_min_memory_bytes}"
  assert_single_gpu_lock_held
  assert_container_accelerator "${WORK_IMAGE}" "${work_uuid}" "Work"
  assert_container_accelerator "${JUDGE_IMAGE}" "${judge_uuid}" "Judge"
  handoff_mode="single_allocation_serial_v1"
  scheduling_mode="single_gpu_lease_manual_judge_then_standalone_final_handoff"
  allocation_key="${SFORGE_ACCELERATOR_ALLOCATION_KEY}"
  trusted_lease_id="${SFORGE_TRUSTED_LEASE_ID}"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/preflight.py" \
    --target-repo "${EAGLE3_TARGET_REPO}" \
    --head-dir "${EAGLE3_HEAD_DIR}" \
    --judge-gpu "${JUDGE_GPU}" \
    --image-manifest "${build_manifest}" >/dev/null

  # Two deliberately separate identities.  deployment_attestation records the
  # physical UUID and observed clocks for audit only.  measurement_profile
  # hashes the stable comparability key: model / compute capability / memory /
  # driver / exact immutable image IDs / software ABI / calibration protocol /
  # campaign epoch.  The UUID and observed clocks never enter that key.
  measurement_gpu_name="$(nvidia-smi --id="${judge_uuid}" --query-gpu=name --format=csv,noheader | xargs)"
  measurement_driver_version="$(nvidia-smi --id="${judge_uuid}" --query-gpu=driver_version --format=csv,noheader | xargs)"
  measurement_compute_capability="$(nvidia-smi --id="${judge_uuid}" --query-gpu=compute_cap --format=csv,noheader | xargs)"
  measurement_memory_mib="$(nvidia-smi --id="${judge_uuid}" --query-gpu=memory.total --format=csv,noheader,nounits | xargs)"
  measurement_sm_clock_mhz="$(nvidia-smi --id="${judge_uuid}" --query-gpu=clocks.current.sm --format=csv,noheader,nounits | xargs)"
  measurement_memory_clock_mhz="$(nvidia-smi --id="${judge_uuid}" --query-gpu=clocks.current.memory --format=csv,noheader,nounits | xargs)"
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
      --arg gpu_uuid "${judge_uuid}" \
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
      --arg work_image_id "${recorded_work_image_id}" \
      --arg judge_image_id "${recorded_judge_image_id}" \
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
else
  scheduling_mode="dual_h100_manual_judge_then_independent_holdout_final"
  WORK_GPU="$(canonical_gpu_index "${WORK_GPU:?set WORK_GPU}")"
  JUDGE_GPU="$(canonical_gpu_index "${JUDGE_GPU:?set JUDGE_GPU}")"
  work_uuid="$(gpu_uuid "${WORK_GPU}")"
  judge_uuid="$(gpu_uuid "${JUDGE_GPU}")"
  acquire_dual_gpu_locks "${work_uuid}" "${judge_uuid}"
  assert_distinct_gpu_pair "${WORK_GPU}" "${JUDGE_GPU}"
  assert_dual_gpu_locks_held
  assert_container_gpu "${WORK_IMAGE}" "${work_uuid}" "Work"
  assert_container_gpu "${JUDGE_IMAGE}" "${judge_uuid}" "Judge"

  "${PYTHON_BIN}" "${SCRIPT_DIR}/preflight.py" \
    --target-repo "${EAGLE3_TARGET_REPO}" \
    --head-dir "${EAGLE3_HEAD_DIR}" \
    --work-gpu "${WORK_GPU}" \
    --judge-gpu "${JUDGE_GPU}" \
    --image-manifest "${build_manifest}" >/dev/null
fi

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$ ]] \
  || die "unsafe RUN_ID: ${RUN_ID}"
RUN_ROOT="${LOGS_DIR}/runs/${RUN_ID}"
[[ ! -e "${RUN_ROOT}" && ! -L "${RUN_ROOT}" ]] \
  || die "run directory already exists: ${RUN_ROOT}"
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${JUDGE_PORT}$"; then
  die "Judge port ${JUDGE_PORT} is already in use"
fi

mkdir -p "${RUN_ROOT}"
chmod 700 "${RUN_ROOT}"
run_image_manifest="${RUN_ROOT}/image-build-manifest.json"
run_policy="${RUN_ROOT}/oracle-policy.json"
cp -- "${build_manifest}" "${run_image_manifest}"
cp -- "${policy_path}" "${run_policy}"
chmod 444 "${run_image_manifest}" "${run_policy}"
policy_sha256="$(sha256sum "${run_policy}" | awk '{print $1}')"
runtime_entry_sha256="$(sha256sum "${SCRIPT_DIR}/sforge_h100_entry.py" | awk '{print $1}')"
restricted_proxy_sha256="$(sha256sum "${SCRIPT_DIR}/restricted_proxy.py" | awk '{print $1}')"

# Pin every SForge Python source used by this run. The second digest binds the
# file set itself so adding or removing a file is detected at the final barrier.
mapfile -d '' -t SFORGE_RUNTIME_FILES < <(
  cd "${SFORGE_ROOT}"
  find sforge -type f -name '*.py' -print0 | LC_ALL=C sort -z
)
(( ${#SFORGE_RUNTIME_FILES[@]} > 0 )) || die "SForge runtime source set is empty"
sforge_runtime_sha256="$(
  (
    cd "${SFORGE_ROOT}"
    sha256sum "${SFORGE_RUNTIME_FILES[@]}"
  ) | sha256sum | awk '{print $1}'
)"
sforge_runtime_file_set_sha256="$(
  printf '%s\n' "${SFORGE_RUNTIME_FILES[@]}" | sha256sum | awk '{print $1}'
)"
runtime_provenance="${RUN_ROOT}/runtime-provenance.json"
jq -n \
  --arg entry "${runtime_entry_sha256}" \
  --arg proxy "${restricted_proxy_sha256}" \
  --arg runtime "${sforge_runtime_sha256}" \
  --arg file_set "${sforge_runtime_file_set_sha256}" \
  --arg agent_tree "${runtime_tree_sha256}" \
  --arg source "${recorded_source_digest}" \
  --arg work_image "${recorded_work_image_id}" \
  --arg judge_image "${recorded_judge_image_id}" \
  --arg policy "${policy_sha256}" \
  '{schema_version:1,runtime_entry_sha256:$entry,restricted_proxy_sha256:$proxy,
    sforge_runtime_sha256:$runtime,sforge_runtime_file_set_sha256:$file_set,
    agent_runtime_tree_sha256:$agent_tree,source_digest:$source,
    work_image_id:$work_image,judge_image_id:$judge_image,policy_sha256:$policy}' \
  >"${runtime_provenance}"
chmod 444 "${runtime_provenance}"
runtime_provenance_sha256="$(sha256sum "${runtime_provenance}" | awk '{print $1}')"

# Create a private per-run cohort secret for HMAC anchor sampling.
cohort_secret_file="${RUN_ROOT}/.cohort-secret"
(
  umask 077
  "${PYTHON_BIN}" -c 'import secrets; print(secrets.token_hex(32))' \
    >"${cohort_secret_file}"
)
cohort_secret="$(<"${cohort_secret_file}")"
[[ "${cohort_secret}" =~ ^[0-9a-f]{64}$ ]] \
  || die "failed to create the HMAC cohort secret"

network="eagle3-${RUN_ID}"
restricted_proxy_name="${RUN_ID}-restricted-proxy"
server_pid=""
active_pid=""

stop_active_process() {
  # The CLI may own Docker children, so terminate its process first and give
  # it a bounded grace period before the resource cleanup below.
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

run_logged_command() {
  # Preserve the CLI exit status while retaining a PID for signal cleanup.
  local log_file="$1"
  shift
  local command_rc=0
  "$@" >"${log_file}" 2>&1 &
  active_pid="$!"
  wait "${active_pid}" || command_rc="$?"
  active_pid=""
  return "${command_rc}"
}

cleanup() {
  # Cleanup is idempotent and never masks the original lifecycle failure.
  stop_active_process
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    server_pid=""
  fi
  remove_task_gpu_containers "work" "${work_uuid}" "${RUN_ID}" \
    >/dev/null 2>&1 || true
  remove_task_gpu_containers "judge" "${judge_uuid}" "${RUN_ID}" \
    >/dev/null 2>&1 || true
  docker rm -f "${restricted_proxy_name}" >/dev/null 2>&1 || true
  docker network rm "${network}" >/dev/null 2>&1 || true
}

handle_signal() {
  local signal_name="$1"
  local exit_code=143
  if [[ "${signal_name}" == "INT" ]]; then
    exit_code=130
  elif [[ "${signal_name}" == "HUP" ]]; then
    exit_code=129
  fi
  trap - INT TERM HUP
  cleanup
  exit "${exit_code}"
}

trap cleanup EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM
trap 'handle_signal HUP' HUP

# Host-local Judge control must bypass any ambient outbound proxy.
bridge_gateway="$(docker network inspect bridge | jq -er '.[0].IPAM.Config[0].Gateway')"
NO_PROXY="${bridge_gateway},127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
no_proxy="${NO_PROXY}"
export NO_PROXY no_proxy

export SFORGE_ADMIN_SECRET="$(${PYTHON_BIN} -c 'import secrets; print(secrets.token_urlsafe(48))')"
# The Judge server is long-lived and parses these values at process startup.
# Export its GPU, immutable-image, and private-context inputs before spawning
# it so interactive submissions use the same role-bound environment as final
# and replay evaluation.
export EAGLE3_JUDGE_GPU_UUID="${judge_uuid}"
export EAGLE3_RUNTIME_RUN_ID="${RUN_ID}"
export EAGLE3_COHORT_SECRET="${cohort_secret}"
export SFORGE_EXPECTED_WORK_IMAGE_ID="${recorded_work_image_id}"
export SFORGE_EXPECTED_JUDGE_IMAGE_ID="${recorded_judge_image_id}"
export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${judge_uuid},CUDA_VISIBLE_DEVICES=0,EAGLE3_GPU_ROLE=judge,EAGLE3_COHORT_SECRET=${cohort_secret},EAGLE3_INTERACTIVE_JUDGE_TIMEOUT=${FINAL_EVAL_TIMEOUT},EAGLE3_FINAL_JUDGE_TIMEOUT=${FINAL_EVAL_TIMEOUT}"
# The adapter refuses any Judge container that lacks one of these variables,
# and the first formal run lost its only interactive round exactly that way.
# Validate the string the server hands to every interactive round with the
# same shared guard smoke_submission_plane.sh uses, so the formal launcher and
# the smoke cannot drift apart.
assert_interactive_judge_env_spec "${SFORGE_JUDGE_EXTRA_ENV}" "${judge_uuid}"
# The coordinator binds the Docker bridge gateway for the Work sidecar.  In
# single-allocation mode it also binds loopback, the only origin the
# standalone-final handoff client accepts; no public interface is ever bound.
judge_serve_extra_hosts=()
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  judge_serve_extra_hosts+=(--host 127.0.0.1)
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/sforge_h100_entry.py" \
  --output-dir "${RUN_ROOT}" \
  --tasks-dir "${TASKS_DIR}" \
  serve --host "${bridge_gateway}" "${judge_serve_extra_hosts[@]}" --port "${JUDGE_PORT}" \
  >"${RUN_ROOT}/judge-server.log" 2>&1 &
server_pid="$!"
wait_for_http "http://${bridge_gateway}:${JUDGE_PORT}/openapi.json" 90
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  wait_for_http "http://127.0.0.1:${JUDGE_PORT}/openapi.json" 30
fi

docker network create \
  --driver bridge \
  --internal \
  --label "mlsysforge.run_id=${RUN_ID}" \
  "${network}" >/dev/null
# The sidecar is the flash-kmeans restricted CONNECT proxy byte for byte: it
# allowlists exactly api.openai.com / auth.openai.com / chatgpt.com:443 itself,
# dials them directly from the default bridge (no host proxy upstream), relays
# origin-form Judge traffic on 3130 to the coordinator, and with
# MODEL_API_BASE_URL set gains one raw TCP relay on MODEL_FORWARD_PORT.
# With the task JSON's internet=true it also runs --allow-public-egress
# (CONNECT to any globally routable host on 80/443; internal addresses
# refused).
docker run \
  --detach \
  --rm \
  --name "${restricted_proxy_name}" \
  --network bridge \
  --add-host host.docker.internal:host-gateway \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --volume "${SCRIPT_DIR}/restricted_proxy.py:/opt/mlsysforge/restricted_proxy.py:ro" \
  --entrypoint python3 \
  "${WORK_IMAGE}" \
  /opt/mlsysforge/restricted_proxy.py \
  ${sidecar_public_egress:+"${sidecar_public_egress}"} \
  --judge-host "${bridge_gateway}" \
  --judge-port "${JUDGE_PORT}" \
  --judge-forward-port 3130 \
  ${model_api_host:+--model-forward "${MODEL_FORWARD_PORT}:${model_api_host}:${model_api_port}"} >/dev/null
docker network connect --alias runtime-proxy "${network}" "${restricted_proxy_name}"
restricted_proxy_ip="$(
  docker inspect "${restricted_proxy_name}" \
    | jq -er --arg network "${network}" \
      '.[0].NetworkSettings.Networks[$network].IPAddress
       | select(type == "string" and length > 0)'
)"
[[ "${restricted_proxy_ip}" =~ ^[0-9]+(\.[0-9]+){3}$ ]] \
  || die "restricted proxy has no address on ${network}"

# SForge's host-side session registration posts to the restricted proxy IP.
# Keep that control-plane hop out of the ambient outbound HTTP proxy.
NO_PROXY="${restricted_proxy_ip},${NO_PROXY}"
no_proxy="${NO_PROXY}"
export NO_PROXY no_proxy

# Preflight from inside the internal network, before any agent budget is
# spent: the sidecar answers, the Judge relay reaches the coordinator, direct
# public egress is impossible, and the CONNECT policy holds: chatgpt.com 200;
# github.com 403 in allowlist mode, or with --allow-public-egress github.com
# 200 and the Docker bridge gateway (a non-global address) 403.
for _ in $(seq 1 30); do
  if docker run --rm \
      --network "${network}" \
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
  --network "${network}" \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint curl \
  "${WORK_IMAGE}" \
  --connect-timeout 2 \
  --max-time 5 \
  -fsS "http://${restricted_proxy_ip}:3128/health" >/dev/null \
  || die "restricted proxy sidecar is not answering on the internal network"
docker run --rm \
  --network "${network}" \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint curl \
  "${WORK_IMAGE}" \
  --connect-timeout 2 \
  --max-time 5 \
  -fsS "http://${restricted_proxy_ip}:3130/openapi.json" >/dev/null \
  || die "Judge relay on the restricted proxy does not reach the coordinator"
if docker run --rm \
    --network "${network}" \
    --env NVIDIA_VISIBLE_DEVICES=void \
    --entrypoint curl \
    "${WORK_IMAGE}" \
    --connect-timeout 2 \
    --max-time 4 \
    -fsS https://example.com >/dev/null 2>&1; then
  die "internal work network unexpectedly has direct public egress"
fi
docker run --rm \
  --network "${network}" \
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
  echo "Public-egress network: PASS (${network}, proxy ${restricted_proxy_ip}; no direct egress; CONNECT to public hosts on 80/443; internal addresses refused)"
else
  echo "Restricted network: PASS (${network}, proxy ${restricted_proxy_ip})"
fi

export EAGLE3_WORK_GPU_UUID="${work_uuid}"
export EAGLE3_GPU_ROLE="work"
if [[ -n "${MODEL_API_BASE_URL}" ]]; then
  export SFORGE_AGENT_API_BASE_URL="http://${restricted_proxy_ip}:${MODEL_FORWARD_PORT}${model_api_path}"
  export SFORGE_AGENT_API_KEY="${MODEL_API_KEY}"
elif [[ "${AGENT_NAME}" == "claude-code" ]]; then
  unset SFORGE_CODEX_AUTH_JSON_PATH
  export MLSYSFORGE_CLAUDE_OAUTH_TOKEN_FILE="${CLAUDE_TOKEN_FILE}"
else
  export SFORGE_CODEX_AUTH_JSON_PATH="${AUTH_JSON}"
fi
# Read by sforge_h100_entry.py (claude_local.install_from_environment).
export MLSYSFORGE_AGENT="${AGENT_NAME}" MLSYSFORGE_AGENT_MODEL="${MODEL}"
export MLSYSFORGE_AGENT_EFFORT="${REASONING_EFFORT}" MLSYSFORGE_AGENT_BUDGET_SECONDS="${TASK_TIMEOUT}"
export MLSYSFORGE_RUN_LOG_ROOT="${RUN_ROOT}"
export SFORGE_CODEX_CLI_VERSION="${CODEX_CLI_VERSION}"
export SFORGE_CODEX_REASONING_EFFORT="${REASONING_EFFORT}"
export SFORGE_ISOLATED_DOCKER_NETWORK="${network}"
export SFORGE_HTTP_PROXY="http://${restricted_proxy_ip}:3128"
export SFORGE_HTTPS_PROXY="http://${restricted_proxy_ip}:3128"
export SFORGE_NO_PROXY="${restricted_proxy_ip},127.0.0.1,localhost"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  # sforge-submit runs the root-owned cooperative drain hook before archive
  # and HTTP admission; the host coordinator still pauses Work and attests.
  export SFORGE_AGENT_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${work_uuid},CUDA_VISIBLE_DEVICES=0,EAGLE3_GPU_ROLE=work,SFORGE_PRE_SUBMIT_HOOK=/app/workdir/spec_lab/.sforge/prepare_submit.sh"
else
  export SFORGE_AGENT_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${work_uuid},CUDA_VISIBLE_DEVICES=0,EAGLE3_GPU_ROLE=work"
fi

experiment_manifest="${RUN_ROOT}/experiment-manifest.json"
jq -n \
  --arg run_id "${RUN_ID}" \
  --arg task_id "${TASK_ID}" \
  --arg agent "${AGENT_NAME}" \
  --arg model "${MODEL}" \
  --arg effort "${REASONING_EFFORT}" \
  --arg work_uuid "${work_uuid}" \
  --arg judge_uuid "${judge_uuid}" \
  --arg work_image "${recorded_work_image_id}" \
  --arg judge_image "${recorded_judge_image_id}" \
  --arg runtime_provenance_sha256 "${runtime_provenance_sha256}" \
  --arg network_mode "docker_internal_restricted_connect_proxy" \
  --arg isolated_entry_sha256 "${runtime_entry_sha256}" \
  --arg restricted_proxy_sha256 "${restricted_proxy_sha256}" \
  --arg codex_web_search "${codex_web_search}" \
  --argjson public_connect_egress "${task_internet}" \
  --argjson timeout "${TASK_TIMEOUT}" \
  --argjson interval "${EVAL_INTERVAL}" \
  '{schema_version:1,run_id:$run_id,task_id:$task_id,agent:$agent,model:$model,
    reasoning_effort:$effort,timeout_seconds:$timeout,
    submission_cooldown_seconds:$interval,scheduled_auto_evaluation:false,
    work_gpu_uuid:$work_uuid,judge_gpu_uuid:$judge_uuid,
    images:{work:$work_image,judge:$judge_image},
    runtime_provenance_sha256:$runtime_provenance_sha256,
    codex_web_search:$codex_web_search,
    network:{
      mode:$network_mode,
      direct_public_egress:false,
      public_connect_egress:$public_connect_egress,
      connect_allowlist:[
        "api.openai.com:443",
        "auth.openai.com:443",
        "chatgpt.com:443"
      ],
      isolated_entry_sha256:$isolated_entry_sha256,
      restricted_proxy_sha256:$restricted_proxy_sha256
    }}' \
  >"${experiment_manifest}"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  jq \
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
    '. + {
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
      })
    }' "${experiment_manifest}" >"${experiment_manifest}.tmp"
  mv -- "${experiment_manifest}.tmp" "${experiment_manifest}"
fi
chmod 444 "${experiment_manifest}"

# Claude Code binds through sforge/harness/agent/claude_local.py around the
# unchanged task entry: the entry's bytes are pinned by the portable software
# ABI and build-resource manifests (and through them the image source digest),
# so the entry module is loaded as-is and only then wrapped.  Codex runs the
# entry directly, exactly as before.
agent_entry_shim=()
if [[ "${AGENT_NAME}" == "claude-code" ]]; then
  agent_entry_shim=(-P -c '
import runpy, sys
entry = sys.argv[1]
sys.argv = [entry, *sys.argv[2:]]
module = runpy.run_path(entry, run_name="sforge_task_entry")
from sforge.harness.agent.claude_local import install_from_environment
install_from_environment(module["_task_internet"], sys.argv)
module["main"]()
')
fi
set +e
run_logged_command "${RUN_ROOT}/sforge-run.log" \
  "${PYTHON_BIN}" "${agent_entry_shim[@]}" "${SCRIPT_DIR}/sforge_h100_entry.py" \
  --output-dir "${RUN_ROOT}" \
  --tasks-dir "${TASKS_DIR}" \
  run \
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
  --work-mem-limit 64g \
  --judge-cpu-limit 8 \
  --judge-mem-limit 64g \
  --run-id "${RUN_ID}" \
  "$@"
run_rc="$?"
set -e
(( run_rc == 0 )) || die "SForge run failed with exit ${run_rc}"

if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  # The Work container has been frozen and destroyed by the harness's
  # never-resume final capture.  The coordinator stays alive: it performs the
  # final Judge on the same allocation under the still-held lease through the
  # explicit trusted standalone-final CLI path, and only then is everything
  # torn down.
  assert_gpu_released "${work_uuid}" "${WORK_GPU}" "Work"
  assert_single_gpu_lock_held
else
  # Stop every mutable participant-facing service before freezing the archive.
  cleanup
  trap - EXIT INT TERM HUP
  assert_gpu_released "${work_uuid}" "${WORK_GPU}" "Work"
  assert_gpu_released "${judge_uuid}" "${JUDGE_GPU}" "interactive Judge"
  assert_dual_gpu_locks_held
fi

TASK_RUN_ROOT="${RUN_ROOT}/${TASK_ID}"
[[ -d "${TASK_RUN_ROOT}" ]] || TASK_RUN_ROOT="${RUN_ROOT}/runs/${RUN_ID}/${TASK_ID}"
final_archive="${TASK_RUN_ROOT}/final_archive.tar.gz"
final_result="${TASK_RUN_ROOT}/final_result.json"
final_snapshot="${TASK_RUN_ROOT}/final_snapshot.json"
run_history="${TASK_RUN_ROOT}/run_history.json"
for required in "${final_archive}" "${final_result}" "${final_snapshot}" "${run_history}"; do
  [[ -s "${required}" && ! -L "${required}" ]] \
    || die "run ended without required artifact: ${required}"
done

# Re-attest every mutable dependency at the final archive barrier.
assert_built_images_current
[[ "$(image_source_digest)" == "${recorded_source_digest}" ]] \
  || die "task sources changed before final evaluation"
[[ "$(sha256sum "${runtime_provenance}" | awk '{print $1}')" == "${runtime_provenance_sha256}" ]] \
  || die "runtime-provenance snapshot changed"
[[ "$(sha256sum "${run_policy}" | awk '{print $1}')" == "${policy_sha256}" ]] \
  || die "oracle policy changed"
current_sforge_runtime_sha256="$(
  (
    cd "${SFORGE_ROOT}"
    sha256sum "${SFORGE_RUNTIME_FILES[@]}"
  ) | sha256sum | awk '{print $1}'
)"
[[ "${current_sforge_runtime_sha256}" == "${sforge_runtime_sha256}" ]] \
  || die "SForge runtime changed before final evaluation"

formal_precheck="${TASK_RUN_ROOT}/formal-precheck.json"
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_formal_artifacts.py" pre \
  --task-root "${TASK_RUN_ROOT}" \
  --run-id "${RUN_ID}" \
  --task-id "${TASK_ID}" \
  --agent "${AGENT_NAME}" \
  --model "${MODEL}" \
  --budget-seconds "${TASK_TIMEOUT}" \
  --interval-seconds "${EVAL_INTERVAL}" \
  --evaluation-mode "${scheduling_mode}" \
  >"${formal_precheck}.tmp"
mv -- "${formal_precheck}.tmp" "${formal_precheck}"
final_archive_sha256="$(jq -er '.final_archive.sha256' "${formal_precheck}")"

final_eval_id="${RUN_ID}-final"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  final_run_root="${LOGS_DIR}/runs/${final_eval_id}"
  [[ ! -e "${final_run_root}" && ! -L "${final_run_root}" ]] \
    || die "final evaluation run directory already exists: ${final_run_root}"
  mkdir -p "${final_run_root}"
  chmod 700 "${final_run_root}"
  # The standalone-final client only accepts a loopback coordinator origin.
  wait_for_http "http://127.0.0.1:${JUDGE_PORT}/openapi.json" 30
  set +e
  run_logged_command "${final_run_root}/eval.log" \
    "${PYTHON_BIN}" "${SCRIPT_DIR}/sforge_h100_entry.py" \
    --output-dir "${final_run_root}" \
    --tasks-dir "${TASKS_DIR}" \
    eval \
    --backend docker \
    --task "${TASK_ID}" \
    --archive "${final_archive}" \
    --run-id "${final_eval_id}" \
    --timeout "${FINAL_EVAL_TIMEOUT}" \
    --judge-cpu-limit 8 \
    --judge-mem-limit 64g \
    --standalone-final-handoff \
    --handoff-source-run-id "${RUN_ID}" \
    --judge-url "http://127.0.0.1:${JUDGE_PORT}" \
    --json
  final_eval_rc="$?"
  set -e
  cleanup
  trap - EXIT INT TERM HUP
  (( final_eval_rc == 0 )) \
    || die "standalone final evaluation failed with exit ${final_eval_rc}"
  assert_gpu_released "${judge_uuid}" "${JUDGE_GPU}" "final Judge"
  assert_single_gpu_lock_held
else
  "${SCRIPT_DIR}/eval_archive.sh" "${final_archive}" "${final_eval_id}"
fi
final_task_root=""
final_task_candidates=(
  "${LOGS_DIR}/runs/${final_eval_id}/${TASK_ID}"
  "${LOGS_DIR}/runs/${final_eval_id}/runs/${final_eval_id}/${TASK_ID}"
  # Single-GPU handoff: the coordinator that ran Work persists the standalone
  # final under its own --output-dir (this run's root), not a sibling run.
  "${RUN_ROOT}/runs/${final_eval_id}/${TASK_ID}"
)
for candidate_root in "${final_task_candidates[@]}"; do
  if [[ -s "${candidate_root}/submissions/manual-1/report.json" ]]; then
    final_task_root="${candidate_root}"
    break
  fi
done
[[ -n "${final_task_root}" ]] \
  || die "final evaluation did not persist a supported task-root layout"
formal_postcheck="${TASK_RUN_ROOT}/formal-postcheck.json"
completion="${TASK_RUN_ROOT}/completion.json"

# Persist a durable record of a rejected final.  The validator fails closed
# (non-zero exit); this writes a complete `invalid_final` postcheck and a
# `failed` completion.json so the reason outlives the launcher's stdout, and
# then aborts the launcher exactly as before.
record_invalid_final() {
  local postcheck_rc="$1"
  local stderr_file="$2"
  local reason
  reason="$(sed -n 's/^error: //p' "${stderr_file}" | head -n 1 || true)"
  [[ -n "${reason}" ]] || reason="$(tail -n 1 "${stderr_file}" || true)"
  [[ -n "${reason}" ]] || reason="formal postcheck exited ${postcheck_rc} without a reason"
  if ! jq -e '.status == "invalid_final" and (.reason | type == "string")' \
      "${formal_postcheck}.tmp" >/dev/null 2>&1; then
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
      >"${formal_postcheck}.tmp"
  fi
  jq --arg stderr "$(cat "${stderr_file}")" --argjson exit_code "${postcheck_rc}" \
    '. + {validator_exit_code:$exit_code, validator_stderr:$stderr}' \
    "${formal_postcheck}.tmp" >"${formal_postcheck}.invalid"
  mv -- "${formal_postcheck}.invalid" "${formal_postcheck}"
  rm -f -- "${formal_postcheck}.tmp" "${stderr_file}"
  jq -n \
    --arg run_id "${RUN_ID}" \
    --arg final_eval_id "${final_eval_id}" \
    --arg archive "${final_archive}" \
    --arg archive_sha256 "${final_archive_sha256}" \
    --arg precheck "${formal_precheck}" \
    --arg postcheck "${formal_postcheck}" \
    --arg scheduling_mode "${scheduling_mode}" \
    --arg reason "${reason}" \
    --arg finished_at "$(date -u +%FT%TZ)" \
    '{schema_version:1,status:"failed",failure_stage:"formal_postcheck",reason:$reason,
      finished_at:$finished_at,run_id:$run_id,final_eval_id:$final_eval_id,
      scheduling:$scheduling_mode,
      final_archive:{path:$archive,sha256:$archive_sha256},
      validation:{precheck:$precheck,postcheck:$postcheck}}' \
    >"${completion}.tmp"
  mv -- "${completion}.tmp" "${completion}"
  chmod 444 "${completion}"
  die "independent final evaluation is invalid (recorded in ${formal_postcheck}): ${reason}"
}

postcheck_stderr="${formal_postcheck}.stderr"
set +e
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_formal_artifacts.py" post \
  --task-root "${TASK_RUN_ROOT}" \
  --final-task-root "${final_task_root}" \
  --final-eval-id "${final_eval_id}" \
  --task-id "${TASK_ID}" \
  --expected-archive-sha256 "${final_archive_sha256}" \
  --evaluation-mode "${scheduling_mode}" \
  >"${formal_postcheck}.tmp" 2>"${postcheck_stderr}"
postcheck_rc="$?"
set -e
if (( postcheck_rc != 0 )); then
  cat -- "${postcheck_stderr}" >&2
  record_invalid_final "${postcheck_rc}" "${postcheck_stderr}"
fi
cat -- "${postcheck_stderr}" >&2
rm -f -- "${postcheck_stderr}"
jq -e '.status == "PASS"' "${formal_postcheck}.tmp" >/dev/null \
  || die "formal postcheck exited 0 without a PASS record"
mv -- "${formal_postcheck}.tmp" "${formal_postcheck}"

# Freeze a compact completion attestation only after post-run validation.
jq -n \
  --arg run_id "${RUN_ID}" \
  --arg final_eval_id "${final_eval_id}" \
  --arg archive "${final_archive}" \
  --arg archive_sha256 "${final_archive_sha256}" \
  --arg precheck "${formal_precheck}" \
  --arg postcheck "${formal_postcheck}" \
  '{schema_version:1,status:"passed",run_id:$run_id,final_eval_id:$final_eval_id,
    final_archive:{path:$archive,sha256:$archive_sha256},
    validation:{precheck:$precheck,postcheck:$postcheck}}' \
  >"${completion}.tmp"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  jq \
    --arg work_gpu_uuid "${work_uuid}" \
    --arg judge_gpu_uuid "${judge_uuid}" \
    --arg scheduling_mode "${scheduling_mode}" \
    --arg evaluation_contract_sha256 "${evaluation_contract_sha256}" \
    --arg execution_profile_id "${execution_profile_id}" \
    --arg measurement_profile_sha256 "${measurement_profile_sha256}" \
    --arg raw_score_comparison_identity_sha256 "${raw_score_comparison_identity_sha256}" \
    --arg qualification_status "${qualification_status}" \
    '. + {
      work_gpu_uuid: $work_gpu_uuid,
      judge_gpu_uuid: $judge_gpu_uuid,
      scheduling: $scheduling_mode,
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
      }
    }' "${completion}.tmp" >"${completion}.single"
  mv -- "${completion}.single" "${completion}.tmp"
else
  jq --arg scheduling_mode "${scheduling_mode}" '. + {scheduling: $scheduling_mode}' \
    "${completion}.tmp" >"${completion}.legacy"
  mv -- "${completion}.legacy" "${completion}.tmp"
fi
mv -- "${completion}.tmp" "${completion}"
chmod 444 "${completion}"

printf 'Completed %s\n' "${RUN_ID}"
printf 'Final archive: %s (%s)\n' "${final_archive}" "${final_archive_sha256}"
