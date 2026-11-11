#!/usr/bin/env bash
# Formal fixed-budget lifecycle for the LLaDA decoder task: single whole-GPU
# lease by default (SFORGE_SINGLE_GPU_HANDOFF=1, exported only by
# run_single_gpu_fixed_budget.sh after its fail-closed preflight), or the
# historical dual-GPU path selected by launch/llada-h100/run.sh
# --legacy-dual-gpu.  The workload, policies, Judge and scoring are identical
# in both modes; only which physical card runs the Judge and how the final
# Judge is reached differ.
set -Eeuo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_sforge
require_command awk
require_command curl
require_command flock
require_command gzip
require_command jq
require_command nvidia-smi
require_command python3
require_command realpath
require_command sha256sum
require_command ss

[[ "${LLADA_CALIBRATION:-0}" != "1" ]] \
  || die "formal fixed-budget runs forbid LLADA_CALIBRATION=1"
unset LLADA_CALIBRATION

AUTH_JSON="${CODEX_AUTH_JSON_PATH:-}"
# MLSYSFORGE_AGENT=claude-code (launch/llada-h100/run.sh --agent claude-code)
# drives the pinned Claude Code runtime with a 0600 file holding one
# `claude setup-token` OAuth token (CLAUDE_OAUTH_TOKEN_FILE); the entry binds
# it through sforge/harness/agent/claude_local.py.
AGENT_NAME="${MLSYSFORGE_AGENT:-codex}"
CLAUDE_TOKEN_FILE="${CLAUDE_OAUTH_TOKEN_FILE:-}"
# Optional OpenAI-compatible model API (e.g. a self-hosted DeepSeek).  With
# MODEL_API_BASE_URL set, the restricted sidecar gains one raw TCP relay to
# that endpoint, Codex talks to it in API-key mode (key from the mode-0600
# MODEL_API_KEY_FILE) and MODEL may be any id the endpoint serves.
MODEL_API_BASE_URL="${MODEL_API_BASE_URL:-}"
MODEL_API_KEY_FILE="${MODEL_API_KEY_FILE:-}"
MODEL_FORWARD_PORT=3131
SFORGE_PYTHON="$(dirname "${SFORGE_BIN}")/python"
H100_ENTRY="${MIGRATION_ROOT}/scripts/sforge_h100_entry.py"
RESTRICTED_PROXY_SOURCE="${MIGRATION_ROOT}/scripts/restricted_connect_proxy.py"
JUDGE_SERVER_SOURCE="${SFORGE_ROOT}/sforge/harness/judge_server.py"
mapfile -d '' -t SFORGE_RUNTIME_FILES < <(
  cd "${SFORGE_ROOT}"
  find sforge -type f -name '*.py' -print0 | LC_ALL=C sort -z
)
(( ${#SFORGE_RUNTIME_FILES[@]} > 0 )) \
  || die "SForge Python runtime source set is empty"
MODEL="${MODEL:-}"
[[ -n "${MODEL}" ]] || die "set MODEL to the agent model id (launch/llada-h100/run.sh --model)"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-0.148.0}"
TASK_TIMEOUT="${TASK_TIMEOUT:-7200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-1800}"
requested_submission_cooldown="${SUBMISSION_COOLDOWN:-}"
requested_final_eval_timeout="${FINAL_EVAL_TIMEOUT:-}"
JUDGE_PORT="${JUDGE_PORT:-18384}"
# Host-local SForge/Judge control traffic must never inherit the user's
# outbound HTTP/SOCKS proxy settings.
NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,${NO_PROXY}}"
no_proxy="${NO_PROXY}"
export NO_PROXY no_proxy
LLADA_MODEL_REPO="${LLADA_MODEL_REPO:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--GSAI-ML--LLaDA-8B-Instruct}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
# Label-safe: gpt-N.M-<name> -> <name>, ZHIPU/GLM-5.3-Flash -> zhipu-glm-5.3-flash.
model_slug="$(tr 'A-Z' 'a-z' <<<"${MODEL}" | sed -E 's/^gpt-[0-9.]+-//; s/[^a-z0-9.]+/-/g; s/^-+//; s/-+$//')"
budget_minutes="$(((TASK_TIMEOUT + 59) / 60))"
RUN_ID="${RUN_ID:-llada-decoder-v2-h100-${model_slug}-${REASONING_EFFORT}-${budget_minutes}m-${RUN_STAMP}}"
RUN_LOG_ROOT="${LOGS_DIR}/runs/${RUN_ID}"
TASK_RUN_ROOT="${RUN_LOG_ROOT}/${TASK_ID}"
WORK_CONTAINER_NAME="sforge.run.${TASK_ID}.${RUN_ID}"

# Any model id: a ChatGPT-auth Codex model or whatever MODEL_API_BASE_URL serves.
[[ "${MODEL}" =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]] || die "unsafe MODEL: ${MODEL}"
case "${REASONING_EFFORT}" in
  high|max|xhigh|ultra) ;;
  *) die "fixed run requires REASONING_EFFORT=high, max, xhigh, or ultra" ;;
esac
case "${AGENT_NAME}" in
  codex) ;;
  claude-code)
    [[ -z "${MODEL_API_BASE_URL}" ]] || die "Claude Code uses its OAuth token; unset MODEL_API_BASE_URL"
    [[ "${REASONING_EFFORT}" != ultra ]] || die "Claude Code efforts are high|xhigh|max"
    ;;
  *) die "MLSYSFORGE_AGENT must be codex or claude-code" ;;
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

for numeric_name in TASK_TIMEOUT EVAL_INTERVAL; do
  numeric_value="${!numeric_name}"
  [[ "${numeric_value}" =~ ^[0-9]+$ ]] \
    || die "${numeric_name} must be a positive integer"
  (( 10#${numeric_value} > 0 )) \
    || die "${numeric_name} must be positive"
done
TASK_TIMEOUT="$((10#${TASK_TIMEOUT}))"
EVAL_INTERVAL="$((10#${EVAL_INTERVAL}))"
(( TASK_TIMEOUT >= 90 )) || die "TASK_TIMEOUT must be at least 90 seconds"
final_minutes="$(((TASK_TIMEOUT + 59) / 60))"
interval_minutes="$(((EVAL_INTERVAL + 59) / 60))"

PILOT_MODE="${LLADA_PILOT:-0}"
[[ "${PILOT_MODE}" == "0" || "${PILOT_MODE}" == "1" ]] \
  || die "LLADA_PILOT must be 0 or 1"
load_frozen_memory_policy
if [[ "${PILOT_MODE}" == "1" ]]; then
  load_provisional_workload_policy
else
  load_frozen_workload_policy
fi
# Judge adapter kill: judge/run_judge.py kills the verifier at
# LLADA_{INTERACTIVE,FINAL}_JUDGE_TIMEOUT.  Owner decision 2026-10-13: 6600 s,
# host-side only (the frozen workload policy, its pins and the images are
# unchanged).  It must stay beyond the verifier's own persistent outer
# deadline (frozen final.outer_timeout_seconds, 3900 s, which starts after
# Judge admission, ~170 s in) so a slow candidate gets the verifier's
# candidate-level "outer deadline" verdict and private artifact instead of an
# adapter kill that the harness records as infrastructure_error.
LLADA_JUDGE_KILL_TIMEOUT="${LLADA_JUDGE_KILL_TIMEOUT:-6600}"
[[ "${LLADA_JUDGE_KILL_TIMEOUT}" =~ ^[0-9]+$ ]] \
  || die "LLADA_JUDGE_KILL_TIMEOUT must be a positive integer"
LLADA_JUDGE_KILL_TIMEOUT="$((10#${LLADA_JUDGE_KILL_TIMEOUT}))"
(( LLADA_JUDGE_KILL_TIMEOUT >= POLICY_FINAL_OUTER_TIMEOUT + 600 )) \
  || die "LLADA_JUDGE_KILL_TIMEOUT must exceed the verifier outer deadline by at least 600 s"
QUALITY_CONTROLS_PATH="${MIGRATION_ROOT}/quality_controls.json"
[[ -f "${QUALITY_CONTROLS_PATH}" && ! -L "${QUALITY_CONTROLS_PATH}" ]] \
  || die "H100 run requires a regular quality-control manifest"
if [[ "${PILOT_MODE}" == "1" ]]; then
  # Pilot runs are development evaluations, not release evidence.  Their
  # quality signal comes directly from the sampled hidden cases, so missing or
  # stale release-control archives must not prevent agent feedback.
  QUALITY_CONTROLS_SHA256="$(
    sha256sum "${QUALITY_CONTROLS_PATH}" | awk '{print $1}'
  )"
else
  QUALITY_CONTROLS_SHA256="$(
    python3 "${MIGRATION_ROOT}/scripts/validate_quality_controls.py" \
      --task-root "${MIGRATION_ROOT}" \
      --manifest "${QUALITY_CONTROLS_PATH}" \
      --require-precommitted \
      --verify-archive-hashes
  )" || die "formal H100 run requires verified precommitted quality controls"
fi
[[ "${QUALITY_CONTROLS_SHA256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "quality-control validator returned an invalid digest"
if [[ "${PILOT_MODE}" != "1" ]]; then
  jq -e --arg controls "${QUALITY_CONTROLS_SHA256}" '
    .quality_policy.status == "QUALIFIED"
    and .quality_policy.quality_controls_sha256 == $controls
  ' "${WORKLOAD_POLICY_PATH}" >/dev/null \
    || die "frozen workload policy does not bind the current quality controls"
fi
if [[ -n "${requested_submission_cooldown}" \
      && "${requested_submission_cooldown}" != "${POLICY_SUBMISSION_COOLDOWN}" ]]; then
  die "SUBMISSION_COOLDOWN must match frozen workload policy (${POLICY_SUBMISSION_COOLDOWN})"
fi
SUBMISSION_COOLDOWN="${POLICY_SUBMISSION_COOLDOWN}"
if [[ "${PILOT_MODE}" == "1" ]]; then
  # The previous pilot used the 3,600 s producer inner deadline as its wrapper
  # and killed a healthy 520-case run after 11/12 rows.  Give the verifier
  # 7,000 s and reserve the final 200 s of the task's doubled 7,200 s ceiling
  # for result serialization and private-artifact collection.
  PILOT_FINAL_VERIFIER_TIMEOUT=7000
  PILOT_FINAL_JUDGE_TIMEOUT=7200
  PILOT_FINAL_ROW_TIMEOUT=3600
  if [[ -n "${requested_final_eval_timeout}" \
        && "${requested_final_eval_timeout}" != "${PILOT_FINAL_JUDGE_TIMEOUT}" ]]; then
    die "pilot FINAL_EVAL_TIMEOUT must be ${PILOT_FINAL_JUDGE_TIMEOUT}"
  fi
  FINAL_EVAL_TIMEOUT="${PILOT_FINAL_JUDGE_TIMEOUT}"
else
  # The coordinator's exec kill for the legacy dual-GPU final is the task
  # Judge ceiling: it must stay outside the adapter kill, which itself stays
  # outside the verifier's persistent outer deadline (LLADA_JUDGE_KILL_TIMEOUT above).
  task_judge_ceiling="$(jq -er '.judge.eval_timeout' "${TASKS_DIR}/${TASK_ID}.json")"
  if [[ -n "${requested_final_eval_timeout}" \
        && "${requested_final_eval_timeout}" != "${task_judge_ceiling}" ]]; then
    die "FINAL_EVAL_TIMEOUT must match the task Judge ceiling (${task_judge_ceiling})"
  fi
  FINAL_EVAL_TIMEOUT="${task_judge_ceiling}"
fi

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
elif [[ "${AGENT_NAME}" == "claude-code" ]]; then
  [[ -n "${CLAUDE_TOKEN_FILE}" ]] || die "set CLAUDE_OAUTH_TOKEN_FILE"
  assert_private_auth_file "${CLAUDE_TOKEN_FILE}"
else
  [[ -n "${AUTH_JSON}" ]] \
    || die "set CODEX_AUTH_JSON_PATH to the host-only Codex auth file"
  assert_private_auth_file "${AUTH_JSON}"
fi
export LLADA_MODEL_REPO
python3 "${MIGRATION_ROOT}/scripts/validate_model_artifact.py" \
  --repo "${LLADA_MODEL_REPO}" --full >/dev/null
[[ -x "${SFORGE_PYTHON}" ]] \
  || die "missing SForge virtualenv Python: ${SFORGE_PYTHON}"
[[ -f "${H100_ENTRY}" && ! -L "${H100_ENTRY}" ]] \
  || die "missing H100 SForge entry: ${H100_ENTRY}"
[[ -f "${RESTRICTED_PROXY_SOURCE}" && ! -L "${RESTRICTED_PROXY_SOURCE}" ]] \
  || die "missing restricted proxy source: ${RESTRICTED_PROXY_SOURCE}"
[[ -f "${JUDGE_SERVER_SOURCE}" && ! -L "${JUDGE_SERVER_SOURCE}" ]] \
  || die "missing persistent Judge server source: ${JUDGE_SERVER_SOURCE}"
for relative_path in "${SFORGE_RUNTIME_FILES[@]}"; do
  runtime_path="${SFORGE_ROOT}/${relative_path}"
  [[ -f "${runtime_path}" && ! -L "${runtime_path}" ]] \
    || die "missing SForge runtime source: ${runtime_path}"
done
assert_built_images_current
reference_quality="$(judge_private_data_dir)/reference_quality.json"
hidden_cases_sha256="$(canonical_hidden_cases_sha256)"
if [[ "${PILOT_MODE}" == "1" ]]; then
  workload_provisional_parent_sha256="${WORKLOAD_POLICY_SHA256}"
else
  workload_provisional_parent_sha256="$(
    frozen_workload_parent_sha256 "${WORKLOAD_POLICY_PATH}"
  )"
fi
jq -e \
  --arg task_id "${TASK_ID}" \
  --arg workload_policy_parent_sha256 "${workload_provisional_parent_sha256}" \
  --arg memory_policy_sha256 "${MEMORY_POLICY_SHA256}" '
  .schema_version == 4
  and .calibrated == true
  and .status == "CALIBRATED_STRATIFIED520"
  and .task_id == $task_id
  and .workload_profile == "fast_dllm_v1_llada_12row_stratified520_batch1"
  and .generation_count == 520
  and .requested_slots == 216320
  and (.case_records | type == "object" and length == 520)
  and (.rows | type == "object" and length == 12)
  and (.source_private_report_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.calibration_provenance.schema_version == 1)
  and (.calibration_provenance.candidate_archive_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.calibration_provenance.canonical_candidate_archive_sha256 == .calibration_provenance.candidate_archive_sha256)
  and (.calibration_provenance.candidate_source_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.calibration_provenance.evaluation_validator_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.calibration_provenance.calibration_tool_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.calibration_provenance.image_manifest_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.calibration_provenance.source_digest | type == "string" and test("^[0-9a-f]{64}$"))
  and (.calibration_provenance.judge_runtime_tree_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.calibration_provenance.work_image.id | type == "string" and test("^sha256:[0-9a-f]{64}$"))
  and (.calibration_provenance.judge_image.id | type == "string" and test("^sha256:[0-9a-f]{64}$"))
  and (.calibration_provenance.model_validation_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.calibration_provenance.evaluation_attestation_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.calibration_provenance.private_artifact_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.calibration_provenance.sforge_report_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.calibration_provenance.private_payload_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
  and (.calibration_provenance.run_id | type == "string" and length > 0)
  and (.calibration_provenance.submission_id == .calibration_provenance.run_id)
  and (.calibration_provenance.model_revision == "6059b30c9531767ce47e0ca6a8161a6bb0c67693")
  and (.calibration_provenance.workload_policy_sha256 == $workload_policy_parent_sha256)
  and (.calibration_provenance.memory_policy_sha256 == $memory_policy_sha256)
' "${reference_quality}" >/dev/null \
  || die "formal H100 run requires a calibrated 520-generation quality reference"
[[ "$(jq -er '.dataset_manifest_sha256' "${reference_quality}")" \
    == "${hidden_cases_sha256}" ]] \
  || die "H100 quality reference does not match the canonical workload"
[[ "${hidden_cases_sha256}" \
    == "$(jq -er '.manifest.hidden_cases_sha256' "${WORKLOAD_POLICY_PATH}")" ]] \
  || die "frozen workload policy does not match hidden_cases.json"
python3 "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}" >/dev/null \
  || die "H100 source/reference deep validation failed"

SINGLE_GPU_HANDOFF="${SFORGE_SINGLE_GPU_HANDOFF:-0}"
[[ "${SINGLE_GPU_HANDOFF}" == "0" || "${SINGLE_GPU_HANDOFF}" == "1" ]] \
  || die "SFORGE_SINGLE_GPU_HANDOFF must be 0 or 1"

# The Judge server runs the mandatory LLM integrity review (docs/LLM_JUDGE.md)
# from the SFORGE_LLM_JUDGE_* environment; refuse anything but a ready,
# enforcing configuration before any lock, container or GPU work.
[[ "${SFORGE_LLM_JUDGE_MODE:-}" == "enforce" ]] \
  || die "set SFORGE_LLM_JUDGE_MODE=enforce and the SFORGE_LLM_JUDGE_* reviewer settings"
review_summary="$(
  cd -- "${REPOSITORY_ROOT}" && "${SFORGE_PYTHON}" -B - <<'PY' 2>&1
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
  dry_internet="$(jq -r '.internet' "${TASKS_DIR}/${TASK_ID}.json")"
  [[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || dry_failures+=("unsafe RUN_ID: ${RUN_ID}")
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
  if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
    dry_uuid="${SFORGE_ACCELERATOR_DEVICE_UUID:-}"
    dry_gpu="${dry_uuid} (Work and Judge, one lease; lock $(gpu_lock_path "${dry_uuid}"))"
    dry_used="$(nvidia-smi --id="${dry_uuid}" --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | xargs)" \
      || dry_failures+=("nvidia-smi cannot see ${dry_uuid}")
    dry_gpu+=" [index,used: ${dry_used:-?}]"
    # Read-only (one nvidia-smi query): the Judge image's reference must match
    # this card's measurement profile, as the post-lock check below requires.
    "${PYTHON_BIN:-python3}" "${SCRIPT_DIR}/select_reference_profile.py" \
      --judge-gpu "${dry_used%%,*}" --check >/dev/null \
      || dry_failures+=("select_reference_profile.py --check failed for GPU ${dry_used%%,*}")
  else
    dry_gpu="Work ${WORK_GPU:-<unset>} / Judge ${JUDGE_GPU:-<unset>} (legacy dual GPU)"
  fi
  echo "Dry run (no lock, container or GPU work): ${RUN_ID}"
  echo "  agent:        ${AGENT_NAME} ${MODEL} effort=${REASONING_EFFORT} runtime ${AGENT_RUNTIME_DIR} (${runtime_version})"
  echo "  credential:   ${dry_credential}"
  echo "  budget:       ${TASK_TIMEOUT}s; auto-eval off; unlimited manual submits, cooldown ${SUBMISSION_COOLDOWN}s"
  echo "  final Judge:  standalone after the deadline, same GPU under the held lock; verifier outer ${POLICY_FINAL_OUTER_TIMEOUT}s (after admission), adapter kill ${LLADA_JUDGE_KILL_TIMEOUT}s, coordinator ceiling $(jq -r '.judge.eval_timeout' "${TASKS_DIR}/${TASK_ID}.json")s"
  echo "  GPU:          ${dry_gpu}"
  echo "  judge port:   ${JUDGE_PORT}"
  echo "  network:      internet=${dry_internet} web_search=$([[ "${dry_internet}" == true ]] && echo live || echo disabled)"
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
  # One scheduler-issued whole GPU shared serially by Work and Judge.
  [[ ! -v WORK_GPU && ! -v JUDGE_GPU ]] \
    || die "single GPU handoff rejects WORK_GPU/JUDGE_GPU"
  [[ "${SFORGE_ACCELERATOR_DEVICE_UUID:-}" =~ ^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]] \
    || die "SFORGE_ACCELERATOR_DEVICE_UUID must be a qualified canonical whole-GPU UUID"
  [[ "${SFORGE_ACCELERATOR_ALLOCATION_KEY:-}" =~ ^allocation-sha256:[0-9a-f]{64}$ ]] \
    || die "SFORGE_ACCELERATOR_ALLOCATION_KEY must be allocation-sha256:<64 lowercase hex>"
  [[ "${SFORGE_TRUSTED_LEASE_ID:-}" =~ ^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$ ]] \
    || die "SFORGE_TRUSTED_LEASE_ID must be a label-safe never-reused identity"
  [[ "${SFORGE_ACCELERATOR_OBSERVER:-}" == "nvml" ]] \
    || die "single GPU handoff requires SFORGE_ACCELERATOR_OBSERVER=nvml"
  work_gpu_uuid="${SFORGE_ACCELERATOR_DEVICE_UUID}"
  judge_gpu_uuid="${SFORGE_ACCELERATOR_DEVICE_UUID}"
  WORK_GPU="$(gpu_index_of_uuid "${work_gpu_uuid}")"
  JUDGE_GPU="${WORK_GPU}"

  # The selected portable execution profile supplies the capability envelope
  # and the software-ABI / calibration identities of the measurement profile.
  readarray -t portable_identity < <(
    "${SFORGE_PYTHON}" - "${TASKS_DIR}/${TASK_ID}.json" <<'PY'
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
  [[ "${execution_profile_id}" == "single-cuda-sm80-34g" && "${profile_count}" == "1" ]] \
    || die "single GPU handoff requires the task's single-accelerator execution profile default"
  [[ "${profile_min_compute_capability}" == "${GPU_MIN_COMPUTE_CAPABILITY}" \
     && "$(( (10#${profile_min_memory_bytes} + 1024 * 1024 - 1) / (1024 * 1024) ))" \
        == "${GPU_MIN_TOTAL_MEMORY_MIB}" ]] \
    || die "portable execution profile envelope differs from the launcher envelope"

  acquire_single_gpu_lock "${work_gpu_uuid}"
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
  # Capability envelope only: compute capability, device capacity, free memory
  # and exclusive ownership.  The GPU model is recorded as provenance below.
  assert_portable_gpu_idle \
    "${work_gpu_uuid}" \
    "single GPU" \
    "${profile_vendor}" \
    "${profile_count}" \
    "${profile_min_compute_capability}" \
    "${profile_min_memory_bytes}"
  assert_gpu_available "${WORK_GPU}" "single allocation" "${GPU_ADMISSION_MIN_FREE_MIB}"
  assert_single_gpu_lock_held
  # The producer-relative quality reference is only comparable within one
  # measurement profile; refuse a Judge image built for a different device.
  "${PYTHON_BIN:-python3}" "${SCRIPT_DIR}/select_reference_profile.py" \
    --judge-gpu "${JUDGE_GPU}" --check
  assert_container_accelerator "${WORK_IMAGE}" "${work_gpu_uuid}" "Work"
  assert_container_accelerator "${JUDGE_IMAGE}" "${judge_gpu_uuid}" "Judge"
  assert_container_gpu "${WORK_IMAGE}" "${work_gpu_uuid}" "Work"
  assert_container_gpu "${JUDGE_IMAGE}" "${judge_gpu_uuid}" "Judge"
  assert_single_gpu_lock_held
  handoff_mode="single_allocation_serial_v1"
  scheduling_mode="single_gpu_lease_manual_judge_then_standalone_final_handoff"
  gpu_lock_scheme="single_uuid_flock_work_judge_final"
  allocation_key="${SFORGE_ACCELERATOR_ALLOCATION_KEY}"
  trusted_lease_id="${SFORGE_TRUSTED_LEASE_ID}"

  # Two deliberately separate identities.  deployment_attestation records the
  # physical UUID and observed clocks for audit only.  measurement_profile
  # hashes the stable comparability key: model / compute capability / memory /
  # driver / exact immutable image IDs / software ABI / calibration protocol /
  # campaign epoch.  The UUID and observed clocks never enter that key.
  measurement_gpu_name="$(gpu_name "${JUDGE_GPU}")"
  measurement_driver_version="$(gpu_query "${JUDGE_GPU}" driver_version)"
  measurement_compute_capability="$(gpu_query "${JUDGE_GPU}" compute_cap)"
  measurement_memory_mib="$(gpu_query "${JUDGE_GPU}" memory.total)"
  measurement_sm_clock_mhz="$(gpu_query "${JUDGE_GPU}" clocks.current.sm)"
  measurement_memory_clock_mhz="$(gpu_query "${JUDGE_GPU}" clocks.current.memory)"
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
  # The run-local image manifest (and the recorded_* IDs read from it) is
  # only frozen further down; the measurement profile needs the exact image
  # IDs now, so resolve the currency-checked images directly.  The later
  # provenance freeze re-asserts that these are the IDs it records.
  profile_work_image_id="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")" \
    || die "cannot resolve the Work image ID for the measurement profile"
  profile_judge_image_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")" \
    || die "cannot resolve the Judge image ID for the measurement profile"
  [[ "${profile_work_image_id}" =~ ^sha256:[0-9a-f]{64}$ \
     && "${profile_judge_image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || die "measurement profile image IDs are not canonical"
  measurement_profile_json="$(
    jq -cn \
      --arg gpu_name "${measurement_gpu_name}" \
      --arg driver_version "${measurement_driver_version}" \
      --arg compute_capability "${measurement_compute_capability}" \
      --arg memory_mib "${measurement_memory_mib}" \
      --arg work_image_id "${profile_work_image_id}" \
      --arg judge_image_id "${profile_judge_image_id}" \
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
  scheduling_mode="dual_gpu_interactive_then_frozen_final"
  execution_profile_id="dual-cuda-sm80-34g"
  gpu_lock_scheme="two_uuid_flocks_deterministic_order"
  [[ -v WORK_GPU && -n "${WORK_GPU}" ]] \
    || die "formal runs require an explicit numeric WORK_GPU"
  [[ -v JUDGE_GPU && -n "${JUDGE_GPU}" ]] \
    || die "formal runs require an explicit numeric JUDGE_GPU"
  WORK_GPU="$(canonical_gpu_index "${WORK_GPU}")"
  JUDGE_GPU="$(canonical_gpu_index "${JUDGE_GPU}")"
  [[ "${WORK_GPU}" != "${JUDGE_GPU}" ]] \
    || die "Work and Judge GPU indices must differ"
  work_gpu_uuid="$(gpu_uuid "${WORK_GPU}")"
  judge_gpu_uuid="$(gpu_uuid "${JUDGE_GPU}")"
  [[ "${work_gpu_uuid}" != "${judge_gpu_uuid}" ]] \
    || die "Work and Judge GPU UUIDs must differ"
  acquire_dual_gpu_locks "${work_gpu_uuid}" "${judge_gpu_uuid}"
  # Capability envelope only: compute capability, device capacity, free memory
  # and exclusive ownership.  The GPU model is recorded as provenance below.
  assert_distinct_gpu_pair \
    "${WORK_GPU}" "${JUDGE_GPU}" "${GPU_ADMISSION_MIN_FREE_MIB}"
  # The producer-relative quality reference is only comparable within one
  # measurement profile; refuse a Judge image built for a different device.
  "${PYTHON_BIN:-python3}" "${SCRIPT_DIR}/select_reference_profile.py" \
    --judge-gpu "${JUDGE_GPU}" --check
  assert_container_gpu "${WORK_IMAGE}" "${work_gpu_uuid}" "Work"
  assert_container_gpu "${JUDGE_IMAGE}" "${judge_gpu_uuid}" "Judge"
  assert_dual_gpu_locks_held
fi
export LLADA_EXECUTION_PROFILE_ID="${execution_profile_id}"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  interactive_evaluation_lifecycle="serial_single_gpu_lease_handoff"
  final_evaluation_lifecycle="coordinator_standalone_final_after_work_retired_and_archive_frozen"
else
  interactive_evaluation_lifecycle="concurrent_distinct_gpus"
  final_evaluation_lifecycle="after_work_stopped_and_archive_frozen"
fi
# The task-contract Judge ceiling is the coordinator's outer kill limit for
# the standalone final; the qualified release deadline is enforced inside the
# Judge by its persistent outer deadline (workload policy).
TASK_EVAL_TIMEOUT="$(jq -er '.judge.eval_timeout' "${TASKS_DIR}/${TASK_ID}.json")"
[[ "${TASK_EVAL_TIMEOUT}" =~ ^[0-9]+$ ]] \
  || die "task JSON lacks an integer judge.eval_timeout"
# The task JSON's `internet` flag decides Work's egress
# (docs/INTERNET_AND_WEB_SEARCH.md).  true: the sidecar also tunnels to public
# internet hosts (never to private/loopback/link-local addresses) and Codex
# runs with live web search.  false: the historical allowlist-only sidecar and
# offline Codex.  Work stays on the internal network either way.
TASK_INTERNET="$(
  jq -er '.internet | if type == "boolean" then tostring
    else error("internet must be a boolean") end' "${TASKS_DIR}/${TASK_ID}.json"
)" || die "task JSON lacks a boolean internet flag"
if [[ "${TASK_INTERNET}" == "true" ]]; then
  restricted_proxy_egress_args=(--allow-public-egress)
  codex_web_search="live"
else
  restricted_proxy_egress_args=()
  codex_web_search="disabled"
fi
# Read by scripts/sforge_h100_entry.py when it builds the Codex command and
# prompt.
export MLSYSFORGE_WORK_INTERNET="${TASK_INTERNET}"
(( POLICY_FINAL_OUTER_TIMEOUT < LLADA_JUDGE_KILL_TIMEOUT \
    && LLADA_JUDGE_KILL_TIMEOUT < TASK_EVAL_TIMEOUT )) \
  || die "deadline order must be verifier outer < Judge adapter kill < task Judge ceiling"

work_gpu_name="$(gpu_name "${WORK_GPU}")"
judge_gpu_name="$(gpu_name "${JUDGE_GPU}")"
work_gpu_total_mib="$(gpu_query "${WORK_GPU}" memory.total)"
judge_gpu_total_mib="$(gpu_query "${JUDGE_GPU}" memory.total)"
work_gpu_free_mib="$(gpu_query "${WORK_GPU}" memory.free)"
judge_gpu_free_mib="$(gpu_query "${JUDGE_GPU}" memory.free)"
work_gpu_compute_capability="$(gpu_query "${WORK_GPU}" compute_cap)"
judge_gpu_compute_capability="$(gpu_query "${JUDGE_GPU}" compute_cap)"
gpu_driver_version="$(gpu_query "${WORK_GPU}" driver_version)"
gpu_observed_at="$(date -u +%FT%TZ)"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
  || die "unsafe RUN_ID: ${RUN_ID}"
[[ ! -e "${RUN_LOG_ROOT}" ]] || die "run directory already exists: ${RUN_LOG_ROOT}"
# A version store that already holds events for this RUN_ID makes the harness
# fail closed only when it persists the final version, i.e. after the whole
# agent budget has been spent (VersionStoreCollision, seen live). Refuse early.
if [[ -n "${SFORGE_VERSION_STORE_ROOT:-}" \
    && -e "${SFORGE_VERSION_STORE_ROOT}/tasks/$(basename "${MIGRATION_ROOT}")/runs/${RUN_ID}" ]]; then
  die "version store already records run id ${RUN_ID}; choose a fresh RUN_ID"
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${JUDGE_PORT}$"; then
  die "judge port ${JUDGE_PORT} is already in use"
fi

mkdir -p "${RUN_LOG_ROOT}"
model_validation_snapshot="${RUN_LOG_ROOT}/full-model-validation.json"
python3 "${MIGRATION_ROOT}/scripts/validate_model_artifact.py" \
  --repo "${LLADA_MODEL_REPO}" --full \
  --output "${model_validation_snapshot}" >/dev/null
model_validation_sha256="$(
  sha256sum "${model_validation_snapshot}" | awk '{print $1}'
)"
cohort_secret_file="${RUN_LOG_ROOT}/.cohort-secret"
(umask 077; python3 -c 'import secrets; print(secrets.token_hex(32))' >"${cohort_secret_file}")
IFS= read -r cohort_secret <"${cohort_secret_file}"
[[ "${cohort_secret}" =~ ^[0-9a-f]{64}$ ]] \
  || die "failed to generate the private cohort secret"
cohort_secret_sha256="$(printf '%s' "${cohort_secret}" | sha256sum | awk '{print $1}')"
global_image_manifest="${LOGS_DIR}/build/image-manifest.json"
run_image_manifest="${RUN_LOG_ROOT}/image-build-manifest.json"
[[ -f "${global_image_manifest}" && ! -L "${global_image_manifest}" ]] \
  || die "image build manifest must be a regular non-symlink file"
cp -- "${global_image_manifest}" "${run_image_manifest}"
chmod 444 "${run_image_manifest}"
image_manifest_sha256="$(sha256sum "${run_image_manifest}" | awk '{print $1}')"
run_memory_policy="${RUN_LOG_ROOT}/memory-policy.json"
run_workload_policy="${RUN_LOG_ROOT}/workload-policy.json"
run_quality_controls="${RUN_LOG_ROOT}/quality-controls-manifest.json"
cp -- "${MEMORY_POLICY_PATH}" "${run_memory_policy}"
cp -- "${WORKLOAD_POLICY_PATH}" "${run_workload_policy}"
cp -- "${QUALITY_CONTROLS_PATH}" "${run_quality_controls}"
chmod 444 \
  "${run_memory_policy}" \
  "${run_workload_policy}" \
  "${run_quality_controls}"
[[ "$(sha256sum "${run_memory_policy}" | awk '{print $1}')" \
    == "${MEMORY_POLICY_SHA256}" ]] \
  || die "run-local memory policy digest mismatch"
[[ "$(sha256sum "${run_workload_policy}" | awk '{print $1}')" \
    == "${WORKLOAD_POLICY_SHA256}" ]] \
  || die "run-local workload policy digest mismatch"
[[ "$(sha256sum "${run_quality_controls}" | awk '{print $1}')" \
    == "${QUALITY_CONTROLS_SHA256}" ]] \
  || die "run-local quality-control manifest digest mismatch"
recorded_image_source_digest="$(jq -er '.source_digest' "${run_image_manifest}")" \
  || die "run-local image manifest lacks source_digest"
recorded_work_image_name="$(jq -er '.work.name' "${run_image_manifest}")" \
  || die "run-local image manifest lacks work image name"
recorded_work_image_id="$(jq -er '.work.id' "${run_image_manifest}")" \
  || die "run-local image manifest lacks work image ID"
recorded_judge_image_name="$(jq -er '.judge.name' "${run_image_manifest}")" \
  || die "run-local image manifest lacks judge image name"
recorded_judge_image_id="$(jq -er '.judge.id' "${run_image_manifest}")" \
  || die "run-local image manifest lacks judge image ID"
[[ "${recorded_image_source_digest}" =~ ^[0-9a-f]{64}$ ]] \
  || die "run-local image manifest has an invalid source digest"
[[ "${recorded_work_image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || die "run-local image manifest has an invalid Work image ID"
[[ "${recorded_judge_image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || die "run-local image manifest has an invalid Judge image ID"
[[ "${recorded_work_image_name}" == "${WORK_IMAGE}" ]] \
  || die "run-local image manifest Work name mismatch"
[[ "${recorded_judge_image_name}" == "${JUDGE_IMAGE}" ]] \
  || die "run-local image manifest Judge name mismatch"
[[ "$(image_source_digest)" == "${recorded_image_source_digest}" ]] \
  || die "image sources changed while freezing run provenance"
[[ "$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")" \
    == "${recorded_work_image_id}" ]] \
  || die "Work image changed while freezing run provenance"
[[ "$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")" \
    == "${recorded_judge_image_id}" ]] \
  || die "Judge image changed while freezing run provenance"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  [[ "${profile_work_image_id}" == "${recorded_work_image_id}" \
     && "${profile_judge_image_id}" == "${recorded_judge_image_id}" ]] \
    || die "measurement profile image IDs differ from the frozen run provenance"
fi
final_eval_id="${RUN_ID}-final-${final_minutes}m"
final_run_root="${LOGS_DIR}/runs/${final_eval_id}"
[[ ! -e "${final_run_root}" && ! -L "${final_run_root}" ]] \
  || die "final eval directory already exists: ${final_eval_id}"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  # The standalone final Judge is created by the long-lived coordinator, so
  # its final run identity and root-owned checkpoint directory must exist and
  # be exported before the server process starts (the adapter binds them only
  # to the coordinator's final Judge container).
  mkdir -- "${final_run_root}"
  prepare_root_owned_checkpoint_dir \
    "${final_run_root}" "${final_eval_id}" "${recorded_judge_image_id}"
  export LLADA_FINAL_RUNTIME_RUN_ID="${final_eval_id}"
  export LLADA_FINAL_JUDGE_CHECKPOINT_DIR="${LLADA_JUDGE_CHECKPOINT_DIR}"
fi
prepare_root_owned_checkpoint_dir \
  "${RUN_LOG_ROOT}" "${RUN_ID}" "${recorded_judge_image_id}"
sforge_runtime_sha256="$(
  (
    cd "${SFORGE_ROOT}"
    sha256sum "${SFORGE_RUNTIME_FILES[@]}"
  ) | sha256sum | awk '{print $1}'
)"
[[ "${sforge_runtime_sha256}" =~ ^[0-9a-f]{64}$ ]] \
  || die "failed to compute SForge runtime digest"
sforge_runtime_file_set_sha256="$(
  printf '%s\n' "${SFORGE_RUNTIME_FILES[@]}" | sha256sum | awk '{print $1}'
)"
sforge_runtime_files_json="$(
  printf '%s\n' "${SFORGE_RUNTIME_FILES[@]}" | jq -Rsc 'split("\n")[:-1]'
)"
mapfile -t judge_core_hashes_start < <(
  docker run --rm --network none --entrypoint sha256sum \
    "${recorded_judge_image_id}" \
    /opt/decoder_judge/verifier/verify_decoder.py \
    /opt/decoder_judge/verifier/candidate_worker.py \
    /opt/decoder_judge/verifier/cohort.py \
    /opt/decoder_judge/verifier/quality.py \
    /opt/decoder_judge/verifier/source_admission.py \
    | awk '{print $1}'
)
(( ${#judge_core_hashes_start[@]} == 5 )) \
  || die "failed to attest the five Judge invariant core files"
for core_sha256 in "${judge_core_hashes_start[@]}"; do
  [[ "${core_sha256}" =~ ^[0-9a-f]{64}$ ]] \
    || die "Judge invariant core returned an invalid SHA-256"
done
judge_verify_decoder_sha256="${judge_core_hashes_start[0]}"
judge_candidate_worker_sha256="${judge_core_hashes_start[1]}"
judge_cohort_sha256="${judge_core_hashes_start[2]}"
judge_quality_sha256="${judge_core_hashes_start[3]}"
judge_source_admission_sha256="${judge_core_hashes_start[4]}"
runtime_provenance_snapshot="${RUN_LOG_ROOT}/runtime-provenance.json"
write_runtime_provenance_snapshot \
  "${runtime_provenance_snapshot}" \
  "${sforge_runtime_sha256}" \
  "${sforge_runtime_file_set_sha256}" \
  "${#SFORGE_RUNTIME_FILES[@]}" \
  "${sforge_runtime_files_json}" \
  "${judge_verify_decoder_sha256}" \
  "${judge_candidate_worker_sha256}" \
  "${judge_cohort_sha256}" \
  "${judge_quality_sha256}" \
  "${judge_source_admission_sha256}"
chmod 444 "${runtime_provenance_snapshot}"
runtime_provenance_sha256="$(
  sha256sum "${runtime_provenance_snapshot}" | awk '{print $1}'
)"
judge_pid=""
active_pid=""
isolated_network=""
restricted_proxy_name=""
final_judge_container_name=""
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
cleanup() {
  stop_active_process
  docker rm -f "${WORK_CONTAINER_NAME}" >/dev/null 2>&1 || true
  if [[ -n "${final_judge_container_name}" ]]; then
    docker rm -f "${final_judge_container_name}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${judge_pid}" ]] && kill -0 "${judge_pid}" 2>/dev/null; then
    kill "${judge_pid}" 2>/dev/null || true
    wait "${judge_pid}" 2>/dev/null || true
  fi
  # Interactive submission IDs are intentionally random, so an interrupted
  # Judge cannot be named in advance. Labels plus our already-held UUID locks
  # scope this cleanup to this task's running containers on the reserved pair.
  remove_task_gpu_containers "work" "${work_gpu_uuid}" "${RUN_ID}" \
    >/dev/null 2>&1 || true
  remove_task_gpu_containers "judge" "${judge_gpu_uuid}" "${RUN_ID}" \
    >/dev/null 2>&1 || true
  if [[ -n "${final_eval_id:-}" ]]; then
    remove_task_gpu_containers \
      "judge" "${judge_gpu_uuid}" "${final_eval_id}" \
      >/dev/null 2>&1 || true
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
    return
  else
    command_rc="$?"
  fi
  rm -f -- "${temporary_file}"
  return "${command_rc}"
}
stop_judge_server() {
  if [[ -n "${judge_pid}" ]] && kill -0 "${judge_pid}" 2>/dev/null; then
    kill "${judge_pid}"
    wait "${judge_pid}" || true
  fi
  judge_pid=""
}
remove_work_network() {
  if [[ -n "${restricted_proxy_name}" ]]; then
    docker rm -f "${restricted_proxy_name}" >/dev/null
    restricted_proxy_name=""
  fi
  if [[ -n "${isolated_network}" ]]; then
    docker network rm "${isolated_network}" >/dev/null
    isolated_network=""
  fi
}
trap cleanup EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

export SFORGE_TASKS_DIR="${TASKS_DIR}"
export SFORGE_LOG_DIR="${LOGS_DIR}"
export LLADA_WORK_GPU_UUID="${work_gpu_uuid}"
export LLADA_JUDGE_GPU_UUID="${judge_gpu_uuid}"
export LLADA_RUNTIME_RUN_ID="${RUN_ID}"
export SFORGE_EXPECTED_WORK_IMAGE_ID="${recorded_work_image_id}"
export SFORGE_EXPECTED_JUDGE_IMAGE_ID="${recorded_judge_image_id}"
runtime_entry_sha256="$(sha256sum "${H100_ENTRY}" | awk '{print $1}')"
restricted_proxy_sha256="$(
  sha256sum "${RESTRICTED_PROXY_SOURCE}" | awk '{print $1}'
)"
judge_server_sha256="$(
  sha256sum "${JUDGE_SERVER_SOURCE}" | awk '{print $1}'
)"
judge_extra_env="NVIDIA_VISIBLE_DEVICES=${judge_gpu_uuid},CUDA_VISIBLE_DEVICES=0,LLADA_TARGET=${execution_profile_id},LLADA_GPU_ROLE=judge,LLADA_INTERACTIVE_JUDGE_TIMEOUT=${LLADA_JUDGE_KILL_TIMEOUT},LLADA_FINAL_JUDGE_TIMEOUT=${LLADA_JUDGE_KILL_TIMEOUT},SFORGE_PRIVATE_CONTEXT_CHECKPOINT_DIR=/logs/verifier/checkpoints,SFORGE_PRIVATE_CONTEXT_COHORT_SECRET=${cohort_secret},SFORGE_PRIVATE_CONTEXT_QUALITY_CONTROLS_SHA256=${QUALITY_CONTROLS_SHA256},SFORGE_PRIVATE_CONTEXT_RUNTIME_PROVENANCE_SHA256=${runtime_provenance_sha256},SFORGE_PRIVATE_CONTEXT_SFORGE_RUNTIME_SHA256=${sforge_runtime_sha256},SFORGE_PRIVATE_CONTEXT_SFORGE_RUNTIME_FILE_SET_SHA256=${sforge_runtime_file_set_sha256},SFORGE_PRIVATE_CONTEXT_SFORGE_RUNTIME_FILE_COUNT=${#SFORGE_RUNTIME_FILES[@]},SFORGE_PRIVATE_CONTEXT_JUDGE_VERIFY_DECODER_SHA256=${judge_verify_decoder_sha256},SFORGE_PRIVATE_CONTEXT_JUDGE_CANDIDATE_WORKER_SHA256=${judge_candidate_worker_sha256},SFORGE_PRIVATE_CONTEXT_JUDGE_COHORT_SHA256=${judge_cohort_sha256},SFORGE_PRIVATE_CONTEXT_JUDGE_QUALITY_SHA256=${judge_quality_sha256},SFORGE_PRIVATE_CONTEXT_JUDGE_SOURCE_ADMISSION_SHA256=${judge_source_admission_sha256}"
# Every full scored round binds the host runtime and image identity. The
# coordinator injects ARCHIVE_SHA256 from each submitted frozen archive.
judge_extra_env="${judge_extra_env},SFORGE_PRIVATE_CONTEXT_WORK_IMAGE_ID=${recorded_work_image_id},SFORGE_PRIVATE_CONTEXT_JUDGE_IMAGE_ID=${recorded_judge_image_id},SFORGE_PRIVATE_CONTEXT_MODEL_VALIDATION_SHA256=${model_validation_sha256},SFORGE_PRIVATE_CONTEXT_RUNTIME_ENTRY_SHA256=${runtime_entry_sha256},SFORGE_PRIVATE_CONTEXT_SOURCE_DIGEST=${recorded_image_source_digest}"
if [[ "${PILOT_MODE}" == "1" ]]; then
  judge_extra_env="${judge_extra_env},SFORGE_PRIVATE_CONTEXT_PILOT=1,LLADA_FINAL_JUDGE_TIMEOUT=${PILOT_FINAL_JUDGE_TIMEOUT},LLADA_PILOT_FINAL_OUTER_TIMEOUT=${PILOT_FINAL_VERIFIER_TIMEOUT},LLADA_PILOT_FINAL_ROW_TIMEOUT=${PILOT_FINAL_ROW_TIMEOUT}"
fi
export SFORGE_JUDGE_EXTRA_ENV="${judge_extra_env}"
export SFORGE_ADMIN_SECRET
SFORGE_ADMIN_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
unset SFORGE_ISOLATED_DOCKER_NETWORK

# Bind the evaluation service only to Docker's host-side bridge address.  The
# Work container reaches it through the run-scoped sidecar below; it is never
# exposed on a public host interface.
docker_bridge_gateway="$(
  docker network inspect bridge \
    | jq -er '.[0].IPAM.Config[0].Gateway
      | select(type == "string" and length > 0)'
)"
# In single-allocation mode the coordinator also binds loopback, the only
# origin the standalone-final handoff client accepts; still no public
# interface.
judge_serve_extra_hosts=()
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  judge_serve_extra_hosts+=(--host 127.0.0.1)
fi
"${SFORGE_PYTHON}" "${H100_ENTRY}" serve \
  --host "${docker_bridge_gateway}" "${judge_serve_extra_hosts[@]}" --port "${JUDGE_PORT}" \
  >"${RUN_LOG_ROOT}/judge-server.log" 2>&1 &
judge_pid="$!"
wait_for_http "http://${docker_bridge_gateway}:${JUDGE_PORT}/openapi.json" 90
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  wait_for_http "http://127.0.0.1:${JUDGE_PORT}/openapi.json" 30
fi

# The Work container gets no default route. A host-controlled sidecar permits
# CONNECT to the three Codex authentication/model hosts and, when the task
# allows internet, to any public host on 443/80 (private, loopback and
# link-local addresses stay refused). This avoids SForge's sudo-iptables
# dependency without exposing local services to Work.
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
  --evaluation-upstream-host "${docker_bridge_gateway}" \
  --evaluation-upstream-port "${JUDGE_PORT}" \
  "${restricted_proxy_egress_args[@]}" \
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
NO_PROXY="${NO_PROXY},${docker_bridge_gateway},${restricted_proxy_ip}"
no_proxy="${NO_PROXY}"
export NO_PROXY no_proxy

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
  -fsS "http://${restricted_proxy_ip}:3128/health" >/dev/null
docker run --rm \
  --network "${isolated_network}" \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint curl \
  "${WORK_IMAGE}" \
  --connect-timeout 2 \
  --max-time 5 \
  -fsS "http://${restricted_proxy_ip}:3130/openapi.json" >/dev/null
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
# CONNECT policy probes (HOST=STATUS, all on port 443).  The model host must
# tunnel in both modes; with public egress a public host tunnels and the
# Docker bridge gateway (literal and via host.docker.internal) stays refused.
connect_probes=("chatgpt.com=200")
if [[ "${TASK_INTERNET}" == "true" ]]; then
  connect_probes+=(
    "github.com=200"
    "${docker_bridge_gateway}=403"
    "host.docker.internal=403"
  )
else
  connect_probes+=("github.com=403")
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
        s.settimeout(30)
        request=f"CONNECT {host}:443 HTTP/1.1\r\nHost: {host}:443\r\n\r\n"
        s.sendall(request.encode("ascii"))
        line=s.recv(256).split(b"\r\n",1)[0]
        if expected not in line:
            raise SystemExit(f"{host}: expected {expected!r}, got {line!r}")
for spec in sys.argv[2:]:
    host, _, status = spec.rpartition("=")
    probe(host, f" {status} ".encode("ascii"))' \
  "${restricted_proxy_ip}" \
  "${connect_probes[@]}"
echo "Restricted network: PASS (${isolated_network}, proxy ${restricted_proxy_ip}, public egress via proxy: ${TASK_INTERNET})"

# Force file-based ChatGPT authentication. Credential bytes are copied only
# into the ephemeral Work container and are rejected from submission archives.
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
# Read by sforge_h100_entry.py (claude_local.install_from_environment).
export MLSYSFORGE_AGENT="${AGENT_NAME}" MLSYSFORGE_AGENT_MODEL="${MODEL}"
export MLSYSFORGE_AGENT_EFFORT="${REASONING_EFFORT}" MLSYSFORGE_AGENT_BUDGET_SECONDS="${TASK_TIMEOUT}"
export MLSYSFORGE_RUN_LOG_ROOT="${RUN_LOG_ROOT}"
export SFORGE_CODEX_REASONING_EFFORT="${REASONING_EFFORT}"
export SFORGE_CODEX_CLI_VERSION="${CODEX_CLI_VERSION}"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  # sforge-submit runs the root-owned cooperative drain hook before archive
  # and HTTP admission; the host coordinator still pauses Work and attests.
  export SFORGE_AGENT_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${work_gpu_uuid},CUDA_VISIBLE_DEVICES=0,LLADA_TARGET=${execution_profile_id},LLADA_GPU_ROLE=work,SFORGE_PRE_SUBMIT_HOOK=/app/workdir/decoder_lab/.sforge/prepare_submit.sh"
else
  export SFORGE_AGENT_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${work_gpu_uuid},CUDA_VISIBLE_DEVICES=0,LLADA_TARGET=${execution_profile_id},LLADA_GPU_ROLE=work"
fi
export SFORGE_ISOLATED_DOCKER_NETWORK="${isolated_network}"
export SFORGE_HTTP_PROXY="http://${restricted_proxy_ip}:3128"
export SFORGE_HTTPS_PROXY="http://${restricted_proxy_ip}:3128"
export SFORGE_NO_PROXY="${restricted_proxy_ip},127.0.0.1,localhost"

# The private cohort's contract fields come from h100_policy_contract.py, the
# same constants validate_formal_artifacts.py pre checks the manifest against.
private_cohort_contract="$(python3 "${MIGRATION_ROOT}/scripts/private_cohort_contract.py")" \
  || die "cannot read the private-cohort contract"
jq -n \
  --arg run_id "${RUN_ID}" \
  --arg task_id "${TASK_ID}" \
  --arg started_at "$(date -u +%FT%TZ)" \
  --arg agent "${AGENT_NAME}" \
  --arg model "${MODEL}" \
  --arg effort "${REASONING_EFFORT}" \
  --arg cli_version "${CODEX_CLI_VERSION}" \
  --arg runtime_dir "${AGENT_RUNTIME_DIR}" \
  --arg runtime_agent "${runtime_agent}" \
  --arg runtime_version "${runtime_version}" \
  --arg runtime_tree_sha256 "${runtime_tree_sha256}" \
  --argjson work_gpu_index "${WORK_GPU}" \
  --arg work_gpu_uuid "${work_gpu_uuid}" \
  --arg work_gpu_name "${work_gpu_name}" \
  --argjson work_gpu_total_mib "${work_gpu_total_mib}" \
  --argjson work_gpu_free_mib "${work_gpu_free_mib}" \
  --arg work_gpu_compute_capability "${work_gpu_compute_capability}" \
  --argjson judge_gpu_index "${JUDGE_GPU}" \
  --arg judge_gpu_uuid "${judge_gpu_uuid}" \
  --arg judge_gpu_name "${judge_gpu_name}" \
  --argjson judge_gpu_total_mib "${judge_gpu_total_mib}" \
  --argjson judge_gpu_free_mib "${judge_gpu_free_mib}" \
  --arg judge_gpu_compute_capability "${judge_gpu_compute_capability}" \
  --arg gpu_driver_version "${gpu_driver_version}" \
  --arg gpu_observed_at "${gpu_observed_at}" \
  --arg model_revision "6059b30c9531767ce47e0ca6a8161a6bb0c67693" \
  --arg network_mode "docker_internal_restricted_connect_proxy" \
  --arg runtime_entry_sha256 "${runtime_entry_sha256}" \
  --arg restricted_proxy_sha256 "${restricted_proxy_sha256}" \
  --arg judge_server_sha256 "${judge_server_sha256}" \
  --arg image_manifest_sha256 "${image_manifest_sha256}" \
  --arg image_source_digest "${recorded_image_source_digest}" \
  --arg work_image_name "${recorded_work_image_name}" \
  --arg work_image_id "${recorded_work_image_id}" \
  --arg judge_image_name "${recorded_judge_image_name}" \
  --arg judge_image_id "${recorded_judge_image_id}" \
  --arg runtime_provenance_sha256 "${runtime_provenance_sha256}" \
  --arg sforge_runtime_sha256 "${sforge_runtime_sha256}" \
  --arg sforge_runtime_file_set_sha256 "${sforge_runtime_file_set_sha256}" \
  --argjson sforge_runtime_files "${sforge_runtime_files_json}" \
  --arg judge_verify_decoder_sha256 "${judge_verify_decoder_sha256}" \
  --arg judge_candidate_worker_sha256 "${judge_candidate_worker_sha256}" \
  --arg judge_cohort_sha256 "${judge_cohort_sha256}" \
  --arg judge_quality_sha256 "${judge_quality_sha256}" \
  --arg judge_source_admission_sha256 "${judge_source_admission_sha256}" \
  --arg cohort_secret_sha256 "${cohort_secret_sha256}" \
  --arg memory_policy_sha256 "${MEMORY_POLICY_SHA256}" \
  --arg workload_policy_sha256 "${WORKLOAD_POLICY_SHA256}" \
  --arg quality_controls_sha256 "${QUALITY_CONTROLS_SHA256}" \
  --arg codex_web_search "${codex_web_search}" \
  --argjson proxy_public_egress "${TASK_INTERNET}" \
  --argjson timeout "${TASK_TIMEOUT}" \
  --argjson eval_interval "${EVAL_INTERVAL}" \
  --argjson submission_cooldown "${SUBMISSION_COOLDOWN}" \
  --argjson qualified_max_peak_gib "${QUALIFIED_MAX_PEAK_GIB}" \
  --argjson headroom_mib "${HEADROOM_MIB}" \
  --argjson admission_min_free_mib "${GPU_ADMISSION_MIN_FREE_MIB}" \
  --argjson interactive_case_timeout "${POLICY_INTERACTIVE_CASE_TIMEOUT}" \
  --argjson interactive_judge_timeout "${POLICY_INTERACTIVE_JUDGE_TIMEOUT}" \
  --argjson final_case_timeout "${POLICY_FINAL_CASE_TIMEOUT}" \
  --argjson final_row_timeout "${POLICY_FINAL_ROW_TIMEOUT}" \
  --argjson final_inner_timeout "${POLICY_FINAL_INNER_TIMEOUT}" \
  --argjson final_outer_timeout "${POLICY_FINAL_OUTER_TIMEOUT}" \
  --arg scheduling_mode "${scheduling_mode}" \
  --arg execution_profile_id "${execution_profile_id}" \
  --arg gpu_lock_scheme "${gpu_lock_scheme}" \
  --arg interactive_evaluation "${interactive_evaluation_lifecycle}" \
  --argjson private_cohort_contract "${private_cohort_contract}" \
  --arg final_evaluation "${final_evaluation_lifecycle}" \
  '{
    schema_version: 3,
    scheduling: $scheduling_mode,
    execution_profile_id: $execution_profile_id,
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
    auto_eval_enabled: false,
    max_agent_submissions: null,
    manual_submission_cooldown_seconds: $submission_cooldown,
    private_cohort: ($private_cohort_contract + {
      interactive_case_timeout_seconds: $interactive_case_timeout,
      interactive_judge_timeout_seconds: $interactive_judge_timeout,
      secret_sha256: $cohort_secret_sha256
    }),
    lifecycle: {
      interactive_evaluation: $interactive_evaluation,
      final_evaluation: $final_evaluation,
      final_work_lock: "held_until_cleanup",
      judge_checkpoint_dir: "judge-checkpoints",
      judge_checkpoint_mount: "/logs/verifier/checkpoints",
      checkpoint_ownership: "root:root-0700"
    },
    final_timeouts: {
      case_seconds: $final_case_timeout,
      row_seconds: $final_row_timeout,
      inner_seconds: $final_inner_timeout,
      outer_seconds: $final_outer_timeout
    },
    model_revision: $model_revision,
    judge_server_sha256: $judge_server_sha256,
    policies: {
      memory: {path: "memory-policy.json", sha256: $memory_policy_sha256},
      workload: {path: "workload-policy.json", sha256: $workload_policy_sha256},
      quality_controls: {
        path: "quality-controls-manifest.json",
        sha256: $quality_controls_sha256
      }
    },
    images: {
      build_manifest: {
        path: "image-build-manifest.json",
        sha256: $image_manifest_sha256
      },
      source_digest: $image_source_digest,
      work: {name: $work_image_name, id: $work_image_id},
      judge: {name: $judge_image_name, id: $judge_image_id}
    },
    sforge_runtime: {
      sha256: $sforge_runtime_sha256,
      scope: "all_sforge_python",
      file_count: ($sforge_runtime_files | length),
      files: $sforge_runtime_files,
      file_set_sha256: $sforge_runtime_file_set_sha256
    },
    runtime_provenance: {
      path: "runtime-provenance.json",
      sha256: $runtime_provenance_sha256
    },
    judge_invariant_core: {
      verify_decoder_sha256: $judge_verify_decoder_sha256,
      candidate_worker_sha256: $judge_candidate_worker_sha256,
      cohort_sha256: $judge_cohort_sha256,
      quality_sha256: $judge_quality_sha256,
      source_admission_sha256: $judge_source_admission_sha256
    },
    network: {
      mode: $network_mode,
      direct_public_egress: false,
      proxy_public_egress: $proxy_public_egress,
      connect_allowlist: [
        "api.openai.com:443",
        "auth.openai.com:443",
        "chatgpt.com:443"
      ],
      runtime_entry_sha256: $runtime_entry_sha256,
      restricted_proxy_sha256: $restricted_proxy_sha256
    },
    gpu_admission: {
      qualified_max_peak_gib: $qualified_max_peak_gib,
      headroom_mib: $headroom_mib,
      minimum_free_mib: $admission_min_free_mib,
      locks: $gpu_lock_scheme,
      process_check: true,
      running_container_assignment_check: true
    },
    work_gpu: {
      index: $work_gpu_index,
      uuid: $work_gpu_uuid,
      name: $work_gpu_name,
      total_memory_mib: $work_gpu_total_mib,
      free_memory_mib_at_admission: $work_gpu_free_mib,
      compute_capability: $work_gpu_compute_capability,
      driver_version: $gpu_driver_version,
      observed_at: $gpu_observed_at,
      container_cuda_visible_devices: "0"
    },
    judge_gpu: {
      index: $judge_gpu_index,
      uuid: $judge_gpu_uuid,
      name: $judge_gpu_name,
      total_memory_mib: $judge_gpu_total_mib,
      free_memory_mib_at_admission: $judge_gpu_free_mib,
      compute_capability: $judge_gpu_compute_capability,
      driver_version: $gpu_driver_version,
      observed_at: $gpu_observed_at,
      container_cuda_visible_devices: "0"
    }
  }' > "${RUN_LOG_ROOT}/experiment-manifest.json"
experiment_manifest="${RUN_LOG_ROOT}/experiment-manifest.json"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  jq \
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
    --argjson final_outer_timeout "${POLICY_FINAL_OUTER_TIMEOUT}" \
    --argjson coordinator_timeout "${TASK_EVAL_TIMEOUT}" \
    '. + {
      resource_handoff: {
        mode: $handoff_mode,
        allocation_key: $allocation_key,
        trusted_lease_id: $trusted_lease_id,
        evaluation_contract_sha256: $evaluation_contract_sha256,
        execution_profile_id: $execution_profile_id,
        final_judge: {
          path: "coordinator_standalone_final_handoff",
          coordinator_ceiling_seconds: $coordinator_timeout,
          verifier_outer_deadline_seconds: $final_outer_timeout,
          archive_context: "SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256 injected by the coordinator from the retired Work lease"
        }
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
experiment_manifest_sha256="$(sha256sum "${experiment_manifest}" | awk '{print $1}')"
chmod 444 "${experiment_manifest}"

echo "Starting ${RUN_ID}"
echo "  agent: ${AGENT_NAME}; model: ${MODEL} (${REASONING_EFFORT})"
echo "  work budget: ${TASK_TIMEOUT}s; unlimited manual evaluations (${SUBMISSION_COOLDOWN}s cooldown)"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  echo "  scheduling: single whole-GPU lease; Judge shares ${work_gpu_uuid} serially with Work; standalone final through the live coordinator; the UUID lock stays held"
else
  echo "  final Judge: independent on Judge GPU after Work stops; both UUID locks stay held"
fi
echo "  Codex web search: ${codex_web_search}"
echo "  Work egress: internal network via sidecar (public internet: ${TASK_INTERNET})"
echo "  Work GPU: ${WORK_GPU} (${work_gpu_uuid})"
echo "  Judge GPU: ${JUDGE_GPU} (${judge_gpu_uuid})"
echo "  GPU admission floor: ${GPU_ADMISSION_MIN_FREE_MIB} MiB"
echo "  logs: ${RUN_LOG_ROOT}"

# --enable-internet bypasses the unavailable sudo-iptables implementation;
# the internal network and the sidecar above are the actual fail-closed
# enforcement boundary.  The entry takes the task's internet flag from
# MLSYSFORGE_WORK_INTERNET (exported above): true keeps Codex web search live
# and the prompt free of network text, false restores the offline Codex flags
# and prompt.
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
run_logged_command "${RUN_LOG_ROOT}/sforge-run.log" \
  "${SFORGE_PYTHON}" "${agent_entry_shim[@]}" \
  "${H100_ENTRY}" \
  run \
  --backend docker \
  --task "${TASK_ID}" \
  --agent "${AGENT_NAME}" \
  --model "${MODEL}" \
  --timeout "${TASK_TIMEOUT}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --disable-auto-eval \
  --submission-cooldown "${SUBMISSION_COOLDOWN}" \
  --judge-url "http://${restricted_proxy_ip}:3130" \
  --enable-internet \
  --work-cpu-limit 8 \
  --work-mem-limit 48g \
  --judge-cpu-limit 8 \
  --judge-mem-limit 48g \
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

# Freeze/final barrier: the Work container and its internal network are gone
# and the UUID lock descriptor(s) intentionally remain held.  On the legacy
# path the interactive Judge service stops here too; under the single GPU
# handoff the Work container has been frozen and destroyed by the harness's
# never-resume final capture and the coordinator stays alive, because it
# performs the final Judge on the same allocation under the still-held lease.
if [[ "${SINGLE_GPU_HANDOFF}" != "1" ]]; then
  stop_judge_server
fi
remove_work_network
unset \
  SFORGE_ISOLATED_DOCKER_NETWORK \
  SFORGE_HTTP_PROXY \
  SFORGE_HTTPS_PROXY \
  SFORGE_NO_PROXY
if docker container inspect "${WORK_CONTAINER_NAME}" >/dev/null 2>&1; then
  die "Work container still exists at the final-evaluation barrier"
fi
assert_gpu_released "${work_gpu_uuid}" "${WORK_GPU}" "Work"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  assert_single_gpu_lock_held
  kill -0 "${judge_pid}" 2>/dev/null \
    || die "handoff coordinator exited before the standalone final evaluation"
else
  assert_gpu_released "${judge_gpu_uuid}" "${JUDGE_GPU}" "interactive Judge"
  assert_dual_gpu_locks_held
  assert_distinct_gpu_pair \
    "${WORK_GPU}" "${JUDGE_GPU}" "${GPU_ADMISSION_MIN_FREE_MIB}"
fi
assert_built_images_current
[[ "$(image_source_digest)" == "${recorded_image_source_digest}" ]] \
  || die "task source digest changed between Work and final evaluation"
[[ "$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")" \
    == "${recorded_work_image_id}" ]] \
  || die "Work image changed between Work and final evaluation"
[[ "$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")" \
    == "${recorded_judge_image_id}" ]] \
  || die "Judge image changed between Work and final evaluation"
final_model_validation="${TASK_RUN_ROOT}/final-model-validation.json"
python3 "${MIGRATION_ROOT}/scripts/validate_model_artifact.py" \
  --repo "${LLADA_MODEL_REPO}" --full \
  --output "${final_model_validation}" >/dev/null
[[ "$(sha256sum "${final_model_validation}" | awk '{print $1}')" \
    == "${model_validation_sha256}" ]] \
  || die "model validation identity changed between Work and final evaluation"
mapfile -d '' -t CURRENT_SFORGE_RUNTIME_FILES < <(
  cd "${SFORGE_ROOT}"
  find sforge -type f -name '*.py' -print0 | LC_ALL=C sort -z
)
[[ "${CURRENT_SFORGE_RUNTIME_FILES[*]}" == "${SFORGE_RUNTIME_FILES[*]}" ]] \
  || die "SForge Python source set changed between Work and final evaluation"
current_sforge_runtime_sha256="$(
  (
    cd "${SFORGE_ROOT}"
    sha256sum "${CURRENT_SFORGE_RUNTIME_FILES[@]}"
  ) | sha256sum | awk '{print $1}'
)"
current_sforge_runtime_file_set_sha256="$(
  printf '%s\n' "${CURRENT_SFORGE_RUNTIME_FILES[@]}" \
    | sha256sum | awk '{print $1}'
)"
[[ "${current_sforge_runtime_sha256}" == "${sforge_runtime_sha256}" \
    && "${current_sforge_runtime_file_set_sha256}" \
      == "${sforge_runtime_file_set_sha256}" ]] \
  || die "SForge runtime changed between Work and final evaluation"
mapfile -t current_judge_core_hashes < <(
  docker run --rm --network none --entrypoint sha256sum \
    "${recorded_judge_image_id}" \
    /opt/decoder_judge/verifier/verify_decoder.py \
    /opt/decoder_judge/verifier/candidate_worker.py \
    /opt/decoder_judge/verifier/cohort.py \
    /opt/decoder_judge/verifier/quality.py \
    /opt/decoder_judge/verifier/source_admission.py \
    | awk '{print $1}'
)
[[ "${current_judge_core_hashes[*]}" == "${judge_core_hashes_start[*]}" ]] \
  || die "Judge invariant core changed between Work and final evaluation"
[[ -f "${runtime_provenance_snapshot}" \
    && ! -L "${runtime_provenance_snapshot}" \
    && "$(stat -c '%a' "${runtime_provenance_snapshot}")" == "444" \
    && "$(sha256sum "${runtime_provenance_snapshot}" | awk '{print $1}')" \
      == "${runtime_provenance_sha256}" ]] \
  || die "runtime provenance snapshot changed before final evaluation"
[[ "$(sha256sum "${H100_ENTRY}" | awk '{print $1}')" == "${runtime_entry_sha256}" ]] \
  || die "H100 runtime entry changed between Work and final evaluation"
[[ "$(sha256sum "${RESTRICTED_PROXY_SOURCE}" | awk '{print $1}')" == "${restricted_proxy_sha256}" ]] \
  || die "restricted proxy changed between Work and final evaluation"
[[ "$(sha256sum "${JUDGE_SERVER_SOURCE}" | awk '{print $1}')" == "${judge_server_sha256}" ]] \
  || die "Judge server changed between Work and final evaluation"
[[ "$(sha256sum "${MEMORY_POLICY_PATH}" | awk '{print $1}')" == "${MEMORY_POLICY_SHA256}" ]] \
  || die "memory policy changed between Work and final evaluation"
[[ "$(sha256sum "${WORKLOAD_POLICY_PATH}" | awk '{print $1}')" == "${WORKLOAD_POLICY_SHA256}" ]] \
  || die "workload policy changed between Work and final evaluation"
[[ "$(sha256sum "${QUALITY_CONTROLS_PATH}" | awk '{print $1}')" \
    == "${QUALITY_CONTROLS_SHA256}" ]] \
  || die "quality-control manifest changed between Work and final evaluation"
[[ "$(sha256sum "${run_quality_controls}" | awk '{print $1}')" \
    == "${QUALITY_CONTROLS_SHA256}" ]] \
  || die "run-local quality-control manifest changed before final evaluation"

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
  --evaluation-mode "${scheduling_mode}"
final_archive_sha256="$(
  jq -er \
    '.final_archive.sha256
     | select(type == "string" and test("^[0-9a-f]{64}$"))' \
    "${formal_precheck}"
)"
echo "Formal precheck: PASS (archive sha256=${final_archive_sha256})"

final_judge_container_name="llada_decoder.judge.${TASK_ID}.${final_eval_id}"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  [[ -d "${final_run_root}" && ! -L "${final_run_root}" ]] \
    || die "final eval directory prepared before Work is missing: ${final_eval_id}"
else
  [[ ! -e "${final_run_root}" && ! -L "${final_run_root}" ]] \
    || die "final eval directory already exists: ${final_eval_id}"
  mkdir -- "${final_run_root}"
fi
final_runtime_provenance_snapshot="${final_run_root}/runtime-provenance.json"
cp -- "${runtime_provenance_snapshot}" "${final_runtime_provenance_snapshot}"
chmod 444 "${final_runtime_provenance_snapshot}"
[[ "$(sha256sum "${final_runtime_provenance_snapshot}" | awk '{print $1}')" \
    == "${runtime_provenance_sha256}" ]] \
  || die "final runtime provenance snapshot digest mismatch"
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  # The coordinator that ran Work still holds the retired lease.  It re-opens
  # its own immutable copy of the frozen archive, checks it against the lease,
  # attests the allocation idle, creates the final Judge through the task
  # adapter (final run scope, its own root-owned checkpoint directory, network
  # none), injects the authenticated archive digest into the private context,
  # strictly destroys the Judge, attests idle again and closes the lease.
  [[ -d "${final_run_root}/judge-checkpoints" \
      && "$(stat -c '%u' "${final_run_root}/judge-checkpoints")" == "0" \
      && "$(stat -c '%a' "${final_run_root}/judge-checkpoints")" == "700" ]] \
    || die "final Judge checkpoint directory is not the prepared root-owned directory"
  wait_for_http "http://127.0.0.1:${JUDGE_PORT}/openapi.json" 30
  set +e
  run_logged_command "${TASK_RUN_ROOT}/final-${final_minutes}m-eval.log" \
  "${SFORGE_PYTHON}" \
  "${H100_ENTRY}" \
  eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${final_archive}" \
  --run-id "${final_eval_id}" \
  --timeout "${TASK_EVAL_TIMEOUT}" \
  --judge-cpu-limit 8 \
  --judge-mem-limit 48g \
  --standalone-final-handoff \
  --handoff-source-run-id "${RUN_ID}" \
  --judge-url "http://127.0.0.1:${JUDGE_PORT}" \
  --json
  final_eval_rc="$?"
  set -e
  stop_judge_server
  (( final_eval_rc == 0 )) \
    || die "standalone final evaluation failed with exit ${final_eval_rc}"
  assert_gpu_released "${judge_gpu_uuid}" "${JUDGE_GPU}" "final Judge"
  assert_single_gpu_lock_held
else
  export LLADA_RUNTIME_RUN_ID="${final_eval_id}"
  prepare_root_owned_checkpoint_dir \
    "${final_run_root}" "${final_eval_id}" "${recorded_judge_image_id}"
  export SFORGE_JUDGE_EXTRA_ENV="${judge_extra_env},SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256=${final_archive_sha256},SFORGE_PRIVATE_CONTEXT_WORK_IMAGE_ID=${recorded_work_image_id},SFORGE_PRIVATE_CONTEXT_JUDGE_IMAGE_ID=${recorded_judge_image_id},SFORGE_PRIVATE_CONTEXT_MODEL_VALIDATION_SHA256=${model_validation_sha256},SFORGE_PRIVATE_CONTEXT_RUNTIME_ENTRY_SHA256=${runtime_entry_sha256},SFORGE_PRIVATE_CONTEXT_SOURCE_DIGEST=${recorded_image_source_digest}"
  set +e
  run_logged_command "${TASK_RUN_ROOT}/final-${final_minutes}m-eval.log" \
  "${SFORGE_PYTHON}" \
  "${H100_ENTRY}" \
  eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${final_archive}" \
  --run-id "${final_eval_id}" \
  --timeout "${FINAL_EVAL_TIMEOUT}" \
  --judge-cpu-limit 8 \
  --judge-mem-limit 48g \
  --json
  final_eval_rc="$?"
  set -e
  (( final_eval_rc == 0 )) \
    || die "independent final evaluation failed with exit ${final_eval_rc}"
  assert_gpu_released "${judge_gpu_uuid}" "${JUDGE_GPU}" "final Judge"
  assert_gpu_released "${work_gpu_uuid}" "${WORK_GPU}" "reserved Work"
  assert_dual_gpu_locks_held
fi
[[ "$(image_source_digest)" == "${recorded_image_source_digest}" ]] \
  || die "task source digest changed during final evaluation"
[[ "$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")" \
    == "${recorded_work_image_id}" ]] \
  || die "Work image changed during final evaluation"
[[ "$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")" \
    == "${recorded_judge_image_id}" ]] \
  || die "Judge image changed during final evaluation"
[[ "$(sha256sum "${MEMORY_POLICY_PATH}" | awk '{print $1}')" == "${MEMORY_POLICY_SHA256}" ]] \
  || die "memory policy changed during final evaluation"
[[ "$(sha256sum "${WORKLOAD_POLICY_PATH}" | awk '{print $1}')" == "${WORKLOAD_POLICY_SHA256}" ]] \
  || die "workload policy changed during final evaluation"
[[ "$(sha256sum "${QUALITY_CONTROLS_PATH}" | awk '{print $1}')" \
    == "${QUALITY_CONTROLS_SHA256}" ]] \
  || die "quality-control manifest changed during final evaluation"
[[ "$(sha256sum "${run_quality_controls}" | awk '{print $1}')" \
    == "${QUALITY_CONTROLS_SHA256}" ]] \
  || die "run-local quality-control manifest changed during final evaluation"
[[ "$(sha256sum "${final_archive}" | awk '{print $1}')" == "${final_archive_sha256}" ]] \
  || die "frozen final archive changed during final evaluation"
post_final_model_validation="${TASK_RUN_ROOT}/post-final-model-validation.json"
python3 "${MIGRATION_ROOT}/scripts/validate_model_artifact.py" \
  --repo "${LLADA_MODEL_REPO}" --full \
  --output "${post_final_model_validation}" >/dev/null
[[ "$(sha256sum "${post_final_model_validation}" | awk '{print $1}')" \
    == "${model_validation_sha256}" ]] \
  || die "model artifact changed during final evaluation"
mapfile -d '' -t POST_FINAL_SFORGE_RUNTIME_FILES < <(
  cd "${SFORGE_ROOT}"
  find sforge -type f -name '*.py' -print0 | LC_ALL=C sort -z
)
[[ "${POST_FINAL_SFORGE_RUNTIME_FILES[*]}" == "${SFORGE_RUNTIME_FILES[*]}" ]] \
  || die "SForge Python source set changed during final evaluation"
post_final_sforge_runtime_sha256="$(
  (
    cd "${SFORGE_ROOT}"
    sha256sum "${POST_FINAL_SFORGE_RUNTIME_FILES[@]}"
  ) | sha256sum | awk '{print $1}'
)"
post_final_sforge_runtime_file_set_sha256="$(
  printf '%s\n' "${POST_FINAL_SFORGE_RUNTIME_FILES[@]}" \
    | sha256sum | awk '{print $1}'
)"
[[ "${post_final_sforge_runtime_sha256}" == "${sforge_runtime_sha256}" \
    && "${post_final_sforge_runtime_file_set_sha256}" \
      == "${sforge_runtime_file_set_sha256}" ]] \
  || die "SForge runtime changed during final evaluation"
mapfile -t post_final_judge_core_hashes < <(
  docker run --rm --network none --entrypoint sha256sum \
    "${recorded_judge_image_id}" \
    /opt/decoder_judge/verifier/verify_decoder.py \
    /opt/decoder_judge/verifier/candidate_worker.py \
    /opt/decoder_judge/verifier/cohort.py \
    /opt/decoder_judge/verifier/quality.py \
    /opt/decoder_judge/verifier/source_admission.py \
    | awk '{print $1}'
)
[[ "${post_final_judge_core_hashes[*]}" == "${judge_core_hashes_start[*]}" ]] \
  || die "Judge invariant core changed during final evaluation"
[[ -f "${final_runtime_provenance_snapshot}" \
    && ! -L "${final_runtime_provenance_snapshot}" \
    && "$(stat -c '%a' "${final_runtime_provenance_snapshot}")" == "444" ]] \
  || die "final runtime provenance snapshot is unsafe"
cmp -s -- "${runtime_provenance_snapshot}" \
  "${final_runtime_provenance_snapshot}" \
  || die "final runtime provenance snapshot changed during evaluation"
[[ "$(sha256sum "${H100_ENTRY}" | awk '{print $1}')" == "${runtime_entry_sha256}" ]] \
  || die "H100 runtime entry changed during final evaluation"
[[ "$(sha256sum "${RESTRICTED_PROXY_SOURCE}" | awk '{print $1}')" == "${restricted_proxy_sha256}" ]] \
  || die "restricted proxy source changed during final evaluation"
[[ "$(sha256sum "${JUDGE_SERVER_SOURCE}" | awk '{print $1}')" == "${judge_server_sha256}" ]] \
  || die "Judge server source changed during final evaluation"

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
  local reason
  reason="$(sed -n 's/^error: //p' "${stderr_file}" | head -n 1 || true)"
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
    --arg reason "${reason}" \
    --arg scheduling_mode "${scheduling_mode}" \
    --arg final_archive "${final_archive}" \
    --arg final_archive_sha256 "${final_archive_sha256}" \
    --arg experiment_manifest "${experiment_manifest}" \
    --arg experiment_manifest_sha256 "${experiment_manifest_sha256}" \
    --arg formal_precheck "${formal_precheck}" \
    --arg formal_postcheck "${formal_postcheck}" \
    --slurpfile experiment "${experiment_manifest}" \
    --argjson sforge_run_exit "${run_rc}" \
    --argjson final_eval_exit "${final_eval_rc}" \
    --argjson formal_postcheck_exit "${postcheck_rc}" \
    '{
      schema_version: 3,
      status: "failed",
      failure_stage: "formal_postcheck",
      reason: $reason,
      finished_at: $finished_at,
      run_id: $run_id,
      task_id: $task_id,
      scheduling: $scheduling_mode,
      lifecycle: $experiment[0].lifecycle,
      work_gpu: $experiment[0].work_gpu,
      judge_gpu: $experiment[0].judge_gpu,
      sforge_run_exit: $sforge_run_exit,
      final_eval_exit: $final_eval_exit,
      formal_postcheck_exit: $formal_postcheck_exit,
      final_archive: {path: $final_archive, sha256: $final_archive_sha256},
      experiment_manifest: {
        path: $experiment_manifest,
        sha256: $experiment_manifest_sha256
      },
      images: $experiment[0].images,
      validation: {precheck: $formal_precheck, postcheck: $formal_postcheck}
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
  --evaluation-mode "${scheduling_mode}" \
  --verify-suite "full" \
  $(if [[ "${PILOT_MODE}" == "1" ]]; then printf '%s' '--pilot'; fi) \
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
  --arg final_checkpoint_dir "${final_run_root}/judge-checkpoints" \
  --arg private_judge_artifact "${final_private_judge_artifact}" \
  --arg private_judge_sha256 "${private_judge_sha256}" \
  --arg private_judge_member_sha256 "${private_judge_member_sha256}" \
  --slurpfile precheck "${formal_precheck}" \
  --slurpfile postcheck "${formal_postcheck}" \
  '{
    snapshot_elapsed_seconds: $precheck[0].snapshot_elapsed_seconds,
    snapshot_captured_at: $precheck[0].snapshot_captured_at,
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

python3 "${MIGRATION_ROOT}/scripts/collect_curve.py" \
  "${TASK_RUN_ROOT}" \
  --final-eval "${final_curve_report}" \
  --interval-minutes "${interval_minutes}" \
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
  --arg final_archive "${final_archive}" \
  --arg final_archive_sha256 "${final_archive_sha256}" \
  --arg final_report "${final_report}" \
  --arg final_report_sha256 "${final_report_sha256}" \
  --arg final_checkpoint_dir "${final_run_root}/judge-checkpoints" \
  --arg evaluated_archive "${final_evaluated_archive}" \
  --arg evaluated_archive_sha256 "${evaluated_archive_sha256}" \
  --arg private_judge_artifact "${final_private_judge_artifact}" \
  --arg private_judge_sha256 "${private_judge_sha256}" \
  --arg private_judge_member_sha256 "${private_judge_member_sha256}" \
  --arg curve_json "${curve_json}" \
  --arg curve_markdown "${curve_markdown}" \
  --arg formal_precheck "${formal_precheck}" \
  --arg formal_postcheck "${formal_postcheck}" \
  --arg post_final_model_validation "${post_final_model_validation}" \
  --arg post_final_model_validation_sha256 "${model_validation_sha256}" \
  --arg experiment_manifest "${experiment_manifest}" \
  --arg experiment_manifest_sha256 "${experiment_manifest_sha256}" \
  --slurpfile experiment "${experiment_manifest}" \
  --slurpfile precheck "${formal_precheck}" \
  --argjson sforge_run_exit "${run_rc}" \
  --argjson final_eval_exit "${final_eval_rc}" \
  --arg scheduling_mode "${scheduling_mode}" \
  '{
    schema_version: 3,
    status: "passed",
    finished_at: $finished_at,
    run_id: $run_id,
    task_id: $task_id,
    scheduling: $scheduling_mode,
    lifecycle: $experiment[0].lifecycle,
    work_gpu: $experiment[0].work_gpu,
    judge_gpu: $experiment[0].judge_gpu,
    manual_evaluations_during_work: $precheck[0].work_evaluations.agent,
    sforge_run_exit: $sforge_run_exit,
    final_eval_exit: $final_eval_exit,
    final_archive: {path: $final_archive, sha256: $final_archive_sha256},
    final_report: {path: $final_report, sha256: $final_report_sha256},
    final_checkpoint_dir: $final_checkpoint_dir,
    evaluated_archive: {
      path: $evaluated_archive,
      sha256: $evaluated_archive_sha256
    },
    private_judge_artifact: {
      path: $private_judge_artifact,
      sha256: $private_judge_sha256,
      member_sha256: $private_judge_member_sha256
    },
    experiment_manifest: {
      path: $experiment_manifest,
      sha256: $experiment_manifest_sha256
    },
    images: $experiment[0].images,
    policies: $experiment[0].policies,
    sforge_runtime: $experiment[0].sforge_runtime,
    curve: {json: $curve_json, markdown: $curve_markdown},
    validation: {
      precheck: $formal_precheck,
      postcheck: $formal_postcheck,
      post_final_model: {
        path: $post_final_model_validation,
        sha256: $post_final_model_validation_sha256
      }
    }
  }'
if [[ "${SINGLE_GPU_HANDOFF}" == "1" ]]; then
  write_command_json_atomically \
    "${completion}" \
    jq \
    --arg evaluation_contract_sha256 "${evaluation_contract_sha256}" \
    --arg execution_profile_id "${execution_profile_id}" \
    --arg measurement_profile_sha256 "${measurement_profile_sha256}" \
    --arg raw_score_comparison_identity_sha256 "${raw_score_comparison_identity_sha256}" \
    --arg qualification_status "${qualification_status}" \
    --slurpfile experiment "${experiment_manifest}" \
    '. + {
      resource_handoff: $experiment[0].resource_handoff,
      deployment_attestation: $experiment[0].deployment_attestation,
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
    }' "${completion}"
fi

echo "Completed ${RUN_ID}"
echo "  curve: ${TASK_RUN_ROOT}/curve/curve.md"
echo "  private Judge rows: ${final_private_judge_artifact}"
echo "  SForge exit: ${run_rc}; final eval exit: ${final_eval_rc}"

(( run_rc == 0 && final_eval_rc == 0 ))
