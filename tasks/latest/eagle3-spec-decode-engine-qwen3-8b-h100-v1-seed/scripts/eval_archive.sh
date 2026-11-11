#!/usr/bin/env bash
# Run-scoped manual evaluation of one immutable candidate archive.
# Lineage: LLaDA v2 eval_archive.sh. The evaluator holds the Judge UUID lease,
# pins the archive/image/runtime identities, writes a root-only provenance record,
# and validates the persisted report before returning.
set -Eeuo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'USAGE'
Usage: eval_archive.sh ARCHIVE [RUN_ID] [--timeout S] [--qualification]

The archive is copied and hashed before evaluation. The original and run-local
copies are hashed again afterwards; any mutation invalidates the attestation.
USAGE
}

active_pid=""

stop_active_process() {
  # Retain a killable PID for interrupted manual evaluations. SForge then
  # receives a bounded grace period to remove its Judge container.
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

cleanup_eval() {
  # Cleanup is idempotent and cannot replace the original failure status.
  stop_active_process
  if [[ -n "${judge_uuid:-}" && -n "${run_id:-}" ]]; then
    remove_task_gpu_containers "judge" "${judge_uuid}" "${run_id}" \
      >/dev/null 2>&1 || true
  fi
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
  cleanup_eval
  exit "${exit_code}"
}

run_logged_command() {
  local log_file="$1"
  shift
  local command_rc=0
  "$@" >"${log_file}" 2>&1 &
  active_pid="$!"
  wait "${active_pid}" || command_rc="$?"
  active_pid=""
  return "${command_rc}"
}

trap cleanup_eval EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM
trap 'handle_signal HUP' HUP

[[ $# -ge 1 ]] || {
  usage >&2
  exit 2
}

archive_argument="$1"
shift
run_id=""
timeout="${FINAL_EVAL_TIMEOUT:-5400}"
qualification=0
if [[ $# -gt 0 && "$1" != --* ]]; then
  run_id="$1"
  shift
fi
while (( $# )); do
  case "$1" in
    --timeout)
      [[ $# -ge 2 ]] || die "--timeout requires a value"
      timeout="$2"
      shift 2
      ;;
    --qualification)
      qualification=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "${timeout}" =~ ^[1-9][0-9]*$ ]] || die "timeout must be a positive integer"

ensure_docker_group "${ORIGINAL_ARGS[@]}"
require_sforge
for required_command in docker flock jq nvidia-smi python3 realpath seq sha256sum; do
  require_command "${required_command}"
done

archive="$(realpath -- "${archive_argument}")"
[[ -f "${archive}" && ! -L "${archive}" && -s "${archive}" ]] \
  || die "archive must be a nonempty regular file: ${archive}"
run_id="${run_id:-manual-$(date -u +%Y%m%d-%H%M%S)}"
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$ ]] \
  || die "unsafe run ID: ${run_id}"

assert_built_images_current
build_manifest="${BUILD_DIR}/image-manifest.json"
recorded_source_digest="$(jq -er '.source_digest' "${build_manifest}")"
work_image_id="$(jq -er '.work.id' "${build_manifest}")"
judge_image_id="$(jq -er '.judge.id' "${build_manifest}")"

JUDGE_GPU="$(canonical_gpu_index "${JUDGE_GPU:?set JUDGE_GPU}")"
judge_uuid="$(gpu_uuid "${JUDGE_GPU}")"
judge_lock="$(gpu_lock_path "${judge_uuid}")"
if [[ ",${EAGLE3_LOCKED_GPU_UUIDS:-}," == *",${judge_uuid},"* ]] \
    && [[ "$(readlink "/proc/$$/fd/8" 2>/dev/null)" == "${judge_lock}" \
          || "$(readlink "/proc/$$/fd/9" 2>/dev/null)" == "${judge_lock}" ]]; then
  # Called by run_fixed_budget.sh for the final evaluation of a two-GPU run: that
  # script holds this lock for the whole run and its lock descriptors are inherited
  # here, so the lease is already held. Taking it again would always fail.
  :
else
  exec 8>"${judge_lock}"
  flock -n 8 || die "Judge GPU lock is already held: ${judge_uuid}"
fi
assert_gpu_available "${JUDGE_GPU}" "Judge"
assert_container_gpu "${JUDGE_IMAGE}" "${judge_uuid}" "Judge"

"${PYTHON_BIN}" "${SCRIPT_DIR}/preflight.py" \
  --target-repo "${EAGLE3_TARGET_REPO}" \
  --head-dir "${EAGLE3_HEAD_DIR}" \
  --judge-gpu "${JUDGE_GPU}" \
  --image-manifest "${build_manifest}" >/dev/null

run_root="${LOGS_DIR}/runs/${run_id}"
task_run_root="${run_root}/${TASK_ID}"
[[ ! -e "${run_root}" && ! -L "${run_root}" ]] \
  || die "evaluation run directory already exists: ${run_root}"
mkdir -p "${task_run_root}"
chmod 700 "${run_root}" "${task_run_root}"

run_archive="${task_run_root}/submission.tar.gz"
cp -- "${archive}" "${run_archive}"
chmod 400 "${run_archive}"
archive_sha256_before="$(sha256sum "${archive}" | awk '{print $1}')"
run_archive_sha256_before="$(sha256sum "${run_archive}" | awk '{print $1}')"
[[ "${archive_sha256_before}" == "${run_archive_sha256_before}" ]] \
  || die "run-local archive copy digest differs"

runtime_entry="${SCRIPT_DIR}/sforge_h100_entry.py"
runtime_entry_sha256="$(sha256sum "${runtime_entry}" | awk '{print $1}')"
policy_sha256="$(sha256sum "${EAGLE3_PRIVATE_DATA}/policy.json" | awk '{print $1}')"
protected_hashes_sha256="$(sha256sum "${MIGRATION_ROOT}/images/judge/protected_hashes.txt" | awk '{print $1}')"

# This root-only record binds the operator request to its exact archive, images,
# runtime entry, policy, and protected manifest. SForge separately creates and
# mounts the verifier context; per-version selection belongs to replay-versions.
context="${task_run_root}/evaluation-context.json"
jq_arguments=(
  -n
  --arg round_id "${run_id}"
  --arg archive_sha256 "${archive_sha256_before}"
  --arg work_image_id "${work_image_id}"
  --arg judge_image_id "${judge_image_id}"
  --arg source_digest "${recorded_source_digest}"
  --arg runtime_entry_sha256 "${runtime_entry_sha256}"
  --arg policy_sha256 "${policy_sha256}"
  --arg protected_hashes_sha256 "${protected_hashes_sha256}"
)
context_program='{
  schema_version: 1,
  round_id: $round_id,
  archive_sha256: $archive_sha256,
  work_image_id: $work_image_id,
  judge_image_id: $judge_image_id,
  source_digest: $source_digest,
  runtime_entry_sha256: $runtime_entry_sha256,
  policy_sha256: $policy_sha256,
  protected_hashes_sha256: $protected_hashes_sha256
}'
jq "${jq_arguments[@]}" "${context_program}" >"${context}"
chmod 600 "${context}"

export EAGLE3_JUDGE_GPU_UUID="${judge_uuid}"
export EAGLE3_RUNTIME_RUN_ID="${run_id}"
export SFORGE_EXPECTED_WORK_IMAGE_ID="${work_image_id}"
export SFORGE_EXPECTED_JUDGE_IMAGE_ID="${judge_image_id}"
qualification_env=""
if (( qualification )); then
  qualification_env=",EAGLE3_QUALIFICATION_MODE=1"
fi
export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${judge_uuid},CUDA_VISIBLE_DEVICES=0,EAGLE3_GPU_ROLE=judge,EAGLE3_INTERACTIVE_JUDGE_TIMEOUT=${EAGLE3_INTERACTIVE_JUDGE_TIMEOUT:-3600},EAGLE3_FINAL_JUDGE_TIMEOUT=${timeout}${qualification_env}"

set +e
run_logged_command "${task_run_root}/eval.log" \
  "${PYTHON_BIN}" "${runtime_entry}" \
  --output-dir "${run_root}" \
  --tasks-dir "${TASKS_DIR}" \
  eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${run_archive}" \
  --run-id "${run_id}" \
  --timeout "${timeout}" \
  --judge-cpu-limit 8 \
  --judge-mem-limit 64g \
  --json
eval_rc="$?"
set -e

assert_gpu_released "${judge_uuid}" "${JUDGE_GPU}" "manual Judge"
archive_sha256_after="$(sha256sum "${archive}" | awk '{print $1}')"
run_archive_sha256_after="$(sha256sum "${run_archive}" | awk '{print $1}')"
[[ "${archive_sha256_before}" == "${archive_sha256_after}" ]] \
  || die "source archive changed during evaluation"
[[ "${run_archive_sha256_before}" == "${run_archive_sha256_after}" ]] \
  || die "run-local archive changed during evaluation"
[[ "$(image_source_digest)" == "${recorded_source_digest}" ]] \
  || die "task source changed during evaluation"
assert_built_images_current

# SForge may write either directly under the supplied output directory or one
# additional runs/<id> level. Resolve both without accepting arbitrary paths.
candidate_task_roots=(
  "${run_root}/${TASK_ID}"
  "${run_root}/runs/${run_id}/${TASK_ID}"
)
evaluated_task_root=""
for candidate_root in "${candidate_task_roots[@]}"; do
  if [[ -s "${candidate_root}/submissions/manual-1/report.json" ]]; then
    evaluated_task_root="${candidate_root}"
    break
  fi
done
[[ -n "${evaluated_task_root}" ]] \
  || die "manual evaluation did not persist a report"

attestation="${task_run_root}/evaluation-attestation.json"
validation_phase="eval"
if (( qualification )); then
  validation_phase="qualify"
fi
"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_formal_artifacts.py" "${validation_phase}" \
  --task-root "${task_run_root}" \
  --final-task-root "${evaluated_task_root}" \
  --final-eval-id "${run_id}" \
  --task-id "${TASK_ID}" \
  --expected-archive-sha256 "${archive_sha256_before}" \
  >"${attestation}.tmp"
mv -- "${attestation}.tmp" "${attestation}"
chmod 444 "${attestation}"

(( eval_rc == 0 )) || die "manual evaluation failed with exit ${eval_rc}"
printf 'Evaluation %s completed; archive sha256=%s\n' \
  "${run_id}" "${archive_sha256_before}"
