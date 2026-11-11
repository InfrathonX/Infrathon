#!/usr/bin/env bash
set -Eeuo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_sforge
require_command cmp
require_command docker
require_command flock
require_command jq
require_command mktemp
require_command nvidia-smi
require_command python3
require_command realpath
require_command sha256sum
require_command stat

archive="${1:-}"
[[ -f "${archive}" ]] || die "usage: $0 ARCHIVE.tar.gz [RUN_ID]"
[[ ! -L "${archive}" ]] || die "submission archive must not be a symlink"
archive_sha256_before="$(sha256sum -- "${archive}" | awk '{print $1}')"
archive_size_before="$(stat -c '%s' -- "${archive}")"
(( archive_size_before > 0 )) || die "submission archive is empty"
qualification_mode="${LLADA_QUALIFICATION:-0}"
calibration_mode="${LLADA_CALIBRATION:-0}"
quality_control_mode="${LLADA_QUALITY_CONTROL:-0}"
reference_replay="${LLADA_REFERENCE_REPLAY:-0}"
resume_mode="${LLADA_RESUME:-0}"
pilot_mode="${LLADA_PILOT:-0}"
[[ "${qualification_mode}" == "0" || "${qualification_mode}" == "1" ]] \
  || die "LLADA_QUALIFICATION must be 0 or 1"
[[ "${calibration_mode}" == "0" || "${calibration_mode}" == "1" ]] \
  || die "LLADA_CALIBRATION must be 0 or 1"
[[ "${quality_control_mode}" == "0" || "${quality_control_mode}" == "1" ]] \
  || die "LLADA_QUALITY_CONTROL must be 0 or 1"
[[ "${reference_replay}" == "0" || "${reference_replay}" == "1" ]] \
  || die "LLADA_REFERENCE_REPLAY must be 0 or 1"
[[ "${resume_mode}" == "0" || "${resume_mode}" == "1" ]] \
  || die "LLADA_RESUME must be 0 or 1"
[[ "${pilot_mode}" == "0" || "${pilot_mode}" == "1" ]] \
  || die "LLADA_PILOT must be 0 or 1"
(( 10#${qualification_mode} + 10#${calibration_mode} \
    + 10#${quality_control_mode} + 10#${reference_replay} \
    + 10#${pilot_mode} <= 1 )) \
  || die "pilot, qualification, calibration, quality-control, and reference-replay modes are mutually exclusive"
evaluation_mode="final"
[[ "${qualification_mode}" != "1" ]] || evaluation_mode="qualification"
[[ "${calibration_mode}" != "1" ]] || evaluation_mode="calibration"
[[ "${quality_control_mode}" != "1" ]] || evaluation_mode="quality_control"
[[ "${pilot_mode}" != "1" ]] || evaluation_mode="pilot"
memory_fixture_kind="${LLADA_MEMORY_FIXTURE_KIND:-}"
if [[ -n "${memory_fixture_kind}" ]]; then
  [[ "${qualification_mode}" == "1" ]] \
    || die "LLADA_MEMORY_FIXTURE_KIND is valid only with LLADA_QUALIFICATION=1"
  [[ "${memory_fixture_kind}" == "corrected_reference" \
      || "${memory_fixture_kind}" == "near_cap_stress" ]] \
    || die "LLADA_MEMORY_FIXTURE_KIND must be corrected_reference or near_cap_stress"
fi
run_id="${2:-manual-eval-$(date -u +%Y%m%d-%H%M%S)}"
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
  || die "unsafe RUN_ID: ${run_id}"
run_root="${LOGS_DIR}/runs/${run_id}"
if [[ "${resume_mode}" == "1" ]]; then
  [[ -d "${run_root}" && ! -L "${run_root}" ]] \
    || die "LLADA_RESUME=1 requires the existing real run directory: ${run_id}"
else
  [[ ! -e "${run_root}" && ! -L "${run_root}" ]] \
    || die "run directory already exists: ${run_id}"
fi
assert_built_images_current
if [[ "${qualification_mode}" == "1" ]]; then
  # Initial stratified-520 timing/memory observations are explicitly non-release
  # and may only consume the tracked UNQUALIFIED/PENDING policy pair.
  load_unqualified_memory_policy
  load_unqualified_workload_policy
else
  load_frozen_memory_policy
fi
if [[ "${pilot_mode}" == "1" || "${calibration_mode}" == "1" \
      || "${quality_control_mode}" == "1" ]]; then
  # Producer calibration and the precommitted independent quality controls
  # are the intentional pre-release evaluations. Both consume the reviewed
  # QUALIFIED/PROVISIONAL policy; their evidence lets the finalizer freeze it.
  load_provisional_workload_policy
elif [[ "${qualification_mode}" != "1" ]]; then
  load_frozen_workload_policy
fi
if [[ "${qualification_mode}" == "1" ]]; then
  # Qualification keeps its operator-specified non-release ceilings.
  LLADA_JUDGE_KILL_TIMEOUT="${POLICY_FINAL_INNER_TIMEOUT}"
else
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
fi
hidden_cases_sha256="$(canonical_hidden_cases_sha256)"
[[ "${hidden_cases_sha256}" \
    == "$(jq -er '.manifest.hidden_cases_sha256' "${WORKLOAD_POLICY_PATH}")" ]] \
  || die "workload policy does not match the canonical hidden workload"

if [[ -n "${memory_fixture_kind}" ]]; then
  memory_fixture_path="${MIGRATION_ROOT}/memory_qualification_fixtures.json"
  [[ -f "${memory_fixture_path}" && ! -L "${memory_fixture_path}" ]] \
    || die "memory qualification requires a safe precommitted fixture manifest"
  memory_fixture_snapshot="$(mktemp)"
  cp -- "${memory_fixture_path}" "${memory_fixture_snapshot}"
  validate_memory_fixture_binding \
    "${memory_fixture_snapshot}" \
    "${memory_fixture_kind}" \
    "${archive}" \
    "${WORKLOAD_POLICY_SHA256}" \
    "${MEMORY_POLICY_SHA256}"
  memory_fixture_identity_json="$(jq -nS \
    --arg manifest_sha256 "${MEMORY_FIXTURE_MANIFEST_SHA256}" \
    --arg approval_id "${MEMORY_FIXTURE_APPROVAL_ID}" \
    --arg kind "${MEMORY_FIXTURE_KIND}" \
    --arg archive_sha256 "${MEMORY_FIXTURE_ARCHIVE_SHA256}" \
    --argjson archive_size_bytes "${MEMORY_FIXTURE_ARCHIVE_SIZE_BYTES}" '
    {
      manifest_sha256: $manifest_sha256,
      approval_id: $approval_id,
      kind: $kind,
      archive_sha256: $archive_sha256,
      archive_size_bytes: $archive_size_bytes
    }
  ')"
else
  MEMORY_FIXTURE_MANIFEST_SHA256=""
  MEMORY_FIXTURE_APPROVAL_ID=""
  MEMORY_FIXTURE_KIND=""
  MEMORY_FIXTURE_ARCHIVE_SHA256=""
  MEMORY_FIXTURE_ARCHIVE_SIZE_BYTES=""
  memory_fixture_identity_json="null"
fi

evaluation_manifest_snapshot="$(mktemp)"
model_validation_snapshot="$(mktemp)"
model_validation_end_snapshot="$(mktemp)"
resume_identity_candidate="$(mktemp)"
quality_controls_snapshot="$(mktemp)"
runtime_provenance_snapshot="$(mktemp)"
cleanup_eval() {
  if [[ -n "${judge_uuid:-}" ]]; then
    remove_task_gpu_containers "judge" "${judge_uuid}" "${run_id}" \
      >/dev/null 2>&1 || true
  fi
  rm -f -- \
    "${evaluation_manifest_snapshot}" \
    "${model_validation_snapshot}" \
    "${model_validation_end_snapshot}" \
    "${resume_identity_candidate}" \
    "${quality_controls_snapshot}" \
    "${runtime_provenance_snapshot}" \
    "${memory_fixture_snapshot:-}" \
    "${enriched_manifest:-}"
}
trap cleanup_eval EXIT

reference_quality="$(judge_private_data_dir)/reference_quality.json"
if [[ "${pilot_mode}" != "1" && "${calibration_mode}" != "1" \
      && "${qualification_mode}" != "1" ]]; then
  if [[ "${quality_control_mode}" == "1" ]]; then
    reference_workload_policy_sha256="${WORKLOAD_POLICY_SHA256}"
  else
    reference_workload_policy_sha256="$(
      frozen_workload_parent_sha256 "${WORKLOAD_POLICY_PATH}"
    )"
  fi
  jq -e \
    --arg task_id "${TASK_ID}" \
    --arg workload_policy_sha256 "${reference_workload_policy_sha256}" \
    --arg memory_policy_sha256 "${MEMORY_POLICY_SHA256}" \
    --arg hidden_cases_sha256 "${hidden_cases_sha256}" '
    .schema_version == 4
    and .calibrated == true
    and .status == "CALIBRATED_STRATIFIED520"
    and .task_id == $task_id
    and .workload_profile == "fast_dllm_v1_llada_12row_stratified520_batch1"
    and .generation_count == 520
    and .requested_slots == 216320
    and (.case_records | type == "object" and length == 520)
    and (.rows | type == "object" and length == 12)
    and .dataset_manifest_sha256 == $hidden_cases_sha256
    and .calibration_provenance.workload_policy_sha256 == $workload_policy_sha256
    and .calibration_provenance.memory_policy_sha256 == $memory_policy_sha256
  ' "${reference_quality}" >/dev/null \
    || die "evaluation requires a calibrated reference bound to the correct policy parent"
  if [[ "${reference_replay}" == "1" ]]; then
    jq -e --arg archive_sha256 "${archive_sha256_before}" '
      .calibration_provenance.candidate_archive_sha256 == $archive_sha256
      and .calibration_provenance.canonical_candidate_archive_sha256 == $archive_sha256
    ' "${reference_quality}" >/dev/null \
      || die "reference replay archive does not match calibration provenance"
  fi
fi
quality_controls_path="${MIGRATION_ROOT}/quality_controls.json"
[[ -f "${quality_controls_path}" && ! -L "${quality_controls_path}" ]] \
  || die "quality-control manifest must be a regular non-symlink file"
cp -- "${quality_controls_path}" "${quality_controls_snapshot}"
quality_controls_sha256="$(
  sha256sum "${quality_controls_snapshot}" | awk '{print $1}'
)"
if [[ "${calibration_mode}" == "1" || "${quality_control_mode}" == "1" ]]; then
  python3 "${MIGRATION_ROOT}/scripts/validate_quality_controls.py" \
    --task-root "${MIGRATION_ROOT}" \
    --manifest "${quality_controls_snapshot}" \
    --require-precommitted \
    --verify-archive-hashes >/dev/null \
    || die "producer/control evaluation requires verified precommitted archives"
  jq -e \
    --arg workload "${WORKLOAD_POLICY_SHA256}" \
    --arg hidden "${hidden_cases_sha256}" '
    .workload_policy_sha256 == $workload
    and .hidden_cases_sha256 == $hidden
  ' "${quality_controls_snapshot}" >/dev/null \
    || die "quality-control manifest is not bound to the provisional workload"
fi
if [[ "${calibration_mode}" == "1" ]]; then
  jq -e --arg archive "${archive_sha256_before}" '
    .producer_calibration.expected_mode == "calibration"
    and .producer_calibration.archive_sha256 == $archive
  ' "${quality_controls_snapshot}" >/dev/null \
    || die "calibration archive is not the precommitted producer archive"
fi
if [[ "${quality_control_mode}" == "1" ]]; then
  control_id="${LLADA_CONTROL_ID:-}"
  expected_classification="${LLADA_EXPECTED_CLASSIFICATION:-}"
  validate_quality_control_binding \
    "${quality_controls_snapshot}" \
    "${control_id}" \
    "${expected_classification}" \
    "${archive_sha256_before}" \
    "${WORKLOAD_POLICY_SHA256}" \
    "${hidden_cases_sha256}"
else
  [[ -z "${LLADA_CONTROL_ID:-}" && -z "${LLADA_EXPECTED_CLASSIFICATION:-}" ]] \
    || die "control identity variables require LLADA_QUALITY_CONTROL=1"
fi
cp -- "${LOGS_DIR}/build/image-manifest.json" "${evaluation_manifest_snapshot}"
source_digest_start="$(jq -er '.source_digest' "${evaluation_manifest_snapshot}")" \
  || die "run-start image manifest lacks source_digest"
work_image_id_start="$(jq -er '.work.id' "${evaluation_manifest_snapshot}")" \
  || die "run-start image manifest lacks Work image ID"
judge_image_id_start="$(jq -er '.judge.id' "${evaluation_manifest_snapshot}")" \
  || die "run-start image manifest lacks Judge image ID"
[[ "${source_digest_start}" =~ ^[0-9a-f]{64}$ ]] \
  || die "run-start source digest is invalid"
[[ "${work_image_id_start}" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || die "run-start Work image ID is invalid"
[[ "${judge_image_id_start}" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || die "run-start Judge image ID is invalid"
SFORGE_PYTHON="$(dirname "${SFORGE_BIN}")/python"
H100_ENTRY="${MIGRATION_ROOT}/scripts/sforge_h100_entry.py"
[[ -x "${SFORGE_PYTHON}" && -f "${H100_ENTRY}" && ! -L "${H100_ENTRY}" ]] \
  || die "missing or unsafe H100 SForge entry/Python"
runtime_entry_sha256="$(sha256sum "${H100_ENTRY}" | awk '{print $1}')"
mapfile -d '' -t SFORGE_RUNTIME_FILES < <(
  cd "${SFORGE_ROOT}"
  find sforge -type f -name '*.py' -print0 | LC_ALL=C sort -z
)
(( ${#SFORGE_RUNTIME_FILES[@]} > 0 )) \
  || die "SForge Python runtime source set is empty"
sforge_runtime_sha256="$(
  (
    cd "${SFORGE_ROOT}"
    sha256sum "${SFORGE_RUNTIME_FILES[@]}"
  ) | sha256sum | awk '{print $1}'
)"
sforge_runtime_file_set_sha256="$(
  printf '%s\n' "${SFORGE_RUNTIME_FILES[@]}" | sha256sum | awk '{print $1}'
)"
sforge_runtime_files_json="$(
  canonical_sforge_runtime_files_json "${SFORGE_RUNTIME_FILES[@]}"
)"
mapfile -t judge_core_hashes_start < <(
  judge_invariant_core_hashes "${JUDGE_IMAGE}"
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
runtime_provenance_sha256="$(
  sha256sum "${runtime_provenance_snapshot}" | awk '{print $1}'
)"
if [[ "${calibration_mode}" == "1" ]]; then
  judge_runtime_tree_sha256_start="$(
    docker run --rm --network none --entrypoint /bin/sh "${JUDGE_IMAGE}" -c \
      'find /opt/decoder_judge -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum' \
      | awk '{print $1}'
  )"
  calibration_hidden_sha256="$(
    docker run --rm --network none --entrypoint sha256sum "${JUDGE_IMAGE}" \
      /opt/decoder_judge/private_data/hidden_cases.json | awk '{print $1}'
  )"
  [[ "${calibration_hidden_sha256}" =~ ^[0-9a-f]{64}$ ]] \
    || die "failed to attest the calibration hidden workload"
  enrich_calibration_image_manifest \
    "${evaluation_manifest_snapshot}" \
    "${judge_runtime_tree_sha256_start}" \
    "${calibration_hidden_sha256}" \
    "${judge_quality_sha256}" \
    "${judge_verify_decoder_sha256}" \
    "${judge_candidate_worker_sha256}" \
    "${judge_cohort_sha256}" \
    "${WORKLOAD_POLICY_SHA256}" \
    "${MEMORY_POLICY_SHA256}"
fi
LLADA_MODEL_REPO="${LLADA_MODEL_REPO:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--GSAI-ML--LLaDA-8B-Instruct}"
export LLADA_MODEL_REPO
python3 "${MIGRATION_ROOT}/scripts/validate_model_artifact.py" \
  --repo "${LLADA_MODEL_REPO}" --full \
  --output "${model_validation_snapshot}" >/dev/null
model_validation_sha256="$(sha256sum "${model_validation_snapshot}" | awk '{print $1}')"
[[ -v JUDGE_GPU && -n "${JUDGE_GPU}" ]] \
  || die "set JUDGE_GPU to an explicit numeric H100 index"
judge_gpu="$(canonical_gpu_index "${JUDGE_GPU}")"
judge_uuid="$(gpu_uuid "${judge_gpu}")"

exec 9>"$(gpu_lock_path "${judge_uuid}")"
flock -n 9 || die "judge GPU ${judge_gpu} lock is already held"
assert_gpu_available "${judge_gpu}" "Judge" "${GPU_ADMISSION_MIN_FREE_MIB}"
judge_total_mib_at_admission="$(gpu_query "${judge_gpu}" memory.total)"
judge_free_mib_at_admission="$(gpu_query "${judge_gpu}" memory.free)"
[[ "${judge_total_mib_at_admission}" =~ ^[0-9]+$ \
    && "${judge_free_mib_at_admission}" =~ ^[0-9]+$ ]] \
  || die "Judge NVML capacity query did not return integral MiB values"
judge_nvml_total_bytes="$((10#${judge_total_mib_at_admission} * 1048576))"
judge_nvml_clean_idle_free_bytes="$((10#${judge_free_mib_at_admission} * 1048576))"
assert_container_gpu "${JUDGE_IMAGE}" "${judge_uuid}" "Judge"
judge_torch_device_total_bytes="${CONTAINER_GPU_TOTAL_BYTES}"
(( judge_torch_device_total_bytes > 0 \
    && judge_torch_device_total_bytes <= judge_nvml_total_bytes )) \
  || die "Judge torch/NVML capacity evidence is inconsistent"

row_order_sha256="$(
  jq -er '.manifest.row_order[]' "${WORKLOAD_POLICY_PATH}" \
    | sha256sum | awk '{print $1}'
)"
jq -nS \
  --arg task_id "${TASK_ID}" \
  --arg run_id "${run_id}" \
  --arg mode "${evaluation_mode}" \
  --arg archive_sha256 "${archive_sha256_before}" \
  --argjson archive_size "${archive_size_before}" \
  --arg work_image_id "${work_image_id_start}" \
  --arg judge_image_id "${judge_image_id_start}" \
  --arg model_validation_sha256 "${model_validation_sha256}" \
  --arg runtime_entry_sha256 "${runtime_entry_sha256}" \
  --arg source_digest "${source_digest_start}" \
  --arg workload_policy_sha256 "${WORKLOAD_POLICY_SHA256}" \
  --arg memory_policy_sha256 "${MEMORY_POLICY_SHA256}" \
  --arg hidden_manifest_sha256 "${hidden_cases_sha256}" \
  --arg row_order_sha256 "${row_order_sha256}" \
  --arg judge_gpu_uuid "${judge_uuid}" \
  --arg control_id "${control_id:-}" \
  --arg expected_classification "${expected_classification:-}" \
  --arg memory_fixture_kind "${memory_fixture_kind}" \
  --arg quality_controls_sha256 "${quality_controls_sha256}" \
  --arg runtime_provenance_sha256 "${runtime_provenance_sha256}" \
  --slurpfile runtime_provenance "${runtime_provenance_snapshot}" \
  --argjson memory_fixture "${memory_fixture_identity_json}" \
  --arg qualification_case_timeout "${POLICY_FINAL_CASE_TIMEOUT}" \
  --arg qualification_row_timeout "${POLICY_FINAL_ROW_TIMEOUT}" \
  --arg qualification_inner_timeout "${POLICY_FINAL_INNER_TIMEOUT}" \
  --arg qualification_outer_timeout "${POLICY_FINAL_OUTER_TIMEOUT}" '
  {
    schema_version: 1,
    task_id: $task_id,
    run_id: $run_id,
    mode: $mode,
    archive_sha256: $archive_sha256,
    archive_size_bytes: $archive_size,
    work_image_id: $work_image_id,
    judge_image_id: $judge_image_id,
    model_validation_sha256: $model_validation_sha256,
    runtime_entry_sha256: $runtime_entry_sha256,
    source_digest: $source_digest,
    workload_policy_sha256: $workload_policy_sha256,
    memory_policy_sha256: $memory_policy_sha256,
    hidden_manifest_sha256: $hidden_manifest_sha256,
    row_order_sha256: $row_order_sha256,
    judge_gpu_uuid: $judge_gpu_uuid,
    control_id: $control_id,
    expected_classification: $expected_classification,
    memory_fixture_kind: $memory_fixture_kind,
    memory_fixture: $memory_fixture,
    quality_controls_sha256: $quality_controls_sha256,
    runtime_provenance_sha256: $runtime_provenance_sha256,
    sforge_runtime: $runtime_provenance[0].sforge_runtime,
    judge_invariant_core: $runtime_provenance[0].judge_invariant_core,
    qualification_safety_timeouts: {
      case: $qualification_case_timeout,
      row: $qualification_row_timeout,
      inner: $qualification_inner_timeout,
      outer: $qualification_outer_timeout
    }
  }
' >"${resume_identity_candidate}"
if [[ "${resume_mode}" == "0" ]]; then
  mkdir -p -- "${LOGS_DIR}/runs"
  mkdir -- "${run_root}"
fi
bind_runtime_resume_identity \
  "${run_root}" "${run_id}" "${resume_mode}" "${resume_identity_candidate}"
run_quality_controls_snapshot="${run_root}/quality-controls-manifest.json"
if [[ "${calibration_mode}" == "1" || "${quality_control_mode}" == "1" ]]; then
  if [[ "${resume_mode}" == "1" ]]; then
    [[ -f "${run_quality_controls_snapshot}" \
        && ! -L "${run_quality_controls_snapshot}" \
        && "$(stat -c '%a' "${run_quality_controls_snapshot}")" == "444" ]] \
      || die "resume requires the immutable quality-control manifest snapshot"
    cmp -s -- "${quality_controls_snapshot}" "${run_quality_controls_snapshot}" \
      || die "quality-control manifest differs from the original run"
  else
    cp -- "${quality_controls_snapshot}" "${run_quality_controls_snapshot}"
    chmod 444 "${run_quality_controls_snapshot}"
  fi
fi
run_runtime_provenance_snapshot="${run_root}/runtime-provenance.json"
if [[ "${resume_mode}" == "1" ]]; then
  [[ -f "${run_runtime_provenance_snapshot}" \
      && ! -L "${run_runtime_provenance_snapshot}" \
      && "$(stat -c '%a' "${run_runtime_provenance_snapshot}")" == "444" ]] \
    || die "resume requires the immutable runtime provenance snapshot"
  cmp -s -- "${runtime_provenance_snapshot}" "${run_runtime_provenance_snapshot}" \
    || die "runtime provenance differs from the original run"
else
  cp -- "${runtime_provenance_snapshot}" "${run_runtime_provenance_snapshot}"
  chmod 444 "${run_runtime_provenance_snapshot}"
fi
run_memory_fixture_snapshot="${run_root}/memory-fixture-manifest.json"
if [[ -n "${memory_fixture_kind}" ]]; then
  if [[ "${resume_mode}" == "1" ]]; then
    [[ -f "${run_memory_fixture_snapshot}" \
        && ! -L "${run_memory_fixture_snapshot}" \
        && "$(stat -c '%a' "${run_memory_fixture_snapshot}")" == "444" ]] \
      || die "memory fixture resume requires its immutable manifest snapshot"
    cmp -s -- "${memory_fixture_snapshot}" "${run_memory_fixture_snapshot}" \
      || die "memory fixture manifest differs from the original run"
  else
    cp -- "${memory_fixture_snapshot}" "${run_memory_fixture_snapshot}"
    chmod 444 "${run_memory_fixture_snapshot}"
  fi
else
  [[ ! -e "${run_memory_fixture_snapshot}" \
      && ! -L "${run_memory_fixture_snapshot}" ]] \
    || die "non-memory evaluation has an unexpected fixture snapshot"
fi
prepare_root_owned_checkpoint_dir \
  "${run_root}" "${run_id}" "${judge_image_id_start}"

submission_index=1
submissions_root="${run_root}/${TASK_ID}/submissions"
while [[ -d "${submissions_root}/manual-${submission_index}" ]]; do
  submission_index="$((submission_index + 1))"
done
submission_round="manual-${submission_index}"
[[ ! -e "${submissions_root}/${submission_round}" \
    && ! -L "${submissions_root}/${submission_round}" ]] \
  || die "next SForge submission path is unsafe: ${submission_round}"

export LLADA_JUDGE_GPU_UUID="${judge_uuid}"
export LLADA_RUNTIME_RUN_ID="${run_id}"
export SFORGE_EXPECTED_WORK_IMAGE_ID="${work_image_id_start}"
export SFORGE_EXPECTED_JUDGE_IMAGE_ID="${judge_image_id_start}"
unset LLADA_WORK_GPU_UUID
judge_env="NVIDIA_VISIBLE_DEVICES=${judge_uuid},CUDA_VISIBLE_DEVICES=0,LLADA_TARGET=dual-cuda-sm80-34g,LLADA_GPU_ROLE=judge,LLADA_INTERACTIVE_JUDGE_TIMEOUT=${LLADA_JUDGE_KILL_TIMEOUT},LLADA_FINAL_JUDGE_TIMEOUT=${LLADA_JUDGE_KILL_TIMEOUT},SFORGE_PRIVATE_CONTEXT_CHECKPOINT_DIR=/logs/verifier/checkpoints,SFORGE_PRIVATE_CONTEXT_ARCHIVE_SHA256=${archive_sha256_before},SFORGE_PRIVATE_CONTEXT_WORK_IMAGE_ID=${work_image_id_start},SFORGE_PRIVATE_CONTEXT_JUDGE_IMAGE_ID=${judge_image_id_start},SFORGE_PRIVATE_CONTEXT_MODEL_VALIDATION_SHA256=${model_validation_sha256},SFORGE_PRIVATE_CONTEXT_RUNTIME_ENTRY_SHA256=${runtime_entry_sha256},SFORGE_PRIVATE_CONTEXT_SOURCE_DIGEST=${source_digest_start},SFORGE_PRIVATE_CONTEXT_QUALITY_CONTROLS_SHA256=${quality_controls_sha256},SFORGE_PRIVATE_CONTEXT_RUNTIME_PROVENANCE_SHA256=${runtime_provenance_sha256},SFORGE_PRIVATE_CONTEXT_SFORGE_RUNTIME_SHA256=${sforge_runtime_sha256},SFORGE_PRIVATE_CONTEXT_SFORGE_RUNTIME_FILE_SET_SHA256=${sforge_runtime_file_set_sha256},SFORGE_PRIVATE_CONTEXT_SFORGE_RUNTIME_FILE_COUNT=${#SFORGE_RUNTIME_FILES[@]},SFORGE_PRIVATE_CONTEXT_JUDGE_VERIFY_DECODER_SHA256=${judge_verify_decoder_sha256},SFORGE_PRIVATE_CONTEXT_JUDGE_CANDIDATE_WORKER_SHA256=${judge_candidate_worker_sha256},SFORGE_PRIVATE_CONTEXT_JUDGE_COHORT_SHA256=${judge_cohort_sha256},SFORGE_PRIVATE_CONTEXT_JUDGE_QUALITY_SHA256=${judge_quality_sha256},SFORGE_PRIVATE_CONTEXT_JUDGE_SOURCE_ADMISSION_SHA256=${judge_source_admission_sha256}"
# sforge's exec kill is the task Judge ceiling, outside the adapter kill.
eval_outer_timeout="$(jq -er '.judge.eval_timeout' "${TASKS_DIR}/${TASK_ID}.json")"
(( LLADA_JUDGE_KILL_TIMEOUT < eval_outer_timeout )) \
  || die "Judge adapter kill must stay inside the task Judge ceiling"
if [[ "${pilot_mode}" == "1" ]]; then
  eval_outer_timeout=7200
  judge_env+=",SFORGE_PRIVATE_CONTEXT_PILOT=1,LLADA_FINAL_JUDGE_TIMEOUT=7200,LLADA_PILOT_FINAL_OUTER_TIMEOUT=7000,LLADA_PILOT_FINAL_ROW_TIMEOUT=3600"
elif [[ "${qualification_mode}" == "1" ]]; then
  judge_env+=",LLADA_QUALIFICATION_CASE_TIMEOUT=${POLICY_FINAL_CASE_TIMEOUT},LLADA_QUALIFICATION_ROW_TIMEOUT=${POLICY_FINAL_ROW_TIMEOUT},LLADA_QUALIFICATION_OUTER_TIMEOUT=${POLICY_FINAL_OUTER_TIMEOUT},SFORGE_PRIVATE_CONTEXT_QUALIFICATION=1"
elif [[ "${calibration_mode}" == "1" ]]; then
  judge_env+=",SFORGE_PRIVATE_CONTEXT_CALIBRATION=1"
elif [[ "${quality_control_mode}" == "1" ]]; then
  judge_env+=",SFORGE_PRIVATE_CONTEXT_QUALITY_CONTROL=1,SFORGE_PRIVATE_CONTEXT_CONTROL_ID=${control_id},SFORGE_PRIVATE_CONTEXT_EXPECTED_CLASSIFICATION=${expected_classification}"
fi
if [[ -n "${memory_fixture_kind}" ]]; then
  judge_env+=",SFORGE_PRIVATE_CONTEXT_MEMORY_FIXTURE_MANIFEST_SHA256=${MEMORY_FIXTURE_MANIFEST_SHA256},SFORGE_PRIVATE_CONTEXT_MEMORY_FIXTURE_APPROVAL_ID=${MEMORY_FIXTURE_APPROVAL_ID},SFORGE_PRIVATE_CONTEXT_MEMORY_FIXTURE_KIND=${MEMORY_FIXTURE_KIND},SFORGE_PRIVATE_CONTEXT_MEMORY_FIXTURE_ARCHIVE_SHA256=${MEMORY_FIXTURE_ARCHIVE_SHA256},SFORGE_PRIVATE_CONTEXT_MEMORY_FIXTURE_ARCHIVE_SIZE_BYTES=${MEMORY_FIXTURE_ARCHIVE_SIZE_BYTES}"
fi
export SFORGE_JUDGE_EXTRA_ENV="${judge_env}"
unset SFORGE_ISOLATED_DOCKER_NETWORK

enriched_manifest="$(mktemp)"
jq \
  --argjson index "${judge_gpu}" \
  --arg uuid "${judge_uuid}" \
  --arg name "$(gpu_name "${judge_gpu}")" \
  --argjson total_mib "${judge_total_mib_at_admission}" \
  --argjson free_mib "${judge_free_mib_at_admission}" \
  --arg compute_capability "$(gpu_query "${judge_gpu}" compute_cap)" \
  --arg driver_version "$(gpu_query "${judge_gpu}" driver_version)" \
  --arg observed_at "$(date -u +%FT%TZ)" \
  --arg memory_policy_sha256 "${MEMORY_POLICY_SHA256}" \
  --arg workload_policy_sha256 "${WORKLOAD_POLICY_SHA256}" \
  --arg quality_controls_sha256 "${quality_controls_sha256}" \
  --arg runtime_provenance_sha256 "${runtime_provenance_sha256}" \
  --slurpfile runtime_provenance "${runtime_provenance_snapshot}" \
  --argjson memory_fixture "${memory_fixture_identity_json}" \
  '. + {
    task_id: "llada_h100_decoder_optimization_v2",
    lifecycle: {final_evaluation: "after_work_stopped_and_archive_frozen"},
    judge_gpu: {
      index: $index,
      uuid: $uuid,
      name: $name,
      total_memory_mib: $total_mib,
      free_memory_mib_at_admission: $free_mib,
      compute_capability: $compute_capability,
      driver_version: $driver_version,
      observed_at: $observed_at,
      container_cuda_visible_devices: "0"
    },
    policies: {
      memory_sha256: $memory_policy_sha256,
      workload_sha256: $workload_policy_sha256,
      quality_controls_sha256: $quality_controls_sha256
    },
    runtime_provenance_sha256: $runtime_provenance_sha256,
    runtime_provenance: $runtime_provenance[0],
    memory_fixture: $memory_fixture
  }' "${evaluation_manifest_snapshot}" >"${enriched_manifest}"
mv -- "${enriched_manifest}" "${evaluation_manifest_snapshot}"

"${SFORGE_PYTHON}" "${H100_ENTRY}" \
  eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${archive}" \
  --run-id "${run_id}" \
  --timeout "${eval_outer_timeout}" \
  --judge-cpu-limit 8 \
  --judge-mem-limit 48g \
  --json
assert_gpu_released "${judge_uuid}" "${judge_gpu}" "Judge"
[[ -d "${run_root}" ]] || die "evaluation did not create run root: ${run_root}"
submission_root="${submissions_root}/${submission_round}"
[[ -d "${submission_root}" && ! -L "${submission_root}" ]] \
  || die "evaluation did not create the expected submission: ${submission_round}"
python3 "${MIGRATION_ROOT}/scripts/validate_model_artifact.py" \
  --repo "${LLADA_MODEL_REPO}" --full \
  --output "${model_validation_end_snapshot}" >/dev/null
[[ "$(sha256sum "${model_validation_end_snapshot}" | awk '{print $1}')" \
    == "${model_validation_sha256}" ]] \
  || die "model validation identity changed during evaluation"
if [[ -e "${run_root}/full-model-validation.json" \
      || -L "${run_root}/full-model-validation.json" ]]; then
  [[ -f "${run_root}/full-model-validation.json" \
      && ! -L "${run_root}/full-model-validation.json" \
      && "$(sha256sum "${run_root}/full-model-validation.json" | awk '{print $1}')" \
        == "${model_validation_sha256}" ]] \
    || die "persisted model validation differs from the resume identity"
else
  cp -- "${model_validation_end_snapshot}" "${run_root}/full-model-validation.json"
  chmod 444 "${run_root}/full-model-validation.json"
fi

# Close the long-running evaluation TOCTOU window before any report or private
# artifact can be accepted. The frozen manifest describes the source and image
# bytes that passed the run-start freshness check; neither mutable source files
# nor the local tags may move while the Judge is running.
source_digest_end="$(image_source_digest)"
work_image_id_end="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}" 2>/dev/null)" \
  || die "Work image disappeared during evaluation"
judge_image_id_end="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}" 2>/dev/null)" \
  || die "Judge image disappeared during evaluation"
[[ "${source_digest_end}" == "${source_digest_start}" ]] \
  || die "task image sources changed during evaluation"
[[ "${work_image_id_end}" == "${work_image_id_start}" ]] \
  || die "Work image tag changed during evaluation"
[[ "${judge_image_id_end}" == "${judge_image_id_start}" ]] \
  || die "Judge image tag changed during evaluation"
[[ "$(sha256sum "${H100_ENTRY}" | awk '{print $1}')" \
    == "${runtime_entry_sha256}" ]] \
  || die "H100 runtime entry changed during evaluation"
[[ "$(sha256sum "${MEMORY_POLICY_PATH}" | awk '{print $1}')" \
    == "${MEMORY_POLICY_SHA256}" ]] \
  || die "memory policy changed during evaluation"
[[ "$(sha256sum "${WORKLOAD_POLICY_PATH}" | awk '{print $1}')" \
    == "${WORKLOAD_POLICY_SHA256}" ]] \
  || die "workload policy changed during evaluation"
[[ "$(sha256sum -- "${archive}" | awk '{print $1}')" == "${archive_sha256_before}" \
    && "$(stat -c '%s' -- "${archive}")" == "${archive_size_before}" ]] \
  || die "evaluated archive changed during evaluation"
mapfile -d '' -t current_sforge_runtime_files < <(
  cd "${SFORGE_ROOT}"
  find sforge -type f -name '*.py' -print0 | LC_ALL=C sort -z
)
current_sforge_runtime_files_json="$(
  canonical_sforge_runtime_files_json "${current_sforge_runtime_files[@]}"
)"
[[ "${current_sforge_runtime_files_json}" == "${sforge_runtime_files_json}" ]] \
  || die "SForge Python runtime file set changed during evaluation"
current_sforge_runtime_sha256="$(
  (
    cd "${SFORGE_ROOT}"
    sha256sum "${current_sforge_runtime_files[@]}"
  ) | sha256sum | awk '{print $1}'
)"
current_sforge_runtime_file_set_sha256="$(
  printf '%s\n' "${current_sforge_runtime_files[@]}" \
    | sha256sum | awk '{print $1}'
)"
[[ "${current_sforge_runtime_sha256}" == "${sforge_runtime_sha256}" \
    && "${current_sforge_runtime_file_set_sha256}" \
    == "${sforge_runtime_file_set_sha256}" ]] \
  || die "SForge Python runtime bytes changed during evaluation"
mapfile -t judge_core_hashes_end < <(
  judge_invariant_core_hashes "${JUDGE_IMAGE}"
)
[[ "${judge_core_hashes_end[*]}" == "${judge_core_hashes_start[*]}" ]] \
  || die "Judge invariant core changed during evaluation"
[[ "$(sha256sum "${runtime_provenance_snapshot}" | awk '{print $1}')" \
    == "${runtime_provenance_sha256}" ]] \
  || die "runtime provenance working snapshot changed during evaluation"
cmp -s -- "${runtime_provenance_snapshot}" "${run_runtime_provenance_snapshot}" \
  || die "run-local runtime provenance snapshot changed during evaluation"
if [[ -n "${memory_fixture_kind}" ]]; then
  cmp -s -- "${memory_fixture_snapshot}" "${memory_fixture_path}" \
    || die "memory fixture manifest changed during evaluation"
  cmp -s -- "${memory_fixture_snapshot}" "${run_memory_fixture_snapshot}" \
    || die "run-local memory fixture manifest snapshot changed during evaluation"
  [[ "$(sha256sum -- "${archive}" | awk '{print $1}')" \
      == "${MEMORY_FIXTURE_ARCHIVE_SHA256}" \
      && "$(stat -c '%s' -- "${archive}")" \
      == "${MEMORY_FIXTURE_ARCHIVE_SIZE_BYTES}" ]] \
    || die "selected memory fixture archive changed during evaluation"
fi
if [[ "${calibration_mode}" == "1" || "${quality_control_mode}" == "1" ]]; then
  cmp -s -- \
    "${quality_controls_snapshot}" \
    "${quality_controls_path}" \
    || die "quality-control manifest changed during evaluation"
  cmp -s -- \
    "${quality_controls_snapshot}" \
    "${run_quality_controls_snapshot}" \
    || die "run-local quality-control manifest snapshot changed"
fi
if [[ "${calibration_mode}" == "1" ]]; then
  judge_runtime_tree_sha256_end="$(
    docker run --rm --network none --entrypoint /bin/sh "${JUDGE_IMAGE}" -c \
      'find /opt/decoder_judge -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum' \
      | awk '{print $1}'
  )"
  [[ "${judge_runtime_tree_sha256_end}" == "${judge_runtime_tree_sha256_start}" ]] \
    || die "Judge runtime tree changed during calibration evaluation"
fi
echo "Evaluation source/image end barrier: PASS (${source_digest_end:0:12})"

if [[ -e "${run_root}/evaluation-image-manifest.json" \
      || -L "${run_root}/evaluation-image-manifest.json" ]]; then
  [[ -f "${run_root}/evaluation-image-manifest.json" \
      && ! -L "${run_root}/evaluation-image-manifest.json" ]] \
    || die "persisted evaluation image manifest is unsafe"
  cmp -s -- \
    "${evaluation_manifest_snapshot}" \
    "${run_root}/evaluation-image-manifest.json" \
    || die "persisted evaluation image manifest differs from the resume identity"
else
  cp -- "${evaluation_manifest_snapshot}" "${run_root}/evaluation-image-manifest.json"
  chmod 444 "${run_root}/evaluation-image-manifest.json"
fi

validation_args=(
  --submission-root "${submission_root}"
  --input-archive "${archive}"
  --expected-archive-sha256 "${archive_sha256_before}"
  --expected-archive-size "${archive_size_before}"
  --run-id "${run_id}"
  --mode "${evaluation_mode}"
  --image-manifest "${run_root}/evaluation-image-manifest.json"
  --model-validation "${run_root}/full-model-validation.json"
  --runtime-provenance "${run_runtime_provenance_snapshot}"
  --judge-gpu-uuid "${judge_uuid}"
  --output "${run_root}/evaluation-attestation.json"
  --task-root "${MIGRATION_ROOT}"
)
if [[ "${reference_replay}" == "1" ]]; then
  validation_args+=(--reference-replay)
fi
if [[ "${calibration_mode}" == "1" || "${quality_control_mode}" == "1" ]]; then
  validation_args+=(
    --quality-controls-manifest "${run_quality_controls_snapshot}"
  )
fi
if [[ -n "${memory_fixture_kind}" ]]; then
  validation_args+=(
    --memory-fixture-manifest "${run_memory_fixture_snapshot}"
    --memory-fixture-kind "${memory_fixture_kind}"
    --nvml-total-bytes "${judge_nvml_total_bytes}"
    --nvml-clean-idle-free-bytes "${judge_nvml_clean_idle_free_bytes}"
    --torch-device-total-bytes "${judge_torch_device_total_bytes}"
  )
fi
python3 "${MIGRATION_ROOT}/scripts/validate_eval_artifacts.py" "${validation_args[@]}"
echo "Evaluation artifact validation: PASS (${run_id}, ${evaluation_mode})"
