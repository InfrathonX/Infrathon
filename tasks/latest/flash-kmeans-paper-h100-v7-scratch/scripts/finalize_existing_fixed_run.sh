#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_command awk
require_command jq
require_command python3
require_command realpath
require_command sha256sum

RUN_ID="${1:-}"
MODEL="${2:-}"
JUDGE_GPU="${3:-}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
  || die "usage: $0 RUN_ID MODEL JUDGE_GPU"
[[ "${MODEL}" =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]] \
  || die "unsupported recovery model: ${MODEL}"
[[ "${JUDGE_GPU}" =~ ^[0-9]+$ ]] || die "JUDGE_GPU must be an integer"

RUN_LOG_ROOT="${LOGS_DIR}/runs/${RUN_ID}"
TASK_RUN_ROOT="${RUN_LOG_ROOT}/${TASK_ID}"
final_archive="${TASK_RUN_ROOT}/final_archive.tar.gz"
[[ -s "${final_archive}" ]] || die "missing final archive: ${final_archive}"

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

formal_precheck="${TASK_RUN_ROOT}/formal-precheck.json"
formal_postcheck="${TASK_RUN_ROOT}/formal-postcheck.json"
recovery_completion="${RUN_LOG_ROOT}/completion-recovered.json"
for output in "${formal_precheck}" "${formal_postcheck}" "${recovery_completion}"; do
  [[ ! -e "${output}" ]] || die "recovery output already exists: ${output}"
done

write_command_json_atomically \
  "${formal_precheck}" \
  python3 "${SCRIPT_DIR}/validate_formal_artifacts.py" pre \
  --task-root "${TASK_RUN_ROOT}" \
  --run-id "${RUN_ID}" \
  --task-id "${TASK_ID}" \
  --model "${MODEL}" \
  --budget-seconds 7200 \
  --interval-seconds 1800
final_archive_sha256="$(
  jq -er '.final_archive.sha256 | select(type == "string")' "${formal_precheck}"
)"

final_eval_id="${RUN_ID}-final-120m-recovered"
[[ ! -e "${LOGS_DIR}/runs/${final_eval_id}" ]] \
  || die "recovered final eval already exists: ${final_eval_id}"
JUDGE_GPU="${JUDGE_GPU}" \
  bash "${SCRIPT_DIR}/eval_archive.sh" "${final_archive}" "${final_eval_id}"

final_task_root="${LOGS_DIR}/runs/${final_eval_id}/${TASK_ID}"
final_submission_root="${final_task_root}/submissions/manual-1"
final_report="${final_submission_root}/report.json"
final_evaluated_archive="${final_submission_root}/submission.tar.gz"

write_command_json_atomically \
  "${formal_postcheck}" \
  python3 "${SCRIPT_DIR}/validate_formal_artifacts.py" post \
  --task-root "${TASK_RUN_ROOT}" \
  --final-task-root "${final_task_root}" \
  --final-eval-id "${final_eval_id}" \
  --task-id "${TASK_ID}" \
  --expected-archive-sha256 "${final_archive_sha256}"

final_report_sha256="$(jq -er '.final_report.sha256' "${formal_postcheck}")"
evaluated_archive_sha256="$(jq -er '.evaluated_archive.sha256' "${formal_postcheck}")"
private_judge_source="$(jq -er '.private_judge_artifact.path' "${formal_postcheck}")"
private_judge_sha256="$(jq -er '.private_judge_artifact.sha256' "${formal_postcheck}")"
private_judge_member_sha256="$(
  jq -er '.private_judge_artifact.member_sha256' "${formal_postcheck}"
)"
final_private_judge_artifact="$(
  realpath -m "${TASK_RUN_ROOT}/final-120m-private-judge-artifacts.tar.gz"
)"
[[ ! -e "${final_private_judge_artifact}" ]] \
  || die "private Judge recovery output already exists: ${final_private_judge_artifact}"
cp -- "${private_judge_source}" "${final_private_judge_artifact}"
chmod 600 "${final_private_judge_artifact}"
[[ "$(sha256sum "${final_private_judge_artifact}" | awk '{print $1}')" \
    == "${private_judge_sha256}" ]] \
  || die "copied private Judge artifact digest mismatch"

final_curve_report="${TASK_RUN_ROOT}/final-120m-report.json"
[[ ! -e "${final_curve_report}" ]] \
  || die "final curve recovery report already exists: ${final_curve_report}"
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
    final_archive_sha256: $final_archive_sha256,
    evaluated_archive_sha256: $evaluated_archive_sha256,
    final_report_sha256: $final_report_sha256,
    recovered_after_snapshot_runtime_reporting_skew_fix: true,
    private_judge_artifact: {
      path: $private_judge_artifact,
      sha256: $private_judge_sha256,
      member_sha256: $private_judge_member_sha256
    },
    report: $postcheck[0].validated_report
  }'

python3 "${SCRIPT_DIR}/collect_curve.py" \
  "${TASK_RUN_ROOT}" \
  --final-eval "${final_curve_report}" \
  --no-auto-snapshots \
  --final-minutes 120 \
  --output-dir "${TASK_RUN_ROOT}/curve"

write_command_json_atomically \
  "${recovery_completion}" \
  jq -n \
  --arg finished_at "$(date -u +%FT%TZ)" \
  --arg run_id "${RUN_ID}" \
  --arg task_id "${TASK_ID}" \
  --arg final_archive "${final_archive}" \
  --arg final_archive_sha256 "${final_archive_sha256}" \
  --arg final_report "${final_report}" \
  --arg final_report_sha256 "${final_report_sha256}" \
  --arg evaluated_archive "${final_evaluated_archive}" \
  --arg evaluated_archive_sha256 "${evaluated_archive_sha256}" \
  --arg private_judge_artifact "${final_private_judge_artifact}" \
  --arg private_judge_sha256 "${private_judge_sha256}" \
  '{
    schema_version: 1,
    status: "passed",
    recovery_reason: "subsecond_snapshot_runtime_reporting_skew",
    immutable_boundary_archive_reused: true,
    finished_at: $finished_at,
    run_id: $run_id,
    task_id: $task_id,
    final_archive: {path: $final_archive, sha256: $final_archive_sha256},
    final_report: {path: $final_report, sha256: $final_report_sha256},
    evaluated_archive: {
      path: $evaluated_archive,
      sha256: $evaluated_archive_sha256
    },
    private_judge_artifact: {
      path: $private_judge_artifact,
      sha256: $private_judge_sha256
    }
  }'

echo "Recovered finalization for ${RUN_ID}"
echo "  report: ${final_report}"
echo "  curve:  ${TASK_RUN_ROOT}/curve/curve.md"
