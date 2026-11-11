#!/usr/bin/env bash
set -Eeuo pipefail

# Trusted-oracle run: package oracle/ as a submission archive and judge it on
# the complete hidden suite through scripts/eval_archive.sh, which locks and
# admits the JUDGE_GPUS (both host lock names per GPU) and sizes the Judge.
#   JUDGE_GPUS=0,1,2,3,4,5,6,7 bash scripts/run_oracle.sh
# While the release is "calibrating" the Judge runs every case and records
# every reading but returns valid=false (uncalibrated_release): the run then
# only has to complete cleanly, and the readings for the gate calibration come
# from the private artifact.  Once the release is "calibrated" the oracle must
# be valid.

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
for command in jq tar; do
  require_command "${command}"
done

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ID="${RUN_ID:-qvg-h100-scratch-v4-trusted-oracle-${RUN_STAMP}}"
release_status="$(jq -er '.status' "${RELEASE_PATH}")" \
  || die "the release has no status"
[[ "${release_status}" == calibrating || "${release_status}" == calibrated ]] \
  || die "unexpected release status: ${release_status}"
# ORACLE_TIMEOUT stays as an explicit override of the derived Judge timeout.
if [[ -n "${ORACLE_TIMEOUT:-}" ]]; then
  export FINAL_EVAL_TIMEOUT="${ORACLE_TIMEOUT}"
fi

archive_root="$(mktemp -d -t qvg-oracle-XXXXXXXX)"
cleanup() { rm -rf -- "${archive_root}"; }
trap cleanup EXIT INT TERM
mkdir -p "${archive_root}/submission"
[[ -f "${TASK_ROOT}/oracle/mlsysforge_codec.py" \
   && -d "${TASK_ROOT}/oracle/codec" ]] \
  || die "trusted scratch v4 oracle implementation is missing"
tar -C "${TASK_ROOT}/oracle" --exclude='__pycache__' --exclude='*.pyc' \
  -cf - . | tar -C "${archive_root}/submission" -xf -
tar -C "${archive_root}" -czf "${archive_root}/oracle.tar.gz" submission

# Trusted oracle smoke: no LLM integrity review (see `sforge eval --help`).
bash "${SCRIPT_DIR}/eval_archive.sh" --skip-integrity-review \
  "${archive_root}/oracle.tar.gz" "${RUN_ID}"

submission_dir="${LOGS_DIR}/runs/${RUN_ID}/${TASK_ID}/submissions/manual-1"
report="${submission_dir}/report.json"
private_archive="${submission_dir}/private_judge_artifacts.tar.gz"
[[ -s "${report}" && ! -L "${report}" ]] || die "trusted oracle report is missing"
jq -e '.timed_out == false' "${report}" >/dev/null \
  || die "trusted oracle run timed out: ${report}"

if [[ "${release_status}" == calibrating ]]; then
  # run_judge.py reports a complete calibration run as evaluation_status
  # "uncalibrated"; the private result names the reason.
  jq -e '.metrics.evaluation_status == "uncalibrated"' "${report}" >/dev/null \
    || die "calibration run did not complete every case (status: $(jq -r '.metrics.evaluation_status' "${report}")): ${report}"
  [[ -s "${private_archive}" && ! -L "${private_archive}" ]] \
    || die "trusted oracle private artifact is missing: ${private_archive}"
  tar -xzOf "${private_archive}" qvg-private.json \
    | jq -e '.outcome == "evaluated" and .reason == "uncalibrated_release"' >/dev/null \
    || die "calibration run has no uncalibrated_release verdict: ${private_archive}"
  echo "Trusted oracle calibration run complete: ${RUN_ID} (release calibrating; readings in ${private_archive})"
  exit 0
fi

jq -e '
  .valid == true
  and (.metrics.candidate_invalid == false)
  and (.metrics.infrastructure_invalid == false)
' "${report}" >/dev/null \
  || die "trusted oracle is not valid on the calibrated release: ${report}"
score="$(jq -r '.score' "${report}")"
# The oracle's score band is per device class, like the quality reference it is
# measured against: oracle_calibration.score_band = [low, high] in this host's
# quality file (absent while a calibrating release has none).
quality_file="$(quality_path)"
band=""
if [[ -f "${quality_file}" ]]; then
  band="$(jq -c '.oracle_calibration.score_band // empty' "${quality_file}")"
fi
if [[ -n "${band}" ]]; then
  jq -e --argjson band "${band}" '
    ($band | length == 2 and (map(type) == ["number", "number"]))
    and .score >= $band[0] and .score <= $band[1]
  ' "${report}" >/dev/null \
    || die "trusted oracle score ${score} is outside the band ${band} of ${quality_file}"
  echo "Trusted oracle PASS: ${RUN_ID} (valid=true, score=${score}, band ${band})"
else
  echo "Trusted oracle PASS: ${RUN_ID} (valid=true, score=${score}; no oracle score band for this device class yet)"
fi
