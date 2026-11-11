#!/usr/bin/env bash
# LOCAL-ONLY forced-oracle Judge run for the paged-GQA decode task.
#
#   oracle/paper_reference/forced_judge/run_forced_oracle.sh --gpu N [--skip-build] [--run-id ID]
#
# 1. builds <judge image>:forced-oracle FROM the task's private Judge image
#    (scripts/lib.sh JUDGE_IMAGE, which holds the frozen holdout) + the
#    FlashAttention-3 package of the paper-reference image FA3_IMAGE (built
#    first from FA3_SOURCE with ../Dockerfile if it is missing) +
#    forced_verify.py / forced_admission.py / forced_worker.py (see Dockerfile
#    and README.md);
# 2. packs the forced oracle archive: FINAL_VERSION = oracle/FINAL_VERSION,
#    candidates/kernel_vNNN/submission.py = oracle/candidates/kernel_vNNN, and
#    oracle/notes.md (the same members scripts/make_oracle_archive.sh packs);
# 3. runs the same `sforge eval` that scripts/eval_archive.sh runs (docker
#    backend, same task id, CPU/memory limits, GPU binding, per-UUID GPU lock,
#    idle and container-pin checks), substituting only the Judge image tag (a
#    temporary copy of tasks/ whose judge.image_tag is the forced tag and whose
#    eval_cmd passes --verifier forced_verify.py) and skipping the LLM
#    integrity review (TRUSTED_CONTROL semantics).  The Judge container also
#    gets --network none.
#
# Environment: FA3_IMAGE (default standalone_paged_gqa_decode.paper-fa3:0251105a),
# FA3_SOURCE (only to build a missing FA3_IMAGE), SFORGE_ROOT / SFORGE_BIN /
# PYTHON_BIN, LOGS_DIR (run output root), FORCED_TAG (default forced-oracle),
# SFORGE_JUDGE_CPUSET (optional CPU pinning, e.g. 12-15,44-47),
# FINAL_EVAL_TIMEOUT (default 3600 s, the Judge's).
# The :local tags are never retagged or overwritten, and nothing is pushed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../../../scripts/lib.sh"
require_sforge
for c in docker jq nvidia-smi flock tar gzip timeout; do require_command "$c"; done

FORCED_TAG="${FORCED_TAG:-forced-oracle}"
[[ "${FORCED_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ && "${FORCED_TAG}" != "local" ]] \
  || die "FORCED_TAG must be a plain tag other than 'local'"
FORCED_IMAGE="${JUDGE_IMAGE%:*}:${FORCED_TAG}"
FA3_COMMIT=0251105a2fb19d2957484b7f023cd8c115286ced
FA3_IMAGE="${FA3_IMAGE:-standalone_paged_gqa_decode.paper-fa3:0251105a}"
FA3_SOURCE="${FA3_SOURCE:-}"
EVAL_TIMEOUT="${FINAL_EVAL_TIMEOUT:-3600}"

gpu=""; skip_build=0; run_id=""
while (( $# )); do
  case "$1" in
    --gpu) [[ $# -ge 2 ]] || die "--gpu needs a value"; gpu="$2"; shift 2 ;;
    --skip-build) skip_build=1; shift ;;
    --run-id) [[ $# -ge 2 ]] || die "--run-id needs a value"; run_id="$2"; shift 2 ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ "${gpu}" =~ ^[0-9]+$ ]] || die "--gpu N is required"
run_id="${run_id:-forced-paged-gqa-$(date -u +%Y%m%d-%H%M%S)}"
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe run id: ${run_id}"
[[ ! -e "${LOGS_DIR}/runs/${run_id}" ]] || die "run directory already exists: ${run_id}"

docker image inspect "${JUDGE_IMAGE}" >/dev/null 2>&1 || die "missing private Judge image ${JUDGE_IMAGE}"
judge_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
manifest="${LOGS_DIR}/build/image-manifest.json"
if [[ -s "${manifest}" ]]; then
  [[ "$(jq -r '.judge.id' "${manifest}")" == "${judge_id}" ]] \
    || die "${JUDGE_IMAGE} is not the image recorded in ${manifest}"
  echo "Judge image ${judge_id:7:12} matches ${manifest}"
else
  echo "WARNING: no build manifest at ${manifest}; using ${JUDGE_IMAGE} (${judge_id:7:12}) as is" >&2
fi

# 1. forced image
if (( ! skip_build )); then
  if ! docker image inspect "${FA3_IMAGE}" >/dev/null 2>&1; then
    [[ -n "${FA3_SOURCE}" && -d "${FA3_SOURCE}/hopper" && -d "${FA3_SOURCE}/csrc/cutlass/include" ]] \
      || die "missing ${FA3_IMAGE}; set FA3_SOURCE to a flash-attention checkout at ${FA3_COMMIT} with csrc/cutlass"
    if [[ -d "${FA3_SOURCE}/.git" ]]; then
      [[ "$(git -C "${FA3_SOURCE}" rev-parse HEAD)" == "${FA3_COMMIT}" ]] || die "${FA3_SOURCE} is not at ${FA3_COMMIT}"
    fi
    docker build --tag "${FA3_IMAGE}" --build-context "harness=${HERE}/.." \
      --file "${HERE}/../Dockerfile" "${FA3_SOURCE}"
  fi
  docker build --tag "${FORCED_IMAGE}" \
    --build-arg "JUDGE_IMAGE=${JUDGE_IMAGE}" \
    --build-arg "FA3_IMAGE=${FA3_IMAGE}" \
    --file "${HERE}/Dockerfile" "${HERE}"
fi
docker image inspect "${FORCED_IMAGE}" >/dev/null 2>&1 || die "missing ${FORCED_IMAGE} (build it first)"
forced_id="$(docker image inspect --format '{{.Id}}' "${FORCED_IMAGE}")"

# 2. forced oracle archive
version="$(tr -d '[:space:]' <"${MIGRATION_ROOT}/oracle/FINAL_VERSION")"
[[ "${version}" =~ ^[0-9]{1,3}$ ]] || die "oracle/FINAL_VERSION must be an integer"
printf -v vvv '%03d' "$((10#${version}))"
source_file="${MIGRATION_ROOT}/oracle/candidates/kernel_v${vvv}/submission.py"
[[ -f "${source_file}" && ! -L "${source_file}" ]] || die "missing ${source_file}"
work="$(mktemp -d)"
cleanup() { rm -rf "${work}"; }
trap cleanup EXIT
mkdir -p "${work}/archive/candidates/kernel_v${vvv}" "${work}/tasks"
install -m 0644 "${source_file}" "${work}/archive/candidates/kernel_v${vvv}/submission.py"
printf '%d\n' "$((10#${version}))" >"${work}/archive/FINAL_VERSION"
install -m 0644 "${MIGRATION_ROOT}/oracle/notes.md" "${work}/archive/notes.md"
mkdir -p "${ARTIFACTS_DIR}"
archive="${ARTIFACTS_DIR}/forced-oracle.tar.gz"
tar -C "${work}/archive" --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  -cf - candidates FINAL_VERSION notes.md | gzip -n >"${archive}"
echo "forced archive: ${archive} sha256=$(sha256sum "${archive}" | awk '{print $1}')"
echo "selected source: $(sha256sum "${source_file}" | awk '{print $1}')"

# 3. temporary tasks dir: only the Judge image tag, the verifier path and the
#    container timeout differ from tasks/<task>.json.
cp "${TASKS_DIR}/BENCHMARK.yaml" "${work}/tasks/"
jq --arg tag "${FORCED_TAG}" --argjson t "${EVAL_TIMEOUT}" \
  '.judge.image_tag = $tag
   | .judge.eval_timeout = $t
   | .judge.eval_cmd = (.judge.eval_cmd | sub("run_judge.py "; "run_judge.py --verifier /opt/standalone_paged_gqa_decode_judge/forced_verify.py "))' \
  "${TASKS_DIR}/${TASK_ID}.json" >"${work}/tasks/${TASK_ID}.json"
jq -r '.judge.eval_cmd' "${work}/tasks/${TASK_ID}.json"
export SFORGE_TASKS_DIR="${work}/tasks"

physical_gpu="$(canonical_gpu_index "${gpu}")"
physical_uuid="$(gpu_uuid "${physical_gpu}")"
exec 8>"$(gpu_lock_path "${physical_uuid}")"
flock -n 8 || die "GPU ${physical_gpu} lock is already held"
assert_gpu_idle "${physical_gpu}" "forced judge"
assert_container_gpu "${FORCED_IMAGE}" "${physical_uuid}" "forced judge"

container="${BENCHMARK_NAME}.judge.${TASK_ID}.${run_id}"
cleanup() { docker rm -f "${container}" >/dev/null 2>&1 || true; rm -rf "${work}"; }
export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${physical_uuid},CUDA_VISIBLE_DEVICES=0"
export SFORGE_LLM_JUDGE_MODE=off
export SFORGE_JUDGE_NETWORK_MODE=none
echo "run ${run_id}: image ${FORCED_IMAGE} (${forced_id:7:12}) from ${JUDGE_IMAGE} (${judge_id:7:12}), GPU ${physical_gpu} ${physical_uuid}, cpuset ${SFORGE_JUDGE_CPUSET:-unpinned}"
status=0
timeout --kill-after=60 "$((EVAL_TIMEOUT + 900))" "${SFORGE_BIN}" eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${archive}" \
  --run-id "${run_id}" \
  --timeout "${EVAL_TIMEOUT}" \
  --judge-cpu-limit 8 \
  --judge-mem-limit 32g \
  --json \
  --skip-integrity-review || status=$?
assert_gpu_released "${physical_uuid}"

out="$(find "${LOGS_DIR}/runs/${run_id}" -name report.json -print -quit 2>/dev/null || true)"
[[ -n "${out}" ]] || die "no report.json for ${run_id} (sforge exit ${status})"
jq -c '{valid, score, summary, failed_gate: .metrics.failed_gate, edge_cases_failed: .metrics.edge_cases_failed}' "${out}"
art="$(dirname "${out}")/private_judge_artifacts.tar.gz"
if [[ -f "${art}" ]]; then
  tar -xzOf "${art}" 2>/dev/null | jq -c '{forced_scored_score_tflops, forced_scored_q1_tflops, forced_scored_q16_tflops,
      score_tflops, forced_failed_gates, edge_cases_failed, sanitizer: (.sanitizer | {status, returncode}),
      forced: (.forced | {bypassed, edge_failures: [.edge_failures[]?.case_id]})}' 2>/dev/null || true
fi
{
  echo "run_id=${run_id}"
  echo "forced_image=${FORCED_IMAGE} ${forced_id}"
  echo "judge_image=${JUDGE_IMAGE} ${judge_id}"
  echo "archive_sha256=$(sha256sum "${archive}" | awk '{print $1}')"
  echo "gpu=${physical_gpu} ${physical_uuid} cpuset=${SFORGE_JUDGE_CPUSET:-unpinned}"
} >"$(dirname "${out}")/forced_receipt.txt"
echo "report: ${out}"
exit "${status}"
