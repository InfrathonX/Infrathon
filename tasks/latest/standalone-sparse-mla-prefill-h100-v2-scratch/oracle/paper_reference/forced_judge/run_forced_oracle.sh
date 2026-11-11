#!/usr/bin/env bash
# FORCED-ORACLE Judge run (LOCAL ONLY).  See README.md here.
#
#   PHYSICAL_GPU=<idx> [SFORGE_JUDGE_CPUSET=<cpus>] \
#     oracle/paper_reference/forced_judge/run_forced_oracle.sh [--skip-build] RUN_ID
#   oracle/paper_reference/forced_judge/run_forced_oracle.sh --build-only
#
# 1. builds <benchmark>.judge.<task_id>:${FORCED_TAG} (default tag forced-oracle) from
#    the task's private Judge image (:local) + the paper library + the admission
#    override (Dockerfile here).  It never retags :local and never pushes.
# 2. packs the oracle archive with scripts/make_oracle_archive.sh (FINAL_VERSION and
#    the candidate it selects) and checks the selected source is the forced candidate.
# 3. runs the task's own scripts/eval_archive.sh unchanged (same sforge eval, GPU pin,
#    lock, cpuset, network none, holdouts, sanitizer, timing, aggregation), with
#    TRUSTED_CONTROL=1 (no LLM integrity review) and TASKS_DIR pointing at a temporary
#    copy of tasks/<task_id>.json whose judge.image_tag is ${FORCED_TAG}: the only
#    substitution is the Judge image.
#
# Environment:
#   EVAL_TASK_ROOT  task tree whose scripts/eval_archive.sh + lib.sh run the Judge and
#                   whose outputs/build manifest matches the local :local images
#                   (default: this task).  Example: a development checkout of the same
#                   task, EVAL_TASK_ROOT=/path/to/mlsys_forge/tasks/latest/<task-dir>.
#   PAPER_IMAGE     image holding the built FlashMLA package (default: the image
#                   ../run_flashmla_reference.sh builds).  If it is missing and
#                   FLASHMLA_SOURCE is set, it is built first with that script.
#   FLASHMLA_SOURCE upstream FlashMLA checkout at ba89a34 with csrc/cutlass (only to
#                   build PAPER_IMAGE).
#   FORCED_TAG      tag of the derived Judge image (default forced-oracle, never local).
#   EXPECTED_SHA256 sha256 the archive's selected submission.py must have (default:
#                   that of oracle/candidates/kernel_v<FINAL_VERSION>/submission.py).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${HERE}/../../.." && pwd)"
EVAL_TASK_ROOT="$(cd "${EVAL_TASK_ROOT:-${TASK_ROOT}}" && pwd)"
FORCED_TAG="${FORCED_TAG:-forced-oracle}"
PAPER_IMAGE="${PAPER_IMAGE:-standalone_sparse_mla_prefill.paper-oracle-flashmla:local}"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ "${FORCED_TAG}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ && "${FORCED_TAG}" != local ]] \
  || die "FORCED_TAG must be a plain tag other than 'local'"

skip_build=0
build_only=0
run_id=""
while (( $# )); do
  case "$1" in
    --skip-build) skip_build=1; shift ;;
    --build-only) build_only=1; shift ;;
    -h|--help) sed -n '2,37p' "$0"; exit 0 ;;
    -*) die "unknown argument: $1" ;;
    *) [[ -z "${run_id}" ]] || die "only one RUN_ID"; run_id="$1"; shift ;;
  esac
done

# Names (TASK_ID, BENCHMARK_NAME, JUDGE_IMAGE, TASKS_DIR, LOGS_DIR) from the eval tree.
# shellcheck source=/dev/null
source "${EVAL_TASK_ROOT}/scripts/lib.sh"
require_command docker
require_command jq
FORCED_IMAGE="${BENCHMARK_NAME}.judge.${TASK_ID}:${FORCED_TAG}"
[[ "${FORCED_IMAGE}" != "${JUDGE_IMAGE}" ]] || die "refusing to overwrite ${JUDGE_IMAGE}"

if (( ! skip_build )); then
  if ! docker image inspect "${PAPER_IMAGE}" >/dev/null 2>&1; then
    [[ -n "${FLASHMLA_SOURCE:-}" ]] \
      || die "missing ${PAPER_IMAGE}; set FLASHMLA_SOURCE (FlashMLA checkout at ba89a34) to build it"
    PAPER_IMAGE="${PAPER_IMAGE}" FLASHMLA_SOURCE="${FLASHMLA_SOURCE}" \
      "${HERE}/../run_flashmla_reference.sh" --build-only
  fi
  docker image inspect "${JUDGE_IMAGE}" >/dev/null 2>&1 \
    || die "missing private Judge image ${JUDGE_IMAGE} (scripts/build_images.sh)"
  judge_id_before="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
  docker build --network none \
    --build-arg "JUDGE_IMAGE=${JUDGE_IMAGE}" \
    --build-arg "PAPER_IMAGE=${PAPER_IMAGE}" \
    --tag "${FORCED_IMAGE}" \
    --file "${HERE}/Dockerfile" "${HERE}"
  [[ "$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")" == "${judge_id_before}" ]] \
    || die "${JUDGE_IMAGE} moved during the forced build"
fi
docker image inspect "${FORCED_IMAGE}" >/dev/null 2>&1 || die "missing ${FORCED_IMAGE}"
echo "forced Judge image: ${FORCED_IMAGE} $(docker image inspect --format '{{.Id}}' "${FORCED_IMAGE}")"
echo "base private Judge: ${JUDGE_IMAGE} $(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
(( build_only )) && exit 0

[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "usage: $0 [--skip-build] RUN_ID"
[[ -n "${PHYSICAL_GPU:-}" ]] || die "set PHYSICAL_GPU=<idx>"

work="$(mktemp -d)"
trap 'rm -rf "${work}"' EXIT

# Oracle archive from THIS tree's oracle/ (FINAL_VERSION selects the forced candidate).
archive="$(ARTIFACTS_DIR="${work}/artifacts" "${TASK_ROOT}/scripts/make_oracle_archive.sh" | sed -n '1p')"
version="$(tr -d '[:space:]' <"${TASK_ROOT}/oracle/FINAL_VERSION")"
printf -v selected 'candidates/kernel_v%03d/submission.py' "$((10#${version}))"
expected="${EXPECTED_SHA256:-$(sha256sum "${TASK_ROOT}/oracle/${selected}" | awk '{print $1}')}"
packed="$(tar -xzOf "${archive}" "${selected}" | sha256sum | awk '{print $1}')"
[[ "${packed}" == "${expected}" ]] || die "archive ${selected} sha256 ${packed} != ${expected}"
mkdir -p "${work}/keep"
cp "${archive}" "${work}/keep/oracle-forced.tar.gz"
archive="${work}/keep/oracle-forced.tar.gz"
echo "archive: ${selected} sha256=${packed} archive_sha256=$(sha256sum "${archive}" | awk '{print $1}')"

# Task spec copy: identical except judge.image_tag.
mkdir -p "${work}/tasks"
cp "${TASKS_DIR}/BENCHMARK.yaml" "${work}/tasks/BENCHMARK.yaml"
jq --arg tag "${FORCED_TAG}" '.judge.image_tag = $tag' \
  "${TASKS_DIR}/${TASK_ID}.json" >"${work}/tasks/${TASK_ID}.json"

TASKS_DIR="${work}/tasks" PHYSICAL_GPU="${PHYSICAL_GPU}" TRUSTED_CONTROL=1 \
  SFORGE_LLM_JUDGE_MODE=off \
  "${EVAL_TASK_ROOT}/scripts/eval_archive.sh" "${archive}" "${run_id}"

report="${LOGS_DIR}/runs/${run_id}/${TASK_ID}/submissions/manual-1/report.json"
[[ -f "${report}" ]] || die "no report at ${report}"
echo "report: ${report}"
jq '{valid, score, summary, source_ok: .metrics.source_ok, memory_safety_ok: .metrics.memory_safety_ok,
       sanitizer_ran: .metrics.sanitizer_ran, edge_cases_failed: .metrics.edge_cases_failed,
       edge_cases_total: .metrics.edge_cases_total}' "${report}" || true
