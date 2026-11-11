#!/usr/bin/env bash
set -euo pipefail

# FORCED ORACLE: score upstream flash-kmeans' unmerged fa3-branch CuTe DSL
# (SM90 TMA + WGMMA) assignment kernel with the task's real Judge pipeline
# (sforge eval -> run_judge.py -> verify_flash_kmeans.py: hidden runtime cases,
# fresh-process deployment cases, correctness/ties/integrity/memory/timing
# gates, weighted geomean) in a LOCAL-ONLY derived Judge image whose source
# admission is bypassed.
#
# Replicates scripts/eval_archive.sh exactly (same sforge eval command,
# CPU/memory limits, network policy, per-GPU Judge lock, idle and release
# barriers, NVIDIA_VISIBLE_DEVICES + FLASH_KMEANS_TARGET=H100) and substitutes
# only the Judge image: a scratch copy of the task JSON selects image tag
# `forced-oracle` instead of `local`.  The LLM integrity review is off
# (SFORGE_LLM_JUDGE_MODE=off).
#
#   FK_SOURCE=/path/to/flash-kmeans@39be82a \
#     host_reference/official_oracle_fa3_cutedsl_forced/forced_judge/run_forced_oracle.sh \
#       --gpu 4 --cpuset 16-19,48-51 --version 2
#
# Never retags or overwrites the :local images and never pushes anything.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFORGE_ENTRY_SCRIPT="${SCRIPT_DIR}/run_forced_oracle.sh"
# shellcheck source=../../../scripts/lib.sh
source "${SCRIPT_DIR}/../../../scripts/lib.sh"

PIN=39be82a52e66964aaea816c7740db5526bb9fc90
FORCED_TAG="${FORCED_TAG:-forced-oracle}"
FORCED_IMAGE="flash_kmeans.judge.${TASK_ID}:${FORCED_TAG}"
FK_SOURCE="${FK_SOURCE:-}"
CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION:-4.5.0}"
REF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

gpu="${JUDGE_GPU:-}"
cpuset="${SFORGE_JUDGE_CPUSET:-}"
version=""
run_id=""
skip_build=0

usage() {
  cat <<'USAGE'
Usage: host_reference/official_oracle_fa3_cutedsl_forced/forced_judge/run_forced_oracle.sh --gpu N [options]

  --gpu N        physical GPU index (required; must be idle)
  --cpuset LIST  Judge container cpuset (SFORGE_JUDGE_CPUSET), e.g. 16-19,48-51
  --version V    candidate kernels/flash_kmeans_vVVV.py to judge (default: ../FINAL_VERSION)
  --run-id ID    sforge run id (default forced-flash-kmeans-v<V>-gpu<N>-<UTC stamp>)
  --skip-build   reuse an existing forced-oracle image
Env: FK_SOURCE (upstream checkout or `git archive` at the pin; needed to build),
     CUTLASS_DSL_VERSION (default 4.5.0), FORCED_TAG, LOGS_DIR, SFORGE_BIN.
USAGE
}

while (( $# )); do
  case "$1" in
    --gpu) [[ $# -ge 2 ]] || die "--gpu requires a value"; gpu="$2"; shift 2 ;;
    --cpuset) [[ $# -ge 2 ]] || die "--cpuset requires a value"; cpuset="$2"; shift 2 ;;
    --version) [[ $# -ge 2 ]] || die "--version requires a value"; version="$2"; shift 2 ;;
    --run-id) [[ $# -ge 2 ]] || die "--run-id requires a value"; run_id="$2"; shift 2 ;;
    --skip-build) skip_build=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

[[ "${gpu}" =~ ^[0-9]+$ ]] || { usage >&2; die "--gpu is required"; }
[[ "${FORCED_TAG}" != "local" ]] || die "FORCED_TAG must not be 'local'"
ensure_docker_group "$@"
require_sforge
require_command docker
require_command flock
require_command jq
require_command nvidia-smi
docker image inspect "${JUDGE_IMAGE}" >/dev/null 2>&1 \
  || die "missing private Judge image ${JUDGE_IMAGE}; run scripts/build_images.sh"

version="${version:-$(tr -d '[:space:]' <"${REF_ROOT}/FINAL_VERSION")}"
[[ "${version}" =~ ^[0-9]{1,3}$ && "$((10#${version}))" -ge 1 ]] || die "bad --version ${version}"
printf -v selected 'kernels/flash_kmeans_v%03d.py' "$((10#${version}))"
[[ -f "${REF_ROOT}/${selected}" && ! -L "${REF_ROOT}/${selected}" ]] || die "missing ${selected}"

# --- image (CPU only) --------------------------------------------------------
if (( ! skip_build )) || ! docker image inspect "${FORCED_IMAGE}" >/dev/null 2>&1; then
  [[ -n "${FK_SOURCE}" && -f "${FK_SOURCE}/flash_kmeans/cutedsl_impl.py" ]] \
    || die "set FK_SOURCE to a flash-kmeans checkout (or git archive) at ${PIN}"
  if [[ -d "${FK_SOURCE}/.git" ]]; then
    [[ "$(git -C "${FK_SOURCE}" rev-parse HEAD)" == "${PIN}" ]] || die "FK_SOURCE is not at ${PIN}"
  fi
  docker build --tag "${FORCED_IMAGE}" \
    --build-arg "JUDGE_IMAGE=${JUDGE_IMAGE}" \
    --build-arg "CUTLASS_DSL_VERSION=${CUTLASS_DSL_VERSION}" \
    --build-context "fk_src=${FK_SOURCE}" \
    --file "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"
fi
python3 - "$(docker image inspect "${JUDGE_IMAGE}" --format '{{json .RootFS.Layers}}')" \
  "$(docker image inspect "${FORCED_IMAGE}" --format '{{json .RootFS.Layers}}')" <<'PY' \
  || die "${FORCED_IMAGE} is not derived from the current ${JUDGE_IMAGE}"
import json, sys
base, derived = json.loads(sys.argv[1]), json.loads(sys.argv[2])
raise SystemExit(0 if derived[: len(base)] == base and len(derived) > len(base) else 1)
PY

# --- archive (kernels/flash_kmeans_vNNN.py, FINAL_VERSION, notes.md) ---------
out_dir="${ARTIFACTS_DIR}/forced_oracle"
mkdir -p "${out_dir}"
stage="$(mktemp -d)"
tasks_dir="$(mktemp -d)"
trap 'rm -rf "${stage}" "${tasks_dir}"' EXIT
mkdir -p "${stage}/kernels"
cp "${REF_ROOT}/${selected}" "${stage}/${selected}"
cp "${REF_ROOT}/notes.md" "${stage}/notes.md"
printf '%d\n' "$((10#${version}))" >"${stage}/FINAL_VERSION"
archive="${out_dir}/official-oracle-fa3-cutedsl-v$(printf '%03d' "$((10#${version}))").tar.gz"
tar -C "${stage}" --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  --no-recursion -cf - FINAL_VERSION "${selected}" notes.md | gzip -n >"${archive}"
echo "archive ${archive} sha256=$(sha256sum "${archive}" | awk '{print $1}')"
echo "selected ${selected} sha256=$(sha256sum "${REF_ROOT}/${selected}" | awk '{print $1}')"

# What the REAL admission says (the gate the forced image bypasses), offline,
# with the unmodified :local Judge image.
chmod -R a+rX "${stage}"
original_admission="${archive%.tar.gz}.original_admission.json"
docker run --rm --network none --user 0:0 -v "${stage}:/submission:ro" --entrypoint sh "${JUDGE_IMAGE}" -c \
  'python3 /opt/flash_kmeans_benchmark/source_admission.py --root /submission --json-output /tmp/admission.json >/dev/null; cat /tmp/admission.json' \
  >"${original_admission}" || true
echo "original admission status=$(jq -r .status "${original_admission}") problems=$(jq -r '.problems | length' "${original_admission}")"

# --- scratch task definition: identical except judge.image_tag ---------------
cp "${TASKS_DIR}/BENCHMARK.yaml" "${tasks_dir}/BENCHMARK.yaml"
jq --arg tag "${FORCED_TAG}" '.judge.image_tag = $tag' \
  "${TASKS_DIR}/${TASK_ID}.json" >"${tasks_dir}/${TASK_ID}.json"
export SFORGE_TASKS_DIR="${tasks_dir}"

# --- Judge run, as scripts/eval_archive.sh -----------------------------------
judge_gpu="$(canonical_gpu_index "${gpu}")"
judge_uuid="$(gpu_uuid "${judge_gpu}")"
run_id="${run_id:-forced-flash-kmeans-v$(printf '%03d' "$((10#${version}))")-gpu${judge_gpu}-$(date -u +%Y%m%d-%H%M%S)}"
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe run id: ${run_id}"
[[ ! -e "${LOGS_DIR}/runs/${run_id}" ]] || die "run directory already exists: ${run_id}"

exec 9>"$(gpu_lock_path "${judge_uuid}")"
flock -n 9 || die "judge GPU ${judge_gpu} lock is already held"
assert_gpu_idle "${judge_gpu}" "judge"
assert_container_gpu "${FORCED_IMAGE}" "${judge_uuid}" "judge"

export SFORGE_LLM_JUDGE_MODE=off
export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${judge_uuid},CUDA_VISIBLE_DEVICES=0,FLASH_KMEANS_TARGET=H100"
if [[ -n "${cpuset}" ]]; then
  export SFORGE_JUDGE_CPUSET="${cpuset}"
fi
"${SFORGE_BIN}" \
  eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${archive}" \
  --run-id "${run_id}" \
  --judge-cpu-limit 8 \
  --judge-mem-limit 32g \
  --json
report_dir="${LOGS_DIR}/runs/${run_id}/${TASK_ID}/submissions/manual-1"
cp "${original_admission}" "${report_dir}/original_source_admission.json"
echo "report: ${report_dir}/report.json"
echo "private artifacts: ${report_dir}/private_judge_artifacts.tar.gz"
