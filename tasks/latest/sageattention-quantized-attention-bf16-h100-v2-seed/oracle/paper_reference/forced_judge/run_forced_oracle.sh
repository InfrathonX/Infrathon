#!/usr/bin/env bash
set -euo pipefail

# FORCED ORACLE: score the non-admissible upstream SageAttention2 sm90 CUDA path
# with the task's real Judge pipeline (sforge eval -> run_judge.py ->
# verify_quant_attention.py -> candidate_worker.py -> quant_attention_runtime)
# in a LOCAL-ONLY derived Judge image whose source admission is bypassed.
#
# This replicates scripts/eval_archive.sh exactly (same sforge eval command,
# timeout, CPU/memory limits, network policy, per-GPU Judge lock, idle and
# release barriers, NVIDIA_VISIBLE_DEVICES binding) and substitutes only the
# Judge image: a scratch copy of the task JSON selects image tag
# `forced-oracle` instead of `local`.  The LLM integrity review is off
# (SFORGE_LLM_JUDGE_MODE=off), as for every trusted control.
#
#   SAGE_SOURCE=/path/to/SageAttention@d1a57a5 \
#     oracle/paper_reference/forced_judge/run_forced_oracle.sh --gpu 2 --cpuset 8-11,40-43 --version 4
#
# Never retags or overwrites the :local images and never pushes anything.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFORGE_ENTRY_SCRIPT="${SCRIPT_DIR}/run_forced_oracle.sh"
# shellcheck source=../../../scripts/lib.sh
source "${SCRIPT_DIR}/../../../scripts/lib.sh"

PIN=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
PAPER_IMAGE="${PAPER_IMAGE:-sageattention-paper-sm90-harness:${PIN:0:7}}"
FORCED_TAG="${FORCED_TAG:-forced-oracle}"
FORCED_IMAGE="${BENCHMARK_NAME}.judge.${TASK_ID}:${FORCED_TAG}"
SAGE_SOURCE="${SAGE_SOURCE:-}"
ORACLE_ROOT="${MIGRATION_ROOT}/oracle"
PAPER_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

gpu=""
cpuset="${SFORGE_JUDGE_CPUSET:-}"
version=""
run_id=""
skip_build=0
rebuild_paper=0

usage() {
  cat <<'USAGE'
Usage: oracle/paper_reference/forced_judge/run_forced_oracle.sh --gpu N [options]

  --gpu N           physical GPU index (required; must be idle)
  --cpuset LIST     Judge container cpuset (SFORGE_JUDGE_CPUSET), e.g. 8-11,40-43
  --version V       oracle candidate to judge (default: oracle/FINAL_VERSION)
  --run-id ID       sforge run id (default forced-sageattention-v<V>-gpu<N>-<UTC stamp>)
  --skip-build      reuse an existing forced-oracle image
  --rebuild-paper   rebuild the paper-reference base image from SAGE_SOURCE first
Env: SAGE_SOURCE (upstream checkout at the pin; needed only to build the base),
     PAPER_IMAGE, FORCED_TAG, LOGS_DIR, SFORGE_BIN, FINAL_EVAL_TIMEOUT.
USAGE
}

while (( $# )); do
  case "$1" in
    --gpu) [[ $# -ge 2 ]] || die "--gpu requires a value"; gpu="$2"; shift 2 ;;
    --cpuset) [[ $# -ge 2 ]] || die "--cpuset requires a value"; cpuset="$2"; shift 2 ;;
    --version) [[ $# -ge 2 ]] || die "--version requires a value"; version="$2"; shift 2 ;;
    --run-id) [[ $# -ge 2 ]] || die "--run-id requires a value"; run_id="$2"; shift 2 ;;
    --skip-build) skip_build=1; shift ;;
    --rebuild-paper) rebuild_paper=1; shift ;;
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

version="${version:-$(tr -d '[:space:]' <"${ORACLE_ROOT}/FINAL_VERSION")}"
[[ "${version}" =~ ^[0-9]{1,3}$ && "$((10#${version}))" -ge 1 ]] || die "bad --version ${version}"
printf -v selected 'candidates/kernel_v%03d/submission.py' "$((10#${version}))"
[[ -f "${ORACLE_ROOT}/${selected}" && ! -L "${ORACLE_ROOT}/${selected}" ]] \
  || die "missing oracle candidate ${selected}"

# --- images (CPU only) -------------------------------------------------------
if (( rebuild_paper )) || ! docker image inspect "${PAPER_IMAGE}" >/dev/null 2>&1; then
  [[ -n "${SAGE_SOURCE}" && -d "${SAGE_SOURCE}" ]] \
    || die "set SAGE_SOURCE to a SageAttention checkout at ${PIN} to build ${PAPER_IMAGE}"
  if [[ -d "${SAGE_SOURCE}/.git" ]]; then
    [[ "$(git -C "${SAGE_SOURCE}" rev-parse HEAD)" == "${PIN}" ]] || die "SAGE_SOURCE is not at ${PIN}"
  fi
  docker build --tag "${PAPER_IMAGE}" \
    --build-arg "JUDGE_IMAGE=${JUDGE_IMAGE}" \
    ${MAX_JOBS:+--build-arg "MAX_JOBS=${MAX_JOBS}"} \
    ${EXT_PARALLEL:+--build-arg "EXT_PARALLEL=${EXT_PARALLEL}"} \
    ${NVCC_THREADS:+--build-arg "NVCC_THREADS=${NVCC_THREADS}"} \
    --build-context "sage_src=${SAGE_SOURCE}" \
    --file "${PAPER_DIR}/Dockerfile" "${PAPER_DIR}"
fi
if (( ! skip_build )) || ! docker image inspect "${FORCED_IMAGE}" >/dev/null 2>&1; then
  docker build --tag "${FORCED_IMAGE}" \
    --build-arg "PAPER_IMAGE=${PAPER_IMAGE}" \
    --file "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"
fi
# The forced image must sit on top of the current private Judge image.
python3 - "$(docker image inspect "${JUDGE_IMAGE}" --format '{{json .RootFS.Layers}}')" \
  "$(docker image inspect "${FORCED_IMAGE}" --format '{{json .RootFS.Layers}}')" <<'PY' \
  || die "${FORCED_IMAGE} is not derived from the current ${JUDGE_IMAGE}"
import json, sys
base, derived = json.loads(sys.argv[1]), json.loads(sys.argv[2])
raise SystemExit(0 if derived[: len(base)] == base and len(derived) > len(base) else 1)
PY

# --- archive (same layout as scripts/make_oracle_archive.sh) -----------------
out_dir="${ARTIFACTS_DIR}/forced_oracle"
mkdir -p "${out_dir}"
stage="$(mktemp -d)"
tasks_dir="$(mktemp -d)"
trap 'rm -rf "${stage}" "${tasks_dir}"' EXIT
mkdir -p "${stage}/$(dirname "${selected}")"
cp "${ORACLE_ROOT}/notes.md" "${stage}/notes.md"
cp "${ORACLE_ROOT}/${selected}" "${stage}/${selected}"
printf '%d\n' "$((10#${version}))" >"${stage}/FINAL_VERSION"
archive="${out_dir}/oracle_v$(printf '%03d' "$((10#${version}))").tar.gz"
tar -C "${stage}" --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  --no-recursion -cf - FINAL_VERSION notes.md "${selected}" | gzip -n >"${archive}"
echo "archive ${archive} sha256=$(sha256sum "${archive}" | awk '{print $1}')"
echo "selected ${selected} sha256=$(sha256sum "${ORACLE_ROOT}/${selected}" | awk '{print $1}')"

# Record what the REAL admission says about this archive (the gate the forced
# image bypasses), using the unmodified :local Judge image, offline.
chmod -R a+rX "${stage}"
original_admission="${out_dir}/oracle_v$(printf '%03d' "$((10#${version}))").original_admission.json"
docker run --rm --network none -v "${stage}:/submission:ro" --entrypoint python3 "${JUDGE_IMAGE}" \
  /opt/sageattention_quant_attention_judge/source_admission.py --root /submission \
  >"${original_admission}" || true
echo "original admission ok=$(jq -r .ok "${original_admission}") errors=$(jq -r '.errors | length' "${original_admission}") -> ${original_admission}"

# --- scratch task definition: identical except judge.image_tag ---------------
cp "${TASKS_DIR}/BENCHMARK.yaml" "${tasks_dir}/BENCHMARK.yaml"
jq --arg tag "${FORCED_TAG}" '.judge.image_tag = $tag' \
  "${TASKS_DIR}/${TASK_ID}.json" >"${tasks_dir}/${TASK_ID}.json"
export SFORGE_TASKS_DIR="${tasks_dir}"

# --- Judge run, as scripts/eval_archive.sh -----------------------------------
physical_gpu="$(canonical_gpu_index "${gpu}")"
physical_uuid="$(gpu_uuid "${physical_gpu}")"
run_id="${run_id:-forced-sageattention-v$(printf '%03d' "$((10#${version}))")-gpu${physical_gpu}-$(date -u +%Y%m%d-%H%M%S)}"
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe run id: ${run_id}"
[[ ! -e "${LOGS_DIR}/runs/${run_id}" ]] || die "run directory already exists: ${run_id}"

exec 8>"$(gpu_lock_path "${physical_uuid}")"
flock -n 8 || die "GPU ${physical_gpu} lock is already held"
assert_gpu_idle "${physical_gpu}" "judge"
assert_container_gpu "${FORCED_IMAGE}" "${physical_uuid}" "judge"

export SFORGE_LLM_JUDGE_MODE=off
export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${physical_uuid},CUDA_VISIBLE_DEVICES=0"
if [[ -n "${cpuset}" ]]; then
  export SFORGE_JUDGE_CPUSET="${cpuset}"
fi
"${SFORGE_BIN}" eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${archive}" \
  --run-id "${run_id}" \
  --timeout "${FINAL_EVAL_TIMEOUT:-3600}" \
  --judge-cpu-limit 8 \
  --judge-mem-limit 32g \
  --json
assert_gpu_released "${physical_uuid}"
report_dir="${LOGS_DIR}/runs/${run_id}/${TASK_ID}/submissions/manual-1"
cp "${original_admission}" "${report_dir}/original_source_admission.json"
echo "report: ${report_dir}/report.json"
echo "private artifacts: ${report_dir}/private_judge_artifacts.tar.gz"
