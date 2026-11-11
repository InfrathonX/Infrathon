#!/usr/bin/env bash
set -euo pipefail

# Host-only paper reference for the SageAttention task: upstream SageAttention2
# sm90 (wgmma) CUDA path scored with the Judge's own trusted runtime
# (quant_attention_runtime.run_suite: 8 edge cases, 62 real layers, the frozen
# final-holdout-v2 cases, 1 warm-up + 7 fresh-input trials with decoy,
# max(host, CUDA event), median, family geomeans, same-card cuDNN BF16
# reference).  It is NOT an admissible submission (it imports a compiled
# extension) and never touches the Work image, the Judge image, the version
# store or the scored oracle archive.  See oracle/paper_reference/README.md.
#
#   git clone https://github.com/thu-ml/SageAttention /path/to/SageAttention
#   git -C /path/to/SageAttention checkout d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
#   SAGE_SOURCE=/path/to/SageAttention oracle/paper_reference/run_paper_reference.sh --gpu 0
#
# Requires the private Judge image (scripts/build_images.sh) because the
# harness reads the frozen holdout and the real layers from it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../../scripts/lib.sh
source "${SCRIPT_DIR}/../../scripts/lib.sh"

PIN=d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5
HARNESS_IMAGE="${HARNESS_IMAGE:-sageattention-paper-sm90-harness:${PIN:0:7}}"
SAGE_SOURCE="${SAGE_SOURCE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARTIFACTS_DIR}/paper_reference}"

gpu=""
cpuset="${PAPER_REFERENCE_CPUSET:-}"
grans="per_warp per_thread"
skip_build=0
kernel_only=0

usage() {
  cat <<'USAGE'
Usage: oracle/paper_reference/run_paper_reference.sh --gpu N [options]

  --gpu N            physical GPU index to pin (required; must be idle)
  --cpuset LIST      pin the harness container, e.g. 0-7 (memory bound to its nodes)
  --gran LIST        QK quantizer granularities to score (default "per_warp per_thread")
  --kernel-only      also run the kernel-only timing (NOT a Judge-protocol score)
  --skip-build       reuse an already built harness image
USAGE
}

while (( $# )); do
  case "$1" in
    --gpu) [[ $# -ge 2 ]] || die "--gpu requires a value"; gpu="$2"; shift 2 ;;
    --cpuset) [[ $# -ge 2 ]] || die "--cpuset requires a value"; cpuset="$2"; shift 2 ;;
    --gran) [[ $# -ge 2 ]] || die "--gran requires a value"; grans="$2"; shift 2 ;;
    --kernel-only) kernel_only=1; shift ;;
    --skip-build) skip_build=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

[[ "${gpu}" =~ ^[0-9]+$ ]] || { usage >&2; die "--gpu is required"; }
for gran in ${grans}; do
  [[ "${gran}" == per_warp || "${gran}" == per_thread ]] || die "unknown granularity: ${gran}"
done
require_command docker
require_command nvidia-smi
ensure_docker_group "$@"
docker image inspect "${JUDGE_IMAGE}" >/dev/null 2>&1 \
  || die "missing private Judge image ${JUDGE_IMAGE}; run scripts/build_images.sh"

if (( ! skip_build )); then
  [[ -n "${SAGE_SOURCE}" && -d "${SAGE_SOURCE}" ]] \
    || die "set SAGE_SOURCE to a SageAttention checkout at ${PIN} (see the header of this script)"
  if [[ -d "${SAGE_SOURCE}/.git" ]]; then
    head="$(git -C "${SAGE_SOURCE}" rev-parse HEAD)"
    [[ "${head}" == "${PIN}" ]] || die "SAGE_SOURCE is at ${head}, expected ${PIN}"
    [[ -z "$(git -C "${SAGE_SOURCE}" status --porcelain --untracked-files=no)" ]] \
      || die "SAGE_SOURCE has local modifications"
  fi
  echo "Building ${HARNESS_IMAGE} (SageAttention from ${SAGE_SOURCE}, sm_90a)"
  docker build \
    --tag "${HARNESS_IMAGE}" \
    --build-arg "JUDGE_IMAGE=${JUDGE_IMAGE}" \
    ${MAX_JOBS:+--build-arg "MAX_JOBS=${MAX_JOBS}"} \
    ${EXT_PARALLEL:+--build-arg "EXT_PARALLEL=${EXT_PARALLEL}"} \
    ${NVCC_THREADS:+--build-arg "NVCC_THREADS=${NVCC_THREADS}"} \
    --build-context "sage_src=${SAGE_SOURCE}" \
    --file "${SCRIPT_DIR}/Dockerfile" \
    "${SCRIPT_DIR}"
fi

physical_gpu="$(canonical_gpu_index "${gpu}")"
physical_uuid="$(gpu_uuid "${physical_gpu}")"
pin_args=(--cpus 8)
if [[ -n "${cpuset}" ]]; then
  pin_args+=(--cpuset-cpus "${cpuset}")
fi
mkdir -p "${OUTPUT_DIR}"
chmod 0777 "${OUTPUT_DIR}"
stamp="$(date -u +%Y%m%d-%H%M%S)"

# Hold the same per-GPU lock the Judge takes, so no Judge shares the card.
exec 8>"$(gpu_lock_path "${physical_uuid}")"
flock -n 8 || die "GPU ${physical_gpu} lock is already held"
assert_gpu_idle "${physical_gpu}" "paper reference"

status=0
for gran in ${grans}; do
  out="paper_sm90_${gran}_gpu${physical_gpu}_${stamp}.json"
  echo "=== sm90 ${gran} -> ${OUTPUT_DIR}/${out}"
  docker run --rm \
    --gpus "device=${physical_uuid}" "${pin_args[@]}" --memory 32g --shm-size 8g \
    -e PAPER_SM90_QK_GRAN="${gran}" -e PYTHONDONTWRITEBYTECODE=1 \
    -e TRITON_CACHE_DIR=/tmp/triton -v "${OUTPUT_DIR}:/out" "${HARNESS_IMAGE}" \
    python -I /harness/run_paper_sm90.py --candidate /harness/paper_sm90_candidate.py \
      --output "/out/${out}" \
    || { echo "sm90 ${gran}: gate failure or error (rc=$?), see ${out}" >&2; status=2; }
done
if (( kernel_only )); then
  out="paper_sm90_kernel_only_gpu${physical_gpu}_${stamp}.json"
  docker run --rm \
    --gpus "device=${physical_uuid}" "${pin_args[@]}" --memory 32g --shm-size 8g \
    -e PYTHONDONTWRITEBYTECODE=1 -e TRITON_CACHE_DIR=/tmp/triton -v "${OUTPUT_DIR}:/out" \
    "${HARNESS_IMAGE}" python -I /harness/kernel_only_bench.py --output "/out/${out}" \
    || { echo "kernel-only bench failed (rc=$?)" >&2; status=2; }
fi
assert_gpu_released "${physical_uuid}"
exit "${status}"
