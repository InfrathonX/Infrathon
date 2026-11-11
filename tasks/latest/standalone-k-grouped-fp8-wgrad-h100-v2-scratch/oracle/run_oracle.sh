#!/usr/bin/env bash
set -euo pipefail

# Host-side performance oracle for the K-grouped FP8 wgrad task.
#
# Builds an oracle-only image (base + upstream DeepGEMM + the trusted
# runtime) and measures, on one pinned H100, with the Judge's own protocol
# (runtime.run_suite over the frozen scored manifest: 1 warm-up, decoy
# pre-call, 5 fresh-tensor trials, max(host wall, CUDA event), median,
# two-family geomean, all correctness layers):
#
#   1. baseline-triton       the qualification control (baseline/candidates/kernel_v001)
#   2. deepgemm-abi-v001     upstream DeepGEMM, upstream test-generator repack glue
#   3. deepgemm-abi          upstream DeepGEMM, lean glue (deepgemm_oracle_v002) -- the oracle number
#   4. deepgemm-hoisted      the DeepGEMM call alone, K-major repack + ks D2H outside the timer
#   5. deepgemm-kernel-only  as 4, plus C->out outside the timer and d == c (upstream's bench condition)
#
# The private edge gate is deliberately not run: upstream's SM90 kernel does not support
# most edge shapes (FP32 D tensor map needs N % 4 == 0; tiny M/N fail JIT static asserts),
# and driving it through runtime.run_suite left a kernel spinning on the GPU for 20+ min
# (2026-10-29).  See oracle/README.md.
#
# Nothing here touches the Work image, the Judge image, the version store, or
# the recorded qualification.  See oracle/README.md.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASK_DIRECTORY="$(basename "${MIGRATION_ROOT}")"

ORACLE_IMAGE="${ORACLE_IMAGE:-standalone_k_grouped_wgrad.oracle:local}"
# Upstream DeepGEMM is a ~150 MB third-party tree that this repository does not
# vendor.  Provide a checkout pinned to the commit this task qualified against:
#
#   git clone --recursive https://github.com/deepseek-ai/DeepGEMM /path/to/DeepGEMM
#   git -C /path/to/DeepGEMM checkout 559d79fb6994a58b8a15b4b93bf13ccc16edf247
#   git -C /path/to/DeepGEMM submodule update --init --recursive
#   DEEPGEMM_SOURCE=/path/to/DeepGEMM oracle/run_oracle.sh --gpu 0
#
# Submodule pins: third-party/cutlass f3fde58372d33e9a5650ba7b80fc48b3b49d40c8,
# third-party/fmt 553ec11ec06fbe0beebfbb45f9dc3c9eabd83d28.  See oracle/README.md.
DEEPGEMM_SOURCE="${DEEPGEMM_SOURCE:-}"
DEEPGEMM_COMMIT=559d79fb6994a58b8a15b4b93bf13ccc16edf247
ASSETS_DIR="${INFRATHON_ASSETS_ROOT:-${HOME}/.cache/infrathon/assets}/${TASK_DIRECTORY}"
# Judge-only frozen inputs: the repository copy if present, else the fetched dataset asset.
default_input() {
  if [[ -f "${MIGRATION_ROOT}/judge/private/$1" ]]; then
    echo "${MIGRATION_ROOT}/judge/private/$1"
  else
    echo "${ASSETS_DIR}/judge/$1"
  fi
}
FINAL_CASES_PATH="${FINAL_CASES_PATH:-$(default_input final_cases.json)}"
OUTPUT_DIR="${OUTPUT_DIR:-${MIGRATION_ROOT}/oracle/results}"
# Optional host pinning of the measurement container (e.g. ORACLE_CPUSET=28-31,60-63).
ORACLE_CPUSET="${ORACLE_CPUSET:-}"
ORACLE_MEMS="${ORACLE_MEMS:-}"

gpu=""
trials="${TRIALS:-5}"
warmups="${WARMUPS:-1}"
skip_build=0

die() { echo "ERROR: $*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage: oracle/run_oracle.sh --gpu N [options]

  --gpu N          physical GPU index to pin (required)
  --trials N       measured trials per case (default 5, the Judge's value)
  --warmups N      warm-up invocations per case (default 1, the Judge's value)
  --skip-build     reuse an already built oracle image

Environment: DEEPGEMM_SOURCE (required), FINAL_CASES_PATH, OUTPUT_DIR,
ORACLE_IMAGE, ORACLE_CPUSET, ORACLE_MEMS.
USAGE
}

while (( $# )); do
  case "$1" in
    --gpu) [[ $# -ge 2 ]] || die "--gpu requires a value"; gpu="$2"; shift 2 ;;
    --trials) [[ $# -ge 2 ]] || die "--trials requires a value"; trials="$2"; shift 2 ;;
    --warmups) [[ $# -ge 2 ]] || die "--warmups requires a value"; warmups="$2"; shift 2 ;;
    --skip-build) skip_build=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown argument: $1" ;;
  esac
done

[[ "${gpu}" =~ ^[0-9]+$ ]] || { usage >&2; die "--gpu is required"; }
command -v docker >/dev/null || die "docker is required"
command -v jq >/dev/null || die "jq is required"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"
[[ -n "${DEEPGEMM_SOURCE}" ]] \
  || die "set DEEPGEMM_SOURCE to a DeepGEMM checkout pinned at ${DEEPGEMM_COMMIT} (see the header of this script)"
[[ -d "${DEEPGEMM_SOURCE}" ]] || die "DeepGEMM checkout not found: ${DEEPGEMM_SOURCE}"
[[ -d "${DEEPGEMM_SOURCE}/third-party/cutlass/include" ]] \
  || die "DeepGEMM snapshot has no CUTLASS submodule: ${DEEPGEMM_SOURCE}"
if git -C "${DEEPGEMM_SOURCE}" rev-parse HEAD >/dev/null 2>&1; then
  [[ "$(git -C "${DEEPGEMM_SOURCE}" rev-parse HEAD)" == "${DEEPGEMM_COMMIT}" ]] \
    || die "DeepGEMM checkout is not at ${DEEPGEMM_COMMIT}"
fi

# The frozen inputs must be the ones assets/manifest.json pins.
check_asset() {  # path asset-relative-path
  local want
  want="$(jq -r --arg p "$2" '.assets[] | select(.path == $p) | .sha256' "${MIGRATION_ROOT}/assets/manifest.json")"
  [[ -f "$1" ]] || die "Judge-only input not found: $1 (run assets/fetch.sh or set the *_PATH variable)"
  [[ "$(sha256sum "$1" | awk '{print $1}')" == "${want}" ]] || die "$1 does not match assets/manifest.json ($2)"
}
check_asset "${FINAL_CASES_PATH}" judge/final_cases.json

gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
  | awk -F', *' -v want="${gpu}" '$1 == want {print $2}')"
[[ -n "${gpu_uuid}" ]] || die "GPU ${gpu} does not exist"
gpu_used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', *' -v want="${gpu}" '$1 == want {print $2}')"
(( gpu_used < 1024 )) || die "GPU ${gpu} is busy (${gpu_used} MiB in use)"

mkdir -p "${OUTPUT_DIR}"
manifest="${OUTPUT_DIR}/.final_cases.json"
install -m 0600 "${FINAL_CASES_PATH}" "${manifest}"
trap 'rm -f "${manifest}"' EXIT

if (( ! skip_build )); then
  echo "Building ${ORACLE_IMAGE} (DeepGEMM from ${DEEPGEMM_SOURCE})"
  docker build \
    --tag "${ORACLE_IMAGE}" \
    --network host \
    --build-arg "http_proxy=${http_proxy:-}" \
    --build-arg "https_proxy=${https_proxy:-}" \
    --build-arg "no_proxy=${no_proxy:-}" \
    --build-arg "HTTP_PROXY=${HTTP_PROXY:-}" \
    --build-arg "HTTPS_PROXY=${HTTPS_PROXY:-}" \
    --build-arg "NO_PROXY=${NO_PROXY:-}" \
    --build-context "deepgemm=${DEEPGEMM_SOURCE}" \
    --build-context "task_runtime=${MIGRATION_ROOT}/runtime" \
    --build-context "task_baseline=${MIGRATION_ROOT}/baseline" \
    --file "${SCRIPT_DIR}/Dockerfile" \
    "${SCRIPT_DIR}" \
    2>&1 | tail -20
fi

cpuset_args=()
if [[ -n "${ORACLE_CPUSET}" ]]; then
  cpuset_args+=(--cpuset-cpus "${ORACLE_CPUSET}")
  [[ -n "${ORACLE_MEMS}" ]] && cpuset_args+=(--cpuset-mems "${ORACLE_MEMS}")
fi

labels=()
run_one() {
  local label="$1"
  shift
  echo
  echo "=== ${label} (GPU ${gpu}) ==="
  docker run --rm \
    --name "kgrouped-oracle-${label}-$$" \
    -e "NVIDIA_VISIBLE_DEVICES=${gpu_uuid}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTHONDONTWRITEBYTECODE=1 \
    --cpus 8 \
    "${cpuset_args[@]}" \
    --memory 32g \
    -v "${manifest}:/manifest/final_cases.json:ro" \
    -v "${OUTPUT_DIR}:/results" \
    "${ORACLE_IMAGE}" \
    python /opt/oracle/oracle_runner.py \
      --label "${label}" \
      --manifest /manifest/final_cases.json \
      --trials "${trials}" \
      --warmups "${warmups}" \
      --output "/results/${label}.json" \
      "$@"
  labels+=("${label}")
}

C=/opt/oracle/candidates
run_one baseline-triton      --mode suite --candidate "${C}/baseline_triton_v001/submission.py"
run_one deepgemm-abi-v001    --mode suite --candidate "${C}/deepgemm_oracle_v001/submission.py"
run_one deepgemm-abi         --mode suite --candidate "${C}/deepgemm_oracle_v002/submission.py"
run_one deepgemm-hoisted     --mode hoisted
run_one deepgemm-kernel-only --mode hoisted --inplace

echo
echo "=== summary (scored grid, GPU ${gpu}) ==="
summary_files=()
for label in "${labels[@]}"; do summary_files+=("${OUTPUT_DIR}/${label}.json"); done
jq -s -r '
  (["implementation","status","score_tflops","coarse","fine"] | @tsv),
  (.[] | .summary | [.label, .status,
          ((.score_tflops // 0) * 100 | round / 100),
          ((.coarse_tflops // 0) * 100 | round / 100),
          ((.fine_tflops // 0) * 100 | round / 100)] | @tsv)
' "${summary_files[@]}" | column -t
