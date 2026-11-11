#!/usr/bin/env bash
set -euo pipefail

# Paper reference for the dynamic grouped FP8 task (host-only; see README.md here).
#
# Builds an image = the task's Judge image + upstream DeepGEMM 559d79f (v2.6.1) and
# measures, on one pinned H100, with the Judge's trusted runtime and the frozen
# final-holdout-v3 seeds:
#
#   deepgemm-abi-suite       upstream DeepGEMM under the task ABI, Judge protocol (the paper number)
#   deepgemm-abi-suite-edge  same, with the 158 private edge cases first (does upstream pass the gate?)
#   deepgemm-hoisted         SFA transpose hoisted out of the timer (DeepEP-layout upper bound)
#   deepgemm-kernel          GPU time only (bench_kineto analogue), transpose hoisted
#   oracle-triton-suite      the scored oracle (../candidates/kernel_v001) in the same container
#   oracle-triton-kernel     its GPU time only
#   deepgemm-abi-suite-r2    the paper number again, after the oracle (order/thermal check)
#
#   GPU=<idx> DEEPGEMM_SOURCE=/path/to/DeepGEMM oracle/paper_reference/run_paper_reference.sh
#   GPU=<idx> SKIP_BUILD=1 ... (reuse the image)
#
# Nothing here touches the Work image, the Judge image, the version store or the scored
# oracle.  It does not take the repository's GPU lock; hold the card's lock yourself
# (e.g. `flock /path/gpu<idx>.lock oracle/paper_reference/run_paper_reference.sh`).

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORACLE_ROOT="$(cd "${HERE}/.." && pwd)"
MIGRATION_ROOT="$(cd "${ORACLE_ROOT}/.." && pwd)"
TASK_DIRECTORY="$(basename "${MIGRATION_ROOT}")"

die() { echo "ERROR: $*" >&2; exit 1; }

: "${GPU:?set GPU=<physical GPU index>}"
DEEPGEMM_SOURCE="${DEEPGEMM_SOURCE:?set DEEPGEMM_SOURCE to a DeepGEMM checkout at 559d79f with submodules (see README.md)}"
DG_REV=559d79fb6994a58b8a15b4b93bf13ccc16edf247
ASSETS_DIR="${INFRATHON_ASSETS_ROOT:-${HOME}/.cache/infrathon/assets}/${TASK_DIRECTORY}"
default_input() {  # repository copy if present, else the fetched dataset asset
  if [[ -f "${MIGRATION_ROOT}/judge/private/$1" ]]; then
    echo "${MIGRATION_ROOT}/judge/private/$1"
  else
    echo "${ASSETS_DIR}/judge/$1"
  fi
}
FINAL_CASES="${FINAL_CASES:-$(default_input final_cases.json)}"
EDGE_CASES="${EDGE_CASES:-$(default_input edge_cases.json)}"
# Base = the task's Judge image (built or pulled by scripts/build_images.sh).
JUDGE_IMAGE="${JUDGE_IMAGE:-standalone_grouped_fp8.judge.standalone_dynamic_grouped_fp8_h100_scratch_v3:local}"
IMAGE="${IMAGE:-standalone_grouped_fp8.paper-reference:local}"
RESULTS="${RESULTS:-${HERE}/results}"
JIT_CACHE="${JIT_CACHE:-${RESULTS}/.deepgemm-jit}"
CPUSET="${HARNESS_CPUSET:-}"
MEMS="${HARNESS_MEMS:-}"

command -v docker >/dev/null || die "docker is required"
command -v jq >/dev/null || die "jq is required"
command -v nvidia-smi >/dev/null || die "nvidia-smi is required"

check_asset() {  # path asset-relative-path
  local want
  want="$(jq -r --arg p "$2" '.assets[] | select(.path == $p) | .sha256' "${MIGRATION_ROOT}/assets/manifest.json")"
  [[ -f "$1" ]] || die "Judge-only input not found: $1 (run assets/fetch.sh or set FINAL_CASES/EDGE_CASES)"
  [[ "$(sha256sum "$1" | awk '{print $1}')" == "${want}" ]] || die "$1 does not match assets/manifest.json ($2)"
}
check_asset "${FINAL_CASES}" judge/final_cases.json
check_asset "${EDGE_CASES}" judge/edge_cases.json
[[ -d "${DEEPGEMM_SOURCE}/third-party/cutlass/include" ]] || die "init DeepGEMM submodules in ${DEEPGEMM_SOURCE}"
if git -C "${DEEPGEMM_SOURCE}" rev-parse HEAD >/dev/null 2>&1; then
  [[ "$(git -C "${DEEPGEMM_SOURCE}" rev-parse HEAD)" == "${DG_REV}" ]] || die "DeepGEMM not at ${DG_REV}"
fi

uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', *' -v g="${GPU}" '$1==g{print $2}')"
used="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', *' -v g="${GPU}" '$1==g{print $2}')"
[[ -n "${uuid}" ]] || die "no GPU ${GPU}"
(( used < 1024 )) || die "GPU ${GPU} busy (${used} MiB)"

mkdir -p "${RESULTS}" "${JIT_CACHE}"
install -m 0600 "${FINAL_CASES}" "${RESULTS}/.final_cases.json"
install -m 0600 "${EDGE_CASES}" "${RESULTS}/.edge_cases.json"
trap 'rm -f "${RESULTS}/.final_cases.json" "${RESULTS}/.edge_cases.json"' EXIT

if [[ "${SKIP_BUILD:-0}" != 1 ]]; then
  docker image inspect "${JUDGE_IMAGE}" >/dev/null 2>&1 \
    || die "base Judge image ${JUDGE_IMAGE} not found (scripts/build_images.sh builds or pulls it)"
  docker build --network host --build-arg "JUDGE_IMAGE=${JUDGE_IMAGE}" \
    --build-arg "http_proxy=${http_proxy:-}" --build-arg "https_proxy=${https_proxy:-}" \
    --build-context "deepgemm=${DEEPGEMM_SOURCE}" -t "${IMAGE}" -f "${HERE}/Dockerfile" "${HERE}" \
    2>&1 | tail -20
fi

cpu_args=(--cpus 8)
if [[ -n "${CPUSET}" ]]; then
  cpu_args+=(--cpuset-cpus "${CPUSET}")
  [[ -n "${MEMS}" ]] && cpu_args+=(--cpuset-mems "${MEMS}")
fi

labels=()
run() {  # label, extra args...
  local label="$1"; shift
  echo "=== ${label} (GPU ${GPU} ${uuid})"
  docker run --rm --gpus "device=${uuid}" -e CUDA_VISIBLE_DEVICES=0 "${cpu_args[@]}" --memory 32g \
    -v "${RESULTS}/.final_cases.json:/cases/final_cases.json:ro" \
    -v "${RESULTS}/.edge_cases.json:/cases/edge_cases.json:ro" \
    -v "${HERE}/candidates:/subs/paper:ro" \
    -v "${ORACLE_ROOT}/candidates:/subs/oracle:ro" \
    -v "${RESULTS}:/results" -v "${JIT_CACHE}:/tmp/deepgemm-jit" \
    "${IMAGE}" python /opt/paper_harness/oracle_runner.py --label "${label}" \
      --final-cases /cases/final_cases.json --output "/results/${label}.json" "$@"
  labels+=("${label}")
}

# DeepGEMM JITs one cubin per (G, N, K, config) with nvcc on first use; the warm-up /
# decoy calls absorb it, and the persistent JIT cache makes reruns fast.  The edge run
# (158 shapes) may spend several minutes compiling on a cold cache.
P=/subs/paper/deepgemm_v001/submission.py
# the admissible Triton oracle (oracle/FINAL_VERSION selects the forced DeepGEMM candidate)
selector="${ORACLE_ROOT}/ADMISSIBLE_VERSION"
[[ -f "${selector}" ]] || selector="${ORACLE_ROOT}/FINAL_VERSION"
version="$(tr -d '[:space:]' <"${selector}")"
printf -v T '/subs/oracle/kernel_v%03d/submission.py' "$((10#${version}))"
run deepgemm-abi-suite      --mode suite --candidate "$P"
run deepgemm-abi-suite-edge --mode suite --candidate "$P" --edge-cases /cases/edge_cases.json
run deepgemm-hoisted        --mode hoisted
run deepgemm-kernel         --mode kernel --hoisted-deepgemm
run oracle-triton-suite     --mode suite --candidate "$T"
run oracle-triton-kernel    --mode kernel --candidate "$T"
run deepgemm-abi-suite-r2   --mode suite --candidate "$P"

echo
echo "=== summary (scored grid, GPU ${GPU}) ==="
for label in "${labels[@]}"; do
  jq -r '[.label, .mode // "suite", (.status // "-"), ((.primary_tflops // 0) * 100 | round / 100),
          ((.family_tflops.masked // 0) * 100 | round / 100), ((.family_tflops.psum // 0) * 100 | round / 100)] | @tsv' \
    "${RESULTS}/${label}.json"
done | (echo -e "label\tmode\tstatus\tscore_tflops\tmasked\tpsum"; cat) | column -t
echo "results in ${RESULTS}"
