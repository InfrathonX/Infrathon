#!/usr/bin/env bash
# Host-only paper reference: time upstream FlashMLA (SM90 sparse prefill,
# ba89a34) on the task ABI with the Judge's own runtime.  See README.md here.
#
#   oracle/paper_reference/run_flashmla_reference.sh --gpu N [--skip-build] [--build-only] [--peaked]
#
# Needs FLASHMLA_SOURCE (a FlashMLA checkout at
# ba89a3466e9470ad08ab39738d4e7bb66989e1e7 with csrc/cutlass at
# 147f5673d0c1c3dcf66f78d677fd647e4a020219) unless --skip-build reuses an
# existing image.  Uses the Judge-only case manifests that scripts/lib.sh
# resolves (FINAL_CASES_PATH / EDGE_CASES_PATH).  Nothing here touches the Work
# image, the Judge image or a run's version store.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../../scripts/lib.sh"
require_command docker
require_command flock
require_command nvidia-smi

PAPER_IMAGE="${PAPER_IMAGE:-standalone_sparse_mla_prefill.paper-oracle-flashmla:local}"
FLASHMLA_SOURCE="${FLASHMLA_SOURCE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${HERE}/results}"
FLASHMLA_COMMIT=ba89a3466e9470ad08ab39738d4e7bb66989e1e7
gpu=""
skip_build=0
build_only=0
peaked=0
while (( $# )); do
  case "$1" in
    --gpu) [[ $# -ge 2 ]] || die "--gpu requires a value"; gpu="$2"; shift 2 ;;
    --skip-build) skip_build=1; shift ;;
    --build-only) build_only=1; shift ;;
    --peaked) peaked=1; shift ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

if (( ! skip_build )); then
  [[ -n "${FLASHMLA_SOURCE}" && -d "${FLASHMLA_SOURCE}/csrc" ]] \
    || die "set FLASHMLA_SOURCE to a FlashMLA checkout at ${FLASHMLA_COMMIT} (see README.md)"
  [[ -f "${FLASHMLA_SOURCE}/csrc/cutlass/include/cute/tensor.hpp" ]] \
    || die "${FLASHMLA_SOURCE} has no csrc/cutlass submodule"
  if [[ -e "${FLASHMLA_SOURCE}/.git" ]]; then
    [[ "$(git -C "${FLASHMLA_SOURCE}" rev-parse HEAD)" == "${FLASHMLA_COMMIT}" ]] \
      || die "${FLASHMLA_SOURCE} is not at ${FLASHMLA_COMMIT}"
  fi
  # CPU-only build (nvcc for sm_90a), 30-60 min with 8 jobs; no GPU is attached.
  docker build --network host --tag "${PAPER_IMAGE}" \
    --build-context "flashmla=${FLASHMLA_SOURCE}" \
    --build-context "task_runtime=${MIGRATION_ROOT}/runtime" \
    --file "${HERE}/Dockerfile" "${HERE}"
fi
(( build_only )) && { echo "image ${PAPER_IMAGE} built"; exit 0; }
docker image inspect "${PAPER_IMAGE}" >/dev/null 2>&1 || die "missing image ${PAPER_IMAGE}"
[[ "${gpu}" =~ ^[0-9]+$ ]] || die "--gpu N is required"

final_cases="${FINAL_CASES_PATH:-${DEFAULT_FINAL_CASES_PATH}}"
edge_cases="${EDGE_CASES_PATH:-${DEFAULT_EDGE_CASES_PATH}}"
[[ -f "${final_cases}" ]] || die "final holdout manifest not found: ${final_cases}"
[[ -f "${edge_cases}" ]] || die "edge holdout manifest not found: ${edge_cases}"
echo "fde2b24100361d8c3d7a23e19c2543274427550697fa3dcfa95fa38fe2d05579  ${final_cases}" | sha256sum -c - >/dev/null \
  || die "final_cases.json is not the frozen final holdout"
echo "882c7dba13c15786209a3f2614de7cef5bd2a1e88d7b5d537a323d82478c2a9f  ${edge_cases}" | sha256sum -c - >/dev/null \
  || die "edge_cases.json is not the frozen edge holdout"

uuid="$(gpu_uuid "$(canonical_gpu_index "${gpu}")")"
exec 8>"$(gpu_lock_path "${uuid}")"
flock -n 8 || die "GPU ${gpu} lock is already held"
assert_gpu_idle "${gpu}" "FlashMLA paper reference"

mkdir -p "${OUTPUT_DIR}"
cases_dir="$(mktemp -d)"
trap 'rm -rf "${cases_dir}"' EXIT
install -m 0644 "${final_cases}" "${cases_dir}/final_cases.json"
install -m 0644 "${edge_cases}" "${cases_dir}/edge_cases.json"

run() {  # label candidate mode [runner args...]
  local label="$1" candidate="$2" mode="$3"
  shift 3
  echo "=== ${label} (${mode}) on GPU ${gpu} ==="
  docker run --rm --name "smla-paper-ref-${label}-$$" --network none \
    --gpus "device=${uuid}" -e CUDA_VISIBLE_DEVICES=0 -e PYTHONDONTWRITEBYTECODE=1 \
    --cpus 8 ${PAPER_CPUSET:+--cpuset-cpus "${PAPER_CPUSET}"} --memory 32g --shm-size 2g \
    -v "${MIGRATION_ROOT}/runtime/sparse_mla_prefill_runtime.py:/opt/oracle/sparse_mla_prefill_runtime.py:ro" \
    -v "${HERE}/oracle_runner.py:/opt/oracle/oracle_runner.py:ro" \
    -v "${HERE}/candidates:/opt/paper:ro" \
    -v "${MIGRATION_ROOT}/oracle/candidates:/opt/task_oracle:ro" \
    -v "${cases_dir}:/m:ro" -v "${OUTPUT_DIR}:/results" \
    --entrypoint python "${PAPER_IMAGE}" -I /opt/oracle/oracle_runner.py \
      --label "${label}" --candidate "${candidate}" --scored /m/final_cases.json \
      --mode "${mode}" --output "/results/${label}-${mode}.json" "$@"
  sleep "${COOLDOWN_SECONDS:-60}"  # power-capped H100 NVL: let clocks settle between runs
}

# 1) FlashMLA writing into the Judge's tensors (glue binding) -- the paper number.
run paper-flashmla-out /opt/paper/flashmla_out_v001/submission.py suite \
  --edges /m/edge_cases.json --trials 7 --warmups 1
# 2) Unpatched public flash_mla_sparse_fwd + copy_ into the Judge's tensors (lower bound).
run paper-flashmla-copy /opt/paper/flashmla_copy_v001/submission.py suite --trials 7 --warmups 1
# 3) The admissible Oracle (Triton) in the SAME image, for a same-protocol comparison.
oracle_selector="${MIGRATION_ROOT}/oracle/ADMISSIBLE_VERSION"
[[ -f "${oracle_selector}" ]] || oracle_selector="${MIGRATION_ROOT}/oracle/FINAL_VERSION"
oracle_version="$(tr -d '[:space:]' <"${oracle_selector}")"
printf -v oracle_name 'kernel_v%03d' "$((10#${oracle_version}))"
run "task-oracle-${oracle_name}" "/opt/task_oracle/${oracle_name}/submission.py" suite --trials 7 --warmups 1
# 4) Optional distribution probe (NOT a task score): q scaled by alpha -> peaked softmax.
if (( peaked )); then
  run paper-flashmla-out /opt/paper/flashmla_out_v001/submission.py peaked --alphas 1,4,8,16,24 --trials 5
  run "task-oracle-${oracle_name}" "/opt/task_oracle/${oracle_name}/submission.py" peaked --alphas 1,4,8,16,24 --trials 5
fi

for f in "${OUTPUT_DIR}"/*-suite.json; do
  [[ -f "${f}" ]] || continue
  python3 -I -c 'import json,sys;s=json.load(open(sys.argv[1]))["summary"];print(sys.argv[1].rsplit("/",1)[-1],"score_tflops=",s.get("score_tflops"),"status=",s.get("status"),"edge_failed=",s.get("edge_failed"),"edge=",s.get("edge_total",s.get("edge_passed")))' "${f}"
done
