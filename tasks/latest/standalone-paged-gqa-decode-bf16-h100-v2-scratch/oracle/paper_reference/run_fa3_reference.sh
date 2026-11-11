#!/usr/bin/env bash
# Host-only paper reference: time upstream FlashAttention-3 (0251105a) on the
# paged-GQA ABI with the Judge's own runtime.  See oracle/paper_reference/README.md.
#
#   oracle/paper_reference/run_fa3_reference.sh --gpu N [--skip-build] [--tuned]
#
# Needs FA3_SOURCE (a full flash-attention checkout at
# 0251105a2fb19d2957484b7f023cd8c115286ced with csrc/cutlass initialised)
# unless --skip-build reuses an existing image.  Uses the Judge-only case
# manifests that scripts/lib.sh resolves (FINAL_CASES_PATH / EDGE_CASES_PATH).
# Nothing here touches the Work image, the Judge image or a run's version store.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../../scripts/lib.sh"
require_command docker
require_command flock
require_command nvidia-smi

FA3_IMAGE="${FA3_IMAGE:-standalone_paged_gqa_decode.paper-fa3:0251105a}"
FA3_SOURCE="${FA3_SOURCE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${HERE}/results}"
FA3_COMMIT=0251105a2fb19d2957484b7f023cd8c115286ced
gpu=""
skip_build=0
tuned=0
while (( $# )); do
  case "$1" in
    --gpu) [[ $# -ge 2 ]] || die "--gpu requires a value"; gpu="$2"; shift 2 ;;
    --skip-build) skip_build=1; shift ;;
    --tuned) tuned=1; shift ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[[ "${gpu}" =~ ^[0-9]+$ ]] || die "--gpu N is required"

final_cases="${FINAL_CASES_PATH:-${DEFAULT_FINAL_CASES_PATH}}"
edge_cases="${EDGE_CASES_PATH:-${DEFAULT_EDGE_CASES_PATH}}"
[[ -f "${final_cases}" ]] || die "final holdout manifest not found: ${final_cases}"
[[ -f "${edge_cases}" ]] || die "edge holdout manifest not found: ${edge_cases}"
echo "09545c07b972e5d0cad59f2a555ed59a1586dca2f1c45cbc0b79869d1e932132  ${final_cases}" | sha256sum -c - >/dev/null \
  || die "final_cases.json is not the frozen final-holdout-v2"
echo "9c9cdddf4f460d932518e82a3550aa086210a3a1feb59a75f60ebc133eedc676  ${edge_cases}" | sha256sum -c - >/dev/null \
  || die "edge_cases.json is not the frozen edge-holdout-v2"

if (( ! skip_build )); then
  [[ -n "${FA3_SOURCE}" && -d "${FA3_SOURCE}/hopper" ]] \
    || die "set FA3_SOURCE to a flash-attention checkout at ${FA3_COMMIT} (see README.md)"
  [[ -d "${FA3_SOURCE}/csrc/cutlass/include" ]] \
    || die "${FA3_SOURCE} has no csrc/cutlass submodule"
  if [[ -d "${FA3_SOURCE}/.git" ]]; then
    [[ "$(git -C "${FA3_SOURCE}" rev-parse HEAD)" == "${FA3_COMMIT}" ]] \
      || die "${FA3_SOURCE} is not at ${FA3_COMMIT}"
  fi
  docker build --tag "${FA3_IMAGE}" \
    --build-context "harness=${HERE}" \
    --file "${HERE}/Dockerfile" "${FA3_SOURCE}"
fi
docker image inspect "${FA3_IMAGE}" >/dev/null 2>&1 || die "missing image ${FA3_IMAGE}"
# CPU-only build check (no GPU): refuse FA3 builds with varlen / paged KV compiled out.
docker run --rm --network none --cpus 2 -v "${HERE}:/opt/fa3:ro" \
  --entrypoint python "${FA3_IMAGE}" -I /opt/fa3/fa3_preflight.py

uuid="$(gpu_uuid "$(canonical_gpu_index "${gpu}")")"
exec 8>"$(gpu_lock_path "${uuid}")"
flock -n 8 || die "GPU ${gpu} lock is already held"
assert_gpu_idle "${gpu}" "FA3 paper reference"

mkdir -p "${OUTPUT_DIR}"
cases_dir="$(mktemp -d)"
trap 'rm -rf "${cases_dir}"' EXIT
install -m 0644 "${final_cases}" "${cases_dir}/final_cases.json"
install -m 0644 "${edge_cases}" "${cases_dir}/edge_cases.json"

run() {  # label candidate mode [extra docker args...]
  local label="$1" candidate="$2" mode="$3"
  shift 3
  docker run --rm --name "pgqa-paper-fa3-${label}-$$" --network none \
    --gpus "device=${uuid}" -e CUDA_VISIBLE_DEVICES=0 -e PYTHONDONTWRITEBYTECODE=1 \
    --cpus 8 ${FA3_CPUSET:+--cpuset-cpus "${FA3_CPUSET}"} --memory 32g --shm-size 8g \
    -v "${MIGRATION_ROOT}/runtime/paged_gqa_decode_runtime.py:/opt/harness/paged_gqa_decode_runtime.py:ro" \
    -v "${HERE}:/opt/fa3:ro" \
    -v "${MIGRATION_ROOT}/oracle/candidates:/opt/cands:ro" \
    -v "${cases_dir}:/cases:ro" -v "${OUTPUT_DIR}:/results" \
    "$@" \
    --entrypoint python "${FA3_IMAGE}" -I /opt/fa3/fa3_runner.py \
      --label "${label}" --candidate "${candidate}" --mode "${mode}" \
      --final-cases /cases/final_cases.json --edge-cases /cases/edge_cases.json \
      --output "/results/${label}-${mode}.json"
}

# 1) FA3 with its own heuristics (num_splits=0, pack_gqa=None) on the scored grid.
run fa3 /opt/fa3/fa3_entry.py scored
# 2) Public flash_attn_with_kvcache API (own output + out.copy_), cross-check.
run fa3-public-api /opt/fa3/fa3_entry_public_api.py scored
# 3) Optional: FA3 with per-shape (num_splits, pack_gqa) picked by a sweep.
if (( tuned )); then
  run fa3-tuned /opt/fa3/fa3_entry_cfg.py scored -e FA3_TUNED_JSON=/opt/fa3/fa3_tuned_pinned.json
fi
# 4) The Oracle's Triton candidates in the SAME image (controls for the
#    torch/Triton/harness differences against the Judge image).
for candidate in "${MIGRATION_ROOT}"/oracle/candidates/kernel_v*/submission.py; do
  name="$(basename "$(dirname "${candidate}")")"
  run "oracle-${name}" "/opt/cands/${name}/submission.py" scored
done
# 5) FA3 over the private edge cases (support-domain coverage).
run fa3 /opt/fa3/fa3_entry.py edge

for f in "${OUTPUT_DIR}"/*-scored.json "${OUTPUT_DIR}"/*-edge.json; do
  [[ -f "${f}" ]] || continue
  python3 -I -c 'import json,sys;s=json.load(open(sys.argv[1]))["summary"];print(s["label"],s["mode"],s["status"],s["failed_gate"],s["score_tflops"],s["edge"],s["private_error"][:120])' "${f}"
done
