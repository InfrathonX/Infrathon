#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="full"
CANDIDATE_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu-only)
      MODE="cpu"
      shift
      ;;
    --quick)
      MODE="quick"
      shift
      ;;
    --candidate-version)
      [[ $# -ge 2 ]] || { printf '%s\n' "--candidate-version requires an integer" >&2; exit 2; }
      CANDIDATE_ARGS=("--candidate-version" "$2")
      shift 2
      ;;
    --help|-h)
      printf '%s\n' \
        "usage: run_public_tests.sh [--cpu-only|--quick] [--candidate-version N]" \
        "  --cpu-only  objective semantics/gradient checks without CUDA" \
        "  --quick     CPU checks plus a short synthetic two-GPU full-model smoke" \
        "  no mode     CPU checks plus frozen-public-TLDR two-GPU replay"
      exit 0
      ;;
    *)
      printf 'unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

python3 "${SCRIPT_DIR}/public_eval.py" --mode cpu "${CANDIDATE_ARGS[@]}"
[[ "${MODE}" != "cpu" ]] || exit 0

MODEL_PATH="${ORPO_MODEL_PATH:-/models/Qwen3-1.7B}"
COMMON_ARGS=(
  "--model-path" "${MODEL_PATH}"
)
if [[ ${#CANDIDATE_ARGS[@]} -gt 0 ]]; then
  COMMON_ARGS+=("${CANDIDATE_ARGS[@]}")
fi

if [[ "${MODE}" == "quick" ]]; then
  exec python3 "${SCRIPT_DIR}/public_train_probe.py" \
    "${COMMON_ARGS[@]}" \
    --pair-batch 1 \
    --sequence-length 128 \
    --prompt-length 64 \
    --warmup-steps 1 \
    --timed-steps 2
fi

DATA_ROOT="${ORPO_DATA_PATH:-/datasets/tldr-preference}"
DATA_PARTITION="${DATA_ROOT}/public"
if [[ -f "${DATA_ROOT}/manifest.json" && -f "${DATA_ROOT}/tensors.safetensors" ]]; then
  DATA_PARTITION="${DATA_ROOT}"
fi
exec python3 "${SCRIPT_DIR}/public_train_probe.py" \
  "${COMMON_ARGS[@]}" \
  --data-path "${DATA_PARTITION}" \
  --expected-data-partition public \
  --expected-manifest-sha256 ba9816131647aefe0d813fc7c7d3706894988943bc58e92c472667f722528c20 \
  --expected-tensors-sha256 e1218b35260ea9e7afd048962b53c25257d7f298e2a787628d8f5b640cfc7e79 \
  --pair-batch 3 \
  --warmup-steps 3 \
  --timed-steps 5
