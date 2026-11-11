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

STUDENT_MODEL_PATH="${GKD_STUDENT_MODEL_PATH:-/models/Qwen3-1.7B}"
TEACHER_MODEL_PATH="${GKD_TEACHER_MODEL_PATH:-/models/Qwen3-4B}"
COMMON_ARGS=(
  "--student-model-path" "${STUDENT_MODEL_PATH}"
  "--teacher-model-path" "${TEACHER_MODEL_PATH}"
)
if [[ ${#CANDIDATE_ARGS[@]} -gt 0 ]]; then
  COMMON_ARGS+=("${CANDIDATE_ARGS[@]}")
fi

if [[ "${MODE}" == "quick" ]]; then
  exec python3 "${SCRIPT_DIR}/public_train_probe.py" \
    "${COMMON_ARGS[@]}" \
    --batch 1 \
    --sequence-length 128 \
    --prompt-length 64 \
    --warmup-steps 1 \
    --timed-steps 2
fi

DATA_ROOT="${GKD_DATA_PATH:-/datasets/tldr-gkd}"
DATA_PARTITION="${DATA_ROOT}/public"
if [[ -f "${DATA_ROOT}/manifest.json" && -f "${DATA_ROOT}/tensors.safetensors" ]]; then
  DATA_PARTITION="${DATA_ROOT}"
fi
exec python3 "${SCRIPT_DIR}/public_train_probe.py" \
  "${COMMON_ARGS[@]}" \
  --data-path "${DATA_PARTITION}" \
  --expected-data-partition public \
  --expected-manifest-sha256 e653d0a59a07d7341197263a28424f85940ef633cce1e3a93891095cf2e04956 \
  --expected-tensors-sha256 9a92460dad2938181bfc51f9cebefc81b6f9b59ac21c2144f4cba66739a953b6 \
  --batch 3 \
  --warmup-steps 3 \
  --timed-steps 5
