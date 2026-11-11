#!/usr/bin/env bash
set -euo pipefail

# Read-only status of the local Judge-only runs (eval_archive.sh,
# run_oracle.sh, replay_candidates.sh); agent runs live on PyroMind.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

run_id="${1:-}"
if [[ -z "${run_id}" ]]; then
  latest="$(find "${LOGS_DIR}/runs" -mindepth 1 -maxdepth 1 -type d \
    -name 'qvg-h100-scratch-v4-*' \
    -printf '%T@ %f\n' 2>/dev/null | sort -nr | head -1 || true)"
  run_id="${latest#* }"
fi

echo "task: ${TASK_ID}"
echo "images: ${WORK_IMAGE} | ${JUDGE_IMAGE}"
echo "outputs: ${LOGS_DIR}"
if [[ -s "${IMAGE_MANIFEST}" && ! -L "${IMAGE_MANIFEST}" ]] && command -v jq >/dev/null 2>&1; then
  echo "image manifest: $(jq -r '"source " + (.source_digest[0:12]) + ", judge " + (.judge.id[7:19])' "${IMAGE_MANIFEST}" 2>/dev/null || echo unreadable)"
else
  echo "image manifest: none (run scripts/build_images.sh)"
fi

echo
echo "host GPU snapshot (selection remains the operator's, through JUDGE_GPUS):"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,uuid,memory.used,utilization.gpu \
    --format=csv,noheader 2>/dev/null || echo "  GPU query failed"
else
  echo "  nvidia-smi unavailable"
fi

echo
echo "latest Judge run:"
if [[ -z "${run_id}" ]]; then
  echo "  no scratch v4 run directory"
  exit 0
fi
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
  || die "unsafe run id: ${run_id}"
run_root="${LOGS_DIR}/runs/${run_id}"
echo "  ${run_root}"
if [[ ! -d "${run_root}" ]]; then
  echo "  run directory is absent"
  exit 0
fi

echo
echo "recent Judge reports:"
mapfile -t reports < <(
  find "${run_root}" -type f -name report.json \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -8 | cut -d' ' -f2-
)
if (( ${#reports[@]} == 0 )); then
  echo "  no report.json yet"
else
  printf '  %s\n' "${reports[@]}"
  if command -v jq >/dev/null 2>&1; then
    echo
    echo "newest report summary:"
    jq '{valid,score,timed_out,runtime_seconds,
         candidate_invalid: .metrics.candidate_invalid,
         infrastructure_invalid: .metrics.infrastructure_invalid,
         evaluation_status: .metrics.evaluation_status}' \
      "${reports[0]}" 2>/dev/null || echo "  report is not valid JSON"
  fi
  private_archive="$(dirname -- "${reports[0]}")/private_judge_artifacts.tar.gz"
  if [[ -s "${private_archive}" ]]; then
    echo "per-case readings (root-private controller JSON): ${private_archive}"
  fi
fi
