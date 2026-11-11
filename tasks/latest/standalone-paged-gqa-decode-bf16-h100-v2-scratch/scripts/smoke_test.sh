#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_sforge

"${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}"
PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" -m pytest \
  -p no:cacheprovider -q "${MIGRATION_ROOT}/tests"
"${SFORGE_BIN}" --tasks-dir "${TASKS_DIR}" list >/dev/null

if [[ "${1:-}" == "--oracle" ]]; then
  SFORGE_ENTRY_SCRIPT="$0"
  ensure_docker_group "$@"
  require_command docker
  require_command flock
  require_command nvidia-smi
  assert_built_images_current
  physical_gpu="$(canonical_gpu_index "${PHYSICAL_GPU:-0}")"
  physical_uuid="$(gpu_uuid "${physical_gpu}")"

  # Compile and exercise the Oracle through the participant-visible runner
  # before spending the single full qualification run.  Both steps use the
  # admissible Oracle (oracle/ADMISSIBLE_VERSION); oracle/FINAL_VERSION may
  # select an upstream candidate that only the local forced-oracle Judge
  # (oracle/paper_reference/forced_judge/) can score.
  (
    exec 8>"$(gpu_lock_path "${physical_uuid}")"
    flock -n 8 || die "GPU ${physical_gpu} lock is already held"
    assert_gpu_idle "${physical_gpu}" "public Oracle smoke"
    assert_container_gpu "${WORK_IMAGE}" "${physical_uuid}" "work"
    docker run --rm \
      -e "NVIDIA_VISIBLE_DEVICES=${physical_uuid}" \
      -e CUDA_VISIBLE_DEVICES=0 \
      -v "${MIGRATION_ROOT}/oracle/ADMISSIBLE_VERSION:/oracle/FINAL_VERSION:ro" \
      -v "${MIGRATION_ROOT}/oracle/notes.md:/oracle/notes.md:ro" \
      -v "${MIGRATION_ROOT}/oracle/candidates:/oracle/candidates:ro" \
      --entrypoint python \
      "${WORK_IMAGE}" \
      /workspace/benchmark/public_runner.py \
      --lab-root /oracle \
      --suite full \
      --candidate-version "$(tr -d '[:space:]' <"${MIGRATION_ROOT}/oracle/ADMISSIBLE_VERSION")"
  )

  oracle_archive="$(${MIGRATION_ROOT}/scripts/make_oracle_archive.sh | sed -n '1p')"
  PHYSICAL_GPU="${physical_gpu}" TRUSTED_CONTROL=1 \
    "${MIGRATION_ROOT}/scripts/eval_archive.sh" "${oracle_archive}"
elif [[ $# -ne 0 ]]; then
  die "usage: $0 [--oracle]"
fi

echo "Focused task smoke: PASS"
