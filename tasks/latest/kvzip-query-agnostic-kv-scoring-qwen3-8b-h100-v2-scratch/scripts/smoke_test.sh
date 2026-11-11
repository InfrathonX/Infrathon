#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_sforge

"${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}"
if [[ "${1:-}" == "--oracle" && ! -f "${MIGRATION_ROOT}/oracle/candidates/scorer_v001/assets.pt" ]]; then
  "${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/prepare_oracle_assets.py"
fi
host_test_python="${HOST_TEST_PYTHON:-${PYTHON_BIN}}"
PYTHONDONTWRITEBYTECODE=1 "${host_test_python}" -m pytest \
  -p no:cacheprovider -q "${MIGRATION_ROOT}/tests"
"${SFORGE_PYTHON}" "${TASK_ENTRY}" --tasks-dir "${TASKS_DIR}" list >/dev/null

if [[ "${1:-}" == "--oracle" ]]; then
  SFORGE_ENTRY_SCRIPT="$0"
  ensure_docker_group "$@"
  require_command docker
  require_command flock
  require_command nvidia-smi
  require_command tar
  assert_built_images_current
  model_snapshot="$(validate_model_repo "${KV_SCORING_MODEL_REPO}")"
  physical_gpu="$(canonical_gpu_index "${PUBLIC_GPU:-0}")"
  physical_uuid="$(gpu_uuid "${physical_gpu}")"
  oracle_archive="$(${MIGRATION_ROOT}/scripts/make_oracle_archive.sh | sed -n '1p')"
  oracle_smoke_root="$(mktemp -d "${MIGRATION_ROOT}/build/oracle-smoke.XXXXXX")"
  trap 'rm -rf -- "${oracle_smoke_root}"' EXIT
  tar -xzf "${oracle_archive}" -C "${oracle_smoke_root}"
  chmod 0755 "${oracle_smoke_root}"

  # Compile and exercise the Oracle through the participant-visible runner
  # before spending the final-holdout qualification run.
  (
    lock_gpu "${physical_uuid}" "public Oracle smoke"
    assert_gpu_idle "${physical_gpu}" "public Oracle smoke"
    assert_container_gpu "${WORK_IMAGE}" "${physical_uuid}" "work"
    docker run --rm \
      -e "NVIDIA_VISIBLE_DEVICES=${physical_uuid}" \
      -e CUDA_VISIBLE_DEVICES=0 \
      -e "KV_SCORING_MODEL_DIR=/opt/models/qwen3-8b/snapshots/${model_snapshot}" \
      -v "${KV_SCORING_MODEL_REPO}:/opt/models/qwen3-8b:ro" \
      -v "${oracle_smoke_root}:/oracle:ro" \
      --entrypoint python \
      "${WORK_IMAGE}" \
      /workspace/benchmark/public_runner.py \
      --lab-root /oracle \
      --suite quick \
      --candidate-version "$(tr -d '[:space:]' < "${oracle_smoke_root}/FINAL_VERSION")"
  )

  [[ -n "${JUDGE_GPUS:-}" ]] || die "JUDGE_GPUS must list the Judge GPU indices for the final holdout evaluation"
  "${MIGRATION_ROOT}/scripts/eval_archive.sh" "${oracle_archive}"
  echo "Final holdout archive evaluation: PASS"
elif [[ $# -ne 0 ]]; then
  die "usage: $0 [--oracle]"
fi

echo "Focused task smoke and final holdout: PASS"
