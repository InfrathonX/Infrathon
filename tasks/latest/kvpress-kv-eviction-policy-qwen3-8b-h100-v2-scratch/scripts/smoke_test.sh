#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_sforge

"${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}"
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
  assert_built_images_current
  model_snapshot="$(validate_model_repo "${KV_EVICTION_MODEL_REPO}")"
  physical_gpu="$(canonical_gpu_index "${PUBLIC_GPU:-0}")"
  physical_uuid="$(gpu_uuid "${physical_gpu}")"

  # Compile and exercise the Oracle through the participant-visible runner
  # before spending the final-holdout qualification run. The runner admits only
  # an archive's root entries, so it runs on the extracted Oracle archive (the
  # oracle/ tree also holds references/), on the version FINAL_VERSION selects.
  oracle_archive="$(${MIGRATION_ROOT}/scripts/make_oracle_archive.sh | sed -n '1p')"
  oracle_version="$(tr -d '[:space:]' < "${MIGRATION_ROOT}/oracle/FINAL_VERSION")"
  oracle_lab="$(mktemp -d)"; trap 'rm -rf "${oracle_lab}"' EXIT
  tar -xzf "${oracle_archive}" -C "${oracle_lab}" --no-same-owner; chmod -R a+rX "${oracle_lab}"
  (
    lock_gpu "${physical_uuid}" "public Oracle smoke"
    assert_gpu_idle "${physical_gpu}" "public Oracle smoke"
    assert_container_gpu "${WORK_IMAGE}" "${physical_uuid}" "work"
    docker run --rm \
      -e "NVIDIA_VISIBLE_DEVICES=${physical_uuid}" \
      -e CUDA_VISIBLE_DEVICES=0 \
      -e "KV_EVICTION_MODEL_DIR=/opt/models/qwen3-8b/snapshots/${model_snapshot}" \
      -v "${KV_EVICTION_MODEL_REPO}:/opt/models/qwen3-8b:ro" \
      -v "${oracle_lab}:/oracle:ro" \
      --entrypoint python \
      "${WORK_IMAGE}" \
      /workspace/benchmark/public_runner.py \
      --lab-root /oracle \
      --suite quick \
      --candidate-version "$((10#${oracle_version}))"
  )

  [[ -n "${JUDGE_GPUS:-}" ]] || die "JUDGE_GPUS must list the Judge GPU indices for the final holdout evaluation"
  "${MIGRATION_ROOT}/scripts/eval_archive.sh" "${oracle_archive}"
  echo "Final holdout archive evaluation: PASS"
elif [[ $# -ne 0 ]]; then
  die "usage: $0 [--oracle]"
fi

echo "Focused task smoke and final holdout: PASS"
