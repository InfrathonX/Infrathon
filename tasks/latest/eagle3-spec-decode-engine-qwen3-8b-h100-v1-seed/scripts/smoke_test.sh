#!/usr/bin/env bash
# Smoke both the host-only lab path and helper D's built-image path.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'USAGE'
Usage: smoke_test.sh [--no-docker | --docker] [--quick | --full]

--no-docker  Use the image-matching lab venv and host GPU 7.
--docker     Validate immutable images/mounts and run the Judge container.
USAGE
}

mode="no-docker"
suite="quick"
while (( $# )); do
  case "$1" in
    --no-docker)
      mode="no-docker"
      ;;
    --docker)
      mode="docker"
      ;;
    --quick)
      suite="quick"
      ;;
    --full)
      suite="full"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown smoke argument: $1"
      ;;
  esac
  shift
done

results_root="${OUTPUT_ROOT}/smokes"
mkdir -p "${results_root}"

if [[ "${mode}" == "no-docker" ]]; then
  # C2 is authorized to use only physical GPU 7. The visible ordinal remains
  # 7 on the host path and becomes cuda:0 inside each Python process.
  export CUDA_VISIBLE_DEVICES=7
  export PYTHONDONTWRITEBYTECODE=1
  export EAGLE3_TARGET_REPO
  export EAGLE3_HEAD_DIR
  # The no-docker path runs the public and judge entrypoints directly against a
  # host interpreter that already has this task's CUDA/transformers stack.  That
  # environment is not part of this repository; point EAGLE3_HOST_PYTHON at it.
  # The default --docker mode needs none of this.
  lab_python="${EAGLE3_HOST_PYTHON:-}"
  [[ -n "${lab_python}" ]] \
    || die "no-docker mode needs EAGLE3_HOST_PYTHON set to an interpreter with this task's CUDA/transformers stack"
  [[ -x "${lab_python}" ]] || die "EAGLE3_HOST_PYTHON is not executable: ${lab_python}"
  public_args=(--out "${results_root}/C2_smoke_public_${suite}.json")
  if [[ "${suite}" == "quick" ]]; then
    public_args+=(--quick)
  else
    public_args+=(--full)
  fi
  "${lab_python}" "${MIGRATION_ROOT}/images/work/public_tests/public_eval.py" \
    "${public_args[@]}"
  "${lab_python}" "${MIGRATION_ROOT}/images/judge/verifier/controller.py" \
    --candidate-root "${MIGRATION_ROOT}/images/work/workdir/spec_lab" \
    --profile anchor \
    --all-anchor \
    --output "${results_root}/C2_smoke_judge_seed.json" \
    --judge-root "${MIGRATION_ROOT}/images/judge" \
    --spec-root "${MIGRATION_ROOT}/images/work/workdir/spec_lab"
  exit 0
fi

SFORGE_ENTRY_SCRIPT="$0"
ensure_docker_group "${ORIGINAL_ARGS[@]}"
for required_command in docker flock jq nvidia-smi; do
  require_command "${required_command}"
done
assert_built_images_current
judge_gpu="$(canonical_gpu_index "${JUDGE_GPU:?set JUDGE_GPU}")"
judge_uuid="$(gpu_uuid "${judge_gpu}")"
exec 8>"$(gpu_lock_path "${judge_uuid}")"
flock -n 8 || die "Judge GPU lock is already held"
assert_gpu_available "${judge_gpu}" "Judge"
assert_container_gpu "${JUDGE_IMAGE}" "${judge_uuid}" "Judge"
"${PYTHON_BIN}" "${SCRIPT_DIR}/preflight.py" \
  --target-repo "${EAGLE3_TARGET_REPO}" \
  --head-dir "${EAGLE3_HEAD_DIR}" \
  --judge-gpu "${judge_gpu}" \
  --image-manifest "${BUILD_DIR}/image-manifest.json" >/dev/null

workspace="$(mktemp -d /tmp/eagle3-smoke.XXXXXX)"
cleanup_smoke() {
  # Delete only the exact mktemp directory created above.
  if [[ "${workspace}" == /tmp/eagle3-smoke.* && -d "${workspace}" ]]; then
    chmod -R u+w -- "${workspace}" 2>/dev/null || true
    rm -rf -- "${workspace}"
  fi
}
trap cleanup_smoke EXIT INT TERM
cp -R "${MIGRATION_ROOT}/images/work/workdir/spec_lab/." "${workspace}/"
find "${workspace}" -type d -exec chmod 0555 {} +
find "${workspace}" -type f -exec chmod 0444 {} +
docker run \
  --rm \
  --network none \
  --gpus "device=${judge_uuid}" \
  --env NVIDIA_VISIBLE_DEVICES="${judge_uuid}" \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env EAGLE3_GPU_ROLE=judge \
  --env EAGLE3_JUDGE_PROFILE=anchor \
  --env EAGLE3_ALL_ANCHOR=1 \
  --env EAGLE3_QUALIFICATION_MODE=1 \
  --volume "${EAGLE3_TARGET_REPO}:/opt/models/qwen3-8b-hf:ro" \
  --volume "${EAGLE3_HEAD_DIR}:/opt/models/eagle3-head:ro" \
  --volume "${workspace}:/app/workdir/spec_lab:ro" \
  --volume "${results_root}:/logs/verifier:rw" \
  "${JUDGE_IMAGE}" \
  python3 /opt/eagle3_judge/run_judge.py \
  --app-dir /app \
  --log-dir /logs/verifier \
  >"${results_root}/C2_smoke_docker_structured.log"
assert_gpu_released "${judge_uuid}" "${judge_gpu}" "Docker smoke Judge"
