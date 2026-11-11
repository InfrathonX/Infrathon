#!/usr/bin/env bash

set -euo pipefail

MIGRATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
find_repository_root() {
  local candidate="${MIGRATION_ROOT}"
  while [[ "${candidate}" != "/" ]]; do
    if [[ -f "${candidate}/pyproject.toml" && -d "${candidate}/sforge" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    candidate="$(dirname "${candidate}")"
  done
  (cd "${MIGRATION_ROOT}/.." && pwd)
}

REPOSITORY_ROOT="${REPOSITORY_ROOT:-$(find_repository_root)}"
SFORGE_ROOT="${SFORGE_ROOT:-${MLSYSFORGE_ROOT:-${REPOSITORY_ROOT}}}"
MLSYSFORGE_ROOT="${SFORGE_ROOT}"
SFORGE_BIN="${SFORGE_BIN:-${SFORGE_ROOT}/.venv/bin/sforge}"
PYTHON_BIN="${PYTHON_BIN:-${SFORGE_ROOT}/.venv/bin/python}"
TASKS_DIR="${TASKS_DIR:-${MIGRATION_ROOT}/tasks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPOSITORY_ROOT}/outputs}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-$(basename "${MIGRATION_ROOT}")}"
LOGS_DIR="${LOGS_DIR:-${OUTPUT_ROOT}/${OUTPUT_SUBDIR}}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${MIGRATION_ROOT}/artifacts}"

export SFORGE_TASKS_DIR="${TASKS_DIR}"
export SFORGE_LOG_DIR="${LOGS_DIR}"

TASK_ID="tilelang_mamba2_ssd_forward_h100_seed_v2"
BENCHMARK_NAME="tilelang_mamba2_ssd"
WORK_IMAGE="${BENCHMARK_NAME}.work.${TASK_ID}:local"
JUDGE_IMAGE="${BENCHMARK_NAME}.judge.${TASK_ID}:local"
# The Judge above is this public Judge (images/judge/Dockerfile, publishable) plus
# the holdout layer images/judge/Dockerfile.private, which is only ever built on
# the host.
JUDGE_PUBLIC_IMAGE="${BENCHMARK_NAME}.judge-public.${TASK_ID}:local"
# The frozen qualification draw of the final holdout ships with the task on
# the Judge-only path; portable/final-suite-manifest.json records its SHA-256
# under frozen_instance.  FINAL_CASES_PATH still overrides it.
DEFAULT_FINAL_CASES_PATH="${MIGRATION_ROOT}/judge/private/final_cases.json"
# Infrathon assets: with INFRATHON_ASSETS_ROOT set, or in a tree without the
# repository copies (the public tree; root defaults to
# ~/.cache/infrathon/assets), the build reads these from
# <INFRATHON_ASSETS_ROOT>/<task directory>/judge/ instead, where assets/fetch.sh
# places them (scripts/infrathon/assets.py).
if [[ -n "${INFRATHON_ASSETS_ROOT:-}" || ! -f "${DEFAULT_FINAL_CASES_PATH}" ]]; then
  INFRATHON_ASSETS_ROOT="${INFRATHON_ASSETS_ROOT:-${HOME}/.cache/infrathon/assets}"
  FINAL_CASES_PATH="${FINAL_CASES_PATH:-${INFRATHON_ASSETS_ROOT}/$(basename "${MIGRATION_ROOT}")/judge/final_cases.json}"
fi
# Captured layers (fixtures/recipe/real_layers/).  The Judge-only layers go into the host-built
# Judge holdout layer and the public layer into the Work image.  REAL_LAYERS_DIR and
# PUBLIC_LAYER_DIR override the asset-root defaults.
_mamba2_ssd_assets="${INFRATHON_ASSETS_ROOT:-${HOME}/.cache/infrathon/assets}/$(basename "${MIGRATION_ROOT}")"
REAL_LAYERS_DIR="${REAL_LAYERS_DIR:-${_mamba2_ssd_assets}/judge/real_layers}"
PUBLIC_LAYER_DIR="${PUBLIC_LAYER_DIR:-${_mamba2_ssd_assets}/public/public_layer}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

require_sforge() {
  [[ -x "${SFORGE_BIN}" ]] || die "missing executable SForge: ${SFORGE_BIN}"
  [[ -x "${PYTHON_BIN}" ]] || die "missing SForge Python: ${PYTHON_BIN}"
}

ensure_docker_group() {
  if docker info >/dev/null 2>&1; then
    return
  fi
  if [[ "${SFORGE_DOCKER_GROUP_REEXEC:-0}" == "1" ]]; then
    die "Docker is unreachable after docker-group re-exec"
  fi
  # The member list follows the last ':' of the group entry, so the first
  # member is preceded by ':' rather than ','.
  getent group docker | grep -Eq "(^|[:,])${USER:-$(id -un)}(,|$)" \
    || die "Docker is unreachable and ${USER:-$(id -un)} is not listed in the docker group"

  local entry_script="${SFORGE_ENTRY_SCRIPT:-$0}"
  local -a command_line=(
    env
    SFORGE_DOCKER_GROUP_REEXEC=1
    REPOSITORY_ROOT="${REPOSITORY_ROOT}"
    SFORGE_ROOT="${SFORGE_ROOT}"
    SFORGE_BIN="${SFORGE_BIN}"
    PYTHON_BIN="${PYTHON_BIN}"
    TASKS_DIR="${TASKS_DIR}"
    LOGS_DIR="${LOGS_DIR}"
    ARTIFACTS_DIR="${ARTIFACTS_DIR}"
  )
  local name
  for name in \
    CODEX_AUTH_JSON_PATH \
    CODEX_CLI_VERSION \
    FINAL_EVAL_TIMEOUT \
    SFORGE_AGENT_RUNTIME_DIR \
    SFORGE_VERSION_STORE_ROOT \
    SFORGE_SINGLE_GPU_HANDOFF \
    SFORGE_ACCELERATOR_DEVICE_UUID \
    SFORGE_ACCELERATOR_ALLOCATION_KEY \
    SFORGE_ACCELERATOR_OBSERVER \
    SFORGE_TRUSTED_LEASE_ID \
    SFORGE_CALIBRATION_EPOCH \
    FINAL_CASES_PATH \
    INFRATHON_ASSETS_ROOT \
    INFRATHON_HARDWARE \
    INFRATHON_PULL \
    GPU_MIN_FREE_MIB \
    H100_MIN_FREE_MIB \
    JUDGE_PORT \
    MODEL \
    MODEL_API_BASE_URL \
    MODEL_API_KEY_FILE \
    WORK_GPU \
    JUDGE_GPU \
    REASONING_EFFORT \
    RUN_ID \
    RUN_STAMP \
    TASK_TIMEOUT \
    PHYSICAL_GPU \
    EVAL_INTERVAL; do
    if [[ -v "${name}" ]]; then
      command_line+=("${name}=${!name}")
    fi
  done
  command_line+=(bash "${entry_script}" "$@")
  local quoted
  printf -v quoted '%q ' "${command_line[@]}"
  exec sg docker -c "${quoted}"
}

assert_private_auth_file() {
  local path="$1"
  [[ -f "${path}" && ! -L "${path}" && -s "${path}" ]] \
    || die "Codex auth JSON must be a nonempty regular, non-symlink file"
  [[ "$(stat -c '%a' "${path}")" == "600" ]] \
    || die "Codex auth JSON must have mode 600"
  [[ "$(stat -c '%u' "${path}")" == "$(id -u)" ]] \
    || die "Codex auth JSON must be owned by the launching user"
}

canonical_gpu_index() {
  local raw="$1"
  [[ "${raw}" =~ ^[0-9]+$ ]] || die "GPU index is not numeric: ${raw}"
  echo "$((10#${raw}))"
}

gpu_uuid() {
  nvidia-smi --id="$1" --query-gpu=uuid --format=csv,noheader | xargs
}

gpu_name() {
  nvidia-smi --id="$1" --query-gpu=name --format=csv,noheader | xargs
}

gpu_lock_path() {
  local uuid="$1"
  local lock_id
  [[ "${uuid}" =~ ^GPU-[0-9a-f-]+$ \
     || "${uuid}" =~ ^MIG-[0-9a-f-]+$ \
     || "${uuid}" =~ ^MIG-GPU-[0-9a-f-]+/(0|[1-9][0-9]*)/(0|[1-9][0-9]*)$ ]] \
    || die "unsafe GPU UUID: ${uuid}"
  lock_id="${uuid//\//-}"
  echo "/tmp/sforge-gpu-${lock_id}.lock"
}

assert_gpu_idle() {
  local index="$1"
  local role="$2"
  local minimum_free="${GPU_MIN_FREE_MIB:-${H100_MIN_FREE_MIB:-74752}}"
  [[ "${minimum_free}" =~ ^[0-9]+$ ]] \
    || die "GPU_MIN_FREE_MIB must be a non-negative integer"
  local name uuid free_mib
  name="$(gpu_name "${index}")"
  uuid="$(gpu_uuid "${index}")"
  free_mib="$(nvidia-smi --id="${index}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  (( free_mib >= minimum_free )) \
    || die "${role} GPU ${index} has only ${free_mib} MiB free"
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
      | grep -Fxq "${uuid}"; then
    die "${role} GPU ${index} (${uuid}) already has a compute process"
  fi
  echo "${role} GPU: physical ${index}, ${name}, ${uuid}, ${free_mib} MiB free"
}

assert_gpu_released() {
  local uuid="$1"
  for _ in $(seq 1 60); do
    if ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
        | grep -Fxq "${uuid}"; then
      echo "GPU release barrier: PASS (${uuid})"
      return 0
    fi
    sleep 1
  done
  die "GPU still has a compute process after the release barrier: ${uuid}"
}

assert_container_gpu() {
  local image="$1"
  local expected_uuid="$2"
  local role="$3"
  local seen
  seen="$(docker run --rm \
    -e "NVIDIA_VISIBLE_DEVICES=${expected_uuid}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    --entrypoint nvidia-smi \
    "${image}" --query-gpu=uuid --format=csv,noheader | xargs)"
  [[ "${seen}" == "${expected_uuid}" ]] \
    || die "${role} container expected ${expected_uuid}, saw ${seen:-nothing}"
  echo "${role} container GPU pin: ${seen}"
}

assert_container_accelerator() {
  local image="$1"
  local expected_uuid="$2"
  local role="$3"
  local seen
  [[ "${expected_uuid}" =~ ^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]] \
    || die "${role} accelerator must be a canonical whole-GPU UUID"
  seen="$(docker run --rm \
    --gpus "device=${expected_uuid}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    --entrypoint nvidia-smi \
    "${image}" --query-gpu=uuid --format=csv,noheader | xargs)"
  [[ "${seen}" == "${expected_uuid}" ]] \
    || die "${role} container expected ${expected_uuid}, saw ${seen:-nothing}"
  echo "${role} container exact DeviceRequest: ${seen}"
}

# Capability-envelope admission of one scheduler-allocated whole GPU.  The
# thresholds come from the task's selected portable execution profile; the GPU
# model name is printed as provenance and never compared.
assert_portable_gpu_idle() {
  local uuid="$1"
  local role="$2"
  local required_vendor="$3"
  local required_count="$4"
  local minimum_capability="$5"
  local minimum_memory_bytes="$6"
  local seen name total_mib free_mib capability capability_major capability_minor
  local minimum_capability_major minimum_capability_minor minimum_memory_mib
  [[ "${required_vendor}" == "nvidia" && "${required_count}" == "1" ]] \
    || die "${role} task profile must select one NVIDIA accelerator"
  [[ "${minimum_capability}" =~ ^[0-9]+\.[0-9]+$ \
     && "${minimum_memory_bytes}" =~ ^[1-9][0-9]*$ ]] \
    || die "${role} task profile has invalid capability requirements"
  minimum_capability_major="${minimum_capability%%.*}"
  minimum_capability_minor="${minimum_capability#*.}"
  minimum_memory_mib="$((
    (10#${minimum_memory_bytes} + 1024 * 1024 - 1) / (1024 * 1024)
  ))"
  seen="$(nvidia-smi --id="${uuid}" --query-gpu=uuid --format=csv,noheader | xargs)"
  [[ "${seen}" == "${uuid}" ]] \
    || die "${role} allocation expected ${uuid}, saw ${seen:-nothing}"
  name="$(nvidia-smi --id="${uuid}" --query-gpu=name --format=csv,noheader | xargs)"
  total_mib="$(nvidia-smi --id="${uuid}" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
  free_mib="$(nvidia-smi --id="${uuid}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  capability="$(nvidia-smi --id="${uuid}" --query-gpu=compute_cap --format=csv,noheader | xargs)"
  [[ "${total_mib}" =~ ^[0-9]+$ && "${free_mib}" =~ ^[0-9]+$ ]] \
    || die "${role} allocation returned invalid memory capability"
  capability_major="${capability%%.*}"
  capability_minor="${capability#*.}"
  [[ "${capability}" =~ ^[0-9]+\.[0-9]+$ ]] \
    || die "${role} allocation returned invalid compute capability: ${capability}"
  (( 10#${capability_major} > 10#${minimum_capability_major} \
     || (10#${capability_major} == 10#${minimum_capability_major} \
         && 10#${capability_minor} >= 10#${minimum_capability_minor}) )) \
    || die "${role} allocation requires compute capability ${minimum_capability} or newer, found ${capability}"
  (( total_mib >= minimum_memory_mib && free_mib >= minimum_memory_mib )) \
    || die "${role} allocation requires ${minimum_memory_bytes} bytes total and free; total=${total_mib} MiB free=${free_mib} MiB"
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
      | grep -Fxq "${uuid}"; then
    die "${role} allocation ${uuid} already has a compute process"
  fi
  echo "${role} allocation: ${name}, ${uuid}, compute ${capability}, ${free_mib} MiB free"
}

assert_container_accelerator() {
  local image="$1"
  local expected_uuid="$2"
  local role="$3"
  local seen
  [[ "${expected_uuid}" =~ ^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]] \
    || die "${role} accelerator must be a canonical whole-GPU UUID"
  seen="$(docker run --rm \
    --gpus "device=${expected_uuid}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    --entrypoint nvidia-smi \
    "${image}" --query-gpu=uuid --format=csv,noheader | xargs)"
  [[ "${seen}" == "${expected_uuid}" ]] \
    || die "${role} container expected ${expected_uuid}, saw ${seen:-nothing}"
  echo "${role} container exact DeviceRequest: ${seen}"
}

image_source_digest() {
  (
    cd "${MIGRATION_ROOT}"
    {
      find images/work images/judge \
        -type f \
        ! -path '*/__pycache__/*' \
        ! -name '*.pyc' \
        -print0
      find runtime \
        -type f \
        ! -path '*/__pycache__/*' \
        ! -name '*.pyc' \
        ! -path '*/.git/*' \
        -print0
      printf '%s\0' judge/run_judge.py
    } \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

assert_built_images_current() {
  local manifest="${LOGS_DIR}/build/image-manifest.json"
  [[ -s "${manifest}" ]] || die "missing build manifest; run scripts/build_images.sh"
  local recorded_source recorded_work recorded_judge
  local final_cases_context_path recorded_final_cases_sha256
  local current_source current_work current_judge current_final_cases_sha256
  recorded_source="$(jq -er '.source_digest' "${manifest}")"
  recorded_work="$(jq -er '.work.id' "${manifest}")"
  recorded_judge="$(jq -er '.judge.id' "${manifest}")"
  recorded_final_cases_sha256="$(jq -er '.final_cases.sha256' "${manifest}")"
  final_cases_context_path="${LOGS_DIR}/build/final-cases-context/final_cases.json"
  current_source="$(image_source_digest)"
  current_work="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}" 2>/dev/null)" \
    || die "missing work image; run scripts/build_images.sh"
  current_judge="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}" 2>/dev/null)" \
    || die "missing judge image; run scripts/build_images.sh"
  [[ -f "${final_cases_context_path}" && ! -L "${final_cases_context_path}" ]] \
    || die "final holdout build-context copy is missing or unsafe; rebuild images"
  current_final_cases_sha256="$(sha256sum "${final_cases_context_path}" | awk '{print $1}')"
  [[ "${current_source}" == "${recorded_source}" ]] \
    || die "image-driving source changed; rebuild images"
  [[ "${current_work}" == "${recorded_work}" ]] \
    || die "work image tag moved after the recorded build"
  [[ "${current_judge}" == "${recorded_judge}" ]] \
    || die "judge image tag moved after the recorded build"
  [[ "${current_final_cases_sha256}" == "${recorded_final_cases_sha256}" ]] \
    || die "final holdout manifest changed after the recorded build"
  echo "Image/source freshness: PASS (${current_source:0:12})"
  echo "Final holdout freshness: PASS (${current_final_cases_sha256:0:12})"
}

wait_for_http() {
  local url="$1"
  local attempts="${2:-60}"
  for _ in $(seq 1 "${attempts}"); do
    if curl --connect-timeout 2 --max-time 5 -fsS "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  die "service did not become healthy: ${url}"
}
