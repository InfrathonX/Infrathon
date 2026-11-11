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
  # Preserve support for copying a task out of the monorepo for isolated
  # validation. In that case the task's parent is the compatibility fallback.
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

# The Judge-only nonce source: the in-repository copy if present, else the copy
# assets/fetch.sh places under the asset root (INFRATHON_ASSETS_ROOT, default
# ~/.cache/infrathon/assets).  SFORGE_KVBENCH_NONCE_SOURCE overrides both.
default_nonce_source() {
  local in_tree="${MIGRATION_ROOT}/judge/private/nonce_source.txt"
  if [[ -e "${in_tree}" || -L "${in_tree}" ]]; then
    printf '%s\n' "${in_tree}"
    return
  fi
  local assets_root="${INFRATHON_ASSETS_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/infrathon/assets}"
  printf '%s\n' "${assets_root}/$(basename "${MIGRATION_ROOT}")/judge/nonce_source.txt"
}

TASK_ID="concurrent_agent_serving_optimizer_qwen36_swe192_v24"
BENCHMARK_NAME="kvbench_qwen36"
WORK_IMAGE="${BENCHMARK_NAME}.work.${TASK_ID}:local"
JUDGE_IMAGE="${BENCHMARK_NAME}.judge.${TASK_ID}:local"
NESTED_ENVIRONMENT_ROOT="${MIGRATION_ROOT}/runtime/frozen/harbor_tasks/concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/environment"
MODEL_API_IMAGE="harbor-kv/model-api:qwen36-swe192-v24-url-v1"
CONTROLLER_IMAGE="harbor-kv/bench-controller:qwen36-swe192-v24-url-v1"
PUBLIC_CONTROLLER_IMAGE="${CONTROLLER_IMAGE}"
MODEL_API_DOCKERFILE="${NESTED_ENVIRONMENT_ROOT}/model-api/Dockerfile"
CONTROLLER_DOCKERFILE="${NESTED_ENVIRONMENT_ROOT}/bench-controller/Dockerfile"
# The two nested images the frozen compose files name by tag.  They are built
# from the frozen environment like model-api and the controller; their image
# IDs are recorded in image-manifest.json, never pinned in the task.
MAIN_IMAGE="harbor-kv/main:qwen36-27b-swe192-v2-1-gpu45-v1"
MAIN_DOCKERFILE="${NESTED_ENVIRONMENT_ROOT}/Dockerfile"
SANDBOX_WORKER_IMAGE="harbor-kv/sandbox-worker:qwen36-27b-tp2-public30-r2"
SANDBOX_WORKER_DOCKERFILE="${NESTED_ENVIRONMENT_ROOT}/sandbox-worker/Dockerfile"
# The external vLLM backend: contract v3 binds it to images/backend/Dockerfile
# (a digest-pinned official vLLM release), not to any host's image ID.
BACKEND_IMAGE="harbor-kv/vllm-qwen36-27b:v3-recipe"
BACKEND_DOCKERFILE="${MIGRATION_ROOT}/images/backend/Dockerfile"

# SForge resolves these pathlib-backed settings at import time.  Keep the
# environment authoritative for both the stock CLI and our custom entrypoint.
export SFORGE_TASKS_DIR="${TASKS_DIR}"
export SFORGE_LOG_DIR="${LOGS_DIR}"
export SFORGE_KVBENCH_MIGRATION_ROOT="${MIGRATION_ROOT}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

user_in_docker_group() {
  local launch_user="$1"
  getent group docker | awk -F: -v wanted="${launch_user}" '
    $1 == "docker" {
      count = split($4, members, ",")
      for (i = 1; i <= count; i++) {
        if (members[i] == wanted) found = 1
      }
    }
    END { exit(found ? 0 : 1) }
  '
}

ensure_docker_group() {
  require_command docker
  if docker info >/dev/null 2>&1; then
    return
  fi
  if [[ "${SFORGE_DOCKER_GROUP_REEXEC:-0}" == "1" ]]; then
    die "Docker is still unreachable after docker-group re-exec"
  fi

  local launch_user
  launch_user="$(id -un)"
  user_in_docker_group "${launch_user}" \
    || die "Docker is unreachable and ${launch_user} is not listed in the docker group"

  local entry_script="${SFORGE_ENTRY_SCRIPT:-$0}"
  entry_script="$(cd "$(dirname "${entry_script}")" && pwd)/$(basename "${entry_script}")"
  [[ -f "${entry_script}" && ! -L "${entry_script}" ]] \
    || die "unsafe docker-group re-exec target: ${entry_script}"

  local -a command_line=(
    env
    SFORGE_DOCKER_GROUP_REEXEC=1
    MLSYSFORGE_ROOT="${MLSYSFORGE_ROOT}"
    SFORGE_BIN="${SFORGE_BIN}"
    PYTHON_BIN="${PYTHON_BIN}"
    TASKS_DIR="${TASKS_DIR}"
    LOGS_DIR="${LOGS_DIR}"
    ARTIFACTS_DIR="${ARTIFACTS_DIR}"
    SFORGE_TASKS_DIR="${TASKS_DIR}"
    SFORGE_LOG_DIR="${LOGS_DIR}"
    SFORGE_KVBENCH_MIGRATION_ROOT="${MIGRATION_ROOT}"
  )

  # Preserve only non-secret, task-scoped controls needed by the formal
  # launcher.  API keys, bearer tokens, proxy credentials, and arbitrary host
  # environment variables are intentionally not forwarded.
  local name
  for name in \
    CODEX_AUTH_JSON_PATH \
    MODEL \
    REASONING_EFFORT \
    CODEX_CLI_VERSION \
    SFORGE_AGENT_RUNTIME_DIR \
    SFORGE_VERSION_STORE_ROOT \
    TASK_TIMEOUT \
    EVAL_INTERVAL \
    SUBMISSION_COOLDOWN \
    EVAL_TIMEOUT \
    DOCKER_BUILD_NETWORK \
    JUDGE_CPU_LIMIT \
    JUDGE_MEM_LIMIT \
    JUDGE_PORT \
    CONNECT_PROXY_PORT \
    RUN_STAMP \
    RUN_ID \
    SFORGE_CODEX_CLI_VERSION \
    SFORGE_CODEX_REASONING_EFFORT \
    SFORGE_ISOLATED_DOCKER_NETWORK \
    SFORGE_KVBENCH_NONCE_SOURCE \
    INFRATHON_ASSETS_ROOT \
    INFRATHON_HARDWARE \
    INFRATHON_PULL \
    SFORGE_KVBENCH_EVAL_ROOT \
    KVBENCH_CORPUS_ROOT \
    KVBENCH_VLLM_BASE_URL \
    KVBENCH_HIDDEN_BASELINE_PATH \
    KVBENCH_PUBLIC_BASELINE_PATH \
    KVBENCH_MODEL_PATH \
    KVBENCH_BACKEND_GPUS \
    KVBENCH_BACKEND_PORT \
    KVBENCH_BACKEND_CONTAINER \
    SFORGE_LLM_JUDGE_MODE; do
    if [[ -v "${name}" ]]; then
      command_line+=("${name}=${!name}")
    fi
  done
  command_line+=(bash "${entry_script}" "$@")

  local quoted
  printf -v quoted '%q ' "${command_line[@]}"
  exec sg docker -c "${quoted}"
}

require_sforge() {
  [[ -x "${SFORGE_BIN}" ]] || die "missing SForge executable: ${SFORGE_BIN}"
  [[ -x "${PYTHON_BIN}" ]] || die "missing MLSys Forge Python: ${PYTHON_BIN}"
  [[ -f "${MIGRATION_ROOT}/scripts/sforge_runtime_entry.py" \
     && ! -L "${MIGRATION_ROOT}/scripts/sforge_runtime_entry.py" ]] \
    || die "missing custom SForge runtime entry"
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

image_source_digest() {
  require_command sha256sum
  (
    cd "${MIGRATION_ROOT}"
    {
      find images/work images/judge \
        -type f \
        ! -path '*/__pycache__/*' \
        ! -name '*.pyc' \
        -print0
      find runtime/frozen/harbor_tasks/concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/environment/bench-controller \
        runtime/frozen/harbor_tasks/concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/environment/model-api \
        -type f \
        ! -path '*/__pycache__/*' \
        ! -name '*.pyc' \
        -print0
      printf '%s\0' \
        .dockerignore \
        runtime/frozen/harbor_tasks/concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45/environment/backend-serving-contract.json \
        judge/run_judge.py \
        judge/run_public.py \
        scripts/run_public_candidate.py
    } \
      | LC_ALL=C sort -zu \
      | while IFS= read -r -d '' source_file; do
          [[ -f "${source_file}" && ! -L "${source_file}" ]] \
            || die "unsafe or missing image source: ${source_file}"
          printf '%s\0' "${source_file}"
          sha256sum "${source_file}" | awk '{print $1}'
        done
  ) | sha256sum | awk '{print $1}'
}

judge_source_digest() {
  require_command sha256sum
  (
    cd "${MIGRATION_ROOT}"
    {
      find images/judge -type f -print0
      printf '%s\0' \
        judge/run_judge.py \
        judge/run_public.py \
        scripts/run_public_candidate.py
    } \
      | LC_ALL=C sort -zu \
      | while IFS= read -r -d '' source_file; do
          [[ -f "${source_file}" && ! -L "${source_file}" ]] \
            || die "unsafe or missing Judge source: ${source_file}"
          printf '%s\0' "${source_file}"
          sha256sum "${source_file}" | awk '{print $1}'
        done
  ) | sha256sum | awk '{print $1}'
}

assert_built_images_current() {
  require_command docker
  require_command jq
  local manifest="${LOGS_DIR}/build/image-manifest.json"
  [[ -s "${manifest}" && ! -L "${manifest}" ]] \
    || die "missing build manifest; run scripts/build_images.sh"

  local recorded_source recorded_work recorded_judge recorded_controller recorded_model_api recorded_task recorded_benchmark
  recorded_source="$(jq -er '.source_digest' "${manifest}")" \
    || die "build manifest lacks source_digest"
  recorded_work="$(jq -er '.work.id' "${manifest}")" \
    || die "build manifest lacks work image ID"
  recorded_judge="$(jq -er '.judge.id' "${manifest}")" \
    || die "build manifest lacks judge image ID"
  recorded_controller="$(jq -er '.public_controller.id' "${manifest}")" \
    || die "build manifest lacks public controller image ID"
  recorded_model_api="$(jq -er '.model_api.id' "${manifest}")" \
    || die "build manifest lacks model-api image ID"
  recorded_task="$(jq -er '.task_id' "${manifest}")" \
    || die "build manifest lacks task_id"
  recorded_benchmark="$(jq -er '.benchmark' "${manifest}")" \
    || die "build manifest lacks benchmark"

  [[ "${recorded_task}" == "${TASK_ID}" ]] \
    || die "build manifest belongs to another task"
  [[ "${recorded_benchmark}" == "${BENCHMARK_NAME}" ]] \
    || die "build manifest belongs to another benchmark"
  [[ "$(image_source_digest)" == "${recorded_source}" ]] \
    || die "image build sources changed; rebuild images"

  local current_work current_judge current_controller current_model_api
  current_work="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}" 2>/dev/null)" \
    || die "missing work image; run scripts/build_images.sh"
  current_judge="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}" 2>/dev/null)" \
    || die "missing judge image; run scripts/build_images.sh"
  current_controller="$(docker image inspect --format '{{.Id}}' "${PUBLIC_CONTROLLER_IMAGE}" 2>/dev/null)" \
    || die "missing public controller image; run scripts/build_images.sh"
  current_model_api="$(docker image inspect --format '{{.Id}}' "${MODEL_API_IMAGE}" 2>/dev/null)" \
    || die "missing model-api image; run scripts/build_images.sh"
  [[ "${current_work}" == "${recorded_work}" ]] \
    || die "work image tag no longer matches the recorded build"
  [[ "${current_judge}" == "${recorded_judge}" ]] \
    || die "judge image tag no longer matches the recorded build"
  [[ "${current_controller}" == "${recorded_controller}" ]] \
    || die "public controller image tag no longer matches the recorded build"
  [[ "${current_model_api}" == "${recorded_model_api}" ]] \
    || die "model-api image tag no longer matches the recorded build"
  echo "Image/source freshness: PASS (${recorded_source:0:12})"
}

wait_for_http() {
  local url="$1"
  local attempts="${2:-90}"
  [[ "${attempts}" =~ ^[0-9]+$ && "${attempts}" -gt 0 ]] \
    || die "HTTP health-check attempts must be a positive integer"
  local _
  for _ in $(seq 1 "${attempts}"); do
    if curl --connect-timeout 2 --max-time 5 -fsS "${url}" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  die "service did not become healthy: ${url}"
}
