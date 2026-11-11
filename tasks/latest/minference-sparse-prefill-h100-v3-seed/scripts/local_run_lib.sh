#!/usr/bin/env bash
# Host helpers for the local launcher (scripts/run_local_v3.sh): the SForge
# install, image names, assets, GPUs and Docker.  The GPU and Docker helpers are
# those of the standalone kernel launchers (standalone-sparse-mla v2 lib.sh).

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SFORGE_ROOT="${SFORGE_ROOT:-${MLSYSFORGE_ROOT:-${REPOSITORY_ROOT}}}"
MLSYSFORGE_ROOT="${SFORGE_ROOT}"
SFORGE_BIN="${SFORGE_BIN:-${SFORGE_ROOT}/.venv/bin/sforge}"
PYTHON_BIN="${PYTHON_BIN:-${SFORGE_ROOT}/.venv/bin/python}"
TASKS_DIR="${TASKS_DIR:-${MIGRATION_ROOT}/tasks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPOSITORY_ROOT}/outputs}"
LOGS_DIR="${LOGS_DIR:-${OUTPUT_ROOT}/$(basename "${MIGRATION_ROOT}")}"
export SFORGE_TASKS_DIR="${TASKS_DIR}"
export SFORGE_LOG_DIR="${LOGS_DIR}"

TASK_DIRECTORY="$(basename "${MIGRATION_ROOT}")"
TASK_ID="minference_sparse_prefill_h100_seed_v3"
BENCHMARK_NAME="minference_sparse_prefill"
# scripts/build_images.sh builds the thin images under these names
# (images/release.json local_tags); SForge looks the task's images up as
# <benchmark>.<role>.<task_id>:local, which the launcher tags from them.
THIN_WORK_IMAGE="mlsysforge/minference-prefill-work:v3-kernel-only-thin"
THIN_JUDGE_IMAGE="mlsysforge/minference-prefill-judge:v3-kernel-only-thin"
WORK_IMAGE="${BENCHMARK_NAME}.work.${TASK_ID}:local"
JUDGE_IMAGE="${BENCHMARK_NAME}.judge.${TASK_ID}:local"

# The three run-time trees (assets/manifest.json) under the Infrathon asset root.
INFRATHON_ASSETS_ROOT="${INFRATHON_ASSETS_ROOT:-${HOME}/.cache/infrathon/assets}"
TASK_ASSETS="${INFRATHON_ASSETS_ROOT}/${TASK_DIRECTORY}"
MODEL_ASSET="${TASK_ASSETS}/public/models/Meta-Llama-3.1-8B-Instruct"
WORK_MODEL_ASSET="${TASK_ASSETS}/public/models/Llama-3-8B-Instruct-262k"
PUBLIC_PROMPTS_ASSET="${TASK_ASSETS}/public/public-prompts"
PRIVATE_PROMPTS_ASSET="${TASK_ASSETS}/judge/kernel-prompts"

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
    INFRATHON_ASSETS_ROOT="${INFRATHON_ASSETS_ROOT}"
  )
  local name
  for name in \
    CODEX_AUTH_JSON_PATH \
    CODEX_CLI_VERSION \
    EVAL_ARCHIVE \
    FINAL_EVAL_TIMEOUT \
    SKIP_INTEGRITY_REVIEW \
    SFORGE_AGENT_RUNTIME_DIR \
    SFORGE_VERSION_STORE_ROOT \
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
  [[ "${uuid}" =~ ^GPU-[0-9a-f-]+$ ]] || die "unsafe GPU UUID: ${uuid}"
  echo "/tmp/sforge-gpu-${uuid}.lock"
}

assert_gpu_idle() {
  local index="$1"
  local role="$2"
  # The Work and Judge each need a whole H100 (deploy/pyromind/minference-v3:
  # min_gpu_memory_bytes 80e9).
  local minimum_free="${GPU_MIN_FREE_MIB:-${H100_MIN_FREE_MIB:-76000}}"
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

# One exact DeviceRequest, as the Work and Judge containers get it
# (scripts/sforge_local_entry_v3.py).
assert_container_accelerator() {
  local image="$1"
  local expected_uuid="$2"
  local role="$3"
  local seen
  seen="$(docker run --rm --pull=never \
    --gpus "device=${expected_uuid}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    --entrypoint nvidia-smi \
    "${image}" --query-gpu=uuid --format=csv,noheader | xargs)"
  [[ "${seen}" == "${expected_uuid}" ]] \
    || die "${role} container expected ${expected_uuid}, saw ${seen:-nothing}"
  echo "${role} container exact DeviceRequest: ${seen}"
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

# The thin images match the build record and carry SForge's task names.
assert_images_current() {
  "${PYTHON_BIN}" "${REPOSITORY_ROOT}/scripts/infrathon/image_record.py" check "${TASK_DIRECTORY}" \
    || die "the MInference images are missing or stale; run scripts/build_images.sh (or scripts/infrathon/pull_images.sh ${TASK_DIRECTORY})"
  docker tag "${THIN_WORK_IMAGE}" "${WORK_IMAGE}"
  docker tag "${THIN_JUDGE_IMAGE}" "${JUDGE_IMAGE}"
}

# Run-scoped copies of the prompt trees: the Work and the Judge refuse prompt
# files with more than one link (an asset root exported for upload is
# hard-linked), and the final prompts become root-only so that no unprivileged
# process in the Judge (UID 1000, the candidate's) can read them, whatever the
# host user's UID.  The model is mounted from the asset root (links allowed).
prepare_run_assets() {
  local destination="$1"
  mkdir -p "${destination}"
  rm -rf "${destination}/public-prompts" "${destination}/kernel-prompts"
  cp -r --no-preserve=links "${PUBLIC_PROMPTS_ASSET}" "${destination}/public-prompts"
  chmod 0755 "${destination}/public-prompts"
  chmod 0444 "${destination}/public-prompts"/*
  cp -r "${PRIVATE_PROMPTS_ASSET}" "${destination}/kernel-prompts"
  docker run --rm --pull=never --network none --user 0 --entrypoint sh \
    -v "${destination}:/target" "${JUDGE_IMAGE}" \
    -c 'chown -R 0:0 /target/kernel-prompts && chmod 0500 /target/kernel-prompts && chmod 0400 /target/kernel-prompts/*' \
    || die "could not make the run's final prompts root-only"
}

remove_run_assets() {
  local destination="$1"
  [[ -d "${destination}" ]] || return 0
  if [[ -e "${destination}/kernel-prompts" ]]; then
    docker run --rm --pull=never --network none --user 0 --entrypoint sh \
      -v "${destination}:/target" "${JUDGE_IMAGE}" -c 'rm -rf /target/kernel-prompts' >/dev/null 2>&1 \
      || echo "WARNING: could not remove ${destination}/kernel-prompts (root-owned)" >&2
  fi
  rm -rf "${destination}/public-prompts"
  rmdir "${destination}" 2>/dev/null || true
}
