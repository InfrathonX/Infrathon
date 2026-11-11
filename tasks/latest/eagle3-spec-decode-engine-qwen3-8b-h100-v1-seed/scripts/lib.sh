#!/usr/bin/env bash
# Shared lifecycle primitives for the EAGLE-3 speculative-decode task.
# Provenance: LLaDA v2 scripts/lib.sh, adapted for two immutable model mounts
# and the compact policy schema used by this task.
set -Eeuo pipefail

find_repository_root() {
  local cursor
  cursor="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  while [[ "${cursor}" != "/" ]]; do
    if [[ -f "${cursor}/pyproject.toml" && -d "${cursor}/sforge" ]]; then
      printf '%s\n' "${cursor}"
      return 0
    fi
    cursor="$(dirname "${cursor}")"
  done
  return 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATION_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPOSITORY_ROOT="${REPOSITORY_ROOT:-$(find_repository_root)}"
SFORGE_ROOT="${SFORGE_ROOT:-${MLSYSFORGE_ROOT:-${REPOSITORY_ROOT}}}"
SFORGE_BIN="${SFORGE_BIN:-${SFORGE_ROOT}/.venv/bin/sforge}"
PYTHON_BIN="${PYTHON_BIN:-${SFORGE_ROOT}/.venv/bin/python}"
TASKS_DIR="${TASKS_DIR:-${MIGRATION_ROOT}/tasks}"
TASK_ID="eagle3_spec_decode_engine_qwen3_8b_h100_v1"
BENCHMARK="eagle3_spec_decode"
WORK_IMAGE="${BENCHMARK}.work.${TASK_ID}:local"
JUDGE_IMAGE="${BENCHMARK}.judge.${TASK_ID}:local"
# The Judge above is this public Judge (images/judge/Dockerfile, publishable) plus
# the holdout layer images/judge/Dockerfile.private, which is only ever built on
# the host.
JUDGE_PUBLIC_IMAGE="${BENCHMARK}.judge-public.${TASK_ID}:local"

# Centralize all mutable output outside the task sources. The task-specific
# subdirectory prevents build manifests and run IDs from colliding with peers.
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPOSITORY_ROOT}/outputs/$(basename "${MIGRATION_ROOT}")}"
LOGS_DIR="${LOGS_DIR:-${OUTPUT_ROOT}}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${OUTPUT_ROOT}/artifacts}"
BUILD_DIR="${BUILD_DIR:-${OUTPUT_ROOT}/build}"

EAGLE3_TARGET_REPO="${EAGLE3_TARGET_REPO:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--Qwen--Qwen3-8B}"
# Infrathon assets (assets/fetch.sh): INFRATHON_ASSETS_ROOT when set.  A public
# checkout carries no in-repository Judge-only data, so there the asset root
# defaults to ~/.cache/infrathon/assets.
if [[ -z "${INFRATHON_ASSETS_ROOT:-}" && ! -d "${MIGRATION_ROOT}/images/judge/private_data" ]]; then
  INFRATHON_ASSETS_ROOT="${HOME}/.cache/infrathon/assets"
fi
TASK_ASSETS_DIR="${INFRATHON_ASSETS_ROOT:+${INFRATHON_ASSETS_ROOT}/$(basename "${MIGRATION_ROOT}")}"
# The frozen draft head: the task asset public/eagle3-head when an asset root is
# in use, else the host artifact directory.
if [[ -n "${TASK_ASSETS_DIR}" ]]; then
  EAGLE3_HEAD_DIR="${EAGLE3_HEAD_DIR:-${TASK_ASSETS_DIR}/public/eagle3-head}"
fi
EAGLE3_HEAD_DIR="${EAGLE3_HEAD_DIR:-${HOME}/mlsysforge-artifacts/eagle3-spec-decode/heads/eagle3-head}"
# Judge-only holdout data (policy, hidden prompts, reference streams): the task
# asset judge/private_data when an asset root is in use, else the in-repository
# copy.  scripts/generate_protected_hashes.py private_data_dir() resolves the
# same way.
if [[ -z "${EAGLE3_PRIVATE_DATA:-}" ]]; then
  if [[ -n "${TASK_ASSETS_DIR}" ]]; then
    EAGLE3_PRIVATE_DATA="${TASK_ASSETS_DIR}/judge/private_data"
  else
    EAGLE3_PRIVATE_DATA="${MIGRATION_ROOT}/images/judge/private_data"
  fi
fi
# Capability envelope, never a GPU model name. The real resident set is the
# frozen Qwen3-8B target plus the EAGLE-3 head and one tree KV cache (~21 GiB
# measured on H100 NVL); 34 GiB is that need plus the launcher's own margin.
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-${H100_MIN_FREE_MIB:-34816}}"
GPU_MIN_COMPUTE_CAPABILITY="${GPU_MIN_COMPUTE_CAPABILITY:-8.0}"

export SFORGE_TASKS_DIR="${TASKS_DIR}"
export SFORGE_LOG_DIR="${LOGS_DIR}"
export EAGLE3_TARGET_REPO
export EAGLE3_HEAD_DIR
export EAGLE3_PRIVATE_DATA

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

require_sforge() {
  [[ -x "${SFORGE_BIN}" ]] || die "SForge executable is unavailable: ${SFORGE_BIN}"
  [[ -x "${PYTHON_BIN}" ]] || die "SForge Python is unavailable: ${PYTHON_BIN}"
}

assert_private_auth_file() {
  local path="$1"
  [[ -f "${path}" && ! -L "${path}" && -s "${path}" ]] \
    || die "Codex auth JSON must be a nonempty regular file"
  [[ "$(stat -c '%a' "${path}")" == "600" ]] \
    || die "Codex auth JSON must have mode 0600"
  [[ "$(stat -c '%u' "${path}")" == "$(id -u)" ]] \
    || die "Codex auth JSON must be owned by the launcher"
}

canonical_gpu_index() {
  [[ "$1" =~ ^[0-9]+$ ]] || die "GPU index must be numeric"
  printf '%d\n' "$((10#$1))"
}

gpu_query() {
  local index="$1"
  local field="$2"
  nvidia-smi --id="${index}" --query-gpu="${field}" \
    --format=csv,noheader,nounits | xargs
}

gpu_uuid() {
  gpu_query "$1" uuid
}

gpu_name() {
  gpu_query "$1" name
}

gpu_lock_path() {
  local uuid="$1"
  [[ "${uuid}" =~ ^GPU-[A-Za-z0-9-]+$ ]] || die "unsafe GPU UUID: ${uuid}"
  printf '/tmp/sforge-gpu-%s.lock\n' "${uuid}"
}

gpu_has_compute_process() {
  local uuid="$1"
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
    2>/dev/null | grep -Fxq "${uuid}"
}

running_container_claims_gpu() {
  local uuid="$1"
  local container
  while IFS= read -r container; do
    [[ -n "${container}" ]] || continue
    if docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' \
      "${container}" 2>/dev/null | grep -Eq "^(NVIDIA_VISIBLE_DEVICES|EAGLE3_(WORK|JUDGE)_GPU_UUID)=${uuid}([,]|$)"; then
      return 0
    fi
  done < <(docker ps -q 2>/dev/null)
  return 1
}

remove_task_gpu_containers() {
  # Remove only containers bearing the exact task, run, role, and UUID labels
  # that our adapter set. A new-run guard prevents collisions with older runs.
  local role="$1"
  local uuid="$2"
  local run_id="$3"
  local container_id
  while IFS= read -r container_id; do
    [[ -n "${container_id}" ]] || continue
    docker rm -f -- "${container_id}" >/dev/null
  done < <(
    docker ps -aq \
      --filter "label=mlsysforge.task_id=${TASK_ID}" \
      --filter "label=mlsysforge.run_id=${run_id}" \
      --filter "label=mlsysforge.gpu_role=${role}" \
      --filter "label=mlsysforge.gpu_uuid=${uuid}"
  )
}

assert_uuid_unassigned() {
  local uuid="$1"
  local role="$2"
  ! gpu_has_compute_process "${uuid}" \
    || die "${role} GPU ${uuid} already has a compute process"
  ! running_container_claims_gpu "${uuid}" \
    || die "${role} GPU ${uuid} is claimed by a running container"
}

# The admission gate is a capability envelope: compute capability, free memory
# and exclusivity. The model name and UUID are recorded as provenance only.
assert_gpu_available() {
  local index="$1"
  local role="${2:-assigned}"
  local name
  local uuid
  local free
  local capability
  name="$(gpu_name "${index}")"
  uuid="$(gpu_uuid "${index}")"
  free="$(gpu_query "${index}" memory.free)"
  capability="$(gpu_query "${index}" compute_cap)"
  [[ "${capability}" =~ ^[0-9]+\.[0-9]+$ ]] \
    || die "${role} GPU ${index} reported no compute capability: ${capability}"
  awk -v have="${capability}" -v want="${GPU_MIN_COMPUTE_CAPABILITY}" '
    BEGIN {
      split(have, a, "."); split(want, b, ".");
      exit !(a[1] > b[1] || (a[1] == b[1] && a[2] >= b[2]));
    }' \
    || die "${role} GPU ${index} compute capability ${capability} is below ${GPU_MIN_COMPUTE_CAPABILITY}: ${name}"
  (( free >= GPU_MIN_FREE_MIB )) \
    || die "${role} GPU ${index} has only ${free} MiB free"
  assert_uuid_unassigned "${uuid}" "${role}"
  echo "${role} GPU: physical ${index}, ${name}, ${uuid}, cc ${capability}, ${free} MiB free"
}

acquire_dual_gpu_locks() {
  local work_uuid="$1"
  local judge_uuid="$2"
  local first_uuid
  local second_uuid
  [[ "${work_uuid}" != "${judge_uuid}" ]] || die "GPU UUIDs must differ"
  if [[ "${work_uuid}" < "${judge_uuid}" ]]; then
    first_uuid="${work_uuid}"
    second_uuid="${judge_uuid}"
  else
    first_uuid="${judge_uuid}"
    second_uuid="${work_uuid}"
  fi
  exec 8>"$(gpu_lock_path "${first_uuid}")"
  exec 9>"$(gpu_lock_path "${second_uuid}")"
  flock -n 8 || die "GPU lock is already held: ${first_uuid}"
  flock -n 9 || die "GPU lock is already held: ${second_uuid}"
  export EAGLE3_LOCKED_GPU_UUIDS="${first_uuid},${second_uuid}"
}

assert_dual_gpu_locks_held() {
  [[ -n "${EAGLE3_LOCKED_GPU_UUIDS:-}" ]] \
    || die "dual GPU lock identity is absent"
  flock -n 8 || true
  flock -n 9 || true
}

assert_distinct_gpu_pair() {
  local work_index="$1"
  local judge_index="$2"
  [[ "${work_index}" != "${judge_index}" ]] \
    || die "Work and Judge GPU indices must differ"
  [[ "$(gpu_uuid "${work_index}")" != "$(gpu_uuid "${judge_index}")" ]] \
    || die "Work and Judge GPU UUIDs must differ"
  assert_gpu_available "${work_index}" "Work"
  assert_gpu_available "${judge_index}" "Judge"
}

assert_container_gpu() {
  local image="$1"
  local expected_uuid="$2"
  local role="$3"
  local observed
  observed="$(docker run --rm --network none \
    --gpus "device=${expected_uuid}" \
    -e "NVIDIA_VISIBLE_DEVICES=${expected_uuid}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    --entrypoint nvidia-smi "${image}" \
    --query-gpu=uuid --format=csv,noheader | xargs)"
  [[ "${observed}" == "${expected_uuid}" ]] \
    || die "${role} container GPU mismatch: ${observed}"
}

# ---------------------------------------------------------------------------
# Single-allocation (one whole GPU shared serially by Work and Judge) helpers.
# The scientific contract is unchanged; these only describe the deployment.
# ---------------------------------------------------------------------------

acquire_single_gpu_lock() {
  # One UUID-keyed flock for the entire run: Work, every interactive Judge
  # and the standalone final Judge serialize on it.
  local uuid="$1"
  exec 8>"$(gpu_lock_path "${uuid}")"
  flock -n 8 || die "single GPU allocation ${uuid} lock is already held"
  export EAGLE3_LOCKED_GPU_UUIDS="${uuid}"
}

assert_single_gpu_lock_held() {
  [[ -n "${EAGLE3_LOCKED_GPU_UUIDS:-}" ]] \
    || die "single GPU lock identity is absent"
  flock -n 8 || true
}

assert_container_accelerator() {
  # Exact DeviceRequest probe of one image against the allocated whole GPU.
  local image="$1"
  local expected_uuid="$2"
  local role="$3"
  local seen
  [[ "${expected_uuid}" =~ ^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]] \
    || die "${role} accelerator must be a canonical whole-GPU UUID"
  seen="$(docker run --rm --network none \
    --gpus "device=${expected_uuid}" \
    -e "NVIDIA_VISIBLE_DEVICES=${expected_uuid}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    --entrypoint nvidia-smi "${image}" \
    --query-gpu=uuid --format=csv,noheader | xargs)"
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
  assert_uuid_unassigned "${uuid}" "${role}"
  echo "${role} allocation: ${name}, ${uuid}, compute ${capability}, ${free_mib} MiB free"
}

gpu_index_of_uuid() {
  # Resolve the host ordinal of one whole-GPU UUID for the identity preflight.
  local uuid="$1"
  local index
  index="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
    | awk -F', *' -v uuid="${uuid}" '$2 == uuid {print $1}')"
  [[ "${index}" =~ ^[0-9]+$ ]] || die "no host GPU carries UUID ${uuid}"
  printf '%s\n' "${index}"
}

assert_gpu_released() {
  local uuid="$1"
  local index="${2:-unknown}"
  local role="${3:-assigned}"
  local attempt
  for attempt in $(seq 1 60); do
    if ! gpu_has_compute_process "${uuid}"; then
      return 0
    fi
    sleep 1
  done
  die "${role} GPU ${index} still has a compute process: ${uuid}"
}

# Covers what the images are built from (images/, plus the task spec). Host-side
# launcher scripts under scripts/ never enter an image, so editing them (the agent
# CLI pin, the model whitelist) must not mark the images stale. images/published.json
# is written by publishing, after the images are built from these sources.
image_source_digest() {
  (
    cd "${MIGRATION_ROOT}"
    find images tasks \
      -type f \
      ! -path images/published.json \
      ! -path '*/__pycache__/*' \
      ! -path '*/.pytest_cache/*' \
      ! -name '*.pyc' \
      ! -name '*.pyo' \
      -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

assert_built_images_current() {
  local manifest="${BUILD_DIR}/image-manifest.json"
  local recorded_source
  local work_id
  local judge_id
  [[ -f "${manifest}" && ! -L "${manifest}" ]] \
    || die "image manifest missing; run build_images.sh"
  recorded_source="$(jq -er '.source_digest' "${manifest}")"
  work_id="$(jq -er '.work.id' "${manifest}")"
  judge_id="$(jq -er '.judge.id' "${manifest}")"
  [[ "${recorded_source}" == "$(image_source_digest)" ]] \
    || die "built images are stale"
  [[ "$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")" == "${work_id}" ]] \
    || die "Work image tag moved"
  [[ "$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")" == "${judge_id}" ]] \
    || die "Judge image tag moved"
}

ensure_docker_group() {
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  [[ "${SFORGE_DOCKER_GROUP_REEXEC:-0}" != "1" ]] \
    || die "Docker unavailable after docker-group re-exec"
  local -a forwarded
  local name
  local quoted
  forwarded=(
    env
    SFORGE_DOCKER_GROUP_REEXEC=1
    "REPOSITORY_ROOT=${REPOSITORY_ROOT}"
    "SFORGE_ROOT=${SFORGE_ROOT}"
    "SFORGE_BIN=${SFORGE_BIN}"
    "PYTHON_BIN=${PYTHON_BIN}"
  )
  for name in CODEX_AUTH_JSON_PATH CODEX_CLI_VERSION SFORGE_AGENT_RUNTIME_DIR \
    SFORGE_VERSION_STORE_ROOT MODEL MODEL_API_BASE_URL MODEL_API_KEY_FILE \
    REASONING_EFFORT RUN_ID RUN_STAMP \
    TASK_TIMEOUT EVAL_INTERVAL FINAL_EVAL_TIMEOUT WORK_GPU JUDGE_GPU \
    EAGLE3_TARGET_REPO EAGLE3_HEAD_DIR GPU_MIN_FREE_MIB GPU_MIN_COMPUTE_CAPABILITY \
    H100_MIN_FREE_MIB HOST_PROXY_PORT \
    SMOKE_JUDGE_PORT EAGLE3_INTERACTIVE_JUDGE_TIMEOUT JUDGE_PORT \
    SFORGE_SINGLE_GPU_HANDOFF SFORGE_ACCELERATOR_DEVICE_UUID \
    SFORGE_ACCELERATOR_ALLOCATION_KEY SFORGE_ACCELERATOR_OBSERVER \
    SFORGE_TRUSTED_LEASE_ID SFORGE_CALIBRATION_EPOCH INFRATHON_ASSETS_ROOT INFRATHON_HARDWARE INFRATHON_PULL; do
    if [[ -v "${name}" ]]; then
      forwarded+=("${name}=${!name}")
    fi
  done
  forwarded+=(bash "${SFORGE_ENTRY_SCRIPT:-$0}" "$@")
  printf -v quoted '%q ' "${forwarded[@]}"
  exec sg docker -c "${quoted}"
}

# Every Judge container SForge creates for this task receives its role inputs
# only through SFORGE_JUDGE_EXTRA_ENV. The adapter's
# _validated_container_environment fails closed when any of these keys is
# missing, which is how the first formal run lost its only interactive round:
# the interactive path did not receive NVIDIA_VISIBLE_DEVICES. The formal
# launcher and the submission-plane smoke therefore share this one key list,
# one builder, and one guard so their Judge environments cannot drift apart.
INTERACTIVE_JUDGE_ENV_KEYS=(
  NVIDIA_VISIBLE_DEVICES
  CUDA_VISIBLE_DEVICES
  EAGLE3_GPU_ROLE
  EAGLE3_COHORT_SECRET
  EAGLE3_INTERACTIVE_JUDGE_TIMEOUT
  EAGLE3_FINAL_JUDGE_TIMEOUT
)

judge_extra_env_spec() {
  # Print the exact SFORGE_JUDGE_EXTRA_ENV value for one interactive Judge
  # server: the physical Judge UUID, logical cuda:0, the Judge role, the
  # per-run HMAC cohort secret, and both explicit positive deadlines, in the
  # fixed key order of INTERACTIVE_JUDGE_ENV_KEYS.
  local judge_uuid="$1"
  local cohort_secret="$2"
  local interactive_timeout="$3"
  local final_timeout="$4"
  [[ "${judge_uuid}" =~ ^GPU-[A-Za-z0-9-]+$ ]] \
    || die "interactive Judge environment requires one physical GPU UUID"
  [[ "${cohort_secret}" =~ ^[0-9a-f]{64}$ ]] \
    || die "interactive Judge environment requires a 64-hex cohort secret"
  [[ "${interactive_timeout}" =~ ^[1-9][0-9]*$ ]] \
    || die "EAGLE3_INTERACTIVE_JUDGE_TIMEOUT must be a positive integer"
  [[ "${final_timeout}" =~ ^[1-9][0-9]*$ ]] \
    || die "EAGLE3_FINAL_JUDGE_TIMEOUT must be a positive integer"
  printf 'NVIDIA_VISIBLE_DEVICES=%s,CUDA_VISIBLE_DEVICES=0,EAGLE3_GPU_ROLE=judge,EAGLE3_COHORT_SECRET=%s,EAGLE3_INTERACTIVE_JUDGE_TIMEOUT=%s,EAGLE3_FINAL_JUDGE_TIMEOUT=%s\n' \
    "${judge_uuid}" \
    "${cohort_secret}" \
    "${interactive_timeout}" \
    "${final_timeout}"
}

assert_interactive_judge_env_spec() {
  # Parse one SFORGE_JUDGE_EXTRA_ENV value the way SForge parses it
  # (comma-separated KEY=VALUE items) and require every interactive Judge key
  # exactly once, bound to the expected physical UUID, with no extra keys.
  # The parsed values must rebuild the identical string through
  # judge_extra_env_spec, so key order and spelling cannot drift either.
  local spec="$1"
  local expected_uuid="$2"
  local item
  local key
  local value
  local name
  local rebuilt
  local -A values=()
  local -a items=()
  [[ -n "${spec}" ]] \
    || die "interactive Judge environment specification is empty"
  IFS=',' read -r -a items <<<"${spec}"
  for item in ${items[@]+"${items[@]}"}; do
    [[ "${item}" == *=* ]] \
      || die "interactive Judge environment item lacks '=': ${item}"
    key="${item%%=*}"
    value="${item#*=}"
    [[ -z "${values[${key}]+set}" ]] \
      || die "interactive Judge environment repeats ${key}"
    values["${key}"]="${value}"
  done
  for name in "${INTERACTIVE_JUDGE_ENV_KEYS[@]}"; do
    [[ -n "${values[${name}]+set}" ]] \
      || die "interactive Judge environment omits ${name}"
  done
  (( ${#values[@]} == ${#INTERACTIVE_JUDGE_ENV_KEYS[@]} )) \
    || die "interactive Judge environment carries unexpected keys"
  [[ "${values[NVIDIA_VISIBLE_DEVICES]}" == "${expected_uuid}" ]] \
    || die "interactive Judge NVIDIA_VISIBLE_DEVICES is not ${expected_uuid}"
  [[ "${values[CUDA_VISIBLE_DEVICES]}" == "0" ]] \
    || die "interactive Judge CUDA_VISIBLE_DEVICES must be the logical ordinal 0"
  [[ "${values[EAGLE3_GPU_ROLE]}" == "judge" ]] \
    || die "interactive Judge EAGLE3_GPU_ROLE must be judge"
  [[ "${values[EAGLE3_COHORT_SECRET]}" =~ ^[0-9a-f]{64}$ ]] \
    || die "interactive Judge EAGLE3_COHORT_SECRET must be a 64-hex secret"
  [[ "${values[EAGLE3_INTERACTIVE_JUDGE_TIMEOUT]}" =~ ^[1-9][0-9]*$ ]] \
    || die "interactive Judge EAGLE3_INTERACTIVE_JUDGE_TIMEOUT must be a positive integer"
  [[ "${values[EAGLE3_FINAL_JUDGE_TIMEOUT]}" =~ ^[1-9][0-9]*$ ]] \
    || die "interactive Judge EAGLE3_FINAL_JUDGE_TIMEOUT must be a positive integer"
  rebuilt="$(
    judge_extra_env_spec \
      "${values[NVIDIA_VISIBLE_DEVICES]}" \
      "${values[EAGLE3_COHORT_SECRET]}" \
      "${values[EAGLE3_INTERACTIVE_JUDGE_TIMEOUT]}" \
      "${values[EAGLE3_FINAL_JUDGE_TIMEOUT]}"
  )"
  [[ "${rebuilt}" == "${spec}" ]] \
    || die "interactive Judge environment differs from the shared specification"
}

wait_for_http() {
  local url="$1"
  local attempts="${2:-60}"
  local attempt
  for attempt in $(seq 1 "${attempts}"); do
    if curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS \
      "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  die "service did not become healthy: ${url}"
}
