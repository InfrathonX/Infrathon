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

TASK_ID="kvzip_query_agnostic_kv_scoring_qwen3_8b_h100_scratch_v2"
BENCHMARK_NAME="kvzip_kv_scoring"
WORK_IMAGE="${BENCHMARK_NAME}.work.${TASK_ID}:local"
JUDGE_IMAGE="${BENCHMARK_NAME}.judge.${TASK_ID}:local"
# The Judge above is this public Judge (images/judge/Dockerfile, publishable) plus
# the holdout layer images/judge/Dockerfile.private, which is only ever built on
# the host.
JUDGE_PUBLIC_IMAGE="${BENCHMARK_NAME}.judge-public.${TASK_ID}:local"
# The frozen final holdout ships as a manifest on the Judge-only path: one
# (dataset, revision, file, row, sha256) record per item.  The items
# themselves are rebuilt from the pinned Hugging Face parquet files at build
# time (scripts/prepare_items.py materialize) and staged 0600 into the Judge
# build context; they are never committed and never reach the Work image.
FINAL_MANIFEST_PATH="${MIGRATION_ROOT}/judge/private/final_manifest.json"
PRIVATE_NEEDLES_PATH="${MIGRATION_ROOT}/judge/private/needle_private.json"
# Infrathon assets: with INFRATHON_ASSETS_ROOT set, the Judge-only inputs come from
# <INFRATHON_ASSETS_ROOT>/<task directory>/judge/ instead (scripts/infrathon/assets.py).  A public
# checkout has no in-repository copy; it uses the default asset root that
# assets/fetch.sh fills when INFRATHON_ASSETS_ROOT is unset.
judge_assets_root="${INFRATHON_ASSETS_ROOT:-}"
if [[ -z "${judge_assets_root}" && ! -f "${FINAL_MANIFEST_PATH}" ]]; then
  judge_assets_root="${XDG_CACHE_HOME:-${HOME}/.cache}/infrathon/assets"
fi
if [[ -n "${judge_assets_root}" ]]; then
  FINAL_MANIFEST_PATH="${judge_assets_root}/$(basename "${MIGRATION_ROOT}")/judge/final_manifest.json"
  PRIVATE_NEEDLES_PATH="${judge_assets_root}/$(basename "${MIGRATION_ROOT}")/judge/needle_private.json"
fi
FINAL_ITEMS_CONTEXT="${LOGS_DIR}/build/final-items-context"
KV_SCORING_MODEL_REPO="${KV_SCORING_MODEL_REPO:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--Qwen--Qwen3-8B}"
# Hugging Face hub cache holding the pinned dataset snapshots (SQuAD, GSM8K,
# SCBench-preprocessed, RULER); scripts/prepare_items.py lists the pins.
KV_SCORING_HF_HUB="${KV_SCORING_HF_HUB:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}}"
# snu-mllab/KVzip checkout at the pinned commit: its PaulGrahamEssays files
# are the NIAH haystack.  Cloned under LOGS_DIR when not given.
KV_SCORING_KVZIP_REPO="${KV_SCORING_KVZIP_REPO:-}"
KVZIP_COMMIT="5d8472975d286e3da6b52178af39d18904330c0d"
SFORGE_PYTHON="${SFORGE_PYTHON:-${PYTHON_BIN}}"
TASK_ENTRY="${MIGRATION_ROOT}/scripts/sforge_entry.py"
export KV_SCORING_MODEL_REPO

validate_model_repo() {
  local repo="${1:-${KV_SCORING_MODEL_REPO}}" revision snapshot
  [[ -d "${repo}" && ! -L "${repo}" && -f "${repo}/refs/main" ]] || die "invalid Qwen3 model repository: ${repo} (set KV_SCORING_MODEL_REPO to a Hugging Face cache-layout directory holding refs/main)"
  IFS= read -r revision < "${repo}/refs/main" || [[ -n "${revision}" ]]
  [[ "${revision}" =~ ^[0-9a-f]{40,64}$ ]] || die "invalid model snapshot hash"
  snapshot="${repo}/snapshots/${revision}"
  [[ -d "${snapshot}" && ! -L "${snapshot}" && -f "${snapshot}/config.json" ]] || die "missing pinned model snapshot"
  python3 - "${snapshot}/config.json" <<'PY'
import json, sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
assert config.get("model_type") == "qwen3"
assert config.get("num_hidden_layers") == 36
PY
  printf '%s\n' "${revision}"
}

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
    SFORGE_PYTHON="${SFORGE_PYTHON}"
    TASK_ENTRY="${TASK_ENTRY}"
    KV_SCORING_MODEL_REPO="${KV_SCORING_MODEL_REPO}"
    KV_SCORING_HF_HUB="${KV_SCORING_HF_HUB}"
    KV_SCORING_KVZIP_REPO="${KV_SCORING_KVZIP_REPO}"
    TASKS_DIR="${TASKS_DIR}"
    LOGS_DIR="${LOGS_DIR}"
    ARTIFACTS_DIR="${ARTIFACTS_DIR}"
  )
  local name
  for name in \
    CODEX_AUTH_JSON_PATH \
    CODEX_CLI_VERSION \
    FINAL_EVAL_TIMEOUT \
    INFRATHON_ASSETS_ROOT \
    INFRATHON_HARDWARE \
    INFRATHON_PULL \
    SFORGE_AGENT_RUNTIME_DIR \
    SFORGE_VERSION_STORE_ROOT \
    SFORGE_SINGLE_GPU_HANDOFF \
    SFORGE_ACCELERATOR_DEVICE_UUID \
    SFORGE_ACCELERATOR_ALLOCATION_KEY \
    SFORGE_ACCELERATOR_OBSERVER \
    SFORGE_TRUSTED_LEASE_ID \
    SFORGE_CALIBRATION_EPOCH \
    KV_SCORING_DATA_PYTHON \
    JUDGE_GPUS \
    GPU_MIN_FREE_MIB \
    GPU_MIN_FREE_MIB \
    H100_MIN_FREE_MIB \
    HOST_TEST_PYTHON \
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

# Hold both host lock names for a GPU until the calling shell exits: sforge's
# /tmp/sforge-gpu-<uuid>.lock and /tmp/mlsysforge-gpu-<uuid>.lock, which other
# launchers on a shared host check before they touch a GPU.
lock_gpu() {
  local uuid="$1" role="$2" path lock_fd
  for path in "$(gpu_lock_path "${uuid}")" "/tmp/mlsysforge-gpu-${uuid//\//-}.lock"; do
    exec {lock_fd}>"${path}"
    flock -n "${lock_fd}" || die "${role} GPU ${uuid}: ${path} is held by another process"
  done
}

assert_gpu_idle() {
  local index="$1"
  local role="$2"
  local minimum_free="${GPU_MIN_FREE_MIB:-${H100_MIN_FREE_MIB:-80896}}"
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
    } \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

assert_built_images_current() {
  local manifest="${LOGS_DIR}/build/image-manifest.json"
  [[ -s "${manifest}" ]] || die "missing build manifest; run scripts/build_images.sh"
  local recorded_source recorded_work recorded_judge recorded_manifest recorded_items
  local current_source current_work current_judge
  recorded_source="$(jq -er '.source_digest' "${manifest}")"
  recorded_work="$(jq -er '.work.id' "${manifest}")"
  recorded_judge="$(jq -er '.judge.id' "${manifest}")"
  recorded_manifest="$(jq -er '.final_suite.manifest_sha256' "${manifest}")"
  recorded_items="$(jq -er '.final_suite.items_sha256' "${manifest}")"
  current_source="$(image_source_digest)"
  current_work="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}" 2>/dev/null)" \
    || die "missing work image; run scripts/build_images.sh"
  current_judge="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}" 2>/dev/null)" \
    || die "missing judge image; run scripts/build_images.sh"
  [[ "${current_source}" == "${recorded_source}" ]] || die "image-driving source changed; rebuild images"
  [[ "${current_work}" == "${recorded_work}" ]] || die "work image tag moved after the recorded build"
  [[ "${current_judge}" == "${recorded_judge}" ]] || die "judge image tag moved after the recorded build"
  [[ "$(sha256sum "${FINAL_MANIFEST_PATH}" | awk '{print $1}')" == "${recorded_manifest}" ]] \
    || die "final holdout manifest changed after the recorded build"
  echo "Image/source freshness: PASS (${current_source:0:12})"
  echo "Final holdout: manifest ${recorded_manifest:0:12}, items ${recorded_items:0:12}"
}

# Judge timeouts in seconds for N GPUs: "candidate total judge eval".  Sized
# from the measured full-suite cost on H100 NVL: Phase A at most
# 4 GPU-s per row for a scorer at the overhead limit, Phase A + B at most 10
# GPU-s per row; 1.5x headroom; the 3-GPU defaults (7200 / 10800) are the
# floor.  Explicit KV_SCORING_*_TIMEOUT values win.
judge_timeouts() {
  local n="$1" rows=1812 candidate total
  candidate=$(( 600 + 600 * rows / (100 * n) ))
  total=$(( 900 + 1500 * rows / (100 * n) ))
  (( candidate >= 7200 )) || candidate=7200
  (( total >= 10800 )) || total=10800
  echo "${candidate} ${total} $(( total + 100 )) $(( total + 200 ))"
}

# Judge GPUs: a comma-separated list of host indices (JUDGE_GPUS, e.g. 1,2,3).
# Prints one canonical whole-GPU UUID per line; duplicates are refused.
judge_gpu_uuids() {
  local raw="${1:-${JUDGE_GPUS:-}}" index uuid
  [[ -n "${raw}" ]] || die "JUDGE_GPUS must list the Judge GPU indices, e.g. JUDGE_GPUS=1,2,3"
  local -A seen=()
  IFS=',' read -r -a indices <<<"${raw}"
  (( ${#indices[@]} >= 1 && ${#indices[@]} <= 8 )) || die "JUDGE_GPUS must name 1 to 8 GPUs"
  for index in "${indices[@]}"; do
    index="$(canonical_gpu_index "${index}")"
    uuid="$(gpu_uuid "${index}")"
    [[ "${uuid}" =~ ^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]] || die "GPU ${index} has no whole-device UUID"
    [[ -z "${seen[${uuid}]:-}" ]] || die "JUDGE_GPUS names GPU ${index} twice"
    seen[${uuid}]=1
    echo "${uuid}"
  done
}

kvzip_repo() {
  local repo="${KV_SCORING_KVZIP_REPO:-${LOGS_DIR}/upstream/KVzip}"
  if [[ ! -d "${repo}/.git" ]]; then
    require_command git; mkdir -p "$(dirname "${repo}")"
    git clone -q https://github.com/snu-mllab/KVzip.git "${repo}" >&2
  fi
  [[ "$(git -C "${repo}" rev-parse HEAD)" == "${KVZIP_COMMIT}" ]] || git -C "${repo}" checkout -q "${KVZIP_COMMIT}" >&2
  [[ "$(git -C "${repo}" rev-parse HEAD)" == "${KVZIP_COMMIT}" ]] || die "KVzip checkout is not at ${KVZIP_COMMIT}"
  echo "${repo}"
}

model_snapshot() {
  local revision; revision="$(validate_model_repo "${KV_SCORING_MODEL_REPO}")"
  echo "${KV_SCORING_MODEL_REPO}/snapshots/${revision}"
}

# Python with pyarrow + tokenizers for scripts/prepare_items.py.  Uses
# KV_SCORING_DATA_PYTHON when set, otherwise a uv-managed venv under LOGS_DIR.
data_python() {
  if [[ -n "${KV_SCORING_DATA_PYTHON:-}" ]]; then
    echo "${KV_SCORING_DATA_PYTHON}"; return
  fi
  local venv="${LOGS_DIR}/data-venv"
  if [[ ! -x "${venv}/bin/python" ]]; then
    require_command uv
    uv venv -q --python 3.11 "${venv}" >&2
    uv pip install -q --python "${venv}/bin/python" pyarrow==25.0.1 transformers==4.57.1 tokenizers==0.22.2 jinja2 >&2
  fi
  echo "${venv}/bin/python"
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
