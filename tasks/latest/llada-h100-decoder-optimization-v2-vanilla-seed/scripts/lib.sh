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
TASKS_DIR="${TASKS_DIR:-${MIGRATION_ROOT}/tasks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPOSITORY_ROOT}/outputs}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-$(basename "${MIGRATION_ROOT}")}"
LOGS_DIR="${LOGS_DIR:-${OUTPUT_ROOT}/${OUTPUT_SUBDIR}}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${MIGRATION_ROOT}/artifacts}"

# SForge resolves these pathlib-backed settings at import time, so export the
# task-specific paths before any CLI module is loaded.
export SFORGE_TASKS_DIR="${TASKS_DIR}"
export SFORGE_LOG_DIR="${LOGS_DIR}"

# Capability envelope, never a GPU model name.  The floors below are the only
# hardware admission criteria: everything else about the device (model, UUID,
# clocks, total capacity) is recorded as measurement provenance.
GPU_MIN_COMPUTE_CAPABILITY="${GPU_MIN_COMPUTE_CAPABILITY:-8.0}"
GPU_MIN_TOTAL_MEMORY_MIB="${GPU_MIN_TOTAL_MEMORY_MIB:-34816}"

TASK_ID="llada_h100_decoder_optimization_v2"
WORK_IMAGE="llada_decoder.work.${TASK_ID}:local"
JUDGE_IMAGE="llada_decoder.judge.${TASK_ID}:local"
# The Judge above is this public Judge (images/judge/Dockerfile, publishable) plus
# the holdout layer images/judge/Dockerfile.private, which is only ever built on
# the host.
JUDGE_PUBLIC_IMAGE="llada_decoder.judge-public.${TASK_ID}:local"
WORKLOAD_ROW_COUNTS_JSON='{"table1_gsm8k_5shot_g256":39,"table1_gsm8k_5shot_g512":39,"table1_math_4shot_g256":65,"table1_math_4shot_g512":65,"table1_humaneval_0shot_g256":65,"table1_humaneval_0shot_g512":65,"table1_mbpp_3shot_g256":65,"table1_mbpp_3shot_g512":65,"table4_gsm8k_5shot_g1024":13,"table5_gsm8k_8shot_g256":13,"table5_gsm8k_8shot_g512":13,"table5_gsm8k_8shot_g1024":13}'
WORKLOAD_ROW_ORDER_JSON='["table1_gsm8k_5shot_g256","table1_gsm8k_5shot_g512","table1_math_4shot_g256","table1_math_4shot_g512","table1_humaneval_0shot_g256","table1_humaneval_0shot_g512","table1_mbpp_3shot_g256","table1_mbpp_3shot_g512","table4_gsm8k_5shot_g1024","table5_gsm8k_8shot_g256","table5_gsm8k_8shot_g512","table5_gsm8k_8shot_g1024"]'
WORKLOAD_ROW_BENCHMARKS_JSON='{"table1_gsm8k_5shot_g256":"gsm8k","table1_gsm8k_5shot_g512":"gsm8k","table1_math_4shot_g256":"math","table1_math_4shot_g512":"math","table1_humaneval_0shot_g256":"humaneval","table1_humaneval_0shot_g512":"humaneval","table1_mbpp_3shot_g256":"mbpp","table1_mbpp_3shot_g512":"mbpp","table4_gsm8k_5shot_g1024":"gsm8k","table5_gsm8k_8shot_g256":"gsm8k","table5_gsm8k_8shot_g512":"gsm8k","table5_gsm8k_8shot_g1024":"gsm8k"}'

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

user_in_docker_group() {
  local user="$1"
  getent group docker | awk -F: -v wanted="${user}" '
    $1 == "docker" {
      count = split($4, members, ",")
      for (i = 1; i <= count; i++) {
        if (members[i] == wanted) {
          found = 1
        }
      }
    }
    END { exit(found ? 0 : 1) }
  '
}

ensure_docker_group() {
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
  local -a command_line=(
    env
    SFORGE_DOCKER_GROUP_REEXEC=1
    MLSYSFORGE_ROOT="${MLSYSFORGE_ROOT}"
    SFORGE_BIN="${SFORGE_BIN}"
    TASKS_DIR="${TASKS_DIR}"
    LOGS_DIR="${LOGS_DIR}"
    ARTIFACTS_DIR="${ARTIFACTS_DIR}"
    bash "${entry_script}"
  )
  # Keep only the migration-specific, non-secret runtime controls. In
  # particular, preserving RUN_ID is required so the tmux launcher, SForge
  # run directory, and status tooling all refer to the same experiment.
  local name
  for name in \
    CODEX_AUTH_JSON_PATH \
    LLADA_MODEL_REPO \
    MODEL \
    MODEL_API_BASE_URL \
    MODEL_API_KEY_FILE \
    REASONING_EFFORT \
    CODEX_CLI_VERSION \
    TASK_TIMEOUT \
    EVAL_INTERVAL \
    SUBMISSION_COOLDOWN \
    FINAL_EVAL_TIMEOUT \
    SFORGE_AGENT_RUNTIME_DIR \
    SFORGE_VERSION_STORE_ROOT \
    LLADA_PILOT \
    LLADA_CALIBRATION \
    LLADA_QUALIFICATION \
    LLADA_QUALITY_CONTROL \
    LLADA_REFERENCE_REPLAY \
    LLADA_BOOTSTRAP \
    LLADA_BOOTSTRAP_INTERACTIVE_JUDGE_TIMEOUT \
    LLADA_BOOTSTRAP_FINAL_JUDGE_TIMEOUT \
    LLADA_BOOTSTRAP_CASE_TIMEOUT \
    LLADA_BOOTSTRAP_ROW_TIMEOUT \
    LLADA_BOOTSTRAP_OUTER_TIMEOUT \
    LLADA_CONTROL_ID \
    LLADA_EXPECTED_CLASSIFICATION \
    LLADA_MEMORY_FIXTURE_KIND \
    LLADA_RESUME \
    LLADA_QUALIFICATION_MIN_FREE_MIB \
    LLADA_QUALIFICATION_INTERACTIVE_JUDGE_TIMEOUT \
    LLADA_QUALIFICATION_CASE_TIMEOUT \
    LLADA_QUALIFICATION_ROW_TIMEOUT \
    LLADA_QUALIFICATION_INNER_TIMEOUT \
    LLADA_QUALIFICATION_OUTER_TIMEOUT \
    GPU_SMOKE_MIN_FREE_MIB \
    SMOKE_JUDGE_PORT \
    LLADA_MEMORY_POLICY_PATH \
    LLADA_WORKLOAD_POLICY_PATH \
    WORK_GPU \
    JUDGE_GPU \
    INFRATHON_ASSETS_ROOT \
    INFRATHON_HARDWARE \
    INFRATHON_PULL \
    JUDGE_PORT \
    SFORGE_SINGLE_GPU_HANDOFF \
    SFORGE_ACCELERATOR_DEVICE_UUID \
    SFORGE_ACCELERATOR_ALLOCATION_KEY \
    SFORGE_ACCELERATOR_OBSERVER \
    SFORGE_TRUSTED_LEASE_ID \
    SFORGE_CALIBRATION_EPOCH \
    RUN_STAMP \
    RUN_ID; do
    if [[ -v "${name}" ]]; then
      command_line=(
        "${command_line[@]:0:${#command_line[@]}-2}"
        "${name}=${!name}"
        "${command_line[@]: -2}"
      )
    fi
  done
  command_line+=("$@")
  local quoted
  printf -v quoted '%q ' "${command_line[@]}"
  exec sg docker -c "${quoted}"
}

require_sforge() {
  [[ -x "${SFORGE_BIN}" ]] || die "missing executable SForge checkout: ${SFORGE_BIN}"
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

gpu_query() {
  local index="$1"
  local field="$2"
  local value
  if ! value="$(
    nvidia-smi --id="${index}" --query-gpu="${field}" \
      --format=csv,noheader,nounits 2>&1
  )"; then
    die "failed to query GPU ${index} field ${field}: ${value}"
  fi
  value="$(xargs <<<"${value}")"
  [[ -n "${value}" && "${value}" != *$'\n'* ]] \
    || die "GPU ${index} returned an invalid ${field}: ${value:-empty}"
  printf '%s\n' "${value}"
}

gpu_uuid() {
  local uuid
  uuid="$(gpu_query "$1" uuid)"
  [[ "${uuid}" =~ ^GPU-[A-Za-z0-9-]+$ ]] \
    || die "GPU $1 returned an unsafe UUID: ${uuid}"
  printf '%s\n' "${uuid}"
}

gpu_name() {
  gpu_query "$1" name
}

gpu_lock_path() {
  local uuid="$1"
  [[ "${uuid}" =~ ^GPU-[A-Za-z0-9-]+$ ]] || die "unsafe GPU UUID: ${uuid}"
  echo "/tmp/sforge-gpu-${uuid}.lock"
}

assert_container_gpu() {
  local image="$1"
  local expected_uuid="$2"
  local role="$3"
  local probe
  [[ "${expected_uuid}" =~ ^GPU-[A-Za-z0-9-]+$ ]] \
    || die "${role} probe received an unsafe GPU UUID: ${expected_uuid}"
  probe="$(docker run --rm \
    --network none \
    --gpus "device=${expected_uuid}" \
    --env "NVIDIA_VISIBLE_DEVICES=${expected_uuid}" \
    --env CUDA_VISIBLE_DEVICES=0 \
    --entrypoint python3 \
    "${image}" -c '
import json, os, subprocess, torch
uuids = [line.strip() for line in subprocess.check_output(
    ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
).splitlines() if line.strip()]
print(json.dumps({
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "torch_device_count": torch.cuda.device_count(),
    "uuid": uuids[0] if len(uuids) == 1 else None,
    "uuid_count": len(uuids),
    "name": torch.cuda.get_device_name(0) if torch.cuda.device_count() == 1 else None,
    "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.device_count() == 1 else None,
    "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.device_count() == 1 else None,
}))')" \
    || die "${role} container GPU probe failed"
  jq -e \
    --arg uuid "${expected_uuid}" \
    --argjson floor_major "${GPU_MIN_COMPUTE_CAPABILITY%%.*}" \
    --argjson floor_minor "${GPU_MIN_COMPUTE_CAPABILITY#*.}" \
    --argjson min_total_mib "${GPU_MIN_TOTAL_MEMORY_MIB}" \
    '.cuda_visible_devices == "0"
     and .torch_device_count == 1
     and .uuid_count == 1
     and .uuid == $uuid
     and (.name | type == "string" and length > 0)
     and (.capability | type == "array" and length == 2
          and (.[0] * 1000 + .[1]) >= ($floor_major * 1000 + $floor_minor))
     and (.total_memory_bytes | type == "number" and . > 0 and floor == .
          and . >= ($min_total_mib * 1048576))' \
    <<<"${probe}" >/dev/null \
    || die "${role} container expected only ${expected_uuid} as cuda:0; probe=${probe}"
  CONTAINER_GPU_TOTAL_BYTES="$(jq -er '.total_memory_bytes | floor' <<<"${probe}")" \
    || die "${role} container did not report a valid device capacity"
  export CONTAINER_GPU_TOTAL_BYTES
  echo "${role} container GPU pin: ${expected_uuid} (logical cuda:0)"
}

GPU_CLAIMING_CONTAINER=""

gpu_has_compute_process() {
  local uuid="$1"
  local processes
  if ! processes="$(
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>&1
  )"; then
    die "failed to inspect GPU compute processes: ${processes}"
  fi
  grep -Fxq -- "${uuid}" <<<"${processes}"
}

running_container_claims_gpu() {
  local uuid="$1"
  local index="$2"
  local container_ids inspect_json inspect_rc container_name
  GPU_CLAIMING_CONTAINER=""
  if ! container_ids="$(docker ps --no-trunc --format '{{.ID}}' 2>&1)"; then
    die "failed to enumerate running Docker containers: ${container_ids}"
  fi
  [[ -n "${container_ids}" ]] || return 1
  while IFS= read -r container_id; do
    [[ -n "${container_id}" ]] || continue
    if ! inspect_json="$(docker inspect --type container "${container_id}" 2>&1)"; then
      # A container that exited between `docker ps` and `docker inspect` no
      # longer owns any device; on a shared host (other runs' Judge
      # containers finishing) this race is routine and must not abort a
      # 30-minute evaluation.  Any other inspect failure still fails closed.
      if [[ "${inspect_json}" == *"No such container"* \
         || "${inspect_json}" == *"No such object"* ]]; then
        continue
      fi
      die "failed to inspect running container ${container_id}: ${inspect_json}"
    fi
    if jq -e --arg uuid "${uuid}" --arg index "${index}" '
      .[0] as $container
      | def gpu_capable:
          ((.Capabilities // []) | flatten | index("gpu")) != null
          or (.Driver // "" | ascii_downcase) == "nvidia";
        def device_request_claim:
          any($container.HostConfig.DeviceRequests[]?;
            gpu_capable and (
              (.Count // 0) == -1
              or (((.DeviceIDs // []) | length) == 0 and (.Count // 0) > 0)
              or ((.DeviceIDs // []) | index($uuid)) != null
              or ((.DeviceIDs // []) | index($index)) != null
            )
          );
        def has_gpu_device_request:
          any($container.HostConfig.DeviceRequests[]?; gpu_capable);
        def visible_values:
          [$container.Config.Env[]?
           | select(startswith("NVIDIA_VISIBLE_DEVICES="))
           | split("=")[1]
           | split(",")[]
           | gsub("^[[:space:]]+|[[:space:]]+$"; "")];
        device_request_claim
        # NVIDIA images commonly retain NVIDIA_VISIBLE_DEVICES=all even when
        # Docker narrows them with an explicit DeviceRequest.  In that case
        # the request is authoritative; treating the image default as a claim
        # on every host GPU creates false conflicts with unrelated containers.
        or ((has_gpu_device_request | not) and any(visible_values[]?;
          . == $uuid or . == $index or (ascii_downcase == "all")))
        or ($container.Config.Labels["mlsysforge.gpu_uuid"] // "") == $uuid
    ' <<<"${inspect_json}" >/dev/null; then
      container_name="$(
        jq -r '.[0].Name // .[0].Id // "unknown" | ltrimstr("/")' \
          <<<"${inspect_json}"
      )"
      GPU_CLAIMING_CONTAINER="${container_name}"
      return 0
    else
      inspect_rc="$?"
      (( inspect_rc == 1 )) \
        || die "failed to parse Docker GPU assignment for ${container_id}"
    fi
  done <<<"${container_ids}"
  return 1
}

assert_uuid_unassigned() {
  local uuid="$1"
  local index="$2"
  local role="$3"
  if gpu_has_compute_process "${uuid}"; then
    die "${role} GPU ${index} (${uuid}) already has a compute process"
  fi
  if running_container_claims_gpu "${uuid}" "${index}"; then
    die "${role} GPU ${index} (${uuid}) is assigned to running container ${GPU_CLAIMING_CONTAINER}"
  fi
}

# Compare "major.minor" compute capabilities without floating-point rounding.
# Returns success when the observed capability is below the floor.
capability_below_floor() {
  local observed="$1"
  local floor="$2"
  local observed_rank floor_rank
  [[ "${observed}" =~ ^([0-9]+)\.([0-9]+)$ ]] || return 0
  observed_rank=$(( 10#${BASH_REMATCH[1]} * 1000 + 10#${BASH_REMATCH[2]} ))
  [[ "${floor}" =~ ^([0-9]+)\.([0-9]+)$ ]] \
    || die "GPU_MIN_COMPUTE_CAPABILITY must be major.minor: ${floor}"
  floor_rank=$(( 10#${BASH_REMATCH[1]} * 1000 + 10#${BASH_REMATCH[2]} ))
  [[ "${observed_rank}" -lt "${floor_rank}" ]]
}

assert_gpu_available() {
  local index role min_free_mib name uuid free_mib total_mib capability
  index="$(canonical_gpu_index "$1")"
  role="$2"
  min_free_mib="$3"
  [[ "${min_free_mib}" =~ ^[0-9]+$ ]] \
    || die "${role} admission floor is not an integer MiB value"
  name="$(gpu_name "${index}")"
  uuid="$(gpu_uuid "${index}")"
  free_mib="$(gpu_query "${index}" memory.free)"
  total_mib="$(gpu_query "${index}" memory.total)"
  capability="$(gpu_query "${index}" compute_cap)"
  [[ "${free_mib}" =~ ^[0-9]+$ && "${total_mib}" =~ ^[0-9]+$ ]] \
    || die "${role} GPU ${index} returned nonnumeric memory values"
  if capability_below_floor "${capability}" "${GPU_MIN_COMPUTE_CAPABILITY}"; then
    die "${role} GPU ${index} compute capability ${capability} is below the task minimum ${GPU_MIN_COMPUTE_CAPABILITY}: ${name}"
  fi
  (( 10#${total_mib} >= 10#${GPU_MIN_TOTAL_MEMORY_MIB} )) \
    || die "${role} GPU ${index} has ${total_mib} MiB total; the envelope requires ${GPU_MIN_TOTAL_MEMORY_MIB} MiB: ${name}"
  (( 10#${free_mib} >= 10#${min_free_mib} )) \
    || die "${role} GPU ${index} has ${free_mib} MiB free; requires ${min_free_mib} MiB"
  assert_uuid_unassigned "${uuid}" "${index}" "${role}"
  echo "${role} GPU: physical ${index}, ${name}, ${uuid}, ${free_mib}/${total_mib} MiB free, CC ${capability}"
}

acquire_dual_gpu_locks() {
  local work_uuid="$1"
  local judge_uuid="$2"
  local first_uuid second_uuid
  local -a sorted_uuids
  [[ "${work_uuid}" != "${judge_uuid}" ]] \
    || die "Work and Judge GPU UUIDs must differ"
  mapfile -t sorted_uuids < <(
    printf '%s\n' "${work_uuid}" "${judge_uuid}" | LC_ALL=C sort
  )
  (( ${#sorted_uuids[@]} == 2 )) || die "failed to sort GPU UUID locks"
  first_uuid="${sorted_uuids[0]}"
  second_uuid="${sorted_uuids[1]}"
  exec 8>"$(gpu_lock_path "${first_uuid}")"
  flock -n 8 || die "GPU UUID lock is held: ${first_uuid}"
  exec 9>"$(gpu_lock_path "${second_uuid}")"
  flock -n 9 || die "GPU UUID lock is held: ${second_uuid}"
  GPU_LOCK_UUID_8="${first_uuid}"
  GPU_LOCK_UUID_9="${second_uuid}"
  export GPU_LOCK_UUID_8 GPU_LOCK_UUID_9
  echo "GPU UUID locks acquired in deterministic order: ${first_uuid}, ${second_uuid}"
}

assert_dual_gpu_locks_held() {
  [[ -e "/proc/$$/fd/8" && -e "/proc/$$/fd/9" ]] \
    || die "dual GPU lock descriptors are no longer open"
  flock -n 8 || die "first GPU UUID lock is no longer held"
  flock -n 9 || die "second GPU UUID lock is no longer held"
}

assert_distinct_gpu_pair() {
  local work_index judge_index work_uuid judge_uuid min_free_mib
  work_index="$(canonical_gpu_index "$1")"
  judge_index="$(canonical_gpu_index "$2")"
  min_free_mib="$3"
  [[ "${work_index}" != "${judge_index}" ]] \
    || die "Work and Judge GPU indices must differ"
  work_uuid="$(gpu_uuid "${work_index}")"
  judge_uuid="$(gpu_uuid "${judge_index}")"
  [[ "${work_uuid}" != "${judge_uuid}" ]] \
    || die "Work and Judge GPU UUIDs must differ"
  assert_gpu_available "${work_index}" "Work" "${min_free_mib}"
  assert_gpu_available "${judge_index}" "Judge" "${min_free_mib}"
}

assert_gpu_released() {
  local uuid="$1"
  local index="$2"
  local role="$3"
  local attempt
  for (( attempt = 1; attempt <= 30; attempt++ )); do
    if ! gpu_has_compute_process "${uuid}" \
        && ! running_container_claims_gpu "${uuid}" "${index}"; then
      echo "${role} GPU release barrier: PASS (${uuid})"
      return
    fi
    (( attempt < 30 )) && sleep 1
  done
  die "${role} GPU ${uuid} still has a process/container assignment after 30s"
}

# ---------------------------------------------------------------------------
# Single-allocation (one whole GPU shared serially by Work and Judge) helpers.
# The scientific contract is unchanged; these only describe the deployment.
# ---------------------------------------------------------------------------

acquire_single_gpu_lock() {
  # One UUID-keyed flock for the entire run: Work, every interactive Judge
  # and the standalone final Judge serialize on it.  Descriptor 8 is the same
  # slot the dual path uses for its first lock, so the "never released before
  # cleanup" audit applies to both modes.
  local uuid="$1"
  exec 8>"$(gpu_lock_path "${uuid}")"
  flock -n 8 || die "single GPU allocation ${uuid} lock is already held"
  GPU_LOCK_UUID_8="${uuid}"
  export GPU_LOCK_UUID_8
  echo "GPU UUID lock acquired for the single allocation: ${uuid}"
}

assert_single_gpu_lock_held() {
  [[ -n "${GPU_LOCK_UUID_8:-}" ]] \
    || die "single GPU lock identity is absent"
  [[ -e "/proc/$$/fd/8" ]] \
    || die "single GPU lock descriptor is no longer open"
  flock -n 8 || die "single GPU UUID lock is no longer held"
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
    --env "NVIDIA_VISIBLE_DEVICES=${expected_uuid}" \
    --env CUDA_VISIBLE_DEVICES=0 \
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
  local index seen name total_mib free_mib capability
  local capability_major capability_minor
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
  index="$(gpu_index_of_uuid "${uuid}")"
  seen="$(gpu_uuid "${index}")"
  [[ "${seen}" == "${uuid}" ]] \
    || die "${role} allocation expected ${uuid}, saw ${seen:-nothing}"
  name="$(gpu_name "${index}")"
  total_mib="$(gpu_query "${index}" memory.total)"
  free_mib="$(gpu_query "${index}" memory.free)"
  capability="$(gpu_query "${index}" compute_cap)"
  [[ "${total_mib}" =~ ^[0-9]+$ && "${free_mib}" =~ ^[0-9]+$ ]] \
    || die "${role} allocation returned invalid memory capability"
  [[ "${capability}" =~ ^[0-9]+\.[0-9]+$ ]] \
    || die "${role} allocation returned invalid compute capability: ${capability}"
  capability_major="${capability%%.*}"
  capability_minor="${capability#*.}"
  (( 10#${capability_major} > 10#${minimum_capability_major} \
     || (10#${capability_major} == 10#${minimum_capability_major} \
         && 10#${capability_minor} >= 10#${minimum_capability_minor}) )) \
    || die "${role} allocation requires compute capability ${minimum_capability} or newer, found ${capability}"
  (( 10#${total_mib} >= minimum_memory_mib && 10#${free_mib} >= minimum_memory_mib )) \
    || die "${role} allocation requires ${minimum_memory_bytes} bytes total and free; total=${total_mib} MiB free=${free_mib} MiB"
  assert_uuid_unassigned "${uuid}" "${index}" "${role}"
  echo "${role} allocation: physical ${index}, ${name}, ${uuid}, compute ${capability}, ${free_mib}/${total_mib} MiB free"
}

gpu_index_of_uuid() {
  # Resolve the host ordinal of one whole-GPU UUID for the identity preflight.
  local uuid="$1"
  local index
  [[ "${uuid}" =~ ^GPU-[A-Za-z0-9-]+$ ]] || die "unsafe GPU UUID: ${uuid}"
  index="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits \
    | awk -F', *' -v uuid="${uuid}" '$2 == uuid {print $1}')"
  [[ "${index}" =~ ^[0-9]+$ ]] || die "no host GPU carries UUID ${uuid}"
  printf '%s\n' "${index}"
}

remove_task_gpu_containers() {
  local role="$1"
  local uuid="$2"
  local run_id="$3"
  local container_ids
  local -a ids
  [[ "${role}" == "work" || "${role}" == "judge" ]] \
    || die "unsafe task container cleanup role: ${role}"
  [[ "${uuid}" =~ ^GPU-[A-Za-z0-9-]+$ ]] \
    || die "unsafe task container cleanup UUID: ${uuid}"
  [[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
    || die "unsafe task container cleanup run ID: ${run_id}"
  if ! container_ids="$(docker ps -aq \
      --filter "label=mlsysforge.task_id=${TASK_ID}" \
      --filter "label=mlsysforge.run_id=${run_id}" \
      --filter "label=mlsysforge.gpu_role=${role}" \
      --filter "label=mlsysforge.gpu_uuid=${uuid}" 2>&1)"; then
    echo "WARNING: failed to enumerate ${role} cleanup containers: ${container_ids}" >&2
    return 1
  fi
  [[ -n "${container_ids}" ]] || return 0
  mapfile -t ids <<<"${container_ids}"
  docker rm -f "${ids[@]}" >/dev/null
}

prepare_root_owned_checkpoint_dir() {
  local run_root="$1"
  local run_id="$2"
  local image="${3:-${JUDGE_IMAGE}}"
  local logs_real run_real checkpoint expected_checkpoint
  [[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
    || die "unsafe checkpoint run ID: ${run_id}"
  [[ -d "${run_root}" && ! -L "${run_root}" ]] \
    || die "checkpoint run root must be a real directory: ${run_root}"
  logs_real="$(realpath "${LOGS_DIR}")"
  run_real="$(realpath "${run_root}")"
  [[ "${run_real}" == "${logs_real}/"* \
      && "$(basename "${run_real}")" == "${run_id}" \
      && "$(basename "$(dirname "${run_real}")")" == "runs" ]] \
    || die "checkpoint run root is outside the exact logs/runs identity"
  checkpoint="${run_real}/judge-checkpoints"
  if [[ ! -e "${checkpoint}" && ! -L "${checkpoint}" ]]; then
    docker run --rm --network none \
      --env NVIDIA_VISIBLE_DEVICES=void \
      --volume "${run_real}:/mlsysforge-run:rw" \
      --entrypoint /bin/sh \
      "${image}" -ceu '
        target=/mlsysforge-run/judge-checkpoints
        test ! -e "$target" && test ! -L "$target"
        umask 077
        mkdir -- "$target"
        chown 0:0 -- "$target"
        chmod 700 -- "$target"
      ' >/dev/null \
      || die "failed to create the root-owned Judge checkpoint directory"
  fi
  [[ -d "${checkpoint}" && ! -L "${checkpoint}" ]] \
    || die "Judge checkpoint path must be a real directory"
  expected_checkpoint="$(realpath "${checkpoint}")"
  [[ "${expected_checkpoint}" == "${checkpoint}" \
      && "$(stat -c '%u' "${checkpoint}")" == "0" \
      && "$(stat -c '%a' "${checkpoint}")" == "700" ]] \
    || die "Judge checkpoint directory must be root-owned mode 700"
  LLADA_JUDGE_CHECKPOINT_DIR="${checkpoint}"
  export LLADA_JUDGE_CHECKPOINT_DIR
}

bind_runtime_resume_identity() {
  local run_root="$1"
  local run_id="$2"
  local resume_mode="$3"
  local candidate="$4"
  local identity="${run_root}/runtime-resume-identity.json"
  [[ "${resume_mode}" == "0" || "${resume_mode}" == "1" ]] \
    || die "resume mode must be 0 or 1"
  [[ -d "${run_root}" && ! -L "${run_root}" \
      && "$(basename "${run_root}")" == "${run_id}" ]] \
    || die "resume identity requires the exact real run directory"
  [[ -f "${candidate}" && ! -L "${candidate}" ]] \
    || die "resume identity candidate must be a regular file"
  exec 7<"${run_root}"
  flock -n 7 || die "run identity is already active: ${run_id}"
  if [[ "${resume_mode}" == "1" ]]; then
    [[ -f "${identity}" && ! -L "${identity}" \
        && "$(stat -c '%u' "${identity}")" == "$(id -u)" \
        && "$(stat -c '%a' "${identity}")" == "444" ]] \
      || die "resume requires an immutable launcher-owned identity snapshot"
    cmp -s -- "${candidate}" "${identity}" \
      || die "resume identity differs from the original run"
    [[ ! -e "${run_root}/evaluation-attestation.json" \
        && ! -L "${run_root}/evaluation-attestation.json" ]] \
      || die "completed/unsafe evaluation runs cannot be resumed"
  else
    [[ ! -e "${identity}" && ! -L "${identity}" ]] \
      || die "new run already has a resume identity"
    cp -- "${candidate}" "${identity}"
    chmod 444 "${identity}"
  fi
  RUNTIME_RESUME_IDENTITY_PATH="${identity}"
  export RUNTIME_RESUME_IDENTITY_PATH
}

enrich_calibration_image_manifest() {
  local manifest="$1"
  local runtime_sha256="$2"
  local hidden_sha256="$3"
  local quality_sha256="$4"
  local verifier_sha256="$5"
  local worker_sha256="$6"
  local cohort_sha256="$7"
  local workload_policy_sha256="$8"
  local memory_policy_sha256="$9"
  local digest enriched
  [[ -f "${manifest}" && ! -L "${manifest}" ]] \
    || die "calibration image manifest must be a regular non-symlink file"
  for digest in \
    "${runtime_sha256}" \
    "${hidden_sha256}" \
    "${quality_sha256}" \
    "${verifier_sha256}" \
    "${worker_sha256}" \
    "${cohort_sha256}" \
    "${workload_policy_sha256}" \
    "${memory_policy_sha256}"; do
    [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
      || die "calibration image manifest received an invalid SHA-256"
  done
  enriched="$(mktemp "${manifest}.calibration.XXXXXX")"
  if ! jq \
      --arg runtime "${runtime_sha256}" \
      --arg hidden "${hidden_sha256}" \
      --arg quality "${quality_sha256}" \
      --arg verifier "${verifier_sha256}" \
      --arg worker "${worker_sha256}" \
      --arg cohort "${cohort_sha256}" \
      --arg workload_policy "${workload_policy_sha256}" \
      --arg memory_policy "${memory_policy_sha256}" \
      '. + {
        judge_runtime_tree_sha256: $runtime,
        calibration_files: {
          hidden_cases_sha256: $hidden,
          quality_runtime_sha256: $quality,
          verifier_sha256: $verifier,
          candidate_worker_sha256: $worker,
          cohort_sha256: $cohort,
          workload_policy_sha256: $workload_policy,
          memory_policy_sha256: $memory_policy
        }
      }' "${manifest}" >"${enriched}"; then
    rm -f -- "${enriched}"
    die "failed to enrich the calibration image manifest"
  fi
  mv -- "${enriched}" "${manifest}"
}

load_unqualified_memory_policy() {
  local policy="${1:-${LLADA_MEMORY_POLICY_PATH:-${MIGRATION_ROOT}/memory_policy.json}}"
  [[ -f "${policy}" && ! -L "${policy}" ]] \
    || die "qualification requires a regular tracked memory policy: ${policy}"
  jq -e --arg task_id "${TASK_ID}" '
    .schema_version == 1
    and .task_id == $task_id
    and (.qualification_hardware | type == "string" and length > 0)
    and .qualification_status == "UNQUALIFIED"
    and .release_status == "PENDING"
    and .qualified_max_peak_gib == null
    and .headroom_mib == null
    and .qualified_clean_idle_free_mib == null
    and .safety_margin_mib == null
    and .qualification_run_sha256 == null
    and .approval_id == null
    and .measurements.reference_and_near_cap_stress_required == true
    and .historical_a6000_policy.max_peak_gib == 47.0
    and .historical_a6000_policy.applicability == "NONE"
    and .historical_a6000_policy.status == "UNCALIBRATED_FOR_H100"
  ' "${policy}" >/dev/null \
    || die "memory policy is not UNQUALIFIED/PENDING for qualification"

  # Admission-only cleanliness floor for non-release qualification.  It is a
  # capability-envelope floor, not a measured or releasable memory cap.
  GPU_ADMISSION_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-${LLADA_QUALIFICATION_MIN_FREE_MIB:-${GPU_MIN_TOTAL_MEMORY_MIB}}}"
  [[ "${GPU_ADMISSION_MIN_FREE_MIB}" =~ ^[0-9]+$ ]] \
    || die "GPU_MIN_FREE_MIB must be a positive integer"
  GPU_ADMISSION_MIN_FREE_MIB="$((10#${GPU_ADMISSION_MIN_FREE_MIB}))"
  (( GPU_ADMISSION_MIN_FREE_MIB > 0 )) \
    || die "GPU_MIN_FREE_MIB must be positive"
  QUALIFIED_MAX_PEAK_GIB=""
  HEADROOM_MIB=""
  MEMORY_POLICY_PATH="$(realpath "${policy}")"
  MEMORY_POLICY_SHA256="$(sha256sum "${policy}" | awk '{print $1}')"
  export QUALIFIED_MAX_PEAK_GIB HEADROOM_MIB GPU_ADMISSION_MIN_FREE_MIB
  export MEMORY_POLICY_PATH MEMORY_POLICY_SHA256
}

load_frozen_memory_policy() {
  local policy="${1:-${LLADA_MEMORY_POLICY_PATH:-${MIGRATION_ROOT}/memory_policy.json}}"
  local values
  [[ -f "${policy}" && ! -L "${policy}" ]] \
    || die "formal H100 run requires a regular tracked memory policy: ${policy}"
  if ! values="$(jq -er --arg task_id "${TASK_ID}" '
    select(.schema_version == 1)
    | select(.task_id == $task_id)
    | select(.qualification_hardware | type == "string" and length > 0)
    | select(.qualification_status == "QUALIFIED")
    | select(.release_status == "FROZEN")
    | select(.qualified_max_peak_gib | type == "number" and . > 0)
    | select(.headroom_mib | type == "number" and . > 0 and floor == .)
    | select(.safety_margin_mib | type == "number" and . > 0 and floor == .)
    | select(.qualification_run_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
    | select(.approval_id | type == "string" and length > 0)
    | [.qualified_max_peak_gib, .headroom_mib, .safety_margin_mib] | @tsv
  ' "${policy}" 2>&1)"; then
    die "memory policy is not QUALIFIED/FROZEN for this task: ${values}"
  fi
  IFS=$'\t' read -r QUALIFIED_MAX_PEAK_GIB HEADROOM_MIB SAFETY_MARGIN_MIB <<<"${values}"
  # The peak cap the Judge enforces is derived on-device from the visible
  # device capacity minus the policy reserve, so the host admission floor is
  # the capability envelope -- never the H100-measured peak.
  GPU_ADMISSION_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-${GPU_MIN_TOTAL_MEMORY_MIB}}"
  [[ "${GPU_ADMISSION_MIN_FREE_MIB}" =~ ^[0-9]+$ ]] \
    || die "GPU_MIN_FREE_MIB must be a positive integer"
  GPU_ADMISSION_MIN_FREE_MIB="$((10#${GPU_ADMISSION_MIN_FREE_MIB}))"
  (( GPU_ADMISSION_MIN_FREE_MIB > 0 )) \
    || die "GPU_MIN_FREE_MIB must be positive"
  MEMORY_POLICY_PATH="$(realpath "${policy}")"
  MEMORY_POLICY_SHA256="$(sha256sum "${policy}" | awk '{print $1}')"
  export QUALIFIED_MAX_PEAK_GIB HEADROOM_MIB SAFETY_MARGIN_MIB
  export GPU_ADMISSION_MIN_FREE_MIB
  export MEMORY_POLICY_PATH MEMORY_POLICY_SHA256
}

load_qualified_workload_policy() {
  local expected_release="$1"
  local policy="$2"
  local values
  [[ "${expected_release}" == "PROVISIONAL" || "${expected_release}" == "FROZEN" ]] \
    || die "unsupported workload policy release state: ${expected_release}"
  [[ -f "${policy}" && ! -L "${policy}" ]] \
    || die "H100 run requires a regular tracked workload policy: ${policy}"
  if ! values="$(jq -er \
    --arg task_id "${TASK_ID}" \
    --arg release "${expected_release}" \
    --argjson expected_rows "${WORKLOAD_ROW_COUNTS_JSON}" \
    --argjson expected_order "${WORKLOAD_ROW_ORDER_JSON}" \
    --argjson expected_benchmarks "${WORKLOAD_ROW_BENCHMARKS_JSON}" '
    select(.schema_version == 1)
    | select(.task_id == $task_id)
    | select(.profile == "fast_dllm_v1_llada_12row_stratified520_batch1")
    | select(.qualification_status == "QUALIFIED")
    | select(.release_status == $release)
    | select(.manifest.generation_count == 520)
    | select(.manifest.requested_slots == 216320)
    | select(.manifest.row_counts == $expected_rows)
    | select(.manifest.row_order == $expected_order)
    | select(.manifest.row_benchmarks == $expected_benchmarks)
    | select(.manifest.hidden_cases_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
    | select(.manifest.dataset_manifest_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
    | select(.interactive.status == "QUALIFIED")
    | select(.interactive.selection == "all_frozen_final_v1")
    | select(.interactive.category_definition == {
        kind: "dataset",
        field: "benchmark",
        categories: ["gsm8k", "math", "humaneval", "mbpp"]
      })
    | select(.interactive.row_atomic == true)
    | select(.interactive.timeout_policy == "same_as_final")
    | select(.interactive.total_cases == 520)
    | select(.interactive.category_base_quota == 130)
    | select(.interactive.bonus_categories == 0)
    | select(.interactive.minimum_per_row == 1)
    | select(.interactive | has("per_row_quotas") | not)
    | select(.interactive | has("per_category_quotas") | not)
    | select(.interactive.cooldown_seconds | type == "number" and . > 0 and floor == .)
    | select(.interactive.case_timeout_seconds | type == "number" and . > 0)
    | select(.interactive.judge_timeout_seconds | type == "number" and . > 0)
    | select(.interactive.case_timeout_seconds == .final.case_timeout_seconds)
    | select(.interactive.judge_timeout_seconds == .final.inner_timeout_seconds)
    | select(.final.selection == "all")
    | select(.final.row_atomic == true)
    | select(.final.max_infrastructure_retries_per_row == 1)
    | select(.final.candidate_failures_retriable == false)
    | select(all([
        .final.case_timeout_seconds,
        .final.row_timeout_seconds,
        .final.inner_timeout_seconds,
        .final.outer_timeout_seconds
      ][]; type == "number" and . > 0))
    | select(.final.outer_timeout_seconds | floor == .)
    | select(.final.outer_timeout_seconds <= 7200)
    | select(
      if $release == "FROZEN" then
          .quality_policy.status == "QUALIFIED"
          and .quality_policy.interactive == "same_as_final"
          and .quality_policy.final == "paired_non_inferiority_hard_gate"
          and (.provisional_parent_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
          and .quality_policy.provisional_parent_sha256 == .provisional_parent_sha256
        else
          .quality_policy.status == "UNQUALIFIED"
          and .quality_policy.interactive == "same_as_final"
          and .quality_policy.final == "PENDING_CONTROL_QUALIFICATION"
        end
      )
    | [
        (.interactive.cooldown_seconds | floor),
        .interactive.case_timeout_seconds,
        .interactive.judge_timeout_seconds,
        .final.case_timeout_seconds,
        .final.row_timeout_seconds,
        .final.inner_timeout_seconds,
        (.final.outer_timeout_seconds | floor)
      ] | @tsv
  ' "${policy}" 2>&1)"; then
    die "workload policy is not QUALIFIED/${expected_release} for this H100 mode: ${values}"
  fi
  IFS=$'\t' read -r \
    POLICY_SUBMISSION_COOLDOWN \
    POLICY_INTERACTIVE_CASE_TIMEOUT \
    POLICY_INTERACTIVE_JUDGE_TIMEOUT \
    POLICY_FINAL_CASE_TIMEOUT \
    POLICY_FINAL_ROW_TIMEOUT \
    POLICY_FINAL_INNER_TIMEOUT \
    POLICY_FINAL_OUTER_TIMEOUT <<<"${values}"
  WORKLOAD_POLICY_PATH="$(realpath "${policy}")"
  WORKLOAD_POLICY_SHA256="$(sha256sum "${policy}" | awk '{print $1}')"
  export POLICY_SUBMISSION_COOLDOWN POLICY_INTERACTIVE_CASE_TIMEOUT
  export POLICY_INTERACTIVE_JUDGE_TIMEOUT POLICY_FINAL_CASE_TIMEOUT
  export POLICY_FINAL_ROW_TIMEOUT POLICY_FINAL_INNER_TIMEOUT
  export POLICY_FINAL_OUTER_TIMEOUT WORKLOAD_POLICY_PATH
  export WORKLOAD_POLICY_SHA256
}

load_frozen_workload_policy() {
  local policy="${1:-${LLADA_WORKLOAD_POLICY_PATH:-${MIGRATION_ROOT}/workload_policy.json}}"
  load_qualified_workload_policy "FROZEN" "${policy}"
}

load_provisional_workload_policy() {
  local policy="${1:-${LLADA_WORKLOAD_POLICY_PATH:-${MIGRATION_ROOT}/workload_policy.json}}"
  load_qualified_workload_policy "PROVISIONAL" "${policy}"
}

load_unqualified_workload_policy() {
  local policy="${1:-${LLADA_WORKLOAD_POLICY_PATH:-${MIGRATION_ROOT}/workload_policy.json}}"
  [[ -f "${policy}" && ! -L "${policy}" ]] \
    || die "qualification requires a regular tracked workload policy: ${policy}"
  jq -e \
    --arg task_id "${TASK_ID}" \
    --argjson expected_rows "${WORKLOAD_ROW_COUNTS_JSON}" \
    --argjson expected_order "${WORKLOAD_ROW_ORDER_JSON}" \
    --argjson expected_benchmarks "${WORKLOAD_ROW_BENCHMARKS_JSON}" '
    .schema_version == 1
    and .task_id == $task_id
    and .profile == "fast_dllm_v1_llada_12row_stratified520_batch1"
    and .qualification_status == "UNQUALIFIED"
    and .release_status == "PENDING"
    and .manifest.generation_count == 520
    and .manifest.requested_slots == 216320
    and .manifest.row_counts == $expected_rows
    and .manifest.row_order == $expected_order
    and .manifest.row_benchmarks == $expected_benchmarks
    and (.manifest.hidden_cases_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
    and (.manifest.dataset_manifest_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
    and .interactive.status == "PROVISIONAL_UNQUALIFIED"
    and .interactive.selection == "all_frozen_final_v1"
    and .interactive.category_definition == {
      kind: "dataset",
      field: "benchmark",
      categories: ["gsm8k", "math", "humaneval", "mbpp"]
    }
    and .interactive.row_atomic == true
    and .interactive.timeout_policy == "same_as_final"
    and .interactive.total_cases == 520
    and .interactive.category_base_quota == 130
    and .interactive.bonus_categories == 0
    and .interactive.minimum_per_row == 1
    and (.interactive | has("per_row_quotas") | not)
    and (.interactive | has("per_category_quotas") | not)
    and .interactive.case_timeout_seconds == null
    and .interactive.judge_timeout_seconds == null
    and .interactive.cooldown_seconds == null
    and .final.selection == "all"
    and .final.row_atomic == true
    and .final.max_infrastructure_retries_per_row == 1
    and .final.candidate_failures_retriable == false
    and .final.case_timeout_seconds == null
    and .final.row_timeout_seconds == null
    and .final.inner_timeout_seconds == null
    and .final.outer_timeout_seconds == null
    and (.qualification.required_stratified520_timing_runs | type == "number" and . >= 3 and floor == .)
    and .quality_policy.status == "UNQUALIFIED"
    and .quality_policy.interactive == "same_as_final"
    and .quality_policy.final == "PENDING_CONTROL_QUALIFICATION"
  ' "${policy}" >/dev/null \
    || die "workload policy is not UNQUALIFIED/PENDING for H100 qualification"

  # Explicit operator-provided non-release safety ceilings allow initial
  # measurement without turning any legacy bootstrap value into a release
  # timeout contract. Qualification fails closed if any ceiling is omitted.
  POLICY_SUBMISSION_COOLDOWN=0
  POLICY_INTERACTIVE_CASE_TIMEOUT="${LLADA_QUALIFICATION_CASE_TIMEOUT:-}"
  POLICY_INTERACTIVE_JUDGE_TIMEOUT="${LLADA_QUALIFICATION_INNER_TIMEOUT:-}"
  POLICY_FINAL_CASE_TIMEOUT="${LLADA_QUALIFICATION_CASE_TIMEOUT:-}"
  POLICY_FINAL_ROW_TIMEOUT="${LLADA_QUALIFICATION_ROW_TIMEOUT:-}"
  POLICY_FINAL_INNER_TIMEOUT="${LLADA_QUALIFICATION_INNER_TIMEOUT:-}"
  POLICY_FINAL_OUTER_TIMEOUT="${LLADA_QUALIFICATION_OUTER_TIMEOUT:-}"
  local name value
  for name in \
    POLICY_INTERACTIVE_CASE_TIMEOUT \
    POLICY_INTERACTIVE_JUDGE_TIMEOUT \
    POLICY_FINAL_CASE_TIMEOUT \
    POLICY_FINAL_ROW_TIMEOUT \
    POLICY_FINAL_INNER_TIMEOUT \
    POLICY_FINAL_OUTER_TIMEOUT; do
    value="${!name}"
    [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be a positive integer"
    printf -v "${name}" '%d' "$((10#${value}))"
    (( ${!name} > 0 )) || die "${name} must be positive"
  done
  (( POLICY_FINAL_CASE_TIMEOUT < POLICY_FINAL_ROW_TIMEOUT \
      && POLICY_FINAL_ROW_TIMEOUT < POLICY_FINAL_INNER_TIMEOUT \
      && POLICY_FINAL_INNER_TIMEOUT <= POLICY_FINAL_OUTER_TIMEOUT \
      && POLICY_FINAL_OUTER_TIMEOUT <= 7200 )) \
    || die "qualification safety ceilings are not coherently nested within task eval_timeout"
  WORKLOAD_POLICY_PATH="$(realpath "${policy}")"
  WORKLOAD_POLICY_SHA256="$(sha256sum "${policy}" | awk '{print $1}')"
  export POLICY_SUBMISSION_COOLDOWN POLICY_INTERACTIVE_CASE_TIMEOUT
  export POLICY_INTERACTIVE_JUDGE_TIMEOUT POLICY_FINAL_CASE_TIMEOUT
  export POLICY_FINAL_ROW_TIMEOUT POLICY_FINAL_INNER_TIMEOUT
  export POLICY_FINAL_OUTER_TIMEOUT WORKLOAD_POLICY_PATH
  export WORKLOAD_POLICY_SHA256
}

frozen_workload_parent_sha256() {
  local policy="${1:-${WORKLOAD_POLICY_PATH:-}}"
  [[ -n "${policy}" && -f "${policy}" && ! -L "${policy}" ]] \
    || die "frozen workload parent lookup requires a safe policy file"
  jq -er '
    select(.qualification_status == "QUALIFIED")
    | select(.release_status == "FROZEN")
    | select(.provisional_parent_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
    | select(.quality_policy.provisional_parent_sha256 == .provisional_parent_sha256)
    | .provisional_parent_sha256
  ' "${policy}" \
    || die "frozen workload policy has no consistent provisional parent chain"
}

validate_quality_control_binding() {
  local manifest="$1"
  local control_id="$2"
  local expected_classification="$3"
  local archive_sha256="$4"
  local workload_policy_sha256="$5"
  local hidden_cases_sha256="$6"
  [[ "${control_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] \
    || die "LLADA_CONTROL_ID is invalid"
  [[ "${expected_classification}" == "ACCEPT" \
      || "${expected_classification}" == "REJECT" ]] \
    || die "LLADA_EXPECTED_CLASSIFICATION must be ACCEPT or REJECT"
  [[ -f "${manifest}" && ! -L "${manifest}" ]] \
    || die "quality-control manifest must be a regular non-symlink file"
  jq -e \
    --arg task_id "${TASK_ID}" \
    --arg control_id "${control_id}" \
    --arg expected "${expected_classification}" \
    --arg archive "${archive_sha256}" \
    --arg workload "${workload_policy_sha256}" \
    --arg hidden "${hidden_cases_sha256}" '
    ([.positive_controls[], .negative_controls[]]
      | map(select(.id == $control_id))) as $matches
    | .schema_version == 1
      and .task_id == $task_id
      and .profile == "fast_dllm_v1_llada_12row_stratified520_batch1"
      and .status == "PRECOMMITTED"
      and .workload_policy_sha256 == $workload
      and .hidden_cases_sha256 == $hidden
      and ($matches | length) == 1
      and $matches[0].archive_sha256 == $archive
      and $matches[0].expected_classification == $expected
  ' "${manifest}" >/dev/null \
    || die "quality-control archive/identity is not exactly precommitted"
}

validate_memory_fixture_binding() {
  local manifest="$1"
  local kind="$2"
  local evaluated_archive="$3"
  local workload_policy_sha256="$4"
  local memory_policy_sha256="$5"
  local task_root="${6:-${MIGRATION_ROOT}}"
  local values fixture_relative fixture_path fixture_real task_real
  [[ "${kind}" == "corrected_reference" || "${kind}" == "near_cap_stress" ]] \
    || die "memory fixture kind is invalid"
  [[ -f "${manifest}" && ! -L "${manifest}" \
      && -f "${evaluated_archive}" && ! -L "${evaluated_archive}" ]] \
    || die "memory fixture manifest/archive must be regular non-symlink files"
  if ! values="$(jq -er \
      --arg task_id "${TASK_ID}" \
      --arg kind "${kind}" \
      --arg workload "${workload_policy_sha256}" \
      --arg memory "${memory_policy_sha256}" '
      select(.schema_version == 1)
      | select(.task_id == $task_id)
      | select(.profile == "fast_dllm_v1_llada_12row_stratified520_batch1")
      | select(.status == "PRECOMMITTED")
      | select(.workload_policy_sha256 == $workload)
      | select(.memory_policy_sha256 == $memory)
      | select(.approval_id | type == "string" and test("^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$"))
      | select(.fixtures | type == "object")
      | select((.fixtures | keys | sort) == ["corrected_reference", "near_cap_stress"])
      | .fixtures[$kind] as $fixture
      | select($fixture | type == "object")
      | select($fixture.kind == $kind)
      | select($fixture.archive | type == "string" and length > 0)
      | select($fixture.archive_sha256 | type == "string" and test("^[0-9a-f]{64}$"))
      | select($fixture.archive_size_bytes | type == "number" and . > 0 and floor == .)
      | [.approval_id, $fixture.archive, $fixture.archive_sha256,
         ($fixture.archive_size_bytes | tostring)] | @tsv
    ' "${manifest}" 2>&1)"; then
    die "memory fixture manifest is not an exact PRECOMMITTED task/policy binding: ${values}"
  fi
  IFS=$'\t' read -r \
    MEMORY_FIXTURE_APPROVAL_ID \
    fixture_relative \
    MEMORY_FIXTURE_ARCHIVE_SHA256 \
    MEMORY_FIXTURE_ARCHIVE_SIZE_BYTES <<<"${values}"
  [[ "${fixture_relative}" != /* ]] \
    || die "memory fixture manifest archive path must be relative"
  [[ -d "${task_root}" && ! -L "${task_root}" ]] \
    || die "memory fixture task root must be a real directory"
  task_real="$(realpath "${task_root}")"
  fixture_path="${task_root}/${fixture_relative}"
  [[ -f "${fixture_path}" && ! -L "${fixture_path}" ]] \
    || die "precommitted memory fixture archive is missing or unsafe"
  fixture_real="$(realpath "${fixture_path}")"
  [[ "${fixture_real}" == "${task_real}/"* ]] \
    || die "precommitted memory fixture archive escapes the task root"
  [[ "$(sha256sum -- "${fixture_real}" | awk '{print $1}')" \
      == "${MEMORY_FIXTURE_ARCHIVE_SHA256}" \
      && "$(stat -c '%s' -- "${fixture_real}")" \
      == "${MEMORY_FIXTURE_ARCHIVE_SIZE_BYTES}" ]] \
    || die "precommitted memory fixture archive bytes do not match the manifest"
  [[ "$(sha256sum -- "${evaluated_archive}" | awk '{print $1}')" \
      == "${MEMORY_FIXTURE_ARCHIVE_SHA256}" \
      && "$(stat -c '%s' -- "${evaluated_archive}")" \
      == "${MEMORY_FIXTURE_ARCHIVE_SIZE_BYTES}" ]] \
    || die "evaluated archive does not match the selected memory fixture"
  MEMORY_FIXTURE_MANIFEST_SHA256="$(sha256sum -- "${manifest}" | awk '{print $1}')"
  MEMORY_FIXTURE_KIND="${kind}"
  export MEMORY_FIXTURE_MANIFEST_SHA256 MEMORY_FIXTURE_APPROVAL_ID
  export MEMORY_FIXTURE_KIND MEMORY_FIXTURE_ARCHIVE_SHA256
  export MEMORY_FIXTURE_ARCHIVE_SIZE_BYTES
}

write_runtime_provenance_snapshot() {
  local output="$1"
  local sforge_runtime_sha256="$2"
  local sforge_file_set_sha256="$3"
  local sforge_file_count="$4"
  local sforge_files_json="$5"
  local verify_decoder_sha256="$6"
  local candidate_worker_sha256="$7"
  local cohort_sha256="$8"
  local quality_sha256="$9"
  local source_admission_sha256="${10}"
  local digest temporary computed_file_set_sha256
  [[ "${output}" == /* && ! -L "${output}" ]] \
    || die "runtime provenance output must be an absolute non-symlink path"
  for digest in \
    "${sforge_runtime_sha256}" \
    "${sforge_file_set_sha256}" \
    "${verify_decoder_sha256}" \
    "${candidate_worker_sha256}" \
    "${cohort_sha256}" \
    "${quality_sha256}" \
    "${source_admission_sha256}"; do
    [[ "${digest}" =~ ^[0-9a-f]{64}$ ]] \
      || die "runtime provenance received an invalid SHA-256"
  done
  [[ "${sforge_file_count}" =~ ^[0-9]+$ ]] \
    && (( 10#${sforge_file_count} > 0 )) \
    || die "runtime provenance file count must be positive"
  jq -e \
    --argjson count "${sforge_file_count}" '
    type == "array"
    and length == $count
    and length == (unique | length)
    and . == sort
    and all(.[];
      type == "string"
      and test("^sforge/[A-Za-z0-9_./-]+[.]py$")
      and (split("/") | index("..") == null)
    )
  ' <<<"${sforge_files_json}" >/dev/null \
    || die "runtime provenance SForge file set is invalid"
  computed_file_set_sha256="$(
    jq -r '.[]' <<<"${sforge_files_json}" | sha256sum | awk '{print $1}'
  )"
  [[ "${computed_file_set_sha256}" == "${sforge_file_set_sha256}" ]] \
    || die "runtime provenance SForge file-set digest is inconsistent"
  temporary="$(mktemp "${output}.tmp.XXXXXX")"
  if ! jq -nS \
      --arg runtime_sha256 "${sforge_runtime_sha256}" \
      --arg file_set_sha256 "${sforge_file_set_sha256}" \
      --argjson file_count "${sforge_file_count}" \
      --argjson files "${sforge_files_json}" \
      --arg verifier "${verify_decoder_sha256}" \
      --arg worker "${candidate_worker_sha256}" \
      --arg cohort "${cohort_sha256}" \
      --arg quality "${quality_sha256}" \
      --arg source_admission "${source_admission_sha256}" '
      {
        schema_version: 1,
        sforge_runtime: {
          scope: "all_sforge_python",
          sha256: $runtime_sha256,
          files: $files,
          file_count: $file_count,
          file_set_sha256: $file_set_sha256
        },
        judge_invariant_core: {
          verify_decoder_sha256: $verifier,
          candidate_worker_sha256: $worker,
          cohort_sha256: $cohort,
          quality_sha256: $quality,
          source_admission_sha256: $source_admission
        }
      }
    ' >"${temporary}"; then
    rm -f -- "${temporary}"
    die "failed to write runtime provenance snapshot"
  fi
  mv -- "${temporary}" "${output}"
}

canonical_sforge_runtime_files_json() {
  (( $# > 0 )) || die "SForge Python runtime source set is empty"
  printf '%s\n' "$@" | jq -Rsc 'split("\n")[:-1]'
}

judge_invariant_core_hashes() {
  local image="$1"
  [[ -n "${image}" ]] || die "Judge image identity is empty"
  docker run --rm --network none --entrypoint sha256sum "${image}" \
    /opt/decoder_judge/verifier/verify_decoder.py \
    /opt/decoder_judge/verifier/candidate_worker.py \
    /opt/decoder_judge/verifier/cohort.py \
    /opt/decoder_judge/verifier/quality.py \
    /opt/decoder_judge/verifier/source_admission.py \
    | awk '{print $1}'
}

# The Judge-only private_data/ directory: the in-repository copy when present,
# else the task's dataset asset that assets/fetch.sh places at
# <INFRATHON_ASSETS_ROOT>/<task directory>/judge/private_data (default asset root:
# ~/.cache/infrathon/assets).  Mirrors scripts/h100_policy_contract.py.
# --prefer-asset-root: an explicitly set INFRATHON_ASSETS_ROOT wins over the
# in-repository copy (image builds stage from the dataset when it is named).
judge_private_data_dir() {
  local in_repository="${MIGRATION_ROOT}/images/judge/private_data"
  if [[ "${1:-}" == --prefer-asset-root && -n "${INFRATHON_ASSETS_ROOT:-}" ]]; then
    in_repository=""
  fi
  if [[ -n "${in_repository}" && -d "${in_repository}" ]] \
      && [[ -n "$(ls -A "${in_repository}")" ]]; then
    printf '%s\n' "${in_repository}"
    return
  fi
  local assets_root="${INFRATHON_ASSETS_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/infrathon/assets}"
  printf '%s\n' "${assets_root}/$(basename "${MIGRATION_ROOT}")/judge/private_data"
}

canonical_hidden_cases_sha256() {
  local plain="${1:-$(judge_private_data_dir)/hidden_cases.json}"
  local compressed="${2:-${plain}.gz}"
  local plain_digest=""
  local compressed_digest=""
  require_command gzip
  if [[ -e "${plain}" ]]; then
    [[ -f "${plain}" && ! -L "${plain}" ]] \
      || die "hidden case source must be a regular non-symlink file: ${plain}"
    plain_digest="$(sha256sum "${plain}" | awk '{print $1}')"
  fi
  if [[ -e "${compressed}" ]]; then
    [[ -f "${compressed}" && ! -L "${compressed}" ]] \
      || die "compressed hidden case source must be a regular non-symlink file: ${compressed}"
    if ! compressed_digest="$(gzip -cd -- "${compressed}" | sha256sum | awk '{print $1}')"; then
      die "failed to validate compressed hidden case source: ${compressed}"
    fi
  fi
  [[ -n "${plain_digest}" || -n "${compressed_digest}" ]] \
    || die "missing canonical hidden case source (${plain} or ${compressed})"
  if [[ -n "${plain_digest}" && -n "${compressed_digest}" ]]; then
    [[ "${plain_digest}" == "${compressed_digest}" ]] \
      || die "plain/compressed hidden case bytes disagree"
  fi
  printf '%s\n' "${plain_digest:-${compressed_digest}}"
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
      # Host-side launcher scripts under scripts/ never enter an image, so they are
      # not image build sources; editing them must not mark the images stale.
      find tasks \
        -type f \
        ! -path '*/__pycache__/*' \
        ! -name '*.pyc' \
        -print0
      printf '%s\0' \
        judge/run_judge.py \
        memory_policy.json \
        workload_policy.json \
        quality_controls.json \
        memory_qualification_fixtures.json \
        docs/dataset_manifest.json \
        docs/model_manifest.json \
        host_reference/corrected_decoder/candidate.py
      # Internal run configs and calibration provenance records: hashed when
      # present (a development checkout), absent from the public tree.
      for optional_provenance in \
        experiment-*.yaml \
        docs/reference_calibration_image_manifest.json \
        docs/reference_model_validation.json \
        docs/reference_calibration_evaluation_attestation.json; do
        [[ ! -f "${optional_provenance}" ]] || printf '%s\0' "${optional_provenance}"
      done
    } \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

assert_built_images_current() {
  local manifest="${LOGS_DIR}/build/image-manifest.json"
  [[ -s "${manifest}" ]] \
    || die "missing build manifest; run scripts/build_images.sh"
  local recorded_source current_source
  local recorded_work current_work recorded_judge current_judge
  recorded_source="$(jq -er '.source_digest' "${manifest}")" \
    || die "build manifest lacks source_digest"
  recorded_work="$(jq -er '.work.id' "${manifest}")" \
    || die "build manifest lacks work image ID"
  recorded_judge="$(jq -er '.judge.id' "${manifest}")" \
    || die "build manifest lacks judge image ID"
  current_source="$(image_source_digest)"
  current_work="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}" 2>/dev/null)" \
    || die "missing work image; run scripts/build_images.sh"
  current_judge="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}" 2>/dev/null)" \
    || die "missing judge image; run scripts/build_images.sh"
  [[ "${current_source}" == "${recorded_source}" ]] \
    || die "image build sources changed; rebuild images before running"
  [[ "${current_work}" == "${recorded_work}" ]] \
    || die "work image tag no longer matches the recorded build"
  [[ "${current_judge}" == "${recorded_judge}" ]] \
    || die "judge image tag no longer matches the recorded build"
  echo "Image/source freshness: PASS (${current_source:0:12})"
}

wait_for_http() {
  local url="$1"
  local attempts="${2:-60}"
  for _ in $(seq 1 "${attempts}"); do
    if curl \
      --noproxy '*' \
      --connect-timeout 2 \
      --max-time 5 \
      -fsS "${url}" >/dev/null 2>&1; then
      return
    fi
    sleep 1
  done
  die "service did not become healthy: ${url}"
}
