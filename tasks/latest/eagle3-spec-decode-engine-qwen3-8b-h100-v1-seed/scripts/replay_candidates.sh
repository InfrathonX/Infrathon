#!/usr/bin/env bash
set -Eeuo pipefail

# Serial checkpoint or implementation replay for the EAGLE-3 decode task.
# Candidate selection is delegated to SForge's persistent version store, but
# this wrapper retains task-specific image, artifact, and GPU identity checks.

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'USAGE'
Usage:
  replay_candidates.sh --origin-run RUN_ID --measurement-set NAME \
    --version-store-root DIR [options]

Required:
  --origin-run, --source-run-id RUN_ID  Closed run with saved versions
  --measurement-set NAME                Stable evaluation series name
  --version-store-root DIR              Persistent SForge version store

Options:
  --replay-run-id RUN_ID                Replay output run ID
  --timeout SECONDS                     Per-candidate Judge timeout
  --candidate vN|CANDIDATE_ID           Repeatable exact checkpoint selector
  --from-v N / --to-v N                 Inclusive checkpoint range
  --every N                             Sample every Nth checkpoint
  --repeat N                            Extra measurements per selected point
  --per-version                         Run replay-versions so every retained
                                        engine_vNNN is scored independently
  --resume                              Resume an identical replay plan
  --dry-run                             Preview selection without GPU or Docker
  --json                                Emit the resulting curve as JSON
USAGE
}

require_value() {
  [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

safe_id() {
  local label="$1"
  local value="$2"
  [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$ ]] \
    || die "unsafe ${label}: ${value}"
}

origin_run=""
measurement_set=""
version_store_root=""
replay_run_id=""
timeout=""
per_version=0
dry_run=0
json_output=0
selection_args=()
repeat_args=()

while (( $# )); do
  case "$1" in
    --origin-run|--source-run-id)
      require_value "$@"
      origin_run="$2"
      shift 2
      ;;
    --measurement-set)
      require_value "$@"
      measurement_set="$2"
      shift 2
      ;;
    --version-store-root)
      require_value "$@"
      version_store_root="$2"
      shift 2
      ;;
    --replay-run-id)
      require_value "$@"
      replay_run_id="$2"
      shift 2
      ;;
    --timeout)
      require_value "$@"
      timeout="$2"
      shift 2
      ;;
    --candidate)
      require_value "$@"
      selection_args+=(--candidate "$2")
      shift 2
      ;;
    --from-v|--to-v|--every)
      require_value "$@"
      selection_args+=("$1" "$2")
      shift 2
      ;;
    --repeat)
      require_value "$@"
      repeat_args+=(--repeat "$2")
      shift 2
      ;;
    --per-version)
      per_version=1
      shift
      ;;
    --resume)
      selection_args+=(--resume)
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    --json)
      json_output=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

[[ -n "${origin_run}" ]] || die "--origin-run is required"
[[ -n "${measurement_set}" ]] || die "--measurement-set is required"
[[ -n "${version_store_root}" ]] || die "--version-store-root is required"
safe_id "origin run" "${origin_run}"
safe_id "measurement set" "${measurement_set}"
[[ -z "${replay_run_id}" ]] || safe_id "replay run" "${replay_run_id}"
[[ -z "${timeout}" || "${timeout}" =~ ^[1-9][0-9]*$ ]] \
  || die "invalid --timeout: ${timeout}"
[[ -d "${version_store_root}" && ! -L "${version_store_root}" ]] \
  || die "version store root is missing or unsafe: ${version_store_root}"

subcommand="replay-candidates"
if (( per_version )); then
  subcommand="replay-versions"
  (( ${#selection_args[@]} == 0 )) \
    || die "--per-version does not take checkpoint selection flags"
fi

command=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/sforge_h100_entry.py"
  "${subcommand}"
  --backend docker
  --task "${TASK_ID}"
  --run-id "${origin_run}"
  --measurement-set "${measurement_set}"
  --version-store-root "${version_store_root}"
  --judge-cpu-limit 8
  --judge-mem-limit 64g
)
[[ -z "${replay_run_id}" ]] || command+=(--replay-run-id "${replay_run_id}")
[[ -z "${timeout}" ]] || command+=(--timeout "${timeout}")
(( ! json_output )) || command+=(--json)
command+=("${selection_args[@]}" "${repeat_args[@]}")

# Dry-run intentionally avoids Docker, artifact, and GPU checks so it remains
# useful on CPU-only planning hosts and in repository integration tests.
if (( dry_run )); then
  exec "${command[@]}" --dry-run
fi

ensure_docker_group "${ORIGINAL_ARGS[@]}"
require_sforge
require_command flock
require_command jq
require_command nvidia-smi
assert_built_images_current

judge_gpu="$(canonical_gpu_index "${JUDGE_GPU:?set JUDGE_GPU}")"
judge_uuid="$(gpu_uuid "${judge_gpu}")"
exec 8>"$(gpu_lock_path "${judge_uuid}")"
flock -n 8 || die "Judge GPU ${judge_gpu} lock is already held"
assert_gpu_available "${judge_gpu}" "Judge"
assert_container_gpu "${JUDGE_IMAGE}" "${judge_uuid}" "Judge"

"${PYTHON_BIN}" "${SCRIPT_DIR}/preflight.py" \
  --target-repo "${EAGLE3_TARGET_REPO}" \
  --head-dir "${EAGLE3_HEAD_DIR}" \
  --judge-gpu "${judge_gpu}" >/dev/null

build_manifest="${LOGS_DIR}/build/image-manifest.json"
scope="${replay_run_id:-candidate-${origin_run}-${measurement_set}}"
export EAGLE3_JUDGE_GPU_UUID="${judge_uuid}"
export EAGLE3_RUNTIME_RUN_ID="${scope}"
export SFORGE_EXPECTED_WORK_IMAGE_ID="$(jq -er '.work.id' "${build_manifest}")"
export SFORGE_EXPECTED_JUDGE_IMAGE_ID="$(jq -er '.judge.id' "${build_manifest}")"
export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${judge_uuid},CUDA_VISIBLE_DEVICES=0,EAGLE3_GPU_ROLE=judge,EAGLE3_INTERACTIVE_JUDGE_TIMEOUT=${EAGLE3_INTERACTIVE_JUDGE_TIMEOUT:-3600},EAGLE3_FINAL_JUDGE_TIMEOUT=${EAGLE3_FINAL_JUDGE_TIMEOUT:-5400}"

"${command[@]}"
assert_gpu_released "${judge_uuid}" "${judge_gpu}" "replay Judge"
