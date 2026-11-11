#!/usr/bin/env bash
set -euo pipefail

# Serial candidate batch replay for the Flash K-Means paper H100 task.
# Reuses this task's lib.sh preflight (image currency, GPU lock, idle
# assertion) and holds the Judge GPU lease for the complete batch, then
# dispatches the shared core batch command. No per-task curve logic here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_sforge
require_command flock
require_command jq
require_command nvidia-smi

usage() {
  cat <<'USAGE'
Usage:
  replay_candidates.sh --origin-run RUN_ID --measurement-set NAME \
    --version-store-root DIR [options]

Required:
  --origin-run, --source-run-id RUN_ID  Closed agent run with saved versions
  --measurement-set NAME                Stable evaluator series name
  --version-store-root DIR              Persistent SForge version store

Options:
  --replay-run-id RUN_ID                Replay output run ID (optional)
  --timeout SECONDS                     Per-candidate Judge timeout
  --candidate vN|CANDIDATE_ID           Repeatable exact candidate selector
  --from-v N / --to-v N                 Inclusive candidate range
  --every N                             Sample every Nth candidate (last always kept)
  --repeat N                            Extra measurements per point (median + spread);
                                        applies to --per-version too
  --per-version                         Score each retained flash_kmeans_vNNN implementation
                                        once instead of replaying whole checkpoints
  --resume                              Continue the same replay plan
  --dry-run                             Preview candidate selection (no GPU)
  --json                                Print the candidate curve as JSON

The full replay holds this task's Judge GPU flock for the whole serial batch.
Dry-run is a side-effect-free preview and takes no GPU lease.
USAGE
}

require_value() {
  [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

safe_id() {
  [[ "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$ ]] || die "unsafe $1: $2"
}

origin_run=""
per_version=0
measurement_set=""
version_store_root=""
replay_run_id=""
timeout=""
dry_run=0
json_output=0
selection_args=()
repeat_args=()

while (( $# )); do
  case "$1" in
    --origin-run|--source-run-id)
      require_value "$@"; origin_run="$2"; shift 2 ;;
    --measurement-set)
      require_value "$@"; measurement_set="$2"; shift 2 ;;
    --version-store-root)
      require_value "$@"; version_store_root="$2"; shift 2 ;;
    --replay-run-id)
      require_value "$@"; replay_run_id="$2"; shift 2 ;;
    --timeout)
      require_value "$@"; timeout="$2"; shift 2 ;;
    --candidate)
      require_value "$@"; selection_args+=(--candidate "$2"); shift 2 ;;
    --repeat)
      require_value "$@"; repeat_args+=("$1" "$2"); shift 2 ;;
    --from-v|--to-v|--every)
      require_value "$@"; selection_args+=("$1" "$2"); shift 2 ;;
    --per-version)
      per_version=1; shift ;;
    --resume)
      selection_args+=(--resume); shift ;;
    --dry-run)
      dry_run=1; shift ;;
    --json)
      json_output=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      usage >&2; die "unknown argument: $1" ;;
  esac
done

[[ -n "${origin_run}" && -n "${measurement_set}" && -n "${version_store_root}" ]] \
  || { usage >&2; die "--origin-run, --measurement-set, and --version-store-root are required"; }
safe_id "origin run" "${origin_run}"
safe_id "measurement set" "${measurement_set}"
[[ -z "${replay_run_id}" ]] || safe_id "replay run" "${replay_run_id}"
[[ -z "${timeout}" || "${timeout}" =~ ^[1-9][0-9]*$ ]] || die "invalid --timeout: ${timeout}"
[[ -d "${version_store_root}" && ! -L "${version_store_root}" ]] \
  || die "version store root is missing or unsafe: ${version_store_root}"

export SFORGE_TASKS_DIR="${TASKS_DIR}"
export SFORGE_LOG_DIR="${LOGS_DIR}"

subcommand=replay-candidates
if (( per_version )); then
  subcommand=replay-versions
  [[ ${#selection_args[@]} -eq 0 ]] \
    || die "--per-version does not take candidate selection flags"
fi
cmd=("${SFORGE_BIN}" "${subcommand}"
  --backend docker
  --task "${TASK_ID}"
  --run-id "${origin_run}"
  --measurement-set "${measurement_set}"
  --version-store-root "${version_store_root}"
  --judge-cpu-limit 8
  --judge-mem-limit 32g)
[[ -z "${replay_run_id}" ]] || cmd+=(--replay-run-id "${replay_run_id}")
[[ -z "${timeout}" ]] || cmd+=(--timeout "${timeout}")
(( ! json_output )) || cmd+=(--json)
cmd+=("${selection_args[@]}" "${repeat_args[@]}")

if (( dry_run )); then
  exec "${cmd[@]}" --dry-run
fi

assert_built_images_current
judge_gpu="$(canonical_gpu_index "${JUDGE_GPU:-2}")"
judge_uuid="$(gpu_uuid "${judge_gpu}")"
exec 8>"$(gpu_lock_path "${judge_uuid}")"
flock -n 8 || die "judge GPU ${judge_gpu} lock is already held"
assert_gpu_idle "${judge_gpu}" "judge"
assert_container_gpu "${JUDGE_IMAGE}" "${judge_uuid}" "judge"

export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${judge_uuid},CUDA_VISIBLE_DEVICES=0,FLASH_KMEANS_TARGET=H100"

"${cmd[@]}"

