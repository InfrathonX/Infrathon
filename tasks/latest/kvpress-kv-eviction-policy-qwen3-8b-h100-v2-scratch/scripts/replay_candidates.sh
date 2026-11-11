#!/usr/bin/env bash
set -euo pipefail

# Serial candidate batch replay for the KVPress eviction H100 task.
# Reuses this task's lib.sh preflight (image currency, GPU locks, idle
# assertions) and holds every Judge GPU named by JUDGE_GPUS for the complete
# batch, then dispatches the shared core batch command. Each candidate runs
# the complete 1-8-GPU sharded final suite. No per-task curve logic here.

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
  --per-version                         Score each retained policy_vNNN implementation
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
cmd=("${SFORGE_PYTHON}" "${TASK_ENTRY}" "${subcommand}"
  --backend docker
  --task "${TASK_ID}"
  --run-id "${origin_run}"
  --measurement-set "${measurement_set}"
  --version-store-root "${version_store_root}")
[[ -z "${replay_run_id}" ]] || cmd+=(--replay-run-id "${replay_run_id}")
[[ -z "${timeout}" ]] || cmd+=(--timeout "${timeout}")
(( ! json_output )) || cmd+=(--json)
cmd+=("${selection_args[@]}" "${repeat_args[@]}")

if (( dry_run )); then
  exec "${cmd[@]}" --dry-run
fi

assert_built_images_current
mapfile -t judge_uuids < <(judge_gpu_uuids "${JUDGE_GPUS:-}")
for uuid in "${judge_uuids[@]}"; do
  lock_gpu "${uuid}" "judge"
  assert_gpu_idle "${uuid}" "judge"
done
n_gpus="${#judge_uuids[@]}"
cmd+=(--judge-cpu-limit "$((8 * n_gpus))" --judge-mem-limit "${JUDGE_MEM_LIMIT:-$((32 + 8 * n_gpus))g}")

validate_model_repo "${KV_EVICTION_MODEL_REPO}" >/dev/null
export KV_EVICTION_MODEL_REPO
export KV_EVICTION_JUDGE_GPU_UUIDS="${judge_uuids[*]}"
read -r t_candidate t_total t_judge t_eval < <(judge_timeouts "${n_gpus}")
export SFORGE_JUDGE_EXTRA_ENV="KV_EVICTION_CANDIDATE_TIMEOUT=${KV_EVICTION_CANDIDATE_TIMEOUT:-${t_candidate}},KV_EVICTION_TOTAL_TIMEOUT=${KV_EVICTION_TOTAL_TIMEOUT:-${t_total}},KV_EVICTION_JUDGE_TIMEOUT=${KV_EVICTION_JUDGE_TIMEOUT:-${t_judge}}"

[[ -n "${timeout}" ]] || cmd+=(--timeout "${t_eval}")
"${cmd[@]}"
for uuid in "${judge_uuids[@]}"; do assert_gpu_released "${uuid}"; done
