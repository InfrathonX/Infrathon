#!/usr/bin/env bash
set -euo pipefail

# Serial candidate batch replay for the Flux MoE AG scatter task (eight GPUs).
# Reuses this task's lib.sh preflight (image currency, GPU locks, idle
# assertions) and holds all eight GPU leases for the complete batch, then
# dispatches the shared core batch command. No per-task curve logic here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_sforge
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'USAGE'
Usage:
  replay_candidates.sh --origin-run RUN_ID [--measurement-set NAME] \
    [--version-store-root DIR] [options]

Required:
  --origin-run, --source-run-id RUN_ID  Closed agent run with saved versions

Options:
  --measurement-set NAME                Stable evaluator series name
                                        (default: flux-moe-ag-replay)
  --version-store-root DIR              Persistent SForge version store;
                                        defaults to SFORGE_VERSION_STORE_ROOT
  --replay-run-id RUN_ID                Replay output run ID (optional)
  --timeout SECONDS                     Per-candidate Judge timeout
  --candidate vN|CANDIDATE_ID           Repeatable exact candidate selector
  --from-v N / --to-v N                 Inclusive candidate range
  --every N                             Sample every Nth candidate (last always kept)
  --repeat N                            Extra measurements per point (median + spread);
                                        applies to --per-version too
  --per-version                         Score each retained kernel_vNNN implementation
                                        once instead of replaying whole checkpoints
  --resume                              Continue the same replay plan
  --dry-run                             Preview candidate selection (no GPU or Docker)
  --json                                Print the candidate curve as JSON

The full replay holds the flocks of all eight selected GPUs for the whole
serial batch. FLUXMOE_GPU_INDICES names the eight GPUs and is required: it has
no default, and the script dies naming the variable when it is unset. Dry-run is a
side-effect-free preview and takes no GPU lease.
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
measurement_set="flux-moe-ag-replay"
version_store_root="${SFORGE_VERSION_STORE_ROOT:-}"
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
  || { usage >&2; die "--origin-run and a version store root are required"; }
safe_id "origin run" "${origin_run}"
safe_id "measurement set" "${measurement_set}"
if [[ -z "${replay_run_id}" ]]; then
  replay_run_id="flux-moe-ag-replay-${origin_run}"
fi
safe_id "replay run" "${replay_run_id}"
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
cmd=("${TASK_ROOT}/scripts/eval_archive.sh" --replay-core "${subcommand}"
  --backend docker
  --task "${TASK_ID}"
  --run-id "${origin_run}"
  --measurement-set "${measurement_set}"
  --version-store-root "${version_store_root}"
  --replay-run-id "${replay_run_id}"
  --judge-cpu-limit 32
  --judge-mem-limit 128g)
[[ -z "${timeout}" ]] || cmd+=(--timeout "${timeout}")
(( ! json_output )) || cmd+=(--json)
cmd+=("${selection_args[@]}" "${repeat_args[@]}")

if (( dry_run )); then
  exec "${cmd[@]}" --dry-run
fi

require_command docker
require_command flock
require_command jq
require_command nvidia-smi
require_command python3
ensure_docker_group "${ORIGINAL_ARGS[@]}"
assert_built_images_current
resolve_gpu_pair "${FLUXMOE_GPU_INDICES:-}"
export FLUXMOE_GPU_INDICES="${GPU_PAIR_INDICES}"
export FLUXMOE_GPU_UUIDS="${GPU_PAIR_UUIDS}"

lock_gpu_group /tmp/mlsysforge-gpu
IFS=, read -r -a group_indices <<<"${GPU_PAIR_INDICES}"
for index in "${group_indices[@]}"; do
  assert_gpu_idle "${index}" "Judge"
done
python3 "${MIGRATION_ROOT}/scripts/preflight.py" \
  --gpu-indices "${GPU_PAIR_INDICES}" \
  --expected-gpu-uuids "${GPU_PAIR_UUIDS}" >/dev/null

mapfile -t container_uuids < <(
  docker run --rm \
    --gpus "\"device=${GPU_PAIR_UUIDS}\"" \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --entrypoint nvidia-smi \
    "${JUDGE_IMAGE}" --query-gpu=uuid --format=csv,noheader
)
[[ "$(IFS=,; echo "${container_uuids[*]}")" == "${GPU_PAIR_UUIDS}" ]] \
  || die "Judge container expected ${GPU_PAIR_UUIDS}, saw ${container_uuids[*]:-nothing}"
echo "Judge container GPU pin: ${GPU_PAIR_UUIDS}"

"${cmd[@]}"
for uuid in "${GPU_GROUP_UUID_LIST[@]}"; do
  assert_gpu_released "${uuid}"
done
