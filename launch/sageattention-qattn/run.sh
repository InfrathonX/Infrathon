#!/usr/bin/env bash
set -euo pipefail

# Unified experiment entrypoint for sageattention-quantized-attention-bf16-h100-v2-seed.
# Defaults to the paper's setup: Work on one GPU and the Judge on another
# (--work-gpu, --judge-gpu).  --shared-gpu runs both on one GPU under the
# fail-closed single-allocation contract, pausing Work while the Judge measures.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
REPO_ROOT="$(launch_repo_root)"
TASK_ROOT="${REPO_ROOT}/tasks/latest/sageattention-quantized-attention-bf16-h100-v2-seed"
SINGLE_ENTRY="${TASK_ROOT}/scripts/run_single_gpu_fixed_budget.sh"
LEGACY_ENTRY="${TASK_ROOT}/scripts/run_fixed_budget.sh"
ENTRY="${SINGLE_ENTRY}"

usage() {
  cat <<'USAGE'
Usage:
  run.sh --model MODEL [options] [-- extra]

Options:
  --agent NAME       codex (default) or claude-code
  --model NAME       A full gpt-*/claude-* model ID,
                     or any ID the MODEL_API_BASE_URL endpoint serves
  --effort LEVEL     Reasoning effort (max|ultra|xhigh|high|medium|low); default max
  --hours N          Agent budget in hours (sets TASK_TIMEOUT=N*3600; default 12 h)
  --timeout SECONDS  Agent budget in seconds (overrides --hours)
  --run-id ID        Explicit run ID (default: task-derived)
  --auth FILE        Codex auth.json path (sets CODEX_AUTH_JSON_PATH)
  --claude-token FILE  0600 file with one `claude setup-token` OAuth token (claude-code)
  --work-gpu INDEX   GPU of the Work container (default 0)
  --judge-gpu INDEX  GPU of the Judge (default 1)
  --shared-gpu       Run Work and the Judge on one GPU; the host pauses Work
                     while the Judge measures
  --gpu INDEX        GPU of a shared run (implies --shared-gpu; default 0)
  --device-uuid UUID Scheduler-issued canonical whole-GPU UUID (implies
                     --shared-gpu; instead of --gpu)
  --allocation-key K Scheduler-issued allocation-sha256:<digest> (shared mode)
  --lease-id ID      Optional scheduler lease ID (shared mode); generated once if omitted
  --calibration-epoch ID  Qualified clock/power campaign epoch (optional)
  --judge-port PORT  Scheduler-unique host coordinator port (default 18602)
  --cpuset VALUE     Pin Work/Judge to GPU-local CPUs: auto|off|CPULIST[@NODES]
                     (default off; sets SFORGE_CPUSET; docs/CPU_AFFINITY.md)
  --legacy-dual-gpu  Older name for the default separate-GPU mode
  --dry-run          Resolve and check the plan (model, runtime, images, integrity
                     review, GPU, port), print it and exit; no lock, container or GPU work
  --tmux             Run detached in a new tmux session
  --config FILE      Launch config (.env or .json); see launch/mlsysforge.env.example
  -- ...             Everything after -- is passed to the task script

Separate mode (the default) also reads WORK_GPU and JUDGE_GPU. Shared mode
accepts SFORGE_ACCELERATOR_DEVICE_UUID (which selects it) and
SFORGE_ACCELERATOR_ALLOCATION_KEY, and optionally SFORGE_TRUSTED_LEASE_ID; the
task launcher generates a never-reused lease ID once when it is omitted. Shared
mode rejects WORK_GPU/JUDGE_GPU and never falls back to separate GPUs. The
Codex CLI pin comes from launch/common.sh unless CODEX_CLI_VERSION is already set.
USAGE
}

agent="codex"
claude_token=""
model=""
effort=""
timeout_seconds=""
hours=""
run_id=""
auth=""
gpu_index=""
work_gpu=""
judge_gpu=""
shared_gpu=0
device_uuid=""
allocation_key=""
lease_id=""
calibration_epoch=""
judge_port=""
cpuset=""
legacy_dual_gpu=0
config_file=""
use_tmux=0
dry_run=0
passthrough=()

while (( $# )); do
  case "$1" in
    --agent)   [[ $# -ge 2 ]] || launch_die "--agent requires a value"; agent="$2"; shift 2 ;;
    --claude-token) [[ $# -ge 2 ]] || launch_die "--claude-token requires a value"; claude_token="$2"; shift 2 ;;
    --model)   [[ $# -ge 2 ]] || launch_die "--model requires a value"; model="$2"; shift 2 ;;
    --effort)  [[ $# -ge 2 ]] || launch_die "--effort requires a value"; effort="$2"; shift 2 ;;
    --hours)   [[ $# -ge 2 ]] || launch_die "--hours requires a value"; hours="$2"; shift 2 ;;
    --timeout) [[ $# -ge 2 ]] || launch_die "--timeout requires a value"; timeout_seconds="$2"; shift 2 ;;
    --run-id)  [[ $# -ge 2 ]] || launch_die "--run-id requires a value"; run_id="$2"; shift 2 ;;
    --auth)    [[ $# -ge 2 ]] || launch_die "--auth requires a value"; auth="$2"; shift 2 ;;
    --gpu)     [[ $# -ge 2 ]] || launch_die "--gpu requires a value"; gpu_index="$2"; shift 2 ;;
    --work-gpu) [[ $# -ge 2 ]] || launch_die "--work-gpu requires a value"; work_gpu="$2"; shift 2 ;;
    --judge-gpu) [[ $# -ge 2 ]] || launch_die "--judge-gpu requires a value"; judge_gpu="$2"; shift 2 ;;
    --shared-gpu) shared_gpu=1; shift ;;
    --device-uuid) [[ $# -ge 2 ]] || launch_die "--device-uuid requires a value"; device_uuid="$2"; shift 2 ;;
    --allocation-key) [[ $# -ge 2 ]] || launch_die "--allocation-key requires a value"; allocation_key="$2"; shift 2 ;;
    --lease-id) [[ $# -ge 2 ]] || launch_die "--lease-id requires a value"; lease_id="$2"; shift 2 ;;
    --calibration-epoch) [[ $# -ge 2 ]] || launch_die "--calibration-epoch requires a value"; calibration_epoch="$2"; shift 2 ;;
    --judge-port) [[ $# -ge 2 ]] || launch_die "--judge-port requires a value"; judge_port="$2"; shift 2 ;;
    --cpuset)  [[ $# -ge 2 ]] || launch_die "--cpuset requires a value"; cpuset="$2"; shift 2 ;;
    --legacy-dual-gpu) legacy_dual_gpu=1; shift ;;
    --config)  [[ $# -ge 2 ]] || launch_die "--config requires a value"; config_file="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --tmux)    use_tmux=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --)        shift; passthrough=("$@"); break ;;
    *)         usage >&2; launch_die "unknown argument: $1" ;;
  esac
done

launch_load_config "${config_file}"
export_cpuset_policy "${cpuset}"

# Shared mode: --shared-gpu, a single-allocation flag, or a scheduler-issued
# UUID in the environment.  Otherwise Work and the Judge get a GPU each.
if (( legacy_dual_gpu )); then
  (( ! shared_gpu )) && [[ -z "${gpu_index}${device_uuid}${allocation_key}${lease_id}" ]] \
    || launch_die "single-allocation flags cannot be combined with --legacy-dual-gpu"
elif [[ -n "${gpu_index}${device_uuid}${allocation_key}${lease_id}" \
        || -n "${SFORGE_ACCELERATOR_DEVICE_UUID:-}" ]]; then
  shared_gpu=1
fi
if (( ! shared_gpu )); then
  [[ "${SFORGE_SINGLE_GPU_HANDOFF:-0}" != "1" \
     && ! -v SFORGE_ACCELERATOR_DEVICE_UUID \
     && ! -v SFORGE_ACCELERATOR_ALLOCATION_KEY \
     && ! -v SFORGE_ACCELERATOR_OBSERVER \
     && ! -v SFORGE_TRUSTED_LEASE_ID ]] \
    || launch_die "separate GPUs reject inherited single-allocation environment; pass --shared-gpu"
  assert_cpuset_legacy_safe
  WORK_GPU="${work_gpu:-${WORK_GPU:-0}}"
  JUDGE_GPU="${judge_gpu:-${JUDGE_GPU:-1}}"
  [[ "${WORK_GPU}" =~ ^[0-9]+$ && "${JUDGE_GPU}" =~ ^[0-9]+$ ]] \
    || launch_die "--work-gpu and --judge-gpu take a GPU index"
  [[ "${WORK_GPU}" != "${JUDGE_GPU}" ]] \
    || launch_die "--work-gpu and --judge-gpu must differ; pass --shared-gpu to use one GPU"
  export WORK_GPU JUDGE_GPU
  export SFORGE_SINGLE_GPU_HANDOFF=0
  ENTRY="${LEGACY_ENTRY}"
else
  [[ -z "${work_gpu}${judge_gpu}" ]] \
    || launch_die "--work-gpu/--judge-gpu name separate GPUs; drop them for a shared run"
  [[ ! -v WORK_GPU && ! -v JUDGE_GPU ]] \
    || launch_die "shared mode rejects WORK_GPU/JUDGE_GPU"
  [[ -z "${gpu_index}" || -z "${device_uuid}" ]] \
    || launch_die "--gpu and --device-uuid both name the GPU; pass one"
  [[ -z "${device_uuid}" ]] || export SFORGE_ACCELERATOR_DEVICE_UUID="${device_uuid}"
  [[ -z "${allocation_key}" ]] || export SFORGE_ACCELERATOR_ALLOCATION_KEY="${allocation_key}"
  [[ -z "${lease_id}" ]] || export SFORGE_TRUSTED_LEASE_ID="${lease_id}"
fi
[[ -z "${calibration_epoch}" ]] \
  || export SFORGE_CALIBRATION_EPOCH="${calibration_epoch}"
if [[ -n "${judge_port}" ]]; then
  [[ "${judge_port}" =~ ^[0-9]+$ ]] \
    || launch_die "--judge-port must be an integer"
  judge_port="$((10#${judge_port}))"
  (( judge_port >= 1024 && judge_port <= 65535 && judge_port != 3128 )) \
    || launch_die "--judge-port must be 1024..65535 and not 3128"
  export JUDGE_PORT="${judge_port}"
fi

[[ -x "${ENTRY}" ]] || launch_die "task entry script is missing: ${ENTRY}"

if [[ -n "${model}" ]]; then
  MODEL="$(resolve_model_id "${model}")"
  export MODEL
fi
effort="${effort:-${REASONING_EFFORT:-max}}"
validate_reasoning_effort "${effort}"
export REASONING_EFFORT="${effort}"
if [[ -n "${timeout_seconds}" ]]; then
  [[ "${timeout_seconds}" =~ ^[1-9][0-9]*$ ]] || launch_die "invalid --timeout"
  export TASK_TIMEOUT="${timeout_seconds}"
elif [[ -n "${hours}" ]]; then
  [[ "${hours}" =~ ^[1-9][0-9]*$ ]] || launch_die "invalid --hours"
  export TASK_TIMEOUT="$(( hours * 3600 ))"
elif [[ -z "${TASK_TIMEOUT:-}" ]]; then
  export TASK_TIMEOUT="43200"
fi
[[ -z "${run_id}" ]] || export RUN_ID="${run_id}"
if [[ -n "${auth}" ]]; then
  [[ -f "${auth}" ]] || launch_die "auth file does not exist: ${auth}"
  export CODEX_AUTH_JSON_PATH="${auth}"
fi
case "${agent}" in
  codex)
    export MLSYSFORGE_AGENT=codex
    export CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-${CODEX_CLI_VERSION_PIN}}"
    ;;
  claude-code)
    export MLSYSFORGE_AGENT=claude-code
    if [[ -n "${claude_token}" ]]; then
      [[ -f "${claude_token}" ]] || launch_die "Claude token file does not exist: ${claude_token}"
      export CLAUDE_OAUTH_TOKEN_FILE="${claude_token}"
    fi
    [[ -n "${CLAUDE_OAUTH_TOKEN_FILE:-}" ]] \
      || launch_die "claude-code needs --claude-token FILE (or CLAUDE_OAUTH_TOKEN_FILE)"
    ;;
  *) launch_die "--agent takes codex or claude-code" ;;
esac
# A shared run without --device-uuid (or SFORGE_ACCELERATOR_DEVICE_UUID) takes
# the GPU named by --gpu, by default GPU 0.
if (( shared_gpu )) && [[ -z "${device_uuid}" ]] \
   && [[ -n "${gpu_index}" || -z "${SFORGE_ACCELERATOR_DEVICE_UUID:-}" ]]; then
  SFORGE_ACCELERATOR_DEVICE_UUID="$(gpu_uuid_for_index "${gpu_index:-0}")" || exit 1
  export SFORGE_ACCELERATOR_DEVICE_UUID
fi
export JUDGE_PORT="${JUDGE_PORT:-18602}"
assert_container_reachable_proxy --entry "${ENTRY}"
export_version_store_root --task-root "${TASK_ROOT}"
if [[ "${agent}" == claude-code ]]; then export_claude_runtime_dir; else export_agent_runtime_dir; fi
(( ! dry_run )) || export SFORGE_LAUNCH_DRY_RUN=1

# run_in_tmux forwards this shell's complete environment, so the detached
# path is equivalent to the direct exec below.
if (( use_tmux && ! dry_run )); then
  stamp="$(date -u +%Y%m%d-%H%M%S)"
  session="sageattention-qattn-v2-${stamp}"
  run_in_tmux "${session}" \
    "${SCRIPT_DIR}/logs/run-${stamp}.log" \
    "${ENTRY}" "${passthrough[@]}"
else
  exec "${ENTRY}" "${passthrough[@]}"
fi
