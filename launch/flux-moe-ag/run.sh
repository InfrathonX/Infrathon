#!/usr/bin/env bash
set -euo pipefail

# Unified experiment entrypoint for the Flux MoE AG scatter task (eight GPUs).
# Work and the Judge share one NVSwitch node of eight GPUs (--gpus, default
# 0-7); every Judge round waits for the GPUs to be idle and pauses Work while
# it measures (two Flux symmetric-memory rendezvous at once on an NVSwitch host
# can fail with a CUDA driver OOM). The task-side launcher owns GPU locks,
# preflight, image checks, and the managed SForge lifecycle; this wrapper only
# normalizes shared CLI knobs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
REPO_ROOT="$(launch_repo_root)"
TASK_ROOT="${REPO_ROOT}/tasks/latest/flux-moe-ag-scatter-grouped-gemm-dual-h100-v1-seed"
ENTRY="${TASK_ROOT}/scripts/launch_fixed_budget_after_gpu_pair_idle.sh"

usage() {
  cat <<'USAGE'
Usage:
  run.sh --model MODEL [options] [-- extra]

Options:
  --agent NAME       codex (default) or claude-code
  --model NAME       A full gpt-*/claude-* model ID,
                     or any ID the MODEL_API_BASE_URL endpoint serves
  --effort LEVEL     Reasoning effort (max|ultra|xhigh|high|medium|low); default max
  --gpus A,...,H     The eight GPUs (one NVSwitch node) that Work and the Judge
                     share (sets FLUXMOE_GPU_INDICES; default 0,1,2,3,4,5,6,7)
  --shared-gpus      Accepted for compatibility; the task always shares its GPUs
  --hours N          Agent budget in hours (default 12 h)
  --timeout SECONDS  Agent budget in seconds (overrides --hours)
  --judge-port PORT  Host port of the Judge server (sets JUDGE_PORT; default 18611)
  --label TEXT       RUN_LABEL override (default: <model-short>-<effort>)
  --auth FILE        Codex auth.json path
  --claude-token FILE  0600 file with one `claude setup-token` OAuth token (claude-code)
  --no-wait          Accepted for compatibility: the launcher sets no minimum
                     start delay (MIN_DELAY_SECONDS=0 unless exported)
  --cpuset VALUE     Pin Work/Judge to GPU-local CPUs: auto|off|CPULIST[@NODES]
                     (default off; sets SFORGE_CPUSET; docs/CPU_AFFINITY.md)
  --config FILE      Launch config (.env or .json)
  -- ...             Forward remaining arguments to the task launcher

The run holds all eight GPUs for its whole life. Without --gpus the group comes
from FLUXMOE_GPU_INDICES (environment or launch config), else 0-7. Work is
always paused for each Judge round: the Judge needs every GPU, and two Flux
symmetric-memory rendezvous at once on one NVSwitch host can fail together
("CUDA driver error: out of memory").

The task launcher manages queueing itself; there is no --tmux here.
USAGE
}

agent="codex"
claude_token=""
model=""
effort=""
gpus=""
work_gpus=""
judge_gpus=""
shared_gpus=0
pause_work_for_judge=""
timeout_seconds=""
hours=""
label=""
judge_port=""
auth=""
cpuset=""
config_file=""
no_wait=0
passthrough=()

while (( $# )); do
  case "$1" in
    --agent)   [[ $# -ge 2 ]] || launch_die "--agent requires a value"; agent="$2"; shift 2 ;;
    --claude-token) [[ $# -ge 2 ]] || launch_die "--claude-token requires a value"; claude_token="$2"; shift 2 ;;
    --model)   [[ $# -ge 2 ]] || launch_die "--model requires a value"; model="$2"; shift 2 ;;
    --effort)  [[ $# -ge 2 ]] || launch_die "--effort requires a value"; effort="$2"; shift 2 ;;
    --gpus)    [[ $# -ge 2 ]] || launch_die "--gpus requires a value"; gpus="$2"; shift 2 ;;
    --work-gpus) [[ $# -ge 2 ]] || launch_die "--work-gpus requires a value"; work_gpus="$2"; shift 2 ;;
    --judge-gpus) [[ $# -ge 2 ]] || launch_die "--judge-gpus requires a value"; judge_gpus="$2"; shift 2 ;;
    --shared-gpus) shared_gpus=1; shift ;;
    --pause-work-for-judge) pause_work_for_judge=1; shift ;;
    --no-pause-work-for-judge) pause_work_for_judge=0; shift ;;
    --hours)   [[ $# -ge 2 ]] || launch_die "--hours requires a value"; hours="$2"; shift 2 ;;
    --timeout) [[ $# -ge 2 ]] || launch_die "--timeout requires a value"; timeout_seconds="$2"; shift 2 ;;
    --judge-port) [[ $# -ge 2 ]] || launch_die "--judge-port requires a value"; judge_port="$2"; shift 2 ;;
    --label)   [[ $# -ge 2 ]] || launch_die "--label requires a value"; label="$2"; shift 2 ;;
    --auth)    [[ $# -ge 2 ]] || launch_die "--auth requires a value"; auth="$2"; shift 2 ;;
    --cpuset)  [[ $# -ge 2 ]] || launch_die "--cpuset requires a value"; cpuset="$2"; shift 2 ;;
    --config)  [[ $# -ge 2 ]] || launch_die "--config requires a value"; config_file="$2"; shift 2 ;;
    --no-wait) no_wait=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --)        shift; passthrough=("$@"); break ;;
    *)         usage >&2; launch_die "unknown argument: $1" ;;
  esac
done

launch_load_config "${config_file}"
export_cpuset_policy "${cpuset}"
[[ -x "${ENTRY}" ]] || launch_die "task entry script is missing: ${ENTRY}"

# Normalize the eight-GPU group to distinct decimal indices, or die naming the flag.
flux_gpu_group() {
  local value="$1" flag="$2" index
  local -a raw=() normalized=()
  local -A seen=()
  [[ "${value}" =~ ^[0-9]+(,[0-9]+)*$ ]] \
    || launch_die "${flag} takes eight comma-separated physical GPU indices"
  IFS=, read -r -a raw <<<"${value}"
  (( ${#raw[@]} == 8 )) || launch_die "${flag} must name exactly eight GPUs"
  for index in "${raw[@]}"; do
    index="$((10#${index}))"
    [[ -z "${seen[${index}]:-}" ]] || launch_die "${flag} must name eight different GPUs"
    seen["${index}"]=1
    normalized+=("${index}")
  done
  (IFS=,; echo "${normalized[*]}")
}

[[ -z "${judge_gpus}" ]] \
  || launch_die "--judge-gpus: the eight-GPU task runs the Judge on the same eight GPUs as Work"
[[ "${pause_work_for_judge}" != 0 ]] \
  || launch_die "--no-pause-work-for-judge: the Judge needs all eight GPUs, so Work is always paused"
if [[ -n "${work_gpus}" ]]; then
  [[ -z "${gpus}" ]] || launch_die "--gpus and --work-gpus both name the GPU group; pass one"
  gpus="${work_gpus}"
fi
FLUXMOE_GPU_INDICES="$(flux_gpu_group "${gpus:-${FLUXMOE_GPU_INDICES:-0,1,2,3,4,5,6,7}}" --gpus)" || exit 1
export FLUXMOE_GPU_INDICES
export FLUXMOE_JUDGE_GPU_INDICES=shared
export FLUXMOE_PAUSE_WORK_FOR_JUDGE=1
: "${shared_gpus}"

model_short=""
if [[ -n "${model}" ]]; then
  AGENT_MODEL="$(resolve_model_id "${model}")"
  export AGENT_MODEL
  case "${AGENT_MODEL}" in
    claude-*) model_short="${AGENT_MODEL}" ;;
    *) model_short="${AGENT_MODEL##*-}" ;;
  esac
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
  export TASK_TIMEOUT=43200
fi
if [[ -n "${label}" ]]; then
  export RUN_LABEL="${label}"
elif [[ -n "${model_short}" && -n "${effort}" ]]; then
  export RUN_LABEL="${model_short}-${effort}"
fi
if [[ -n "${auth}" ]]; then
  [[ -f "${auth}" ]] || launch_die "auth file does not exist: ${auth}"
  export CODEX_AUTH_JSON_PATH="${auth}"
fi
export MIN_DELAY_SECONDS="${MIN_DELAY_SECONDS:-0}"
(( ! no_wait )) || export MIN_DELAY_SECONDS=0
if [[ -n "${judge_port}" ]]; then
  [[ "${judge_port}" =~ ^[0-9]+$ ]] || launch_die "--judge-port must be an integer"
  export JUDGE_PORT="$((10#${judge_port}))"
fi
export JUDGE_PORT="${JUDGE_PORT:-18611}"
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
assert_container_reachable_proxy --entry "${ENTRY}"
export_version_store_root --task-root "${TASK_ROOT}"
if [[ "${agent}" == claude-code ]]; then export_claude_runtime_dir; else export_agent_runtime_dir; fi

exec "${ENTRY}" "${passthrough[@]}"
