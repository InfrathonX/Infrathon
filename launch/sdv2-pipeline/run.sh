#!/usr/bin/env bash
set -euo pipefail

# Unified experiment entrypoint for the StreamDiffusionV2 dual-GPU pipeline task, v2.
# The task-side launcher owns GPU locks, preflight, image checks, and the
# managed SForge lifecycle; this wrapper only normalizes shared CLI knobs.
# Defaults to the paper's setup: Work on one GPU pair and the Judge on another
# (--work-gpus, --judge-gpus), Work never paused.  --shared-gpus (or --gpus)
# runs both on one pair, pausing Work while the Judge measures.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
REPO_ROOT="$(launch_repo_root)"
TASK_ROOT="${REPO_ROOT}/tasks/latest/streamdiffusionv2-dual-h100-pipeline-runtime-v2-seed"
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
  --work-gpus A,B    GPU pair of the Work container (sets SDV2_GPU_INDICES;
                     default 0,1)
  --judge-gpus C,D   GPU pair of the Judge: interactive Judges and the final
                     evaluation (sets SDV2_JUDGE_GPU_INDICES; default 2,3)
  --shared-gpus      Run Work and the Judge on the Work pair; the host pauses
                     Work while the Judge measures
  --gpus A,B         GPU pair of a shared run (implies --shared-gpus)
  --hours N          Agent budget in hours (default 24 h)
  --timeout SECONDS  Agent budget in seconds (overrides --hours)
  --judge-port PORT  Host port of the Judge server (sets JUDGE_PORT; default 18612)
  --label TEXT       RUN_LABEL override (default: <model-short>-<effort>)
  --auth FILE        Codex auth.json path
  --claude-token FILE  0600 file with one `claude setup-token` OAuth token (claude-code)
  --no-wait          Accepted for compatibility: the launcher sets no minimum
                     start delay (MIN_DELAY_SECONDS=0 unless exported)
  --cpuset VALUE     Pin Work/Judge to GPU-local CPUs: auto|off|CPULIST[@NODES]
                     (default off; sets SFORGE_CPUSET; docs/CPU_AFFINITY.md)
  --config FILE      Launch config (.env or .json)
  -- ...             Forward remaining arguments to the task launcher

The task launcher manages queueing itself; there is no --tmux here.
Each pair is two physical GPU indices joined by NVLink. Separate mode (the
default) locks, probes and preflights all four GPUs for the whole run; the
Judge pair must not overlap the Work pair. Without flags, the Work pair comes
from SDV2_GPU_INDICES and the Judge pair from SDV2_JUDGE_GPU_INDICES (the value
`shared` selects shared mode), else 2,3; either can be set in the environment
or the launch config, and flags win over both.
A Work pair pinned through SDV2_GPU_INDICES alone still gets the default Judge
pair 2,3; set SDV2_JUDGE_GPU_INDICES (or --judge-gpus / --shared-gpus) so
concurrent runs do not queue on that pair.
The task script refuses to start without SDV2_JUDGE_GPU_INDICES.
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

# Separate mode (default): Work on SDV2_GPU_INDICES, the Judge on
# SDV2_JUDGE_GPU_INDICES, four distinct GPUs. Shared mode: --shared-gpus,
# --gpus, or SDV2_JUDGE_GPU_INDICES=shared without --judge-gpus; one pair,
# Work paused during each Judge round.
[[ -z "${gpus}" || -z "${work_gpus}" ]] \
  || launch_die "--gpus and --work-gpus both name the Work pair; pass one"
[[ -z "${gpus}" ]] || shared_gpus=1
[[ -n "${judge_gpus}" || "${SDV2_JUDGE_GPU_INDICES:-}" != shared ]] || shared_gpus=1
sdv2_pair_ok() {
  [[ "$1" =~ ^([0-9]+),([0-9]+)$ ]] && (( 10#${BASH_REMATCH[1]} != 10#${BASH_REMATCH[2]} ))
}
export SDV2_GPU_INDICES="${work_gpus:-${gpus:-${SDV2_GPU_INDICES:-0,1}}}"
if (( shared_gpus )); then
  [[ -z "${judge_gpus}" ]] \
    || launch_die "--judge-gpus names a separate Judge pair; drop it for a shared run"
  sdv2_pair_ok "${SDV2_GPU_INDICES}" \
    || launch_die "--gpus takes two distinct comma-separated physical GPU indices"
  unset SDV2_JUDGE_GPU_UUIDS
  export SDV2_JUDGE_GPU_INDICES=shared
else
  assert_cpuset_legacy_safe
  judge_pair_defaulted=0
  [[ -n "${judge_gpus}" || -n "${SDV2_JUDGE_GPU_INDICES:-}" ]] || judge_pair_defaulted=1
  export SDV2_JUDGE_GPU_INDICES="${judge_gpus:-${SDV2_JUDGE_GPU_INDICES:-2,3}}"
  sdv2_pair_ok "${SDV2_GPU_INDICES}" && sdv2_pair_ok "${SDV2_JUDGE_GPU_INDICES}" \
    || launch_die "--work-gpus and --judge-gpus take two distinct comma-separated physical GPU indices"
  for work_index in ${SDV2_GPU_INDICES/,/ }; do
    for judge_index in ${SDV2_JUDGE_GPU_INDICES/,/ }; do
      (( 10#${work_index} != 10#${judge_index} )) \
        || launch_die "the Judge pair ${SDV2_JUDGE_GPU_INDICES} overlaps the Work pair ${SDV2_GPU_INDICES}; pass --judge-gpus with two other GPUs (or set SDV2_JUDGE_GPU_INDICES), or --shared-gpus to run both on one pair"
    done
  done
  unset SDV2_JUDGE_GPU_UUIDS
  if (( judge_pair_defaulted )); then
    # Hosts used to pack one shared-pair run per NVLink pair; say which pair
    # this run will hold for its Judge when nobody named one.
    echo "sdv2-pipeline: Work pair ${SDV2_GPU_INDICES}, Judge pair ${SDV2_JUDGE_GPU_INDICES} (default; pass --judge-gpus C,D or --shared-gpus to change)" >&2
  fi
fi

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
  export TASK_TIMEOUT=86400
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
export JUDGE_PORT="${JUDGE_PORT:-18612}"
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
