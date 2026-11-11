#!/usr/bin/env bash
set -euo pipefail

# Unified experiment entrypoint for quant-videogen-cross-model-kv-codec-h100-v4-scratch.
# One Work GPU plus exactly 6 Judge GPUs on this host (the paper's setup). The task-side launcher owns
# GPU locks, preflight, image and model checks, the sidecar network, the
# Judge server and the independent final Judge; this wrapper only normalizes
# shared CLI knobs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
REPO_ROOT="$(launch_repo_root)"
TASK_ROOT="${REPO_ROOT}/tasks/latest/quant-videogen-cross-model-kv-codec-h100-v4-scratch"
ENTRY="${TASK_ROOT}/scripts/run_agent.sh"

usage() {
  cat <<'USAGE'
Usage:
  run.sh --model MODEL [--work-gpu G --judge-gpus A,B,...] [options]

Options:
  --agent NAME       codex (default) or claude-code
  --model NAME       A full gpt-*/claude-* model ID,
                     or any ID the MODEL_API_BASE_URL endpoint serves
  --effort LEVEL     Reasoning effort (max|ultra|xhigh|high|medium|low); default max
  --work-gpu G       The one Work GPU (physical index or UUID; sets WORK_GPU; default 0)
  --judge-gpus LIST  Exactly 6 Judge GPUs, comma separated (indices or UUIDs; sets
                     JUDGE_GPUS; default 1,2,3,4,5,6). v1 uses exactly 6 (the paper's setup)
  --judge-port PORT  Run-unique host port of the Judge server (sets JUDGE_PORT; default 18623)
  --hours N          Agent budget in hours (default 24 h)
  --timeout SECONDS  Agent budget in seconds (overrides --hours)
  --label TEXT       RUN_LABEL override (default: <model>-<effort>-<budget>)
  --auth FILE        Codex auth.json path
  --claude-token FILE  0600 file with one `claude setup-token` OAuth token (claude-code)
  --no-wait          Accepted for symmetry with the pair launchers; this one never queues
  --dry-run          Resolve and check the plan (GPUs, Judge plan, timeouts, agent,
                     review), print it and exit; no lock, container or GPU work
  --cpuset VALUE     Pin Work/Judge to GPU-local CPUs: auto|off|CPULIST[@NODES]
                     (default off; sets SFORGE_CPUSET; docs/CPU_AFFINITY.md)
  --tmux             Run detached in a new tmux session
  --config FILE      Launch config (.env or .json); see launch/mlsysforge.env.example
  -- ...             Forward remaining arguments to the task launcher

Every interactive and final Judge runs the full hidden suite on the Judge GPUs,
with the timeout scripts/lib.sh judge_plan gives for six GPUs. The
SFORGE_LLM_JUDGE_* integrity review must be configured with MODE=enforce.
USAGE
}

agent="codex"
claude_token=""
model=""
effort=""
work_gpu=""
judge_gpus=""
judge_port=""
timeout_seconds=""
hours=""
label=""
auth=""
cpuset=""
config_file=""
dry_run=0
use_tmux=0
passthrough=()

while (( $# )); do
  case "$1" in
    --agent)   [[ $# -ge 2 ]] || launch_die "--agent requires a value"; agent="$2"; shift 2 ;;
    --claude-token) [[ $# -ge 2 ]] || launch_die "--claude-token requires a value"; claude_token="$2"; shift 2 ;;
    --model)   [[ $# -ge 2 ]] || launch_die "--model requires a value"; model="$2"; shift 2 ;;
    --effort)  [[ $# -ge 2 ]] || launch_die "--effort requires a value"; effort="$2"; shift 2 ;;
    --work-gpu) [[ $# -ge 2 ]] || launch_die "--work-gpu requires a value"; work_gpu="$2"; shift 2 ;;
    --judge-gpus) [[ $# -ge 2 ]] || launch_die "--judge-gpus requires a value"; judge_gpus="$2"; shift 2 ;;
    --judge-port) [[ $# -ge 2 ]] || launch_die "--judge-port requires a value"; judge_port="$2"; shift 2 ;;
    --hours)   [[ $# -ge 2 ]] || launch_die "--hours requires a value"; hours="$2"; shift 2 ;;
    --timeout) [[ $# -ge 2 ]] || launch_die "--timeout requires a value"; timeout_seconds="$2"; shift 2 ;;
    --label)   [[ $# -ge 2 ]] || launch_die "--label requires a value"; label="$2"; shift 2 ;;
    --auth)    [[ $# -ge 2 ]] || launch_die "--auth requires a value"; auth="$2"; shift 2 ;;
    --cpuset)  [[ $# -ge 2 ]] || launch_die "--cpuset requires a value"; cpuset="$2"; shift 2 ;;
    --config)  [[ $# -ge 2 ]] || launch_die "--config requires a value"; config_file="$2"; shift 2 ;;
    --no-wait) shift ;;
    --dry-run) dry_run=1; shift ;;
    --tmux)    use_tmux=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --)        shift; passthrough=("$@"); break ;;
    *)         usage >&2; launch_die "unknown argument: $1" ;;
  esac
done

launch_load_config "${config_file}"
export_cpuset_policy "${cpuset}"
[[ -x "${ENTRY}" ]] || launch_die "task entry script is missing: ${ENTRY}"

# GPUs are only named here; the task launcher resolves, locks and checks them.
if [[ -n "${work_gpu}" ]]; then
  [[ "${work_gpu}" =~ ^([0-9]+|GPU-[A-Za-z0-9-]+)$ ]] \
    || launch_die "--work-gpu takes one physical GPU index or UUID"
  export WORK_GPU="${work_gpu}"
fi
if [[ -n "${judge_gpus}" ]]; then
  export JUDGE_GPUS="${judge_gpus}"
fi
export WORK_GPU="${WORK_GPU:-0}" JUDGE_GPUS="${JUDGE_GPUS:-1,2,3,4,5,6}"
launch_require_judge_gpus 6
if [[ -n "${judge_port}" ]]; then
  [[ "${judge_port}" =~ ^[0-9]+$ ]] || launch_die "--judge-port must be an integer"
  judge_port="$((10#${judge_port}))"
  (( judge_port >= 1024 && judge_port <= 65535 && judge_port != 3128 )) \
    || launch_die "--judge-port must be 1024..65535 and not 3128"
  export JUDGE_PORT="${judge_port}"
fi
export JUDGE_PORT="${JUDGE_PORT:-18623}"

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
  export TASK_TIMEOUT=86400
fi
[[ -z "${label}" ]] || export RUN_LABEL="${label}"
if [[ -n "${auth}" ]]; then
  [[ -f "${auth}" ]] || launch_die "auth file does not exist: ${auth}"
  export CODEX_AUTH_JSON_PATH="${auth}"
fi
(( ! dry_run )) || export QVG_DRY_RUN=1
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

# run_in_tmux forwards this shell's complete environment, so the detached
# path is equivalent to the direct exec below.
if (( use_tmux && ! dry_run )); then
  stamp="$(date -u +%Y%m%d-%H%M%S)"
  session="quant-videogen-${RUN_LABEL:-run}-${stamp}"
  session="${session//[^A-Za-z0-9_-]/-}"
  run_in_tmux "${session}" \
    "${SCRIPT_DIR}/logs/run-${session}.log" \
    "${ENTRY}" "${passthrough[@]}"
else
  exec "${ENTRY}" "${passthrough[@]}"
fi
