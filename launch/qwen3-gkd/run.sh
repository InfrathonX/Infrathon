#!/usr/bin/env bash
set -euo pipefail

# Unified experiment entrypoint for
# qwen3-4b-to-1p7b-gkd-distillation-optimizer-dual-h100-v1-official-trl-seed.
# This task's parameterized engine is launch_fixed_budget_after_gpu_pair_idle.sh:
# it reads AGENT_MODEL / REASONING_EFFORT / TASK_TIMEOUT /
# RUN_LABEL / GKD_GPU_INDICES / GKD_JUDGE_GPU_INDICES and waits for the named
# GPUs to go idle before starting. No GPU pair is baked into the task.  By
# default Work runs on --work-gpus (GKD_GPU_INDICES, default 0,1) and the Judge
# on --judge-gpus (GKD_JUDGE_GPU_INDICES, default 2,3), as in the paper's runs;
# --shared-gpus (or --gpus A,B) runs both on one pair and pauses Work while the
# Judge measures.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
REPO_ROOT="$(launch_repo_root)"
TASK_ROOT="${REPO_ROOT}/tasks/latest/qwen3-4b-to-1p7b-gkd-distillation-optimizer-dual-h100-v1-official-trl-seed"
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
  --work-gpus A,B    GPU pair of the Work container (sets GKD_GPU_INDICES;
                     default 0,1)
  --judge-gpus C,D   GPU pair of the Judge (sets GKD_JUDGE_GPU_INDICES; default
                     2,3). The four GPUs must be distinct; Work is never paused
  --shared-gpus      Run Work and the Judge on the Work pair; the host pauses
                     Work while the Judge measures
  --gpus A,B         GPU pair of a shared run (implies --shared-gpus; sets
                     GKD_GPU_INDICES). Containers always see their pair as
                     CUDA devices 0,1.
  --hours N          Agent budget in hours (sets TASK_TIMEOUT=N*3600; default 24 h)
  --timeout SECONDS  Agent budget in seconds (overrides --hours)
  --judge-port PORT  Host port of the Judge server (sets JUDGE_PORT; default 18614)
  --label TEXT       RUN_LABEL override (default: <label-safe model>-<effort>)
  --auth FILE        Codex auth.json path (sets CODEX_AUTH_JSON_PATH)
  --claude-token FILE  0600 file with one `claude setup-token` OAuth token (claude-code)
  --no-wait          Accepted for compatibility: the launcher sets no minimum
                     start delay (MIN_DELAY_SECONDS=0 unless exported)
  --dry-run          Resolve and check the plan (model, runtime, images, integrity
                     review, GPU pair, port), print it and exit; no wait, lock,
                     container or GPU work
  --cpuset VALUE     Pin Work/Judge to GPU-local CPUs: auto|off|CPULIST[@NODES]
                     (default off; sets SFORGE_CPUSET; docs/CPU_AFFINITY.md)
  --config FILE      Launch config (.env or .json); see launch/mlsysforge.env.example
  -- ...             Everything after -- is passed to the task script

The task script manages its own tmux/queueing; there is no --tmux here.
After the Agent deadline the task script runs the standalone final Judge of
the final archive on the Judge pair (the Work pair of a shared run) while every
pair lock stays held. GKD_GPU_INDICES names the Work pair (the only pair of a
shared run); GKD_JUDGE_GPU_INDICES names the Judge pair, or is `shared` for a
shared run. Either can be set in the environment or the launch config; flags
win over both.
A Work pair pinned through GKD_GPU_INDICES alone still gets the default Judge
pair 2,3; set GKD_JUDGE_GPU_INDICES (or --judge-gpus / --shared-gpus) so
concurrent runs do not queue on that pair.
The task script refuses to start without GKD_JUDGE_GPU_INDICES.
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
dry_run=0
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
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --)        shift; passthrough=("$@"); break ;;
    *)         usage >&2; launch_die "unknown argument: $1" ;;
  esac
done

launch_load_config "${config_file}"
export_cpuset_policy "${cpuset}"

[[ -x "${ENTRY}" ]] || launch_die "task entry script is missing: ${ENTRY}"

# A GPU pair is two distinct physical indices.
assert_gpu_pair() {
  [[ "$2" =~ ^([0-9]+),([0-9]+)$ && "$((10#${BASH_REMATCH[1]}))" != "$((10#${BASH_REMATCH[2]}))" ]] \
    || launch_die "$1 takes two different comma-separated physical GPU indices, got '$2'"
}
# Separate GPUs (the default): Work on the Work pair, the Judge on the Judge
# pair, Work never paused.  --shared-gpus or --gpus: one pair for both.
[[ -z "${gpus}" ]] || shared_gpus=1
# GKD_JUDGE_GPU_INDICES=shared (environment or launch config) without
# --judge-gpus is the same explicit opt-in.
[[ -n "${judge_gpus}" || "${GKD_JUDGE_GPU_INDICES:-}" != shared ]] || shared_gpus=1
[[ -z "${gpus}" || -z "${work_gpus}" ]] \
  || launch_die "--gpus and --work-gpus both name the Work pair; pass one"
if [[ -n "${gpus}" ]]; then
  assert_gpu_pair --gpus "${gpus}"
  work_gpus="${gpus}"
elif [[ -n "${work_gpus}" ]]; then
  assert_gpu_pair --work-gpus "${work_gpus}"
fi
if [[ -n "${work_gpus}" ]]; then
  export GKD_GPU_INDICES="${work_gpus}"
fi
export GKD_GPU_INDICES="${GKD_GPU_INDICES:-0,1}"
assert_gpu_pair GKD_GPU_INDICES "${GKD_GPU_INDICES}"
if (( shared_gpus )); then
  [[ -z "${judge_gpus}" ]] \
    || launch_die "--judge-gpus names a separate Judge pair; drop it for a shared run"
  unset GKD_JUDGE_GPU_UUIDS
  export GKD_JUDGE_GPU_INDICES=shared
else
  assert_cpuset_legacy_safe
  if [[ -n "${judge_gpus}" ]]; then
    assert_gpu_pair --judge-gpus "${judge_gpus}"
    export GKD_JUDGE_GPU_INDICES="${judge_gpus}"
  fi
  judge_pair_defaulted=0
  [[ -n "${GKD_JUDGE_GPU_INDICES:-}" ]] || judge_pair_defaulted=1
  export GKD_JUDGE_GPU_INDICES="${GKD_JUDGE_GPU_INDICES:-2,3}"
  assert_gpu_pair GKD_JUDGE_GPU_INDICES "${GKD_JUDGE_GPU_INDICES}"
  for judge_index in ${GKD_JUDGE_GPU_INDICES//,/ }; do
    for work_index in ${GKD_GPU_INDICES//,/ }; do
      (( 10#${judge_index} != 10#${work_index} )) \
        || launch_die "the Judge pair ${GKD_JUDGE_GPU_INDICES} overlaps the Work pair ${GKD_GPU_INDICES}; pass --judge-gpus with two other GPUs (or set GKD_JUDGE_GPU_INDICES), or --shared-gpus to run Work and the Judge on one pair"
    done
  done
  if (( judge_pair_defaulted )); then
    # Hosts used to pack one shared-pair run per NVLink pair; say which pair
    # this run will hold for its Judge when nobody named one.
    echo "qwen3-gkd: Work pair ${GKD_GPU_INDICES}, Judge pair ${GKD_JUDGE_GPU_INDICES} (default; pass --judge-gpus C,D or --shared-gpus to change)" >&2
  fi
fi

model_short=""
if [[ -n "${model}" ]]; then
  AGENT_MODEL="$(resolve_model_id "${model}")"
  export AGENT_MODEL
  # Label-safe, as the engine's default label: gpt-N.M-NAME -> name,
  # ZHIPU/GLM-5.3-Flash -> zhipu-glm-5.3-flash.
  model_short="$(sed -E 's/^gpt-[0-9.]+-//' <<<"${AGENT_MODEL}" | tr 'A-Z' 'a-z' | sed -E 's/[^a-z0-9.]+/-/g; s/^-+//; s/-+$//')"
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
export JUDGE_PORT="${JUDGE_PORT:-18614}"
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
(( ! dry_run )) || export SFORGE_LAUNCH_DRY_RUN=1

exec "${ENTRY}" "${passthrough[@]}"
