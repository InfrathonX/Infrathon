#!/usr/bin/env bash
set -euo pipefail

# Unified experiment entrypoint for kvpress-kv-eviction-policy-qwen3-8b-h100-v2-scratch.
# One Work GPU plus exactly seven Judge GPUs on this host (the paper's setup:
# the whole host). The task-side launcher owns both
# GPU lock names of every GPU, preflight, image and model checks, the sidecar
# network, the Judge server and the independent final Judge; this wrapper only
# normalizes shared CLI knobs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
REPO_ROOT="$(launch_repo_root)"
TASK_ROOT="${REPO_ROOT}/tasks/latest/kvpress-kv-eviction-policy-qwen3-8b-h100-v2-scratch"
ENTRY="${TASK_ROOT}/scripts/run_agent.sh"

usage() {
  cat <<'USAGE'
Usage:
  run.sh --model MODEL [--work-gpu G --judge-gpus A,B,...] [options]
  run.sh [--work-gpu G --judge-gpus A,B,...] --eval-archive FILE [--public-smoke] [options]

Options:
  --agent NAME       codex (default) or claude-code
  --model NAME       A full gpt-*/claude-* model ID,
                     or any ID the MODEL_API_BASE_URL endpoint serves
  --effort LEVEL     Reasoning effort (max|ultra|xhigh|high|medium|low); default max
  --work-gpu G       The one Work GPU (physical index or UUID; sets WORK_GPU; default 0)
  --judge-gpus LIST  Exactly 7 Judge GPUs, comma separated (indices or UUIDs; sets
                     JUDGE_GPUS; default 1,2,3,4,5,6,7). v1 uses exactly 7 (the paper's setup)
  --judge-port PORT  Run-unique host port of the Judge server (sets JUDGE_PORT; default 18621)
  --hours N          Agent budget in hours (default 24 h)
  --timeout SECONDS  Agent budget in seconds (overrides --hours)
  --label TEXT       RUN_LABEL override (default: <model>-<effort>-<budget>)
  --run-id ID        Explicit RUN_ID (default: kvpress-kv-eviction-scratch-v2-<label>-<UTC stamp>)
  --auth FILE        Codex auth.json path
  --claude-token FILE  0600 file with one `claude setup-token` OAuth token (claude-code)
  --model-repo DIR   Hugging Face cache repo of Qwen/Qwen3-8B (sets KV_EVICTION_MODEL_REPO;
                     mounted read-only into Work and Judge)
  --eval-archive FILE  Run no agent: evaluate FILE through the final-Judge path (same
                     locks, Judge plan and timeouts); qualification
  --public-smoke     With --eval-archive: first run the public quick suite of the task
                     oracle on the Work GPU
  --no-wait          Accepted for symmetry with the pair launchers; this one never queues
  --dry-run          Resolve and check the plan (GPUs and their locks, Judge plan,
                     timeouts, agent, review, model, images), print it and exit;
                     no lock, container or GPU work
  --cpuset VALUE     Pin Work/Judge to GPU-local CPUs: auto|off|CPULIST[@NODES]
                     (default off; sets SFORGE_CPUSET; docs/CPU_AFFINITY.md)
  --tmux             Run detached in a new tmux session (under sg docker for a
                     docker-group user)
  --config FILE      Launch config (.env or .json); see launch/mlsysforge.env.example
  -- ...             Forward remaining arguments to the task launcher

Every interactive and final Judge runs the complete frozen 5,000-item suite
sharded over the Judge GPUs, with the deadlines scripts/lib.sh judge_timeouts
gives for seven GPUs. The submission cooldown is EVAL_INTERVAL (default
1800 s). The SFORGE_LLM_JUDGE_* integrity review must be configured with
MODE=enforce. Before any lock the run refuses a GPU that is locked, busy,
claimed by a container or reserved in GPU_RESERVATIONS_FILE (with
GPU_RESERVATIONS_HOST_ID).
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
run_id=""
auth=""
model_repo=""
eval_archive=""
public_smoke=0
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
    --run-id)  [[ $# -ge 2 ]] || launch_die "--run-id requires a value"; run_id="$2"; shift 2 ;;
    --auth)    [[ $# -ge 2 ]] || launch_die "--auth requires a value"; auth="$2"; shift 2 ;;
    --model-repo) [[ $# -ge 2 ]] || launch_die "--model-repo requires a value"; model_repo="$2"; shift 2 ;;
    --eval-archive) [[ $# -ge 2 ]] || launch_die "--eval-archive requires a value"; eval_archive="$2"; shift 2 ;;
    --public-smoke) public_smoke=1; shift ;;
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
export WORK_GPU="${WORK_GPU:-0}" JUDGE_GPUS="${JUDGE_GPUS:-1,2,3,4,5,6,7}"
launch_require_judge_gpus 7
if [[ -n "${judge_port}" ]]; then
  [[ "${judge_port}" =~ ^[0-9]+$ ]] || launch_die "--judge-port must be an integer"
  judge_port="$((10#${judge_port}))"
  (( judge_port >= 1024 && judge_port <= 65535 && judge_port != 3128 \
     && judge_port != 3130 && judge_port != 3131 )) \
    || launch_die "--judge-port must be 1024..65535 and not a sidecar port (3128, 3130, 3131)"
  export JUDGE_PORT="${judge_port}"
fi
export JUDGE_PORT="${JUDGE_PORT:-18621}"

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
if [[ -n "${run_id}" ]]; then
  [[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ && ${#run_id} -le 120 ]] \
    || launch_die "--run-id must be label-safe ([A-Za-z0-9][A-Za-z0-9_.-]*, at most 120 characters)"
  export RUN_ID="${run_id}"
fi
if [[ -n "${auth}" ]]; then
  [[ -f "${auth}" ]] || launch_die "auth file does not exist: ${auth}"
  export CODEX_AUTH_JSON_PATH="${auth}"
fi
if [[ -n "${model_repo}" ]]; then
  [[ -d "${model_repo}" ]] || launch_die "model repo does not exist: ${model_repo}"
  KV_EVICTION_MODEL_REPO="$(cd -- "${model_repo}" && pwd -P)"
  export KV_EVICTION_MODEL_REPO
fi
# Eval-only mode comes from the flags alone: a shell that still exports these
# from a qualification must not turn an agent launch into an archive evaluation.
unset KV_EVICTION_EVAL_ARCHIVE KV_EVICTION_PUBLIC_SMOKE
if [[ -n "${eval_archive}" ]]; then
  [[ -f "${eval_archive}" && ! -L "${eval_archive}" ]] \
    || launch_die "--eval-archive must be a regular, non-symlink file: ${eval_archive}"
  KV_EVICTION_EVAL_ARCHIVE="$(cd -- "$(dirname -- "${eval_archive}")" && pwd -P)/$(basename -- "${eval_archive}")"
  export KV_EVICTION_EVAL_ARCHIVE
fi
if (( public_smoke )); then
  [[ -n "${eval_archive}" ]] || launch_die "--public-smoke runs only with --eval-archive"
  export KV_EVICTION_PUBLIC_SMOKE=1
fi

assert_container_reachable_proxy --entry "${ENTRY}"
if [[ -z "${eval_archive}" ]]; then
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
  export_version_store_root --task-root "${TASK_ROOT}"
  if [[ "${agent}" == claude-code ]]; then export_claude_runtime_dir; else export_agent_runtime_dir; fi
fi
(( ! dry_run )) || export SFORGE_LAUNCH_DRY_RUN=1

# run_in_tmux forwards this shell's complete environment, so the detached
# path is equivalent to the direct exec below.  The session is started by the
# tmux server, with the groups the server was started with (often without
# docker), and the task launcher never re-execs itself through sg: a user in
# the docker group therefore runs it under `sg docker -c` (sg keeps the
# environment).
if (( use_tmux && ! dry_run )); then
  stamp="$(date -u +%Y%m%d-%H%M%S)"
  session="kvpress-v2-${RUN_LABEL:-run}-${stamp}"
  session="${session//[^A-Za-z0-9_-]/-}"
  entry_command=("${ENTRY}" "${passthrough[@]}")
  docker_group="$(getent group docker 2>/dev/null || true)"
  if grep -Eq "(^|[:,])$(id -un)(,|\$)" <<<"${docker_group}"; then
    entry_command=(sg docker -c "exec $(printf '%q ' "${ENTRY}" "${passthrough[@]}")")
  fi
  run_in_tmux "${session}" \
    "${SCRIPT_DIR}/logs/run-${session}.log" \
    "${entry_command[@]}"
else
  exec "${ENTRY}" "${passthrough[@]}"
fi
