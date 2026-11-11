#!/usr/bin/env bash
set -euo pipefail

# Unified experiment entrypoint for concurrent-agent-serving-optimizer-qwen36-swe192-v24.
# Delegates to the task's own hardened launch script; nothing here bypasses
# the task-side preflight, locks, or provenance checks.
# NOTE: the task script validates agent, model, effort, and budget
# independently (Codex: any model ID, high|max|xhigh|ultra;
# Claude Code: claude-opus-5-5, low|medium|high|xhigh|max; any minute-aligned
# budget of at least one hour); invalid values fail closed inside the task
# script.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
REPO_ROOT="$(launch_repo_root)"
TASK_ROOT="${REPO_ROOT}/tasks/latest/concurrent-agent-serving-optimizer-qwen36-swe192-v24"
ENTRY="${TASK_ROOT}/scripts/run_fixed_budget.sh"

usage() {
  cat <<'USAGE'
Usage:
  run.sh --model MODEL_ID [options] [-- extra]
  run.sh --agent claude-code --claude-token FILE --model claude-opus-5-5 [options]

Options:
  --agent NAME       codex (default) or claude-code
  --model NAME       A full gpt-* model ID, or
                     claude-opus-5-5 (claude-code)
  --effort LEVEL     Reasoning effort (max|ultra|xhigh|high|medium|low); default max
  --hours N          Agent budget in hours (sets TASK_TIMEOUT=N*3600; default 24 h)
  --timeout SECONDS  Agent budget in seconds (overrides --hours)
  --run-id ID        Explicit run ID (default: task-derived)
  --auth FILE        Codex auth.json path (sets CODEX_AUTH_JSON_PATH)
  --claude-token FILE  0600 file with one `claude setup-token` OAuth token
                     (claude-code; sets CLAUDE_OAUTH_TOKEN_FILE)
  --tmux             Run detached in a new tmux session
  --config FILE      Launch config (.env or .json); see launch/mlsysforge.env.example
  -- ...             Everything after -- is passed to the task script

Environment already exported (MODEL, REASONING_EFFORT, TASK_TIMEOUT,
CODEX_CLI_VERSION, CODEX_AUTH_JSON_PATH, RUN_ID, WORK_GPU, JUDGE_GPU, ...)
is respected; flags override it. The Codex CLI pin comes from
launch/common.sh unless CODEX_CLI_VERSION is already set; claude-code uses
the Claude Code runtime at the launch/common.sh pin unless
SFORGE_AGENT_RUNTIME_DIR is already set. A claude-code run must reach Docker
without the task script's docker-group re-exec (its allowlist does not carry
the agent selection): this wrapper runs it under `sg docker` when needed.
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
config_file=""
use_tmux=0
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
    --config)  [[ $# -ge 2 ]] || launch_die "--config requires a value"; config_file="$2"; shift 2 ;;
    --tmux)    use_tmux=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --)        shift; passthrough=("$@"); break ;;
    *)         usage >&2; launch_die "unknown argument: $1" ;;
  esac
done

launch_load_config "${config_file}"

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
  export TASK_TIMEOUT="86400"
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
assert_container_reachable_proxy --entry "${ENTRY}"
export_version_store_root --task-root "${TASK_ROOT}"
if [[ "${agent}" == claude-code ]]; then export_claude_runtime_dir; else export_agent_runtime_dir; fi

entry_command=("${ENTRY}" "${passthrough[@]}")
# The task script's docker-group re-exec (scripts/lib.sh ensure_docker_group)
# forwards a fixed allowlist without the Claude Code selection, so a
# claude-code run that cannot reach Docker yet starts under `sg docker`, which
# keeps the whole environment. The Codex path is unchanged.
if [[ "${agent}" == claude-code ]] && ! docker info >/dev/null 2>&1; then
  docker_group="$(getent group docker 2>/dev/null || true)"
  grep -Eq "(^|[:,])$(id -un)(,|\$)" <<<"${docker_group}" \
    || launch_die "claude-code needs Docker access (the docker group) for the task launcher"
  entry_command=(sg docker -c "exec $(printf '%q ' "${ENTRY}" "${passthrough[@]}")")
fi

# run_in_tmux forwards this shell's complete environment, so the detached
# path is equivalent to the direct exec below.
if (( use_tmux )); then
  stamp="$(date -u +%Y%m%d-%H%M%S)"
  session="concurrent-serving-${stamp}"
  run_in_tmux "${session}" \
    "${SCRIPT_DIR}/logs/run-${stamp}.log" \
    "${entry_command[@]}"
else
  exec "${entry_command[@]}"
fi
