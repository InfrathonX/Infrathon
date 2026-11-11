#!/usr/bin/env bash
set -euo pipefail

# Local Docker entrypoint for minference-sparse-prefill-h100-v3-seed: one Work
# GPU and one Judge GPU of this host.
# The task launcher (scripts/run_local_v3.sh) owns the GPU locks, image and
# asset checks, the restricted Work network, the Judge server and the final
# Judge; this wrapper only normalizes the shared CLI knobs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
REPO_ROOT="$(launch_repo_root)"
TASK_ROOT="${REPO_ROOT}/tasks/latest/minference-sparse-prefill-h100-v3-seed"
ENTRY="${TASK_ROOT}/scripts/run_local_v3.sh"

usage() {
  cat <<'USAGE'
Usage:
  run.sh --model MODEL [--work-gpu G --judge-gpu J] [options]
  run.sh --judge-gpu J --eval-archive FILE [--skip-integrity-review] [options]

Options:
  --model NAME         A full gpt-* model ID,
                       or any ID the MODEL_API_BASE_URL endpoint serves
  --effort LEVEL       Reasoning effort (max|ultra|xhigh|high); default max
  --work-gpu G         The Work GPU (physical index; sets WORK_GPU; default 0)
  --judge-gpu J        The Judge GPU (physical index; sets JUDGE_GPU; default 1), used
                       by every submission and the final
  --hours N            Agent budget in hours (sets TASK_TIMEOUT=N*3600; default 12 h)
  --timeout SECONDS    Agent budget in seconds (overrides --hours)
  --judge-port PORT    Run-unique host port of the Judge server (default 18497)
  --run-id ID          Explicit RUN_ID
  --auth FILE          Codex auth.json path (sets CODEX_AUTH_JSON_PATH)
  --eval-archive FILE  Run no agent: FILE goes through the final Judge path (same
                       GPU checks, phase final); a qualification or a replay
  --skip-integrity-review
                       With --eval-archive: skip the LLM integrity review (oracle
                       and qualification archives)
  --dry-run            Resolve and check the plan (images, assets, GPUs, port),
                       print it and exit; no lock, container or GPU work
  --tmux               Run detached in a new tmux session
  --config FILE        Launch config (.env or .json); see launch/mlsysforge.env.example

Before the first run: build or pull the images (tasks/latest/<task>/scripts/
build_images.sh, or scripts/infrathon/pull_images.sh <task>) and fetch the assets
(tasks/latest/<task>/assets/fetch.sh: the model from Hugging Face, the prompt
sets from the Infrathon dataset) into INFRATHON_ASSETS_ROOT.  Agent runs need
the integrity review configured (SFORGE_LLM_JUDGE_MODE=enforce and the
SFORGE_LLM_JUDGE_* settings) and a Codex agent runtime
(launch/build-agent-runtime.sh).  Submissions have an EVAL_INTERVAL cooldown
(default 300 s); a Judge takes about 28 minutes (README).
USAGE
}

model=""; effort=""; work_gpu=""; judge_gpu=""; hours=""; timeout_seconds=""
judge_port=""; run_id=""; auth=""; eval_archive=""; skip_review=0
config_file=""; dry_run=0; use_tmux=0

while (( $# )); do
  case "$1" in
    --model)   [[ $# -ge 2 ]] || launch_die "--model requires a value"; model="$2"; shift 2 ;;
    --effort)  [[ $# -ge 2 ]] || launch_die "--effort requires a value"; effort="$2"; shift 2 ;;
    --work-gpu) [[ $# -ge 2 ]] || launch_die "--work-gpu requires a value"; work_gpu="$2"; shift 2 ;;
    --judge-gpu) [[ $# -ge 2 ]] || launch_die "--judge-gpu requires a value"; judge_gpu="$2"; shift 2 ;;
    --hours)   [[ $# -ge 2 ]] || launch_die "--hours requires a value"; hours="$2"; shift 2 ;;
    --timeout) [[ $# -ge 2 ]] || launch_die "--timeout requires a value"; timeout_seconds="$2"; shift 2 ;;
    --judge-port) [[ $# -ge 2 ]] || launch_die "--judge-port requires a value"; judge_port="$2"; shift 2 ;;
    --run-id)  [[ $# -ge 2 ]] || launch_die "--run-id requires a value"; run_id="$2"; shift 2 ;;
    --auth)    [[ $# -ge 2 ]] || launch_die "--auth requires a value"; auth="$2"; shift 2 ;;
    --eval-archive) [[ $# -ge 2 ]] || launch_die "--eval-archive requires a value"; eval_archive="$2"; shift 2 ;;
    --skip-integrity-review) skip_review=1; shift ;;
    --config)  [[ $# -ge 2 ]] || launch_die "--config requires a value"; config_file="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --tmux)    use_tmux=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)         usage >&2; launch_die "unknown argument: $1" ;;
  esac
done

launch_load_config "${config_file}"
[[ -x "${ENTRY}" ]] || launch_die "task entry script is missing: ${ENTRY}"

# Eval-only mode comes from the flags alone: a shell that still exports these
# from an earlier qualification must not turn an agent launch into an evaluation.
unset EVAL_ARCHIVE SKIP_INTEGRITY_REVIEW
if [[ -n "${eval_archive}" ]]; then
  [[ -f "${eval_archive}" && ! -L "${eval_archive}" ]] \
    || launch_die "--eval-archive must be a regular, non-symlink file: ${eval_archive}"
  export EVAL_ARCHIVE="${eval_archive}"
  (( ! skip_review )) || export SKIP_INTEGRITY_REVIEW=1
else
  (( ! skip_review )) || launch_die "--skip-integrity-review runs only with --eval-archive"
  work_gpu="${work_gpu:-0}"
fi
judge_gpu="${judge_gpu:-1}"
for gpu in "${work_gpu}" "${judge_gpu}"; do
  [[ -z "${gpu}" || "${gpu}" =~ ^[0-9]+$ ]] || launch_die "GPUs are physical indices, got: ${gpu}"
done
[[ -z "${work_gpu}" ]] || export WORK_GPU="${work_gpu}"
export JUDGE_GPU="${judge_gpu}"
if [[ -n "${judge_port}" ]]; then
  [[ "${judge_port}" =~ ^[0-9]+$ ]] || launch_die "--judge-port must be an integer"
  export JUDGE_PORT="$((10#${judge_port}))"
fi
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
  export TASK_TIMEOUT=43200
fi
[[ -z "${run_id}" ]] || export RUN_ID="${run_id}"
if [[ -n "${auth}" ]]; then
  [[ -f "${auth}" ]] || launch_die "auth file does not exist: ${auth}"
  export CODEX_AUTH_JSON_PATH="${auth}"
fi
if [[ -z "${eval_archive}" ]]; then
  export CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-${CODEX_CLI_VERSION_PIN}}"
  assert_container_reachable_proxy --entry "${ENTRY}"
  export_version_store_root --task-root "${TASK_ROOT}"
  export_agent_runtime_dir
fi
(( ! dry_run )) || export SFORGE_LAUNCH_DRY_RUN=1

# run_in_tmux forwards this shell's complete environment, so the detached path
# is equivalent to the direct exec below.
if (( use_tmux && ! dry_run )); then
  stamp="$(date -u +%Y%m%d-%H%M%S)"
  run_in_tmux "minference-v3-${stamp}" "${SCRIPT_DIR}/logs/run-${stamp}.log" "${ENTRY}"
else
  exec "${ENTRY}"
fi
