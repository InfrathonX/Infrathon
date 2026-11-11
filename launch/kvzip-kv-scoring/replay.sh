#!/usr/bin/env bash
set -euo pipefail

# Unified replay entrypoint for kvzip-query-agnostic-kv-scoring-qwen3-8b-h100-v2-scratch.
# Injects the default version store root, then delegates every argument to
# the task's own hardened replay_candidates.sh.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
REPO_ROOT="$(launch_repo_root)"
TASK_ROOT="${REPO_ROOT}/tasks/latest/kvzip-query-agnostic-kv-scoring-qwen3-8b-h100-v2-scratch"
ENTRY="${TASK_ROOT}/scripts/replay_candidates.sh"
[[ -x "${ENTRY}" ]] || launch_die "task replay script is missing: ${ENTRY}"

# A leading --config FILE and --cpuset VALUE (docs/CPU_AFFINITY.md) are this
# launcher's own; every other argument goes to the task's replay script.
config_file=""; cpuset=""
while [[ "${1:-}" == "--config" || "${1:-}" == "--cpuset" ]]; do
  [[ $# -ge 2 ]] || launch_die "$1 requires a value"
  case "$1" in
    --config) config_file="$2" ;;
    --cpuset) cpuset="$2" ;;
  esac
  shift 2
done
launch_load_config "${config_file}"
export_cpuset_policy "${cpuset}"

# v1 judges every replay on exactly 7 GPUs (the paper's setup). JUDGE_GPUS
# picks which; a --dry-run or --help preview may leave it unset.
if [[ -n "${JUDGE_GPUS:-}" ]]; then
  launch_require_judge_gpus 7
else
  case " $* " in
    *" --dry-run "*|*" --help "*|*" -h "*) ;;
    *) launch_die "set JUDGE_GPUS to exactly 7 Judge GPUs (v1 uses exactly 7, the paper's setup)" ;;
  esac
fi

store_args=()
case " $* " in
  *" --version-store-root "*) ;;
  *)
    export_version_store_root --task-root "${TASK_ROOT}"
    [[ -n "${SFORGE_VERSION_STORE_ROOT:-}" ]] \
      || launch_die "no version store found; pass --version-store-root"
    store_args=(--version-store-root "${SFORGE_VERSION_STORE_ROOT}")
    ;;
esac

exec "${ENTRY}" "${store_args[@]}" "$@"
