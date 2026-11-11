#!/usr/bin/env bash
set -euo pipefail

# Unified replay entrypoint for qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed.
# Injects the default version store root, then delegates every argument to
# the task's own hardened replay_candidates.sh.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
REPO_ROOT="$(launch_repo_root)"
TASK_ROOT="${REPO_ROOT}/tasks/latest/qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed"
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
