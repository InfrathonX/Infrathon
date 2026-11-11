#!/usr/bin/env bash
set -euo pipefail

# Unified replay entrypoint for concurrent-agent-serving-optimizer-qwen36-swe192-v24.
# Injects the default version store root, then delegates every argument to
# the task's own hardened replay_candidates.sh.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
REPO_ROOT="$(launch_repo_root)"
TASK_ROOT="${REPO_ROOT}/tasks/latest/concurrent-agent-serving-optimizer-qwen36-swe192-v24"
ENTRY="${TASK_ROOT}/scripts/replay_candidates.sh"
[[ -x "${ENTRY}" ]] || launch_die "task replay script is missing: ${ENTRY}"

config_file=""
if [[ "${1:-}" == "--config" ]]; then
  [[ $# -ge 2 ]] || launch_die "--config requires a value"
  config_file="$2"; shift 2
fi
launch_load_config "${config_file}"

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
