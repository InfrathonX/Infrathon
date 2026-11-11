#!/usr/bin/env bash
set -euo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_sforge
require_command jq
assert_built_images_current

archive="${1:-${ARTIFACTS_DIR}/oracle-submission.tar.gz}"
[[ -f "${archive}" && ! -L "${archive}" && -s "${archive}" ]] \
  || die "missing or unsafe archive: ${archive}"
archive="$(cd "$(dirname "${archive}")" && pwd)/$(basename "${archive}")"

run_id="${2:-${RUN_ID:-kvbench-qwen36-eval-$(date -u +%Y%m%d-%H%M%S)}}"
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
  || die "unsafe run id: ${run_id}"

eval_timeout="${EVAL_TIMEOUT:-3000}"
judge_cpu_limit="${JUDGE_CPU_LIMIT:-4}"
judge_mem_limit="${JUDGE_MEM_LIMIT:-8g}"
[[ "${eval_timeout}" =~ ^[0-9]+$ && "${eval_timeout}" -ge 2400 ]] \
  || die "EVAL_TIMEOUT must be an integer of at least 2400 seconds"
[[ "${judge_cpu_limit}" =~ ^[0-9]+$ && "${judge_cpu_limit}" -gt 0 ]] \
  || die "JUDGE_CPU_LIMIT must be a positive integer"
[[ "${judge_mem_limit}" =~ ^[0-9]+[kKmMgGtT]?[bB]?$ ]] \
  || die "unsafe JUDGE_MEM_LIMIT: ${judge_mem_limit}"

# The custom entry is mandatory: it keeps Work mount-free and gives only the
# trusted Judge the narrow host mounts needed to orchestrate the frozen stack.
export SFORGE_TASKS_DIR="${TASKS_DIR}"
export SFORGE_LOG_DIR="${LOGS_DIR}"
export SFORGE_KVBENCH_MIGRATION_ROOT="${MIGRATION_ROOT}"
exec "${PYTHON_BIN}" \
  "${MIGRATION_ROOT}/scripts/sforge_runtime_entry.py" eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${archive}" \
  --run-id "${run_id}" \
  --timeout "${eval_timeout}" \
  --judge-cpu-limit "${judge_cpu_limit}" \
  --judge-mem-limit "${judge_mem_limit}" \
  --json
