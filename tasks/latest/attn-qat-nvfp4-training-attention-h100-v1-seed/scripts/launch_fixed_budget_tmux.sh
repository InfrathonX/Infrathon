#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_command tmux

auth_json="${CODEX_AUTH_JSON_PATH:-}"
[[ -n "${auth_json}" ]] || die "set CODEX_AUTH_JSON_PATH"
assert_private_auth_file "${auth_json}"
reasoning_effort="${REASONING_EFFORT:-max}"
case "${reasoning_effort}" in
  high|max|xhigh|ultra) ;;
  *) die "unsupported reasoning effort: ${reasoning_effort}" ;;
esac
model="${MODEL:-}"
[[ -n "${model}" ]] || die "set MODEL to the agent model id"
[[ "${model}" =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]] || die "unsafe MODEL: ${model}"
model_slug="$(tr 'A-Z' 'a-z' <<<"${model}" | sed -E 's/[^a-z0-9.]+/-/g; s/^-+//; s/-+$//')"

run_stamp="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
session="${TMUX_SESSION:-attn-qat-v1-${model_slug}-${reasoning_effort}-2h-${run_stamp}}"
run_id="${RUN_ID:-attn-qat-v1-h100-${model_slug}-${reasoning_effort}-120m-${run_stamp}}"
for value in "${run_stamp}" "${session}" "${run_id}"; do
  [[ "${value}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
    || die "unsafe launcher identity: ${value}"
done

mkdir -p "${LOGS_DIR}"
launcher_log="${LOGS_DIR}/launch-${run_id}.log"
launcher_exit="${LOGS_DIR}/launch-${run_id}.exit"
tmux has-session -t "${session}" 2>/dev/null \
  && die "tmux session already exists: ${session}"
[[ ! -e "${launcher_exit}" ]] \
  || die "launcher exit artifact already exists: ${launcher_exit}"

run_env=(
  env
  "CODEX_AUTH_JSON_PATH=${auth_json}"
  "MODEL=${model}"
  "REASONING_EFFORT=${reasoning_effort}"
  "CODEX_CLI_VERSION=0.148.0"
  "TASK_TIMEOUT=7200"
  "EVAL_INTERVAL=1800"
  "FINAL_EVAL_TIMEOUT=${FINAL_EVAL_TIMEOUT:-3600}"
  "WORK_GPU=${WORK_GPU:-0}"
  "JUDGE_GPU=${JUDGE_GPU:-1}"
  "JUDGE_PORT=${JUDGE_PORT:-18096}"
  "RUN_STAMP=${run_stamp}"
  "RUN_ID=${run_id}"
  "REPOSITORY_ROOT=${REPOSITORY_ROOT}"
  "SFORGE_ROOT=${SFORGE_ROOT}"
  "SFORGE_BIN=${SFORGE_BIN}"
  "PYTHON_BIN=${PYTHON_BIN}"
  "TASKS_DIR=${TASKS_DIR}"
  "LOGS_DIR=${LOGS_DIR}"
  "ARTIFACTS_DIR=${ARTIFACTS_DIR}"
)
for proxy_name in SFORGE_HTTP_PROXY SFORGE_HTTPS_PROXY SFORGE_NO_PROXY \
  SFORGE_AGENT_RUNTIME_DIR SFORGE_VERSION_STORE_ROOT; do
  if [[ -n "${!proxy_name:-}" ]]; then
    run_env+=("${proxy_name}=${!proxy_name}")
  fi
done
run_env+=(bash "${MIGRATION_ROOT}/scripts/run_fixed_budget.sh")

printf -v quoted_runner '%q ' "${run_env[@]}"
printf -v quoted_root '%q' "${MIGRATION_ROOT}"
printf -v quoted_log '%q' "${launcher_log}"
printf -v quoted_exit '%q' "${launcher_exit}"
inner_command="cd ${quoted_root} && ${quoted_runner}2>&1 | tee ${quoted_log}; \
rc=\$?; \
exit_tmp=${quoted_exit}.tmp.\$\$; \
printf '%s\n' \"\${rc}\" > \"\${exit_tmp}\"; \
status_rc=\$?; \
if (( status_rc == 0 )); then mv -f -- \"\${exit_tmp}\" ${quoted_exit}; status_rc=\$?; fi; \
if (( status_rc != 0 )); then rm -f -- \"\${exit_tmp}\"; fi; \
if (( rc == 0 && status_rc != 0 )); then rc=\${status_rc}; fi; \
exit \"\${rc}\""
printf -v command_line 'bash -o pipefail -c %q' "${inner_command}"

tmux new-session -d -s "${session}" -n run "${command_line}"
echo "tmux session: ${session}"
echo "run id:       ${run_id}"
echo "launcher log: ${launcher_log}"
echo "final exit:   ${launcher_exit}"
echo "attach:       tmux attach -t ${session}"
