#!/usr/bin/env bash
set -euo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_sforge
require_command curl
require_command flock
require_command jq
require_command nvidia-smi
require_command python3
require_command ss
require_command tar

AUTH_JSON="${CODEX_AUTH_JSON_PATH:-}"
SFORGE_PYTHON="$(dirname "${SFORGE_BIN}")/python"
ISOLATED_ENTRY="${MIGRATION_ROOT}/scripts/sforge_isolated_entry.py"
RESTRICTED_PROXY_SOURCE="${MIGRATION_ROOT}/scripts/restricted_connect_proxy.py"
MODEL="${MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-0.148.0}"
TASK_TIMEOUT="${TASK_TIMEOUT:-180}"
EVAL_INTERVAL="${EVAL_INTERVAL:-60}"
SUBMISSION_COOLDOWN="${SUBMISSION_COOLDOWN:-${EVAL_INTERVAL}}"
WORK_GPU="${WORK_GPU:-1}"
JUDGE_GPU="${JUDGE_GPU:-2}"
JUDGE_PORT="${JUDGE_PORT:-18285}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ID="${RUN_ID:-flash-kmeans-v7-scratch-h100-codex-smoke-${RUN_STAMP}}"
RUN_LOG_ROOT="${LOGS_DIR}/runs/${RUN_ID}"
TASK_RUN_ROOT="${RUN_LOG_ROOT}/${TASK_ID}"

[[ "${MODEL}" =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]] \
  || die "Codex smoke requires MODEL (the agent's model ID)"
[[ "${REASONING_EFFORT}" == "max" ]] \
  || die "Codex smoke requires REASONING_EFFORT=max"
[[ "${CODEX_CLI_VERSION}" == "0.148.0" ]] \
  || die "Codex smoke requires CODEX_CLI_VERSION=0.148.0"

# The Work image is agent-free; a host-provided runtime supplies the CLI.
AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}"
[[ -n "${AGENT_RUNTIME_DIR}" ]] \
  || die "set SFORGE_AGENT_RUNTIME_DIR (build one with launch/build-agent-runtime.sh)"
runtime_marker="${AGENT_RUNTIME_DIR}/.sforge-agent-runtime.json"
[[ -d "${AGENT_RUNTIME_DIR}" && ! -L "${AGENT_RUNTIME_DIR}" && -f "${runtime_marker}" ]] \
  || die "agent runtime directory is missing or unmarked: ${AGENT_RUNTIME_DIR}"
runtime_kind="$(jq -er '.kind' "${runtime_marker}")"
runtime_agent="$(jq -er '.agent' "${runtime_marker}")"
runtime_version="$(jq -er '.version' "${runtime_marker}")"
[[ "${runtime_kind}" == "sforge-agent-runtime" ]] \
  || die "invalid agent runtime marker: ${runtime_marker}"
[[ "${runtime_agent}" == "codex" ]] \
  || die "Codex smoke requires a codex runtime, found: ${runtime_agent}"
[[ "${runtime_version}" == "${CODEX_CLI_VERSION}" ]] \
  || die "agent runtime version ${runtime_version} differs from the pinned ${CODEX_CLI_VERSION}"
export SFORGE_AGENT_RUNTIME_DIR="${AGENT_RUNTIME_DIR}"

[[ "${TASK_TIMEOUT}" =~ ^[0-9]+$ ]] \
  || die "TASK_TIMEOUT must be a positive integer"
[[ "${EVAL_INTERVAL}" =~ ^[0-9]+$ ]] \
  || die "EVAL_INTERVAL must be a positive integer"
[[ "${SUBMISSION_COOLDOWN}" =~ ^[0-9]+$ ]] \
  || die "SUBMISSION_COOLDOWN must be a positive integer"
TASK_TIMEOUT="$((10#${TASK_TIMEOUT}))"
EVAL_INTERVAL="$((10#${EVAL_INTERVAL}))"
SUBMISSION_COOLDOWN="$((10#${SUBMISSION_COOLDOWN}))"
(( TASK_TIMEOUT >= 90 )) || die "Codex smoke timeout must be at least 90 seconds"
(( EVAL_INTERVAL >= 30 && EVAL_INTERVAL < TASK_TIMEOUT )) \
  || die "Codex smoke compatibility interval must be at least 30s and below the timeout"
(( SUBMISSION_COOLDOWN > 0 )) \
  || die "Codex smoke submission cooldown must be positive"

[[ "${JUDGE_PORT}" =~ ^[0-9]+$ ]] || die "invalid JUDGE_PORT: ${JUDGE_PORT}"
JUDGE_PORT="$((10#${JUDGE_PORT}))"
(( JUDGE_PORT >= 1024 && JUDGE_PORT <= 65535 )) \
  || die "invalid unprivileged JUDGE_PORT: ${JUDGE_PORT}"
[[ "${JUDGE_PORT}" != "3128" ]] \
  || die "judge port conflicts with the restricted CONNECT proxy"
[[ -n "${AUTH_JSON}" ]] \
  || die "set CODEX_AUTH_JSON_PATH to the host-only Codex auth file"
assert_private_auth_file "${AUTH_JSON}"
[[ -x "${SFORGE_PYTHON}" ]] \
  || die "missing SForge virtualenv Python: ${SFORGE_PYTHON}"
[[ -f "${ISOLATED_ENTRY}" && ! -L "${ISOLATED_ENTRY}" ]] \
  || die "missing isolated SForge entry: ${ISOLATED_ENTRY}"
[[ -f "${RESTRICTED_PROXY_SOURCE}" && ! -L "${RESTRICTED_PROXY_SOURCE}" ]] \
  || die "missing restricted proxy source: ${RESTRICTED_PROXY_SOURCE}"
assert_built_images_current
[[ "$(docker info --format '{{.DefaultRuntime}}')" == "nvidia" ]] \
  || die "this local GPU pinning adapter requires Docker default runtime=nvidia"

WORK_GPU="$(canonical_gpu_index "${WORK_GPU}")"
JUDGE_GPU="$(canonical_gpu_index "${JUDGE_GPU}")"
work_gpu_uuid="$(gpu_uuid "${WORK_GPU}")"
judge_gpu_uuid="$(gpu_uuid "${JUDGE_GPU}")"
[[ "${work_gpu_uuid}" != "${judge_gpu_uuid}" ]] \
  || die "work and judge resolve to the same physical GPU: ${work_gpu_uuid}"

exec 8>"$(gpu_lock_path "${work_gpu_uuid}")"
exec 9>"$(gpu_lock_path "${judge_gpu_uuid}")"
flock -n 8 || die "work GPU ${WORK_GPU} lock is already held"
flock -n 9 || die "judge GPU ${JUDGE_GPU} lock is already held"
assert_gpu_idle "${WORK_GPU}" "work"
assert_gpu_idle "${JUDGE_GPU}" "judge"
assert_container_gpu "${WORK_IMAGE}" "${work_gpu_uuid}" "work"
assert_container_gpu "${JUDGE_IMAGE}" "${judge_gpu_uuid}" "judge"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
  || die "unsafe RUN_ID: ${RUN_ID}"
[[ ! -e "${RUN_LOG_ROOT}" ]] \
  || die "run directory already exists: ${RUN_LOG_ROOT}"
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${JUDGE_PORT}$"; then
  die "judge port ${JUDGE_PORT} is already in use"
fi

mkdir -p "${RUN_LOG_ROOT}"
judge_pid=""
active_pid=""
isolated_network=""
restricted_proxy_name=""
cleanup() {
  if [[ -n "${judge_pid}" ]] && kill -0 "${judge_pid}" 2>/dev/null; then
    kill "${judge_pid}" 2>/dev/null || true
    wait "${judge_pid}" 2>/dev/null || true
  fi
  if [[ -n "${restricted_proxy_name}" ]]; then
    docker rm -f "${restricted_proxy_name}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${isolated_network}" ]]; then
    docker network rm "${isolated_network}" >/dev/null 2>&1 || true
  fi
}
handle_signal() {
  local signal_name="$1"
  local exit_code=143
  [[ "${signal_name}" == "INT" ]] && exit_code=130
  trap - INT TERM
  if [[ -n "${active_pid}" ]] && kill -0 "${active_pid}" 2>/dev/null; then
    kill "-${signal_name}" "${active_pid}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${active_pid}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${active_pid}" 2>/dev/null; then
      kill -KILL "${active_pid}" 2>/dev/null || true
    fi
    wait "${active_pid}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}
run_logged_command() {
  local log_file="$1"
  shift
  "$@" > >(tee "${log_file}") 2>&1 &
  active_pid="$!"
  local command_rc=0
  wait "${active_pid}" || command_rc="$?"
  active_pid=""
  return "${command_rc}"
}
trap cleanup EXIT
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

export SFORGE_TASKS_DIR="${TASKS_DIR}"
export SFORGE_LOG_DIR="${LOGS_DIR}"
export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${judge_gpu_uuid},CUDA_VISIBLE_DEVICES=0,FLASH_KMEANS_TARGET=H100,FLASH_KMEANS_VERIFY_SUITE=smoke,FLASH_KMEANS_VERIFY_WARMUP=1,FLASH_KMEANS_VERIFY_ITERS=2,FLASH_KMEANS_VERIFY_REPEATS=1"
SFORGE_ADMIN_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
export SFORGE_ADMIN_SECRET

"${SFORGE_BIN}" serve --port "${JUDGE_PORT}" \
  >"${RUN_LOG_ROOT}/judge-server.log" 2>&1 &
judge_pid="$!"
wait_for_http "http://127.0.0.1:${JUDGE_PORT}/openapi.json" 90

isolated_network="${RUN_ID}-internal"
restricted_proxy_name="${RUN_ID}-restricted-proxy"
docker network create \
  --driver bridge \
  --internal \
  --label "mlsysforge.run_id=${RUN_ID}" \
  "${isolated_network}" >/dev/null
docker run --detach --rm \
  --name "${restricted_proxy_name}" \
  --network bridge \
  --add-host host.docker.internal:host-gateway \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --volume "${RESTRICTED_PROXY_SOURCE}:/opt/mlsysforge/restricted_proxy.py:ro" \
  --entrypoint python3 \
  "${WORK_IMAGE}" \
  /opt/mlsysforge/restricted_proxy.py \
  --judge-port "${JUDGE_PORT}" \
  --judge-forward-port "${JUDGE_PORT}" >/dev/null
docker network connect \
  --alias runtime-proxy \
  "${isolated_network}" \
  "${restricted_proxy_name}"
restricted_proxy_ip="$(
  docker inspect "${restricted_proxy_name}" \
    | jq -er --arg network "${isolated_network}" \
      '.[0].NetworkSettings.Networks[$network].IPAddress
       | select(type == "string" and length > 0)'
)"
for _ in $(seq 1 30); do
  if docker run --rm \
      --network "${isolated_network}" \
      --env NVIDIA_VISIBLE_DEVICES=void \
      --entrypoint curl \
      "${WORK_IMAGE}" \
      --connect-timeout 2 \
      --max-time 3 \
      -fsS "http://${restricted_proxy_ip}:3128/health" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
docker run --rm \
  --network "${isolated_network}" \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --entrypoint curl \
  "${WORK_IMAGE}" \
  --connect-timeout 2 \
  --max-time 5 \
  -fsS "http://${restricted_proxy_ip}:${JUDGE_PORT}/openapi.json" >/dev/null
if docker run --rm \
    --network "${isolated_network}" \
    --env NVIDIA_VISIBLE_DEVICES=void \
    --entrypoint curl \
    "${WORK_IMAGE}" \
    --connect-timeout 2 \
    --max-time 4 \
    -fsS https://example.com >/dev/null 2>&1; then
  die "internal work network unexpectedly has direct public egress"
fi
echo "Restricted smoke network: PASS (${isolated_network})"

unset \
  SFORGE_AGENT_API_KEY \
  SFORGE_AGENT_API_BASE_URL \
  SFORGE_AGENT_MODEL \
  SFORGE_AGENT_TIMEOUT \
  CODEX_API_KEY \
  OPENAI_API_KEY \
  OPENAI_BASE_URL
export SFORGE_CODEX_AUTH_JSON_PATH="${AUTH_JSON}"
export SFORGE_CODEX_REASONING_EFFORT="${REASONING_EFFORT}"
export SFORGE_CODEX_CLI_VERSION="${CODEX_CLI_VERSION}"
export SFORGE_AGENT_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${work_gpu_uuid},CUDA_VISIBLE_DEVICES=0,FLASH_KMEANS_TARGET=H100"
export SFORGE_ISOLATED_DOCKER_NETWORK="${isolated_network}"
export SFORGE_HTTP_PROXY="http://${restricted_proxy_ip}:3128"
export SFORGE_HTTPS_PROXY="http://${restricted_proxy_ip}:3128"
export SFORGE_NO_PROXY="${restricted_proxy_ip},127.0.0.1,localhost"

# Host-local Judge traffic must bypass inherited workstation proxies. Agent
# egress remains restricted by the explicit SFORGE_* sidecar configuration.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${restricted_proxy_ip},127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"

echo "Starting short Codex/SForge smoke ${RUN_ID}"
echo "  model: ${MODEL} (${REASONING_EFFORT})"
echo "  budget: ${TASK_TIMEOUT}s; compatibility interval: ${EVAL_INTERVAL}s (auto-eval disabled)"

set +e
run_logged_command "${RUN_LOG_ROOT}/sforge-run.log" \
  "${SFORGE_PYTHON}" "${ISOLATED_ENTRY}" run \
  --backend docker \
  --task "${TASK_ID}" \
  --agent codex \
  --model "${MODEL}" \
  --timeout "${TASK_TIMEOUT}" \
  --eval-interval "${EVAL_INTERVAL}" \
  --disable-auto-eval \
  --submission-cooldown "${SUBMISSION_COOLDOWN}" \
  --judge-url "http://${restricted_proxy_ip}:${JUDGE_PORT}" \
  --enable-internet \
  --work-cpu-limit 8 \
  --work-mem-limit 32g \
  --judge-cpu-limit 8 \
  --judge-mem-limit 32g \
  --run-id "${RUN_ID}"
run_rc="$?"
set -e
(( run_rc == 0 )) || die "short SForge run failed with exit ${run_rc}"

final_result="${TASK_RUN_ROOT}/final_result.json"
final_snapshot="${TASK_RUN_ROOT}/final_snapshot.json"
run_history="${TASK_RUN_ROOT}/run_history.json"
final_archive="${TASK_RUN_ROOT}/final_archive.tar.gz"
for required in \
  "${final_result}" \
  "${final_snapshot}" \
  "${run_history}" \
  "${final_archive}"; do
  [[ -s "${required}" ]] || die "missing or empty smoke artifact: ${required}"
done

python3 - \
  "${final_result}" \
  "${final_snapshot}" \
  "${run_history}" \
  "${TASK_TIMEOUT}" \
  "${EVAL_INTERVAL}" \
  "${MODEL}" <<'PY'
import json
import math
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
snapshot = json.load(open(sys.argv[2], encoding="utf-8"))
history = json.load(open(sys.argv[3], encoding="utf-8"))
budget = int(sys.argv[4])
interval = int(sys.argv[5])
expected_model = sys.argv[6]

runtime = result.get("runtime_seconds")
elapsed = snapshot.get("snapshot_elapsed_seconds")
if result.get("model") != expected_model:
    raise SystemExit(f"unexpected model: {result.get('model')!r}")
if result.get("timed_out") is not True:
    raise SystemExit(f"agent did not reach the timeout boundary: {result}")
for name, value in (("runtime_seconds", runtime), ("snapshot_elapsed_seconds", elapsed)):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"invalid {name}: {value!r}")
    if not math.isfinite(float(value)):
        raise SystemExit(f"non-finite {name}: {value!r}")
if not budget - 5.0 <= float(runtime) <= budget + 60.0:
    raise SystemExit(f"runtime does not prove the {budget}s smoke budget: {runtime}")
if abs(float(elapsed) - budget) > 60.0:
    raise SystemExit(f"final snapshot is too far from {budget}s: {elapsed}")
if int(result.get("archive_size_bytes") or 0) <= 0:
    raise SystemExit("final archive is empty")

expected_rounds = []
auto_entries = [
    entry
    for entry in history.get("entries", [])
    if str(entry.get("round") or "").startswith("auto-")
]
rounds = [str(entry.get("round")) for entry in auto_entries]
if rounds != expected_rounds:
    raise SystemExit(f"unexpected auto-eval rounds: {rounds}; expected {expected_rounds}")
unfinished = [
    entry
    for entry in auto_entries
    if entry.get("status") in {"queued", "running"}
]
if unfinished:
    raise SystemExit(f"auto evaluations were not drained: {unfinished}")
if any(entry.get("status") != "completed" for entry in auto_entries):
    raise SystemExit(f"an auto evaluation did not complete: {auto_entries}")
print(
    f"Short timing proof: runtime={float(runtime):.3f}s, "
    f"snapshot={float(elapsed):.3f}s, rounds={rounds}"
)
PY

python3 - "${final_archive}" <<'PY'
import sys
import tarfile

archive = sys.argv[1]
with tarfile.open(archive, "r:gz") as tf:
    names = []
    kernel_files = []
    for member in tf.getmembers():
        name = member.name
        while name.startswith("./"):
            name = name[2:]
        if not name:
            continue
        parts = name.split("/")
        if name.startswith("/") or ".." in parts:
            raise SystemExit(f"unsafe final archive member: {member.name}")
        if not (
            name in {"FINAL_VERSION", "notes.md", "kernels"}
            or name.startswith("kernels/")
        ):
            raise SystemExit(f"unexpected final archive member: {member.name}")
        names.append(name)
        if name.startswith("kernels/") and member.isfile():
            kernel_files.append(name)
if "FINAL_VERSION" not in names or "notes.md" not in names:
    raise SystemExit(f"final archive lacks required files: {names}")
if not kernel_files:
    raise SystemExit(f"final archive contains no candidate kernel file: {names}")
print(f"Final archive whitelist: PASS ({len(names)} members)")
PY

final_eval_id="${RUN_ID}-final"
[[ ! -e "${LOGS_DIR}/runs/${final_eval_id}" ]] \
  || die "final smoke eval directory already exists: ${final_eval_id}"
set +e
run_logged_command "${TASK_RUN_ROOT}/final-smoke-eval.log" \
  "${SFORGE_BIN}" eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${final_archive}" \
  --run-id "${final_eval_id}" \
  --timeout 900 \
  --judge-cpu-limit 8 \
  --judge-mem-limit 32g \
  --json
final_eval_rc="$?"
set -e
(( final_eval_rc == 0 )) \
  || die "independent final smoke evaluation failed with exit ${final_eval_rc}"

final_report="${LOGS_DIR}/runs/${final_eval_id}/${TASK_ID}/submissions/manual-1/report.json"
[[ -s "${final_report}" ]] \
  || die "independent final smoke evaluation did not produce a report"
python3 - "${final_report}" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
if report.get("valid") is not True:
    raise SystemExit(f"independent final smoke is invalid: {report}")
if report.get("timed_out") is not False:
    raise SystemExit(f"independent final smoke timed out: {report}")
if report.get("pass_rate") != 1.0:
    raise SystemExit(f"independent final smoke pass_rate is not 1: {report}")
metrics = report.get("metrics")
if not isinstance(metrics, dict):
    raise SystemExit(f"independent final smoke lacks adapter metrics: {report}")
if metrics.get("source") != "flash_kmeans_h100_verifier_adapter":
    raise SystemExit(f"unexpected final smoke adapter source: {metrics!r}")
for field in ("candidate_output_suppressed", "verifier_completed"):
    if metrics.get(field) is not True:
        raise SystemExit(f"final smoke adapter {field} is not true: {metrics!r}")
for field, expected in (
    ("adapter_schema_version", 8),
    ("feedback_schema_version", 7),
    ("result_protocol_records", 1),
    ("verifier_returncode", 0),
):
    value = metrics.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise SystemExit(
            f"final smoke adapter {field} must be integer {expected}: {metrics!r}"
        )
score = report.get("score")
if (
    isinstance(score, bool)
    or not isinstance(score, (int, float))
    or not math.isfinite(float(score))
    or float(score) <= 0.0
):
    raise SystemExit(f"independent final smoke score is not finite/positive: {score!r}")
print(f"Independent final smoke: PASS, score={float(score):.6f}")
PY

jq -n \
  --arg finished_at "$(date -u +%FT%TZ)" \
  --arg run_id "${RUN_ID}" \
  --arg final_report "${final_report}" \
  '{
    finished_at: $finished_at,
    run_id: $run_id,
    final_report: $final_report,
    status: "passed"
  }' > "${RUN_LOG_ROOT}/smoke-completion.json"

echo "Short Codex/SForge smoke: PASS"
echo "  run artifacts: ${TASK_RUN_ROOT}"
echo "  final report:  ${final_report}"
