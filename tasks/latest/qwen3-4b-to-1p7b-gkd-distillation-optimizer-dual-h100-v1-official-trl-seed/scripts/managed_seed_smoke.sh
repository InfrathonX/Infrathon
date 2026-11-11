#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

require_command curl
require_command docker
require_command jq
require_command nvidia-smi
require_command tar
# Refresh supplementary groups before any docker call; see scripts/lib.sh.
ensure_docker_group "$@"

SFORGE_PYTHON="${SFORGE_PYTHON:-${REPOSITORY_ROOT}/.venv/bin/python}"
PORT="${GKD_MANAGED_SMOKE_PORT:-18474}"
RUN_ID="${GKD_MANAGED_SMOKE_RUN_ID:-qwen3-gkd-v1-managed-seed-$(date -u +%Y%m%d-%H%M%S)}"
SUBMISSION_KIND="${GKD_MANAGED_SMOKE_KIND:-agent}"
STUDENT_MODEL_REPO="${GKD_STUDENT_MODEL_HOST_PATH:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--Qwen--Qwen3-1.7B}"
TEACHER_MODEL_REPO="${GKD_TEACHER_MODEL_HOST_PATH:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--Qwen--Qwen3-4B}"
DATA_ROOT="${GKD_DATA_HOST_ROOT:-${HOME}/mlsysforge-artifacts/qwen3-gkd/tldr_gkd_v1}"
GPU_INDICES="${GKD_GPU_INDICES:-}"
GPU_UUIDS="${GKD_GPU_UUIDS:-}"
if [[ -z "${GPU_UUIDS}" ]]; then
  resolve_gpu_pair "${GPU_INDICES}"
  GPU_INDICES="${GPU_PAIR_INDICES}"
  GPU_UUIDS="${GPU_PAIR_UUIDS}"
fi
[[ -n "${GPU_INDICES}" ]] \
  || die "set GKD_GPU_INDICES to the two physical GPU indices for this smoke run"
WORK_NAME="sforge.run.${TASK_ID}.${RUN_ID}"
LOG_ROOT="${LOGS_DIR}/managed-seed-smoke/${RUN_ID}"
ARCHIVE="${LOG_ROOT}/official-v001.tar.gz"
RESULT="${LOG_ROOT}/result.json"
SERVER_LOG="${LOG_ROOT}/judge-server.log"
ADMIN_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"
server_pid=""

cleanup() {
  set +e
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" >/dev/null 2>&1
    wait "${server_pid}" >/dev/null 2>&1
  fi
  docker rm -f "${WORK_NAME}" >/dev/null 2>&1
}
trap cleanup EXIT INT TERM

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe RUN_ID"
[[ "${SUBMISSION_KIND}" == "agent" ]] \
  || die "GKD_MANAGED_SMOKE_KIND must be agent; autojudge is disabled"
[[ -x "${SFORGE_PYTHON}" ]] || die "missing SForge Python: ${SFORGE_PYTHON}"
docker image inspect "${WORK_IMAGE}" "${JUDGE_IMAGE}" >/dev/null
python3 "${TASK_ROOT}/scripts/preflight.py" \
  --gpu-indices "${GPU_INDICES}" \
  --expected-gpu-uuids "${GPU_UUIDS}" \
  --max-used-mib 256 \
  --model-path "${STUDENT_MODEL_REPO}/snapshots/${STUDENT_MODEL_REVISION}" \
  --teacher-model-path "${TEACHER_MODEL_REPO}/snapshots/${TEACHER_MODEL_REVISION}" \
  --data-root "${DATA_ROOT}" \
  --private-data-root "${DATA_ROOT}"

mkdir -p "${LOG_ROOT}"
tar -C "${TASK_ROOT}/images/work/workdir/gkd_lab" \
  --exclude='__pycache__' --exclude='*.pyc' \
  -czf "${ARCHIVE}" solutions FINAL_VERSION notes.md

export SFORGE_TASKS_DIR="${TASK_ROOT}/tasks"
export SFORGE_LOG_DIR="${LOGS_DIR}"
export SFORGE_ADMIN_SECRET="${ADMIN_SECRET}"
export GKD_GPU_UUIDS="${GPU_UUIDS}"
export GKD_STUDENT_MODEL_HOST_PATH="${STUDENT_MODEL_REPO}"
export GKD_TEACHER_MODEL_HOST_PATH="${TEACHER_MODEL_REPO}"
export GKD_DATA_HOST_ROOT="${DATA_ROOT}"
export GKD_PRIVATE_DATA_HOST_ROOT="${DATA_ROOT}"
export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${GPU_UUIDS},CUDA_VISIBLE_DEVICES=0,1,PYTHONDONTWRITEBYTECODE=1"

"${SFORGE_PYTHON}" "${TASK_ROOT}/scripts/sforge_dual_h100_entry.py" serve \
  --port "${PORT}" >"${SERVER_LOG}" 2>&1 &
server_pid=$!
for _ in $(seq 1 90); do
  curl --noproxy '*' --connect-timeout 2 --max-time 3 -fsS \
    "http://127.0.0.1:${PORT}/openapi.json" >/dev/null 2>&1 && break
  sleep 1
done
curl --noproxy '*' --connect-timeout 2 --max-time 5 -fsS \
  "http://127.0.0.1:${PORT}/openapi.json" >/dev/null

# The container is intentionally otherwise idle.  It represents the exact
# managed lifecycle boundary: live Work owns the same pair until Judge pauses
# it after receiving the immutable v001 archive.
docker run --detach --rm --name "${WORK_NAME}" \
  --gpus "\"device=${GPU_UUIDS}\"" \
  --env "NVIDIA_VISIBLE_DEVICES=${GPU_UUIDS}" \
  --env CUDA_VISIBLE_DEVICES=0,1 \
  "${WORK_IMAGE}" tail -f /dev/null >/dev/null

token="$(curl --noproxy '*' -fsS -X POST \
  "http://127.0.0.1:${PORT}/api/v1/register" \
  -H 'Content-Type: application/json' \
  --data "$(jq -cn \
    --arg task "${TASK_ID}" --arg run "${RUN_ID}" --arg secret "${ADMIN_SECRET}" \
    '{task_id:$task,run_id:$run,admin_secret:$secret,judge_cpu_limit:16,judge_mem_limit:"96g",submission_cooldown:300}')" \
  | jq -er '.token')"

submission_id="$(curl --noproxy '*' -fsS -X POST \
  "http://127.0.0.1:${PORT}/api/v1/submit" \
  -H 'Content-Type: application/gzip' \
  -H "X-SForge-Token: ${token}" \
  -H "X-SForge-Submission-Kind: ${SUBMISSION_KIND}" \
  -H "X-SForge-Admin-Secret: ${ADMIN_SECRET}" \
  --data-binary "@${ARCHIVE}" \
  | tee "${LOG_ROOT}/submit.json" | jq -er '.submission_id')"

observed_pause=false
# A formal pass is five rows x four counterbalanced paired repeats x two roles
# plus the hidden CPU cases, so the wait has to cover a full matrix, not a smoke.
deadline=$((SECONDS + ${GKD_MANAGED_SMOKE_DEADLINE_SECONDS:-7200}))
while (( SECONDS < deadline )); do
  status="$(docker inspect -f '{{.State.Status}}' "${WORK_NAME}")"
  [[ "${status}" == "paused" ]] && observed_pause=true
  curl --noproxy '*' -fsS \
    "http://127.0.0.1:${PORT}/api/v1/result/${submission_id}" >"${RESULT}.tmp"
  result_status="$(jq -r '.status' "${RESULT}.tmp")"
  if [[ "${result_status}" == "completed" || "${result_status}" == "error" ]]; then
    mv "${RESULT}.tmp" "${RESULT}"
    break
  fi
  sleep 2
done
[[ -f "${RESULT}" ]] \
  || die "managed seed Judge did not finish within ${GKD_MANAGED_SMOKE_DEADLINE_SECONDS:-7200} seconds"
[[ "${observed_pause}" == true ]] || die "managed Judge never paused Work"
[[ "$(docker inspect -f '{{.State.Status}}' "${WORK_NAME}")" == "running" ]] \
  || die "Work was not resumed after Judge"
jq -e '
  .status == "completed"
  and .report.valid == true
  and (.report.score >= 90 and .report.score <= 110)
  and .report.metrics.auto_feedback_safe == true
' "${RESULT}" >/dev/null || {
  jq . "${RESULT}" >&2
  die "managed official-v001 self-check failed"
}

jq -n \
  --arg run_id "${RUN_ID}" \
  --arg submission_kind "${SUBMISSION_KIND}" \
  --arg submission_id "${submission_id}" \
  --argjson score "$(jq '.report.score' "${RESULT}")" \
  --arg work_image "$(docker image inspect -f '{{.Id}}' "${WORK_IMAGE}")" \
  --arg judge_image "$(docker image inspect -f '{{.Id}}' "${JUDGE_IMAGE}")" \
  '{status:"PASS",run_id:$run_id,submission_kind:$submission_kind,submission_id:$submission_id,score:$score,work_paused:true,work_resumed:true,work_image:$work_image,judge_image:$judge_image}' \
  | tee "${LOG_ROOT}/summary.json"
