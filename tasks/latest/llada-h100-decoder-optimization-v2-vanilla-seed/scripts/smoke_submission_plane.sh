#!/usr/bin/env bash
set -Eeuo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
for command in curl docker flock jq nvidia-smi python3 realpath sha256sum ss; do
  require_command "${command}"
done

archive="${1:-}"
[[ -f "${archive}" && ! -L "${archive}" ]] \
  || die "usage: WORK_GPU=N JUDGE_GPU=M $0 KNOWN_VALID_ARCHIVE"
[[ -v WORK_GPU && -n "${WORK_GPU}" ]] \
  || die "submission-plane smoke requires an explicit numeric WORK_GPU"
[[ -v JUDGE_GPU && -n "${JUDGE_GPU}" ]] \
  || die "submission-plane smoke requires an explicit numeric JUDGE_GPU"
WORK_GPU="$(canonical_gpu_index "${WORK_GPU}")"
JUDGE_GPU="$(canonical_gpu_index "${JUDGE_GPU}")"
[[ "${WORK_GPU}" != "${JUDGE_GPU}" ]] \
  || die "Work and Judge GPU indices must differ"
work_uuid="$(gpu_uuid "${WORK_GPU}")"
judge_uuid="$(gpu_uuid "${JUDGE_GPU}")"
[[ "${work_uuid}" != "${judge_uuid}" ]] \
  || die "Work and Judge GPU UUIDs must differ"

memory_policy="${LLADA_MEMORY_POLICY_PATH:-${MIGRATION_ROOT}/memory_policy.json}"
workload_policy="${LLADA_WORKLOAD_POLICY_PATH:-${MIGRATION_ROOT}/workload_policy.json}"
memory_pending=0
memory_frozen=0
workload_pending=0
workload_frozen=0
workload_provisional=0
jq -e '
  .qualification_status == "UNQUALIFIED"
  and .release_status == "PENDING"
' "${memory_policy}" >/dev/null 2>&1 && memory_pending=1
jq -e '
  .qualification_status == "QUALIFIED"
  and .release_status == "FROZEN"
' "${memory_policy}" >/dev/null 2>&1 && memory_frozen=1
jq -e '
  .qualification_status == "UNQUALIFIED"
  and .release_status == "PENDING"
' "${workload_policy}" >/dev/null 2>&1 && workload_pending=1
jq -e '
  .qualification_status == "QUALIFIED"
  and .release_status == "FROZEN"
' "${workload_policy}" >/dev/null 2>&1 && workload_frozen=1
jq -e '
  .qualification_status == "QUALIFIED"
  and .release_status == "PROVISIONAL"
' "${workload_policy}" >/dev/null 2>&1 && workload_provisional=1

if (( memory_pending == 1 && workload_pending == 1 )); then
  smoke_policy_mode="bootstrap-unqualified-policies"
  smoke_min_free_mib="${GPU_SMOKE_MIN_FREE_MIB:-1}"
  [[ "${smoke_min_free_mib}" =~ ^[0-9]+$ ]] \
    || die "GPU_SMOKE_MIN_FREE_MIB must be a positive integer"
  (( 10#${smoke_min_free_mib} > 0 )) \
    || die "GPU_SMOKE_MIN_FREE_MIB must be positive"
elif (( memory_frozen == 1 && workload_frozen == 1 )); then
  load_frozen_memory_policy "${memory_policy}"
  load_frozen_workload_policy "${workload_policy}"
  smoke_min_free_mib="${GPU_ADMISSION_MIN_FREE_MIB}"
  smoke_policy_mode="frozen-release-policies"
  # Judge adapter kill: judge/run_judge.py kills the verifier at
  # LLADA_{INTERACTIVE,FINAL}_JUDGE_TIMEOUT.  Owner decision 2026-10-13: 6600 s,
  # host-side only (the frozen workload policy, its pins and the images are
  # unchanged).  It must stay beyond the verifier's own persistent outer
  # deadline (frozen final.outer_timeout_seconds, 3900 s, which starts after
  # Judge admission, ~170 s in) so a slow candidate gets the verifier's
  # candidate-level "outer deadline" verdict and private artifact instead of an
  # adapter kill that the harness records as infrastructure_error.
  LLADA_JUDGE_KILL_TIMEOUT="${LLADA_JUDGE_KILL_TIMEOUT:-6600}"
  [[ "${LLADA_JUDGE_KILL_TIMEOUT}" =~ ^[0-9]+$ ]] \
    || die "LLADA_JUDGE_KILL_TIMEOUT must be a positive integer"
  LLADA_JUDGE_KILL_TIMEOUT="$((10#${LLADA_JUDGE_KILL_TIMEOUT}))"
  (( LLADA_JUDGE_KILL_TIMEOUT >= POLICY_FINAL_OUTER_TIMEOUT + 600 )) \
    || die "LLADA_JUDGE_KILL_TIMEOUT must exceed the verifier outer deadline by at least 600 s"
elif [[ "${LLADA_PILOT:-0}" == "1" ]] \
    && (( memory_frozen == 1 && workload_provisional == 1 )); then
  load_frozen_memory_policy "${memory_policy}"
  load_provisional_workload_policy "${workload_policy}"
  smoke_min_free_mib="${GPU_ADMISSION_MIN_FREE_MIB}"
  smoke_policy_mode="pilot-provisional-workload"
else
  die "submission-plane smoke requires pending, frozen, or explicit pilot policies"
fi
quality_controls_path="${MIGRATION_ROOT}/quality_controls.json"
[[ -f "${quality_controls_path}" && ! -L "${quality_controls_path}" ]] \
  || die "submission-plane smoke requires a safe quality-control manifest"
quality_controls_sha256="$(
  sha256sum "${quality_controls_path}" | awk '{print $1}'
)"

acquire_dual_gpu_locks "${work_uuid}" "${judge_uuid}"
assert_distinct_gpu_pair \
  "${WORK_GPU}" "${JUDGE_GPU}" "${smoke_min_free_mib}"

LLADA_MODEL_REPO="${LLADA_MODEL_REPO:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--GSAI-ML--LLaDA-8B-Instruct}"
export LLADA_MODEL_REPO
model_validation_json="$(python3 "${MIGRATION_ROOT}/scripts/validate_model_artifact.py" \
  --repo "${LLADA_MODEL_REPO}" --full)"
assert_built_images_current
work_image_id="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")"
judge_image_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
[[ "${work_image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || die "smoke Work image ID is invalid"
[[ "${judge_image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || die "smoke Judge image ID is invalid"
assert_container_gpu "${WORK_IMAGE}" "${work_uuid}" "Work"
assert_container_gpu "${JUDGE_IMAGE}" "${judge_uuid}" "Judge"

smoke_id="decoder-v2-h100-submit-smoke-$(date -u +%H%M%S)"
smoke_net="${smoke_id}-net"
smoke_proxy="${smoke_id}-proxy"
work_sentinel="${smoke_id}-work-sentinel"
smoke_port="${SMOKE_JUDGE_PORT:-18491}"
[[ "${smoke_port}" =~ ^[0-9]+$ ]] || die "SMOKE_JUDGE_PORT must be numeric"
smoke_port="$((10#${smoke_port}))"
(( smoke_port >= 1024 && smoke_port <= 65535 && smoke_port != 3128 && smoke_port != 3130 )) \
  || die "invalid or conflicting SMOKE_JUDGE_PORT: ${smoke_port}"
if ss -H -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${smoke_port}$"; then
  die "smoke Judge port ${smoke_port} is already in use"
fi
admin_secret="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
cohort_secret="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"
smoke_log_root="${LOGS_DIR}/smoke/${smoke_id}"
smoke_run_root="${smoke_log_root}/runs/${smoke_id}"
mkdir -p -- "${smoke_log_root}/runs"
mkdir -- "${smoke_run_root}"
prepare_root_owned_checkpoint_dir \
  "${smoke_run_root}" "${smoke_id}" "${judge_image_id}"
# A submission-plane smoke now executes the full row-atomic workload and
# therefore supplies the same trusted identity fields as terminal evaluation.
model_validation_snapshot="${smoke_run_root}/full-model-validation.json"
printf '%s\n' "${model_validation_json}" >"${model_validation_snapshot}"
model_validation_sha256="$(sha256sum "${model_validation_snapshot}" | awk '{print $1}')"
runtime_entry_sha256="$(sha256sum "${MIGRATION_ROOT}/scripts/sforge_h100_entry.py" | awk '{print $1}')"
smoke_source_digest="$(image_source_digest)"
mapfile -d '' -t SFORGE_RUNTIME_FILES < <(
  cd "${SFORGE_ROOT}"
  find sforge -type f -name '*.py' -print0 | LC_ALL=C sort -z
)
sforge_runtime_sha256="$(
  (cd "${SFORGE_ROOT}"; sha256sum "${SFORGE_RUNTIME_FILES[@]}") | sha256sum | awk '{print $1}'
)"
sforge_runtime_file_set_sha256="$(printf '%s\n' "${SFORGE_RUNTIME_FILES[@]}" | sha256sum | awk '{print $1}')"
sforge_runtime_files_json="$(printf '%s\n' "${SFORGE_RUNTIME_FILES[@]}" | jq -Rsc 'split("\n")[:-1]')"
mapfile -t smoke_core_hashes < <(
  docker run --rm --network none --entrypoint sha256sum "${judge_image_id}" \
    /opt/decoder_judge/verifier/verify_decoder.py \
    /opt/decoder_judge/verifier/candidate_worker.py \
    /opt/decoder_judge/verifier/cohort.py \
    /opt/decoder_judge/verifier/quality.py \
    /opt/decoder_judge/verifier/source_admission.py | awk '{print $1}'
)
(( ${#smoke_core_hashes[@]} == 5 )) || die "cannot attest the smoke Judge invariant core"
runtime_provenance_snapshot="${smoke_run_root}/runtime-provenance.json"
write_runtime_provenance_snapshot \
  "${runtime_provenance_snapshot}" "${sforge_runtime_sha256}" \
  "${sforge_runtime_file_set_sha256}" "${#SFORGE_RUNTIME_FILES[@]}" \
  "${sforge_runtime_files_json}" "${smoke_core_hashes[0]}" \
  "${smoke_core_hashes[1]}" "${smoke_core_hashes[2]}" \
  "${smoke_core_hashes[3]}" "${smoke_core_hashes[4]}"
runtime_provenance_sha256="$(sha256sum "${runtime_provenance_snapshot}" | awk '{print $1}')"
bridge_gateway="$(docker network inspect bridge | jq -er '.[0].IPAM.Config[0].Gateway')"
server_pid=""
cleanup_smoke() {
  if [[ -n "${server_pid}" ]]; then
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    server_pid=""
  fi
  docker rm -f "${smoke_proxy}" >/dev/null 2>&1 || true
  docker rm -f "${work_sentinel}" >/dev/null 2>&1 || true
  remove_task_gpu_containers "judge" "${judge_uuid}" "${smoke_id}" \
    >/dev/null 2>&1 || true
  docker network rm "${smoke_net}" >/dev/null 2>&1 || true
}
handle_smoke_signal() {
  local rc=143
  [[ "$1" == INT ]] && rc=130
  trap - EXIT INT TERM
  cleanup_smoke
  exit "${rc}"
}
trap cleanup_smoke EXIT
trap 'handle_smoke_signal INT' INT
trap 'handle_smoke_signal TERM' TERM

# Keep a live CUDA context on Work while the ephemeral Judge uses the other
# UUID. This makes accidental same-GPU pinning visible during the smoke.
docker run --detach --rm \
  --name "${work_sentinel}" \
  --network none \
  --gpus "device=${work_uuid}" \
  --env "NVIDIA_VISIBLE_DEVICES=${work_uuid}" \
  --env CUDA_VISIBLE_DEVICES=0 \
  --entrypoint python3 \
  "${WORK_IMAGE}" -c '
import subprocess, time, torch
assert torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability(0) >= (8, 0)
assert torch.cuda.get_device_properties(0).total_memory >= 34 * 1024 ** 3
uuid = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
).strip()
torch.zeros(1, device="cuda")
print(f"WORK_SENTINEL_READY {uuid}", flush=True)
time.sleep(7200)
' >/dev/null
for _ in $(seq 1 60); do
  docker logs "${work_sentinel}" 2>&1 | grep -Fxq "WORK_SENTINEL_READY ${work_uuid}" && break
  sleep 1
done
docker logs "${work_sentinel}" 2>&1 \
  | grep -Fxq "WORK_SENTINEL_READY ${work_uuid}" \
  || die "Work CUDA sentinel did not become ready on ${work_uuid}"

docker network create --driver bridge --internal "${smoke_net}" >/dev/null
docker run -d --rm \
  --name "${smoke_proxy}" \
  --network bridge \
  --add-host host.docker.internal:host-gateway \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --volume "${MIGRATION_ROOT}/scripts/restricted_connect_proxy.py:/opt/mlsysforge/restricted_proxy.py:ro" \
  --entrypoint python3 \
  "${WORK_IMAGE}" \
  /opt/mlsysforge/restricted_proxy.py \
  --evaluation-upstream-host "${bridge_gateway}" \
  --evaluation-upstream-port "${smoke_port}" >/dev/null
docker network connect --alias runtime-proxy "${smoke_net}" "${smoke_proxy}"
proxy_ip="$(
  docker inspect "${smoke_proxy}" \
    | jq -er --arg network "${smoke_net}" \
      '.[0].NetworkSettings.Networks[$network].IPAddress'
)"

export LLADA_WORK_GPU_UUID="${work_uuid}"
export LLADA_JUDGE_GPU_UUID="${judge_uuid}"
judge_extra="NVIDIA_VISIBLE_DEVICES=${judge_uuid},CUDA_VISIBLE_DEVICES=0,LLADA_TARGET=dual-cuda-sm80-34g,LLADA_GPU_ROLE=judge,SFORGE_PRIVATE_CONTEXT_CHECKPOINT_DIR=/logs/verifier/checkpoints,SFORGE_PRIVATE_CONTEXT_COHORT_SECRET=${cohort_secret},SFORGE_PRIVATE_CONTEXT_QUALITY_CONTROLS_SHA256=${quality_controls_sha256}"
judge_extra+=",SFORGE_PRIVATE_CONTEXT_WORK_IMAGE_ID=${work_image_id},SFORGE_PRIVATE_CONTEXT_JUDGE_IMAGE_ID=${judge_image_id},SFORGE_PRIVATE_CONTEXT_MODEL_VALIDATION_SHA256=${model_validation_sha256},SFORGE_PRIVATE_CONTEXT_RUNTIME_ENTRY_SHA256=${runtime_entry_sha256},SFORGE_PRIVATE_CONTEXT_SOURCE_DIGEST=${smoke_source_digest},SFORGE_PRIVATE_CONTEXT_RUNTIME_PROVENANCE_SHA256=${runtime_provenance_sha256},SFORGE_PRIVATE_CONTEXT_SFORGE_RUNTIME_SHA256=${sforge_runtime_sha256},SFORGE_PRIVATE_CONTEXT_SFORGE_RUNTIME_FILE_SET_SHA256=${sforge_runtime_file_set_sha256},SFORGE_PRIVATE_CONTEXT_SFORGE_RUNTIME_FILE_COUNT=${#SFORGE_RUNTIME_FILES[@]},SFORGE_PRIVATE_CONTEXT_JUDGE_VERIFY_DECODER_SHA256=${smoke_core_hashes[0]},SFORGE_PRIVATE_CONTEXT_JUDGE_CANDIDATE_WORKER_SHA256=${smoke_core_hashes[1]},SFORGE_PRIVATE_CONTEXT_JUDGE_COHORT_SHA256=${smoke_core_hashes[2]},SFORGE_PRIVATE_CONTEXT_JUDGE_QUALITY_SHA256=${smoke_core_hashes[3]},SFORGE_PRIVATE_CONTEXT_JUDGE_SOURCE_ADMISSION_SHA256=${smoke_core_hashes[4]}"
if [[ "${smoke_policy_mode}" == "pilot-provisional-workload" ]]; then
  judge_extra+=",LLADA_INTERACTIVE_JUDGE_TIMEOUT=${POLICY_FINAL_INNER_TIMEOUT},LLADA_FINAL_JUDGE_TIMEOUT=${POLICY_FINAL_INNER_TIMEOUT},SFORGE_PRIVATE_CONTEXT_PILOT=1"
elif (( workload_frozen == 1 )); then
  judge_extra+=",LLADA_INTERACTIVE_JUDGE_TIMEOUT=${LLADA_JUDGE_KILL_TIMEOUT},LLADA_FINAL_JUDGE_TIMEOUT=${LLADA_JUDGE_KILL_TIMEOUT}"
else
  # These are explicit, smoke-only liveness ceilings. The Judge adapter has no
  # timeout fallback; formal, producer, qualification, and replay paths inject
  # their frozen-policy or operator-specified values and fail closed otherwise.
  bootstrap_interactive_timeout="${LLADA_BOOTSTRAP_FINAL_JUDGE_TIMEOUT:-}"
  bootstrap_final_timeout="${LLADA_BOOTSTRAP_FINAL_JUDGE_TIMEOUT:-}"
  bootstrap_case_timeout="${LLADA_BOOTSTRAP_CASE_TIMEOUT:-}"
  bootstrap_row_timeout="${LLADA_BOOTSTRAP_ROW_TIMEOUT:-}"
  bootstrap_outer_timeout="${LLADA_BOOTSTRAP_OUTER_TIMEOUT:-}"
  for timeout_value in \
    "${bootstrap_interactive_timeout}" \
    "${bootstrap_final_timeout}" \
    "${bootstrap_case_timeout}" \
    "${bootstrap_row_timeout}" \
    "${bootstrap_outer_timeout}"; do
    [[ "${timeout_value}" =~ ^[0-9]+$ ]] && (( 10#${timeout_value} > 0 )) \
      || die "bootstrap Judge timeouts must be positive integers"
  done
  (( 10#${bootstrap_case_timeout} < 10#${bootstrap_interactive_timeout} \
      && 10#${bootstrap_case_timeout} < 10#${bootstrap_row_timeout} \
      && 10#${bootstrap_row_timeout} < 10#${bootstrap_final_timeout} \
      && 10#${bootstrap_final_timeout} <= 10#${bootstrap_outer_timeout} \
      && 10#${bootstrap_outer_timeout} <= 7200 )) \
    || die "bootstrap Judge timeouts must be coherently nested"
  judge_extra+=",LLADA_INTERACTIVE_JUDGE_TIMEOUT=${bootstrap_interactive_timeout},LLADA_FINAL_JUDGE_TIMEOUT=${bootstrap_final_timeout},LLADA_BOOTSTRAP_CASE_TIMEOUT=${bootstrap_case_timeout},LLADA_BOOTSTRAP_ROW_TIMEOUT=${bootstrap_row_timeout},LLADA_BOOTSTRAP_OUTER_TIMEOUT=${bootstrap_outer_timeout},SFORGE_PRIVATE_CONTEXT_BOOTSTRAP=1"
fi
env \
  SFORGE_TASKS_DIR="${TASKS_DIR}" \
  SFORGE_LOG_DIR="${smoke_log_root}" \
  SFORGE_ADMIN_SECRET="${admin_secret}" \
  SFORGE_ISOLATED_DOCKER_NETWORK="${smoke_net}" \
  SFORGE_JUDGE_EXTRA_ENV="${judge_extra}" \
  LLADA_MODEL_REPO="${LLADA_MODEL_REPO}" \
  LLADA_WORK_GPU_UUID="${work_uuid}" \
  LLADA_JUDGE_GPU_UUID="${judge_uuid}" \
  LLADA_RUNTIME_RUN_ID="${smoke_id}" \
  LLADA_JUDGE_CHECKPOINT_DIR="${LLADA_JUDGE_CHECKPOINT_DIR}" \
  SFORGE_EXPECTED_WORK_IMAGE_ID="${work_image_id}" \
  SFORGE_EXPECTED_JUDGE_IMAGE_ID="${judge_image_id}" \
  "$(dirname "${SFORGE_BIN}")/python" \
  "${MIGRATION_ROOT}/scripts/sforge_h100_entry.py" \
  serve --host "${bridge_gateway}" --port "${smoke_port}" \
  >"${LOGS_DIR}/smoke-${smoke_id}-server.log" 2>&1 &
server_pid="$!"
for _ in $(seq 1 90); do
  curl --noproxy '*' -fsS "http://${proxy_ip}:3130/openapi.json" >/dev/null 2>&1 && break
  sleep 1
done
curl --noproxy '*' -fsS "http://${proxy_ip}:3130/openapi.json" >/dev/null

register="$(
  curl --noproxy '*' -fsS -X POST \
    "http://${proxy_ip}:3130/api/v1/register" \
    -H 'Content-Type: application/json' \
    -d "{\"task_id\":\"${TASK_ID}\",\"run_id\":\"${smoke_id}\",\"admin_secret\":\"${admin_secret}\",\"submission_cooldown\":1,\"max_agent_submissions\":null,\"backend\":\"docker\",\"judge_cpu_limit\":8,\"judge_mem_limit\":\"48g\"}"
)"
token="$(jq -er '.token' <<<"${register}")"
submit="$(
  curl --noproxy '*' -fsS -X POST \
    "http://${proxy_ip}:3130/api/v1/submit" \
    -H "Content-Type: application/gzip" \
    -H "X-SForge-Token: ${token}" \
    -H "X-SForge-Submission-Kind: agent" \
    --data-binary "@${archive}"
)"
submission_id="$(jq -er '.submission_id' <<<"${submit}")"
result=""
# Every scored submission runs the complete 520-generation cohort (~30 min on
# H100 NVL), so poll up to the 7200 s coordinator ceiling.
for _ in $(seq 1 3600); do
  result="$(curl --noproxy '*' -fsS "http://${proxy_ip}:3130/api/v1/result/${submission_id}")"
  status="$(jq -r '.status' <<<"${result}")"
  [[ "${status}" == completed || "${status}" == error ]] && break
  sleep 2
done
jq -e '.status == "completed" or .status == "error"' <<<"${result}" >/dev/null
jq '{status, report: {valid: .report.valid, score: .report.score, pass_rate: .report.pass_rate, summary: .report.summary, details: .report.details, metrics: .report.metrics}}' <<<"${result}"
jq -e '.status == "completed" and .report.valid == true' \
  <<<"${result}" >/dev/null \
  || die "known-valid submission did not pass the bootstrap Judge"

# SForge must remove the ephemeral Judge before another submission could be
# admitted, while the Work sentinel is still alive on the distinct UUID.
assert_gpu_released "${judge_uuid}" "${JUDGE_GPU}" "interactive Judge"
docker inspect --format '{{.State.Running}}' "${work_sentinel}" \
  | grep -Fxq true || die "Work sentinel stopped during Judge evaluation"
docker logs "${work_sentinel}" 2>&1 \
  | grep -Fxq "WORK_SENTINEL_READY ${work_uuid}" \
  || die "Work sentinel UUID evidence changed"

history="{}"
for _ in $(seq 1 20); do
  history="$(curl --noproxy '*' -fsS "http://${proxy_ip}:3130/api/v1/history?token=${token}")"
  [[ "$(jq -r '.agent_submissions // 0' <<<"${history}")" == 1 ]] && break
  sleep 1
done
jq '{agent_submissions, auto_submissions, public_submissions, best_score, best_round, entries: [.entries[] | {round,status,valid,score}]}' <<<"${history}"
jq -e '.agent_submissions == 1 and .auto_submissions == 0 and (.public_submissions // 0) == 0' <<<"${history}" >/dev/null

cleanup_smoke
trap - EXIT INT TERM
assert_gpu_released "${work_uuid}" "${WORK_GPU}" "Work"
assert_gpu_released "${judge_uuid}" "${JUDGE_GPU}" "Judge"
assert_dual_gpu_locks_held
echo "Dual-H100 submission-plane smoke PASS"
