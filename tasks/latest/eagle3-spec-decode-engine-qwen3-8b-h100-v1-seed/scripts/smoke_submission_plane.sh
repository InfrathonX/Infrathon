#!/usr/bin/env bash
# Two-GPU submission-plane smoke for the EAGLE-3 speculative-decode task.
# Lineage: LLaDA v2 scripts/smoke_submission_plane.sh. With a known-valid
# archive, start the Work sentinel, restricted proxy, and SForge Judge server
# the way run_fixed_budget.sh does, submit once as agent-1 through the proxy,
# and refuse to pass unless the ephemeral Judge container received the role
# environment and the round was graded rather than lost to infrastructure.
# The first formal run lost its only interactive round because that Judge
# container did not receive NVIDIA_VISIBLE_DEVICES; this smoke exists so the
# next defect of that class fails here, before any budgeted agent run.
set -Eeuo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'USAGE'
Usage: WORK_GPU=N JUDGE_GPU=M smoke_submission_plane.sh KNOWN_VALID_ARCHIVE

KNOWN_VALID_ARCHIVE is a persisted submission.tar.gz or final_archive.tar.gz
that already passed this task's Judge. Optional environment:
  SMOKE_JUDGE_PORT                  host Judge port (default 18491)
  EAGLE3_INTERACTIVE_JUDGE_TIMEOUT  complete-suite Judge deadline (default 5400)
  FINAL_EVAL_TIMEOUT                final Judge deadline (default 5400)
  GPU_MIN_FREE_MIB                  admission floor for both GPUs
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# Re-exec under the docker group before any Docker or GPU query, preserving
# the already supplied archive argument.
ensure_docker_group "${ORIGINAL_ARGS[@]}"
require_sforge
for required_command in curl docker flock jq nvidia-smi python3 realpath seq sha256sum ss; do
  require_command "${required_command}"
done

# The archive must be a real, nonempty, non-symlink file whose bytes are
# hashed now so the persisted submission record can be bound to it later.
archive_argument="${1:-}"
if [[ -z "${archive_argument}" ]]; then
  usage >&2
  die "usage: WORK_GPU=N JUDGE_GPU=M $0 KNOWN_VALID_ARCHIVE"
fi
archive="$(realpath -- "${archive_argument}")"
[[ -f "${archive}" && ! -L "${archive}" && -s "${archive}" ]] \
  || die "known-valid archive must be a nonempty regular file: ${archive}"
archive_sha256="$(sha256sum -- "${archive}" | awk '{print $1}')"

# Both GPU ordinals must be explicit, numeric, distinct, and resolve to
# distinct physical UUIDs; a default would silently pin both roles together.
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
[[ "${work_uuid}" =~ ^GPU-[A-Za-z0-9-]+$ ]] \
  || die "Work GPU ${WORK_GPU} returned an unsafe UUID: ${work_uuid}"
[[ "${judge_uuid}" =~ ^GPU-[A-Za-z0-9-]+$ ]] \
  || die "Judge GPU ${JUDGE_GPU} returned an unsafe UUID: ${judge_uuid}"
[[ "${work_uuid}" != "${judge_uuid}" ]] \
  || die "Work and Judge GPU UUIDs must differ"

# This task has one private policy whose `status` is either QUALIFIED (all
# eight frozen A2 tiers, formal scoring enabled) or PENDING_QUALIFICATION
# (the verifier grades but withholds the score). Both states exercise the
# same submission plane; any other status is not a smoke-able task.
policy_path="${EAGLE3_PRIVATE_DATA}/policy.json"
[[ -f "${policy_path}" && ! -L "${policy_path}" ]] \
  || die "submission-plane smoke requires a regular policy file: ${policy_path}"
policy_status="$(jq -er '.status' "${policy_path}")"
case "${policy_status}" in
  QUALIFIED)
    [[ "$(jq -er '.tiers | length' "${policy_path}")" == "8" ]] \
      || die "QUALIFIED policy must carry all eight frozen A2 tiers"
    smoke_policy_mode="qualified-policy"
    ;;
  PENDING_QUALIFICATION)
    smoke_policy_mode="pending-qualification"
    ;;
  *)
    die "submission-plane smoke requires a QUALIFIED or PENDING_QUALIFICATION policy; found ${policy_status}"
    ;;
esac
policy_sha256="$(sha256sum -- "${policy_path}" | awk '{print $1}')"

# The Judge adapter has no timeout fallback: both deadlines must be explicit
# positive integers, nested inside SForge's own kill deadline for the task.
task_spec="${TASKS_DIR}/${TASK_ID}.json"
[[ -f "${task_spec}" && ! -L "${task_spec}" ]] \
  || die "task metadata is missing or unsafe: ${task_spec}"
task_eval_timeout="$(jq -er '.judge.eval_timeout' "${task_spec}")"
interactive_timeout="${EAGLE3_INTERACTIVE_JUDGE_TIMEOUT:-5400}"
final_timeout="${FINAL_EVAL_TIMEOUT:-5400}"
for timeout_value in "${task_eval_timeout}" "${interactive_timeout}" "${final_timeout}"; do
  [[ "${timeout_value}" =~ ^[1-9][0-9]*$ ]] \
    || die "Judge deadlines must be positive integers"
done
(( interactive_timeout <= final_timeout )) \
  || die "interactive Judge deadline must not exceed the final Judge deadline"
(( final_timeout <= task_eval_timeout )) \
  || die "final Judge deadline must not exceed the task eval_timeout (${task_eval_timeout})"

# The admission floor is the same operator-visible value every other entry
# uses; a garbage value must fail here rather than inside arithmetic.
[[ "${GPU_MIN_FREE_MIB}" =~ ^[1-9][0-9]*$ ]] \
  || die "GPU_MIN_FREE_MIB must be a positive integer"

# Hold both UUID locks for the whole smoke and prove both devices are idle,
# distinct envelope-conforming GPUs before any container is created.
acquire_dual_gpu_locks "${work_uuid}" "${judge_uuid}"
assert_distinct_gpu_pair "${WORK_GPU}" "${JUDGE_GPU}"
assert_dual_gpu_locks_held

# Only the recorded immutable images may serve the smoke; a moved tag or a
# changed source tree would test something other than the formal images.
assert_built_images_current
build_manifest="${BUILD_DIR}/image-manifest.json"
work_image_id="$(jq -er '.work.id' "${build_manifest}")"
judge_image_id="$(jq -er '.judge.id' "${build_manifest}")"
[[ "${work_image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || die "smoke Work image ID is invalid"
[[ "${judge_image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || die "smoke Judge image ID is invalid"
assert_container_gpu "${WORK_IMAGE}" "${work_uuid}" "Work"
assert_container_gpu "${JUDGE_IMAGE}" "${judge_uuid}" "Judge"

# Validate both frozen read-only model artifacts and the role GPUs exactly as
# the formal launcher does before it spends any budget.
"${PYTHON_BIN}" "${SCRIPT_DIR}/preflight.py" \
  --target-repo "${EAGLE3_TARGET_REPO}" \
  --head-dir "${EAGLE3_HEAD_DIR}" \
  --work-gpu "${WORK_GPU}" \
  --judge-gpu "${JUDGE_GPU}" \
  --image-manifest "${build_manifest}" >/dev/null

# Smoke identity: every container, network, label, and log path is scoped to
# this ID so cleanup can never touch another run's resources.
smoke_id="eagle3-submit-smoke-$(date -u +%Y%m%d-%H%M%S)"
[[ "${smoke_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$ ]] \
  || die "unsafe smoke ID: ${smoke_id}"
smoke_net="${smoke_id}-net"
smoke_proxy="${smoke_id}-proxy"
work_sentinel="${smoke_id}-work-sentinel"

# The host Judge port must be numeric, unprivileged, distinct from the two
# proxy listeners, and currently unused.
smoke_port="${SMOKE_JUDGE_PORT:-18491}"
[[ "${smoke_port}" =~ ^[0-9]+$ ]] || die "SMOKE_JUDGE_PORT must be numeric"
smoke_port="$((10#${smoke_port}))"
(( smoke_port >= 1024 && smoke_port <= 65535 && smoke_port != 3128 && smoke_port != 3130 )) \
  || die "invalid or conflicting SMOKE_JUDGE_PORT: ${smoke_port}"
if ss -H -ltn 2>/dev/null | awk '{print $4}' | grep -Eq ":${smoke_port}$"; then
  die "smoke Judge port ${smoke_port} is already in use"
fi

# Fresh secrets per smoke: the admin secret authenticates host-side
# registration; the cohort secret is retained for legacy host diagnostics only.
admin_secret="$("${PYTHON_BIN}" -c 'import secrets; print(secrets.token_urlsafe(32))')"
cohort_secret="$("${PYTHON_BIN}" -c 'import secrets; print(secrets.token_hex(32))')"
[[ "${cohort_secret}" =~ ^[0-9a-f]{64}$ ]] \
  || die "failed to create the HMAC cohort secret"

# All smoke output lives under a private, write-once directory so a rerun
# can never read a stale record as fresh evidence.
smoke_log_root="${LOGS_DIR}/smoke/${smoke_id}"
[[ ! -e "${smoke_log_root}" && ! -L "${smoke_log_root}" ]] \
  || die "smoke directory already exists: ${smoke_log_root}"
mkdir -p -- "${smoke_log_root}"
chmod 700 -- "${smoke_log_root}"
server_log="${smoke_log_root}/judge-server.log"
judge_env_snapshot="${smoke_log_root}/judge-container-env.json"
bridge_gateway="$(docker network inspect bridge | jq -er '.[0].IPAM.Config[0].Gateway')"

server_pid=""

cleanup_smoke() {
  # Cleanup is idempotent and never masks the original failure status.
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
  local signal_name="$1"
  local exit_code=143
  if [[ "${signal_name}" == "INT" ]]; then
    exit_code=130
  elif [[ "${signal_name}" == "HUP" ]]; then
    exit_code=129
  fi
  trap - EXIT INT TERM HUP
  cleanup_smoke
  exit "${exit_code}"
}

trap cleanup_smoke EXIT
trap 'handle_smoke_signal INT' INT
trap 'handle_smoke_signal TERM' TERM
trap 'handle_smoke_signal HUP' HUP

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
uuid = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
).strip()
torch.zeros(1, device="cuda")
print(f"WORK_SENTINEL_READY {uuid}", flush=True)
time.sleep(7200)
' >/dev/null
for _ in $(seq 1 60); do
  if docker logs "${work_sentinel}" 2>&1 | grep -Fxq "WORK_SENTINEL_READY ${work_uuid}"; then
    break
  fi
  sleep 1
done
docker logs "${work_sentinel}" 2>&1 \
  | grep -Fxq "WORK_SENTINEL_READY ${work_uuid}" \
  || die "Work CUDA sentinel did not become ready on ${work_uuid}"

# The submission plane is the restricted proxy's dedicated port 3130, reached
# from an internal network under the alias runtime-proxy, exactly as Work
# reaches it in a formal run. The proxy allowlists the model-auth hosts
# itself; the smoke never sends outbound traffic through it.
docker network create \
  --driver bridge \
  --internal \
  --label "mlsysforge.run_id=${smoke_id}" \
  "${smoke_net}" >/dev/null
docker run \
  --detach \
  --rm \
  --name "${smoke_proxy}" \
  --network bridge \
  --add-host host.docker.internal:host-gateway \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --volume "${SCRIPT_DIR}/restricted_proxy.py:/opt/mlsysforge/restricted_proxy.py:ro" \
  --entrypoint python3 \
  "${WORK_IMAGE}" \
  /opt/mlsysforge/restricted_proxy.py \
  --judge-host "${bridge_gateway}" \
  --judge-port "${smoke_port}" \
  --judge-forward-port 3130 >/dev/null
docker network connect --alias runtime-proxy "${smoke_net}" "${smoke_proxy}"
proxy_ip="$(
  docker inspect "${smoke_proxy}" \
    | jq -er --arg network "${smoke_net}" \
      '.[0].NetworkSettings.Networks[$network].IPAddress'
)"
judge_url="http://${proxy_ip}:3130"

# Build the interactive Judge environment from the shared specification and
# validate it with the same guard the formal launcher applies. A missing key
# here is the exact defect class that lost the first formal run's only
# interactive round.
judge_extra="$(
  judge_extra_env_spec \
    "${judge_uuid}" \
    "${cohort_secret}" \
    "${interactive_timeout}" \
    "${final_timeout}"
)"
assert_interactive_judge_env_spec "${judge_extra}" "${judge_uuid}"

# The Judge server parses its GPU, immutable-image, run-scope, and private
# context inputs at process startup, so they are exported before spawning it,
# mirroring run_fixed_budget.sh.
env \
  SFORGE_ADMIN_SECRET="${admin_secret}" \
  SFORGE_ISOLATED_DOCKER_NETWORK="${smoke_net}" \
  SFORGE_JUDGE_EXTRA_ENV="${judge_extra}" \
  SFORGE_EXPECTED_WORK_IMAGE_ID="${work_image_id}" \
  SFORGE_EXPECTED_JUDGE_IMAGE_ID="${judge_image_id}" \
  EAGLE3_WORK_GPU_UUID="${work_uuid}" \
  EAGLE3_JUDGE_GPU_UUID="${judge_uuid}" \
  EAGLE3_RUNTIME_RUN_ID="${smoke_id}" \
  EAGLE3_COHORT_SECRET="${cohort_secret}" \
  EAGLE3_TARGET_REPO="${EAGLE3_TARGET_REPO}" \
  EAGLE3_HEAD_DIR="${EAGLE3_HEAD_DIR}" \
  "${PYTHON_BIN}" "${SCRIPT_DIR}/sforge_h100_entry.py" \
  --output-dir "${smoke_log_root}" \
  --tasks-dir "${TASKS_DIR}" \
  serve --host "${bridge_gateway}" --port "${smoke_port}" \
  >"${server_log}" 2>&1 &
server_pid="$!"
# First the server itself, then the proxied hop that agents actually use.
wait_for_http "http://${bridge_gateway}:${smoke_port}/openapi.json" 90
wait_for_http "${judge_url}/openapi.json" 30

# Register one session with a one-second cooldown and the formal Judge
# resource limits; unlimited agent rounds keep the count assertion honest.
register_body="$(
  jq -n \
    --arg task_id "${TASK_ID}" \
    --arg run_id "${smoke_id}" \
    --arg admin_secret "${admin_secret}" \
    '{task_id:$task_id,run_id:$run_id,admin_secret:$admin_secret,
      submission_cooldown:1,max_agent_submissions:null,backend:"docker",
      judge_cpu_limit:8,judge_mem_limit:"64g"}'
)"
register="$(
  curl --noproxy '*' -fsS -X POST \
    "${judge_url}/api/v1/register" \
    -H 'Content-Type: application/json' \
    --data "${register_body}"
)"
token="$(jq -er '.token' <<<"${register}")"

# Submit exactly as sforge-submit does: the raw gzip body, the session token,
# and the agent submission kind. The server must assign round agent-1.
submit="$(
  curl --noproxy '*' -fsS -X POST \
    "${judge_url}/api/v1/submit" \
    -H 'Content-Type: application/gzip' \
    -H "X-SForge-Token: ${token}" \
    -H 'X-SForge-Submission-Kind: agent' \
    --data-binary "@${archive}"
)"
submission_id="$(jq -er '.submission_id' <<<"${submit}")"
round_id="$(jq -er '.round_id' <<<"${submit}")"
[[ "${submission_id}" =~ ^[0-9a-f]{12}$ ]] \
  || die "submission ID is not the SForge hex form: ${submission_id}"
[[ "${round_id}" == "agent-1" ]] \
  || die "first interactive submission was not assigned round agent-1: ${round_id}"

observe_judge_container() {
  # Snapshot the ephemeral interactive Judge container while it is alive.
  # Only containers bearing this smoke's task, run, role, and UUID labels are
  # candidates; the cohort secret value is redacted, its presence is kept.
  local -a container_ids=()
  mapfile -t container_ids < <(
    docker ps -q \
      --filter "label=mlsysforge.task_id=${TASK_ID}" \
      --filter "label=mlsysforge.run_id=${smoke_id}" \
      --filter "label=mlsysforge.gpu_role=judge" \
      --filter "label=mlsysforge.gpu_uuid=${judge_uuid}" 2>/dev/null
  )
  (( ${#container_ids[@]} > 0 )) || return 1
  (
    umask 077
    docker inspect "${container_ids[@]}" \
      | jq '[.[] | {
          name: .Name,
          image: .Image,
          network_mode: .HostConfig.NetworkMode,
          device_requests: .HostConfig.DeviceRequests,
          labels: .Config.Labels,
          cohort_secret_present: (
            [.Config.Env[] | select(startswith("EAGLE3_COHORT_SECRET="))] | length
          ),
          env: [.Config.Env[] | select(startswith("EAGLE3_COHORT_SECRET=") | not)]
        }]' >"${judge_env_snapshot}.tmp"
  )
  mv -- "${judge_env_snapshot}.tmp" "${judge_env_snapshot}"
}

# Poll the round to a terminal state within the interactive deadline plus
# container start/stop slack, snapshotting the Judge container on the way.
poll_deadline="$(( SECONDS + interactive_timeout + 300 ))"
result=""
status="queued"
judge_observed=0
while (( SECONDS < poll_deadline )); do
  result="$(curl --noproxy '*' -fsS "${judge_url}/api/v1/result/${submission_id}")"
  status="$(jq -er '.status' <<<"${result}")"
  if (( judge_observed == 0 )) && observe_judge_container; then
    judge_observed=1
  fi
  if [[ "${status}" == completed || "${status}" == error ]]; then
    break
  fi
  sleep 2
done
[[ "${status}" == completed || "${status}" == error ]] \
  || die "interactive round ${round_id} did not reach a terminal state within the deadline"

assert_judge_container_environment() {
  # The container SForge created for agent-1 must carry the complete role
  # environment, the single explicit device request, network mode none, and
  # the immutable Judge image. A missing NVIDIA_VISIBLE_DEVICES fails here.
  (( judge_observed == 1 )) \
    || die "the interactive Judge container was never observed under this smoke's labels"
  [[ -s "${judge_env_snapshot}" ]] \
    || die "Judge container snapshot is missing: ${judge_env_snapshot}"
  jq -e \
    --arg uuid "${judge_uuid}" \
    --arg image "${judge_image_id}" \
    --arg run_id "${smoke_id}" \
    --arg interactive "${interactive_timeout}" \
    --arg final "${final_timeout}" '
    length == 1
    and .[0].image == $image
    and .[0].network_mode == "none"
    and .[0].labels["mlsysforge.run_id"] == $run_id
    and .[0].labels["mlsysforge.gpu_role"] == "judge"
    and .[0].labels["mlsysforge.gpu_uuid"] == $uuid
    and (.[0].device_requests | length) == 1
    and .[0].device_requests[0].DeviceIDs == [$uuid]
    and .[0].cohort_secret_present == 1
    and any(.[0].env[]; . == "NVIDIA_VISIBLE_DEVICES=" + $uuid)
    and any(.[0].env[]; . == "CUDA_VISIBLE_DEVICES=0")
    and any(.[0].env[]; . == "EAGLE3_GPU_ROLE=judge")
    and any(.[0].env[]; . == "EAGLE3_TARGET=single-cuda-sm80-34g")
    and any(.[0].env[]; . == "EAGLE3_INTERACTIVE_JUDGE_TIMEOUT=" + $interactive)
    and any(.[0].env[]; . == "EAGLE3_FINAL_JUDGE_TIMEOUT=" + $final)
    and any(.[0].env[]; . == "HF_HUB_OFFLINE=1")
  ' "${judge_env_snapshot}" >/dev/null \
    || die "interactive Judge container lacked the role environment; see ${judge_env_snapshot}"
  echo "Interactive Judge container environment: PASS (${judge_uuid} as cuda:0)"
}

assert_report_graded() {
  # One predicate for both the API result and the persisted report.json:
  # the EAGLE-3 adapter completed the verifier and produced a graded outcome,
  # never SForge's infrastructure_error shape.
  local report_json="$1"
  local origin="$2"
  jq -e '
    type == "object"
    and .timed_out == false
    and (.metrics.evaluation_status // "") != "infrastructure_error"
    and ((.summary // "") | startswith("Evaluation infrastructure") | not)
    and .metrics.source == "eagle3_spec_decode_v1_verifier_adapter"
    and .metrics.verifier_completed == true
    and (.metrics.verifier_returncode == 0 or .metrics.verifier_returncode == 1)
    and (.score | type == "number")
  ' <<<"${report_json}" >/dev/null \
    || die "${origin}: interactive round was not graded by the EAGLE-3 verifier adapter"
  if [[ "${smoke_policy_mode}" == "qualified-policy" ]]; then
    # A known-valid archive under a QUALIFIED policy must pass with a
    # positive score; a trusted gate failure is graded but still wrong here.
    if jq -e '.valid == true and .score > 0' <<<"${report_json}" >/dev/null; then
      return 0
    fi
    if jq -e '.metrics.verifier_returncode == 1
        and any(.details[]; .status == "FAILED")' <<<"${report_json}" >/dev/null; then
      die "${origin}: known-valid archive was graded but failed gates: $(
        jq -r '[.details[] | select(.status == "FAILED") | .name] | join(",")' \
          <<<"${report_json}"
      )"
    fi
    die "${origin}: known-valid archive did not pass the QUALIFIED Judge"
  fi
  # Under PENDING_QUALIFICATION the verifier grades, exits 0, and withholds
  # the score behind the pending summary; anything else is not that path.
  jq -e '.valid == false
    and .metrics.verifier_returncode == 0
    and ((.summary // "") | startswith("PENDING_QUALIFICATION"))' \
    <<<"${report_json}" >/dev/null \
    || die "${origin}: pending-qualification Judge did not produce the pending graded outcome"
}

assert_round_graded() {
  # The API record is the agent-visible truth: a status of error means the
  # round died in SForge before or during Judge creation, which is exactly
  # the class of defect this smoke exists to catch.
  if [[ "${status}" == error ]]; then
    die "interactive round ${round_id} failed in infrastructure before grading: $(jq -r '.error // "unknown"' <<<"${result}")"
  fi
  jq --arg round "${round_id}" \
    '{status, round: $round, report: {valid: .report.valid, score: .report.score, pass_rate: .report.pass_rate, summary: .report.summary, details: .report.details, metrics: .report.metrics}}' \
    <<<"${result}"
  assert_report_graded "$(jq -c '.report' <<<"${result}")" "api result"
}

assert_persisted_submission_record() {
  # SForge may nest the run one extra level below --output-dir. Resolve only
  # the two supported layouts and require the complete agent-1 record.
  local -a candidate_roots=(
    "${smoke_log_root}/runs/${smoke_id}/${TASK_ID}/submissions/${round_id}"
    "${smoke_log_root}/${TASK_ID}/submissions/${round_id}"
  )
  local candidate_root
  local submission_dir=""
  local required
  local persisted_sha256
  local api_score
  local persisted_score
  for candidate_root in "${candidate_roots[@]}"; do
    if [[ -s "${candidate_root}/report.json" ]]; then
      submission_dir="${candidate_root}"
      break
    fi
  done
  [[ -n "${submission_dir}" ]] \
    || die "interactive round ${round_id} did not persist a report under ${smoke_log_root}"
  for required in report.json submission.tar.gz run_instance.log test_output.txt; do
    [[ -s "${submission_dir}/${required}" && ! -L "${submission_dir}/${required}" ]] \
      || die "persisted submission record lacks ${required}: ${submission_dir}"
  done
  # The persisted bytes must be the archive this smoke submitted.
  persisted_sha256="$(sha256sum -- "${submission_dir}/submission.tar.gz" | awk '{print $1}')"
  [[ "${persisted_sha256}" == "${archive_sha256}" ]] \
    || die "persisted submission.tar.gz differs from the submitted archive"
  # The persisted report must name this task and submission and carry the
  # same graded outcome and score as the API record.
  jq -e \
    --arg task_id "${TASK_ID}" \
    --arg submission_id "${submission_id}" \
    '.task_id == $task_id and .submission_id == $submission_id' \
    "${submission_dir}/report.json" >/dev/null \
    || die "persisted report.json does not identify this task and submission"
  assert_report_graded "$(jq -c '.' "${submission_dir}/report.json")" "persisted report"
  api_score="$(jq -r '.report.score' <<<"${result}")"
  persisted_score="$(jq -r '.score' "${submission_dir}/report.json")"
  [[ "${api_score}" == "${persisted_score}" ]] \
    || die "persisted score ${persisted_score} differs from API score ${api_score}"
  # The Judge log must show the container starting and no infrastructure
  # failure or traceback from the adapter or SForge.
  grep -Fq "Judge container started" "${submission_dir}/run_instance.log" \
    || die "Judge log does not record the interactive Judge container starting"
  if grep -Eqi 'traceback|evaluation infrastructure' "${submission_dir}/run_instance.log"; then
    die "Judge log records an infrastructure failure: ${submission_dir}/run_instance.log"
  fi
  # The root-only private verifier bundle is collected for every graded
  # non-public round; its absence would already be an infrastructure error.
  [[ -e "${submission_dir}/private_judge_artifacts.tar.gz" ]] \
    || die "persisted submission record lacks the private Judge artifact bundle"
  echo "Persisted submission record: PASS (${submission_dir})"
}

assert_judge_container_environment
assert_round_graded
assert_persisted_submission_record

# SForge must remove the ephemeral Judge before another submission could be
# admitted, while the Work sentinel is still alive on the distinct UUID.
assert_gpu_released "${judge_uuid}" "${JUDGE_GPU}" "interactive Judge"
docker inspect --format '{{.State.Running}}' "${work_sentinel}" \
  | grep -Fxq true || die "Work sentinel stopped during Judge evaluation"
docker logs "${work_sentinel}" 2>&1 \
  | grep -Fxq "WORK_SENTINEL_READY ${work_uuid}" \
  || die "Work sentinel UUID evidence changed"

# The host-side history view must show exactly one completed agent round and
# no automatic or public rounds.
history="{}"
for _ in $(seq 1 20); do
  history="$(
    curl --noproxy '*' -fsS \
      -H "X-SForge-Admin-Secret: ${admin_secret}" \
      "${judge_url}/api/v1/history?token=${token}"
  )"
  if [[ "$(jq -r '.agent_submissions // 0' <<<"${history}")" == 1 ]]; then
    break
  fi
  sleep 1
done
jq '{agent_submissions, auto_submissions, public_submissions, best_score, best_round, entries: [.entries[] | {round,status,valid,score,error}]}' <<<"${history}"
jq -e \
  --arg round "${round_id}" '
  .agent_submissions == 1
  and .auto_submissions == 0
  and (.public_submissions // 0) == 0
  and (.entries | length) == 1
  and .entries[0].round == $round
  and .entries[0].status == "completed"
  and (.entries[0].error // null) == null
' <<<"${history}" >/dev/null \
  || die "run history does not show exactly one completed agent round"

cleanup_smoke
trap - EXIT INT TERM HUP
assert_gpu_released "${work_uuid}" "${WORK_GPU}" "Work"
assert_gpu_released "${judge_uuid}" "${JUDGE_GPU}" "Judge"
assert_dual_gpu_locks_held
echo "Two-GPU submission-plane smoke PASS (${smoke_policy_mode}, policy ${policy_sha256:0:12}, ${round_id} score $(jq -r '.report.score' <<<"${result}"))"
