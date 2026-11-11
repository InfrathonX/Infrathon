#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_sforge
for command_name in curl docker jq python3 ss; do
  require_command "${command_name}"
done
assert_built_images_current

with_live=false
if [[ "${1:-}" == "--with-live" ]]; then
  with_live=true
elif [[ -n "${1:-}" ]]; then
  die "usage: $0 [--with-live]"
fi

original_nonce_source="${SFORGE_KVBENCH_NONCE_SOURCE:-}"
[[ -n "${original_nonce_source}" ]] \
  || die "SFORGE_KVBENCH_NONCE_SOURCE must point to the external nonce source"
[[ -f "${original_nonce_source}" && ! -L "${original_nonce_source}" ]] \
  || die "public API smoke nonce source is unavailable"
vllm_base_url="${KVBENCH_VLLM_BASE_URL:-}"
hidden_baseline_path="${KVBENCH_HIDDEN_BASELINE_PATH:-}"
public_baseline_path="${KVBENCH_PUBLIC_BASELINE_PATH:-}"
[[ -n "${vllm_base_url}" ]] \
  || die "KVBENCH_VLLM_BASE_URL is required"
[[ -f "${hidden_baseline_path}" && ! -L "${hidden_baseline_path}" ]] \
  || die "KVBENCH_HIDDEN_BASELINE_PATH is unavailable"
[[ -f "${public_baseline_path}" && ! -L "${public_baseline_path}" ]] \
  || die "KVBENCH_PUBLIC_BASELINE_PATH is unavailable"
public_direct_rate="$(jq -er \
  '.legs.baseline.valid_steps_per_min | select(type == "number" and . > 0)' \
  "${public_baseline_path}")" \
  || die "selected public baseline has no positive Direct rate"

stamp="$(date -u +%Y%m%d-%H%M%S)"
run_id="kvbench-public-api-smoke-${stamp}"
smoke_root="${LOGS_DIR}/smoke/${stamp}-public-api"
mkdir -p "${smoke_root}"
chmod 0700 "${smoke_root}"
port="$(${PYTHON_BIN} -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
admin_secret="$(${PYTHON_BIN} -c 'import secrets; print(secrets.token_urlsafe(48))')"
judge_pid=""
live_container=""
nonce_runtime_parent="/run/user/$(id -u)"
nonce_runtime_dir=""
nonce_source=""

cleanup() {
  local original_rc="$?"
  local cleanup_rc=0
  trap - EXIT INT TERM
  if [[ -n "${judge_pid}" ]] && kill -0 "${judge_pid}" 2>/dev/null; then
    kill -TERM "${judge_pid}" 2>/dev/null || true
    wait "${judge_pid}" 2>/dev/null || true
  fi
  if [[ -n "${live_container}" ]] \
      && docker container inspect "${live_container}" >/dev/null 2>&1; then
    docker rm --force "${live_container}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${nonce_source}" ]] \
      && [[ -f "${nonce_source}" && ! -L "${nonce_source}" ]]; then
    python3 "${MIGRATION_ROOT}/scripts/prepare_nonce_source.py" delete \
      --path "${nonce_source}" \
      --run-root "${nonce_runtime_dir}" || cleanup_rc=1
  fi
  if [[ -n "${nonce_runtime_dir}" && -d "${nonce_runtime_dir}" \
      && ! -L "${nonce_runtime_dir}" ]]; then
    rmdir -- "${nonce_runtime_dir}" || cleanup_rc=1
  fi
  if (( original_rc == 0 && cleanup_rc != 0 )); then
    original_rc=1
  fi
  exit "${original_rc}"
}
trap cleanup EXIT INT TERM

[[ -d "${nonce_runtime_parent}" && ! -L "${nonce_runtime_parent}" ]] \
  || die "public API smoke nonce runtime parent is unavailable"
[[ "$(stat -c '%u:%a' -- "${nonce_runtime_parent}")" == "$(id -u):700" ]] \
  || die "public API smoke nonce runtime parent must be current-uid mode 0700"
nonce_runtime_dir="$(mktemp -d -- "${nonce_runtime_parent}/mlsysforge-v24-nonce-smoke.XXXXXX")"
chmod 0700 -- "${nonce_runtime_dir}"
nonce_source="${nonce_runtime_dir}/nonce-source"
python3 "${MIGRATION_ROOT}/scripts/prepare_nonce_source.py" create \
  --source "${original_nonce_source}" \
  --output "${nonce_source}" \
  --run-root "${nonce_runtime_dir}"
python3 "${MIGRATION_ROOT}/scripts/prepare_nonce_source.py" validate \
  --path "${nonce_source}" \
  --run-root "${nonce_runtime_dir}"

export SFORGE_TASKS_DIR="${TASKS_DIR}"
export SFORGE_LOG_DIR="${LOGS_DIR}"
export SFORGE_KVBENCH_MIGRATION_ROOT="${MIGRATION_ROOT}"
export SFORGE_KVBENCH_EVAL_ROOT="${HOME}/.cache/mlsysforge-kvbench-qwen36-swe192-evals"
export SFORGE_KVBENCH_NONCE_SOURCE="${nonce_source}"
export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=void"
export SFORGE_ADMIN_SECRET="${admin_secret}"

if [[ "${with_live}" == true ]]; then
  "${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/preflight.py" \
    --nonce-source "${nonce_source}" \
    --vllm-base-url "${vllm_base_url}" \
    --hidden-baseline "${hidden_baseline_path}" \
    --public-baseline "${public_baseline_path}" \
    --output "${smoke_root}/preflight-before.json" \
    >"${smoke_root}/preflight-before.log"
fi

"${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/sforge_runtime_entry.py" \
  serve --port "${port}" >"${smoke_root}/judge-server.log" 2>&1 &
judge_pid="$!"
wait_for_http "http://127.0.0.1:${port}/openapi.json" 90

register_payload="$(jq -cn \
  --arg task_id "${TASK_ID}" \
  --arg run_id "${run_id}" \
  --arg admin_secret "${admin_secret}" \
  '{task_id:$task_id,run_id:$run_id,admin_secret:$admin_secret,submission_cooldown:300}')"
token="$(curl -fsS --max-time 30 \
  -H 'Content-Type: application/json' \
  --data-binary "${register_payload}" \
  "http://127.0.0.1:${port}/api/v1/register" | jq -er '.token')"

docker run --rm \
  --add-host host.docker.internal:host-gateway \
  --env "SFORGE_JUDGE_URL=http://host.docker.internal:${port}" \
  --env "SFORGE_TOKEN=${token}" \
  --env "KVBENCH_PUBLIC_DIRECT_RATE=${public_direct_rate}" \
  --env NVIDIA_VISIBLE_DEVICES=void \
  "${WORK_IMAGE}" kvbench doctor --json \
  >"${smoke_root}/work-doctor.json"
jq -e --argjson expected_rate "${public_direct_rate}" '
  .status == "ok"
  and .workload.workflows == 192
  and .workload.concurrency == 192
  and .workload.max_steps == 100
  and .workload.measurement_warmup_sec == 300
  and .workload.measurement_window_sec == 600
  and .frozen_direct_valid_steps_per_min == $expected_rate
' "${smoke_root}/work-doctor.json" >/dev/null

invalid_archive="${ARTIFACTS_DIR}/invalid-source.tar.gz"
[[ -s "${invalid_archive}" && ! -L "${invalid_archive}" ]] \
  || die "run scripts/smoke_test.sh first to create the invalid archive"
submit_response="$(curl -fsS --max-time 120 \
  -X POST "http://127.0.0.1:${port}/api/v1/submit" \
  -H 'Content-Type: application/gzip' \
  -H "X-SForge-Token: ${token}" \
  -H 'X-SForge-Submission-Kind: public' \
  --data-binary "@${invalid_archive}")"
submission_id="$(jq -er '.submission_id' <<<"${submit_response}")"
round_id="$(jq -er '.round_id' <<<"${submit_response}")"
[[ "${round_id}" == "public-1" ]] || die "public round allocation changed"

result=""
for _ in $(seq 1 120); do
  result="$(curl -fsS --max-time 30 \
    "http://127.0.0.1:${port}/api/v1/result/${submission_id}")"
  status="$(jq -r '.status // empty' <<<"${result}")"
  if [[ "${status}" == "completed" || "${status}" == "error" ]]; then
    break
  fi
  sleep 1
done
jq -e '
  .status == "completed"
  and .error == null
  and .report.valid == false
  and .report.score == 0
  and .report.metrics.source == "kvbench_public_candidate_only_vs_frozen_direct"
  and .report.metrics.evaluation_status == "candidate_invalid"
' <<<"${result}" >/dev/null

expected_public_count=1
live_candidate_rate=""
if [[ "${with_live}" == true ]]; then
  expected_public_count=2
  live_container="${run_id}-work"
  docker create \
    --name "${live_container}" \
    --label "mlsysforge.run_id=${run_id}" \
    --add-host host.docker.internal:host-gateway \
    --env "SFORGE_JUDGE_URL=http://host.docker.internal:${port}" \
    --env "SFORGE_TOKEN=${token}" \
    --env "KVBENCH_PUBLIC_DIRECT_RATE=${public_direct_rate}" \
    --env NVIDIA_VISIBLE_DEVICES=void \
    --entrypoint bash \
    "${WORK_IMAGE}" -lc '
      set -e
      kvbench run --mode baseline --profile public-swe192-distinct --json \
        >/app/public_evals/baseline-command.json
      kvbench run --mode candidate --profile public-swe192-distinct --json \
        >/app/public_evals/candidate-command.json
      kvbench compare --profile public-swe192-distinct --json \
        >/app/public_evals/compare-command.json
      artifact_dir="$(jq -er .candidate_eval_artifact \
        /app/public_evals/candidate-command.json)"
      test -d "${artifact_dir}"
      test -f "${artifact_dir}/candidate_result.json"
      test -f "${artifact_dir}/code_snapshot.tar.gz"
    ' >/dev/null
  docker start --attach "${live_container}" \
    >"${smoke_root}/work-live-command.log" 2>&1
  mkdir -p "${smoke_root}/work-public-evals"
  for public_output in \
    baseline-command.json candidate-command.json compare-command.json; do
    docker cp \
      "${live_container}:/app/public_evals/${public_output}" \
      "${smoke_root}/work-public-evals/${public_output}"
  done
  docker rm "${live_container}" >/dev/null
  live_container=""
  jq -e --argjson expected_rate "${public_direct_rate}" '
    .status == "completed"
    and .error == null
    and .report.valid == true
    and (.report.score | type == "number" and . > 0)
    and .report.metrics.profile == "public-swe192-distinct"
    and .report.metrics.workflows == 192
    and .report.metrics.concurrency == 192
    and .report.metrics.max_steps == 100
    and .report.metrics.measurement_warmup_sec == 300
    and .report.metrics.measurement_window_sec == 600
    and .comparison.valid == true
  ' "${smoke_root}/work-public-evals/candidate-command.json" >/dev/null
  jq -e --argjson expected_rate "${public_direct_rate}" '
    .valid == true
    and .profile == "public-swe192-distinct"
    and .baseline_valid_steps_per_min == $expected_rate
    and (.candidate_valid_steps_per_min | type == "number" and . > 0)
  ' "${smoke_root}/work-public-evals/compare-command.json" >/dev/null
  live_candidate_rate="$(jq -er '.candidate_valid_steps_per_min' \
    "${smoke_root}/work-public-evals/compare-command.json")"
fi

public_history="$(curl -fsS --max-time 30 \
  --get --data-urlencode "token=${token}" \
  "http://127.0.0.1:${port}/api/v1/public/history")"
agent_history="$(curl -fsS --max-time 30 \
  --get --data-urlencode "token=${token}" \
  "http://127.0.0.1:${port}/api/v1/history")"
admin_history="$(curl -fsS --max-time 30 \
  --get --data-urlencode "token=${token}" \
  -H "X-SForge-Admin-Secret: ${admin_secret}" \
  "http://127.0.0.1:${port}/api/v1/history")"
jq -e --argjson expected_count "${expected_public_count}" '
  .enabled == true
  and .public_submissions == $expected_count
  and (.entries | length) == $expected_count
  and .entries[0].type == "public"
  and .entries[0].round == "public-1"
' <<<"${public_history}" >/dev/null
jq -e '
  .best_score == null
  and .best_round == ""
  and .agent_submissions == 0
  and (.entries | length) == 0
' <<<"${agent_history}" >/dev/null
jq -e --argjson expected_count "${expected_public_count}" '
  .best_score == null
  and .best_round == ""
  and .public_submissions == $expected_count
  and (.entries | length) == $expected_count
  and ([.entries[].type] | all(. == "public"))
' <<<"${admin_history}" >/dev/null

for public_number in $(seq 1 "${expected_public_count}"); do
  private_artifact="${LOGS_DIR}/runs/${run_id}/${TASK_ID}/submissions/public-${public_number}/private_judge_artifacts.tar.gz"
  [[ -s "${private_artifact}" && ! -L "${private_artifact}" ]] \
    || die "public Judge omitted private evidence for public-${public_number}"
  [[ "$(stat -c '%a' "${private_artifact}")" == "600" ]] \
    || die "public private evidence is not mode 0600"
done

if [[ "${with_live}" == true ]]; then
  "${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/preflight.py" \
    --nonce-source "${nonce_source}" \
    --vllm-base-url "${vllm_base_url}" \
    --hidden-baseline "${hidden_baseline_path}" \
    --public-baseline "${public_baseline_path}" \
    --output "${smoke_root}/preflight-after.json" \
    >"${smoke_root}/preflight-after.log"
  [[ "$(jq -er '.backend_serving_contract_sha256' "${smoke_root}/preflight-before.json")" \
      == "$(jq -er '.backend_serving_contract_sha256' "${smoke_root}/preflight-after.json")" ]] \
    || die "backend serving contract changed during live public smoke"
fi

jq -n \
  --arg finished_at "$(date -u +%FT%TZ)" \
  --arg run_id "${run_id}" \
  --arg round_id "${round_id}" \
  --argjson live "${with_live}" \
  --arg live_candidate_rate "${live_candidate_rate}" \
  '{
    status:"passed",
    finished_at:$finished_at,
    run_id:$run_id,
    invalid_round_id:$round_id,
    gpu_workload_executed:$live,
    live_candidate_valid_steps_per_min:(
      if $live_candidate_rate == "" then null else ($live_candidate_rate | tonumber) end
    )
  }' \
  >"${smoke_root}/result.json"

if docker ps --all --quiet \
    --filter label=com.docker.compose.project=kvbench-qwen36-swe192-v2-1-gpu67-validation \
    | grep -q .; then
  die "public API smoke leaked the validation Compose project"
fi
echo "Public API smoke PASS: ${smoke_root}"
