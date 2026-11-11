#!/usr/bin/env bash
set -Eeuo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_sforge
for command_name in docker jq pytest sha256sum tar; do
  if [[ "${command_name}" == pytest ]]; then
    continue
  fi
  require_command "${command_name}"
done
assert_built_images_current

with_oracle=false
if [[ "${1:-}" == "--with-oracle" ]]; then
  with_oracle=true
elif [[ -n "${1:-}" ]]; then
  die "usage: $0 [--with-oracle]"
fi

# The nonce source is a Judge-only input (judge/private/nonce_source.txt, else the
# asset-root copy from assets/fetch.sh: default_nonce_source in lib.sh);
# SFORGE_KVBENCH_NONCE_SOURCE only overrides it.
nonce_source="${SFORGE_KVBENCH_NONCE_SOURCE:-$(default_nonce_source)}"
[[ -n "${nonce_source}" ]] \
  || die "SFORGE_KVBENCH_NONCE_SOURCE must point to the external nonce source"
vllm_base_url="${KVBENCH_VLLM_BASE_URL:-}"
hidden_baseline_path="${KVBENCH_HIDDEN_BASELINE_PATH:-}"
public_baseline_path="${KVBENCH_PUBLIC_BASELINE_PATH:-}"
[[ -n "${vllm_base_url}" ]] || die "KVBENCH_VLLM_BASE_URL is required"
[[ -f "${hidden_baseline_path}" && ! -L "${hidden_baseline_path}" ]] \
  || die "KVBENCH_HIDDEN_BASELINE_PATH is unavailable"
[[ -f "${public_baseline_path}" && ! -L "${public_baseline_path}" ]] \
  || die "KVBENCH_PUBLIC_BASELINE_PATH is unavailable"
stamp="$(date -u +%Y%m%d-%H%M%S)"
smoke_root="${LOGS_DIR}/smoke/${stamp}"
mkdir -p "${smoke_root}"
chmod 0700 "${smoke_root}"
mkdir -p "${ARTIFACTS_DIR}"
export SFORGE_KVBENCH_NONCE_SOURCE="${nonce_source}"

"${PYTHON_BIN}" -m pytest -q "${MIGRATION_ROOT}/tests" \
  | tee "${smoke_root}/pytest.log"
"${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/preflight.py" \
  --nonce-source "${nonce_source}" \
  --vllm-base-url "${vllm_base_url}" \
  --hidden-baseline "${hidden_baseline_path}" \
  --public-baseline "${public_baseline_path}" \
  --output "${smoke_root}/preflight.json" \
  | tee "${smoke_root}/preflight.log"

SFORGE_TASKS_DIR="${TASKS_DIR}" "${SFORGE_BIN}" list \
  | grep -F "${TASK_ID}" \
  | tee "${smoke_root}/task-load.log"
docker run --rm --network none --env NVIDIA_VISIBLE_DEVICES=void \
  "${WORK_IMAGE}" bash /workspace/protocol_tests/run_protocol_smoke.sh \
  | tee "${smoke_root}/work-protocol.log"
docker run --rm --network none --entrypoint sh "${JUDGE_IMAGE}" -c \
  'docker --version && docker compose version && test ! -e /var/run/docker.sock' \
  | tee "${smoke_root}/judge-static.log"

invalid_archive="${ARTIFACTS_DIR}/invalid-source.tar.gz"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  -C "${MIGRATION_ROOT}/fixtures/invalid_submission" \
  -czf "${invalid_archive}" submission
negative_id="kvbench-negative-smoke-${stamp}"
"${SCRIPT_DIR}/eval_archive.sh" "${invalid_archive}" "${negative_id}" \
  >"${smoke_root}/negative-eval.log" 2>&1
negative_report="${LOGS_DIR}/runs/${negative_id}/${TASK_ID}/submissions/manual-1/report.json"
jq -e '
  .valid == false
  and .score == 0
  and .metrics.evaluation_status == "candidate_invalid"
' "${negative_report}" >/dev/null

if [[ "${with_oracle}" == true ]]; then
  oracle_archive="${ARTIFACTS_DIR}/oracle-submission.tar.gz"
  "${SCRIPT_DIR}/make_oracle_archive.sh" "${oracle_archive}" \
    >"${smoke_root}/make-oracle.log" 2>&1
  oracle_id="kvbench-oracle-smoke-${stamp}"
  "${SCRIPT_DIR}/eval_archive.sh" "${oracle_archive}" "${oracle_id}" \
    >"${smoke_root}/oracle-eval.log" 2>&1
  oracle_root="${LOGS_DIR}/runs/${oracle_id}/${TASK_ID}/submissions/manual-1"
  jq -e '
    .valid == true
    and (.score | type == "number" and . > 0)
    and (.metrics.frozen_direct_valid_steps_per_min | type == "number" and . > 0)
    and .score > .metrics.frozen_direct_valid_steps_per_min
    and .metrics.correctness_gate_passed == true
    and .metrics.primary_score_name == "valid_steps_per_min"
  ' "${oracle_root}/report.json" >/dev/null
  [[ -s "${oracle_root}/private_judge_artifacts.tar.gz" ]] \
    || die "Oracle evaluation omitted private evidence"
fi

[[ -z "$(docker ps --all --quiet \
  --filter label=com.docker.compose.project=kvbench-qwen36-swe192-v2-1-gpu67-validation)" ]] \
  || die "validation Compose project leaked after smoke test"

jq -n \
  --arg finished_at "$(date -u +%FT%TZ)" \
  --argjson oracle "${with_oracle}" \
  '{status:"passed", finished_at:$finished_at, oracle_hidden_executed:$oracle}' \
  >"${smoke_root}/result.json"
echo "Smoke test PASS: ${smoke_root}"
