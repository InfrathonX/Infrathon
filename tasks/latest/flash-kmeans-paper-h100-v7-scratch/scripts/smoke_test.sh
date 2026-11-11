#!/usr/bin/env bash
set -euo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_sforge
require_command flock
require_command jq
require_command nvidia-smi
require_command python3
require_command tar

python3 "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}"
python3 -m unittest discover -s "${MIGRATION_ROOT}/tests" -v

assert_built_images_current

if docker run --rm --entrypoint /bin/bash "${WORK_IMAGE}" -lc \
    'test ! -e /opt/flash_kmeans_judge \
      && test ! -e /opt/flash_kmeans_benchmark \
      && test ! -e /workspace/benchmark \
      && test -r /workspace/public_tests/public_test.py \
      && test -r /workspace/public_tests/public_fresh_worker.py \
      && test -x /workspace/public_tests/run_public_tests.sh \
      && test ! -e /tests \
      && test ! -e /solution \
      && test ! -e /home/agent/.codex/auth.json'; then
  echo "Work image isolation: PASS"
else
  die "work image boundary is wrong: private material is visible or public tests are missing"
fi
if docker run --rm --user agent --entrypoint /bin/bash "${JUDGE_IMAGE}" -lc \
    '! sudo -n true >/dev/null 2>&1 \
      && test ! -r /opt/flash_kmeans_judge/verify_flash_kmeans.py \
      && test ! -r /opt/flash_kmeans_benchmark/kernel_bench.py \
      && ! ls /opt/flash_kmeans_benchmark >/dev/null 2>&1'; then
  echo "Judge candidate privilege boundary: PASS"
else
  die "judge candidate can use sudo or inspect private Judge code"
fi

work_gpu="$(canonical_gpu_index "${WORK_GPU:-1}")"
judge_gpu="$(canonical_gpu_index "${JUDGE_GPU:-2}")"
work_uuid="$(gpu_uuid "${work_gpu}")"
judge_uuid="$(gpu_uuid "${judge_gpu}")"
[[ "${work_uuid}" != "${judge_uuid}" ]] \
  || die "work and judge resolve to the same physical GPU: ${work_uuid}"
exec 8>"$(gpu_lock_path "${work_uuid}")"
exec 9>"$(gpu_lock_path "${judge_uuid}")"
flock -n 8 || die "work GPU ${work_gpu} lock is already held"
flock -n 9 || die "judge GPU ${judge_gpu} lock is already held"
assert_gpu_idle "${work_gpu}" "work"
assert_gpu_idle "${judge_gpu}" "judge"
assert_container_gpu "${WORK_IMAGE}" "${work_uuid}" "work"
assert_container_gpu "${JUDGE_IMAGE}" "${judge_uuid}" "judge"

"${MIGRATION_ROOT}/scripts/make_host_calibration_archives.sh" >/dev/null
official_archive="${MIGRATION_ROOT}/artifacts/official-oracle.tar.gz"
official_adapter="${MIGRATION_ROOT}/host_reference/official_oracle/generated_task_adapter.py"
official_workspace="$(mktemp -d)"
cleanup_official_workspace() {
  chmod -R u+w "${official_workspace}" 2>/dev/null || true
  rm -rf "${official_workspace}"
}
trap cleanup_official_workspace EXIT
tar -C "${official_workspace}" -xzf "${official_archive}"
find "${official_workspace}" -type d -exec chmod 0555 {} +
find "${official_workspace}" -type f -exec chmod 0444 {} +

docker run --rm \
  -e "NVIDIA_VISIBLE_DEVICES=${work_uuid}" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e FLASH_KMEANS_TARGET=H100 \
  --volume \
  "${MIGRATION_ROOT}/tests/gpu_numerical_regression.py:/tmp/gpu_numerical_regression.py:ro" \
  --volume \
  "${official_adapter}:/tmp/flash_kmeans_oracle.py:ro" \
  --entrypoint python3 \
  "${JUDGE_IMAGE}" \
  /tmp/gpu_numerical_regression.py \
  --oracle /tmp/flash_kmeans_oracle.py

run_public_candidate() {
  local label="$1"
  shift
  local stamp
  stamp="$(date -u +%Y%m%d-%H%M%S)"
  local results_dir="${LOGS_DIR}/public-full-${label}-${stamp}"
  local run_log="${LOGS_DIR}/public-full-${label}-${stamp}.log"
  mkdir -p "${results_dir}"
  chmod 0777 "${results_dir}"

  docker run --rm \
    -e "NVIDIA_VISIBLE_DEVICES=${work_uuid}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e FLASH_KMEANS_TARGET=H100 \
    --volume "${results_dir}:/workspace/public_tests/results" \
    "$@" \
    --entrypoint /bin/bash \
    "${WORK_IMAGE}" \
    -lc 'bash /workspace/public_tests/run_public_tests.sh' \
    2>&1 | tee "${run_log}"

  local result="${results_dir}/public_results.json"
  [[ -s "${result}" ]] \
    || die "${label} full public suite did not produce public_results.json"
  python3 - "${result}" "${label}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
label = sys.argv[2]
if result.get("authoritative") is not False:
    raise SystemExit(f"{label} public result claims authority: {result}")
if "score" in result:
    raise SystemExit(f"{label} public result must not expose a score: {result}")
if result.get("all_public_gates_ok") is not True:
    raise SystemExit(f"{label} full public gates failed: {result}")
tests = result.get("tests")
performance = result.get("performance")
if not isinstance(tests, list) or len(tests) != 3:
    raise SystemExit(f"{label} public golden suite is incomplete: {tests!r}")
if not isinstance(performance, list) or len(performance) != 3:
    raise SystemExit(
        f"{label} candidate-only probes are incomplete: {performance!r}"
    )
stages = {row.get("stage") for row in performance}
if stages != {"assign", "update", "full"}:
    raise SystemExit(
        f"{label} public probe stages are incomplete: {sorted(stages)!r}"
    )
for row in performance:
    if row.get("status") != "passed":
        raise SystemExit(f"{label} public performance gate failed: {row!r}")
    if "candidate-only" not in str(row.get("note")):
        raise SystemExit(f"{label} public row lacks candidate-only label: {row!r}")
deployment = result.get("deployment_cases")
if not isinstance(deployment, list) or len(deployment) != 5:
    raise SystemExit(f"{label} fresh-process ladder is incomplete: {deployment!r}")
if len({row.get("pid") for row in deployment}) != len(deployment):
    raise SystemExit(f"{label} public ladder did not use a fresh process per row")
for row in deployment:
    if (
        row.get("status") != "passed"
        or row.get("measurement_mode") != "fresh_process"
        or row.get("cache_was_empty") is not True
        or row.get("correctness", {}).get("status") != "passed"
    ):
        raise SystemExit(f"{label} fresh-process row failed: {row!r}")
ladders = result.get("scale_ladders")
if not isinstance(ladders, list) or len(ladders) != 2:
    raise SystemExit(f"{label} scale trends are incomplete: {ladders!r}")
for ladder in ladders:
    if (
        ladder.get("status") != "available"
        or ladder.get("scaling_band")
        not in {"healthy", "moderate", "poor", "pathological"}
    ):
        raise SystemExit(f"{label} scale trend is unavailable: {ladder!r}")
serialized = json.dumps(result, sort_keys=True)
for forbidden in ("baseline_ms", "speedup", "denominator"):
    if forbidden in serialized:
        raise SystemExit(f"{label} public result leaks {forbidden!r}: {result}")
print(f"{label} visible public tests: PASS (diagnostic, no score)")
PY
}

run_public_candidate "official-oracle" \
  --volume \
  "${official_workspace}:/app/workdir/flash_kmeans_lab:ro"

smoke_id="official-oracle-smoke-$(date -u +%Y%m%d-%H%M%S)"

export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${judge_uuid},CUDA_VISIBLE_DEVICES=0,FLASH_KMEANS_TARGET=H100,FLASH_KMEANS_VERIFY_SUITE=smoke,FLASH_KMEANS_VERIFY_WARMUP=1,FLASH_KMEANS_VERIFY_ITERS=2,FLASH_KMEANS_VERIFY_REPEATS=1"
"${SFORGE_BIN}" \
  eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${official_archive}" \
  --run-id "${smoke_id}" \
  --timeout 900 \
  --judge-cpu-limit 8 \
  --judge-mem-limit 32g \
  --json | tee "${LOGS_DIR}/${smoke_id}.log"

report="${LOGS_DIR}/runs/${smoke_id}/${TASK_ID}/submissions/manual-1/report.json"
[[ -f "${report}" ]] || die "official-oracle smoke report was not produced"
python3 - "${report}" <<'PY'
import json
import math
import sys

report = json.load(open(sys.argv[1]))
if report.get("valid") is not True:
    raise SystemExit(f"official-oracle smoke was invalid: {report}")
if report.get("timed_out") is not False:
    raise SystemExit(f"official-oracle smoke timed out: {report}")
metrics = report.get("metrics")
if not isinstance(metrics, dict):
    raise SystemExit(f"official-oracle smoke lacks adapter metrics: {report}")
if metrics.get("source") != "flash_kmeans_h100_verifier_adapter":
    raise SystemExit(f"unexpected official-oracle smoke adapter source: {metrics!r}")
for field in ("candidate_output_suppressed", "verifier_completed"):
    if metrics.get(field) is not True:
        raise SystemExit(
            f"official-oracle smoke adapter {field} is not true: {metrics!r}"
        )
for field, expected in (
    ("adapter_schema_version", 8),
    ("feedback_schema_version", 7),
    ("result_protocol_records", 1),
    ("verifier_returncode", 0),
):
    value = metrics.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise SystemExit(
            f"official-oracle smoke adapter {field} must be integer "
            f"{expected}: {metrics!r}"
        )
score = report.get("score")
if (
    isinstance(score, bool)
    or not isinstance(score, (int, float))
    or not math.isfinite(float(score))
    or float(score) <= 0.0
    or report.get("pass_rate") != 1.0
):
    raise SystemExit(f"official-oracle smoke did not pass: {report}")
print(f"Official repository hidden smoke: PASS, score={float(score):.6f}")
PY

assert_invalid_fixture() {
  local fixture_name="$1"
  local run_prefix="$2"
  local fixture_root="${MIGRATION_ROOT}/tests/fixtures/${fixture_name}"
  local archive="${ARTIFACTS_DIR}/${fixture_name}.tar.gz"
  local run_id="${run_prefix}-$(date -u +%Y%m%d-%H%M%S)"
  [[ -d "${fixture_root}/kernels" ]] \
    || die "missing negative smoke fixture: ${fixture_root}"
  mkdir -p "${ARTIFACTS_DIR}"
  tar -C "${fixture_root}" -czf "${archive}" \
    kernels FINAL_VERSION notes.md

  "${SFORGE_BIN}" \
    eval \
    --backend docker \
    --task "${TASK_ID}" \
    --archive "${archive}" \
    --run-id "${run_id}" \
    --timeout 900 \
    --judge-cpu-limit 8 \
    --judge-mem-limit 32g \
    --json | tee "${LOGS_DIR}/${run_id}.log"

  local report="${LOGS_DIR}/runs/${run_id}/${TASK_ID}/submissions/manual-1/report.json"
  [[ -f "${report}" ]] \
    || die "${fixture_name} smoke report was not produced"
  python3 - "${report}" "${fixture_name}" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
name = sys.argv[2]
if report.get("valid") is not False:
    raise SystemExit(f"{name} was incorrectly valid: {report}")
score = report.get("score")
if score is not None:
    raise SystemExit(f"{name} invalid score must be null: {report}")
metrics = report.get("metrics")
details = report.get("details")
if not isinstance(metrics, dict) or not isinstance(details, list) or not details:
    raise SystemExit(f"{name} lacks structured feedback: {report}")
if metrics.get("adapter_schema_version") != 8:
    raise SystemExit(f"{name} did not use adapter schema v8: {metrics}")
if metrics.get("feedback_schema_version") != 7:
    raise SystemExit(f"{name} lacks feedback schema v7: {metrics}")
if metrics.get("evaluation_outcome") != "gate_failed":
    raise SystemExit(f"{name} did not report an intentional gate failure: {metrics}")
if metrics.get("rollback_required") is not True:
    raise SystemExit(f"{name} did not require FINAL_VERSION rollback: {metrics}")
family_feedback = metrics.get("deployment_family_feedback")
if not isinstance(family_feedback, dict) or set(family_feedback) != {
    "large-K", "large-N/high-D"
}:
    raise SystemExit(f"{name} lacks coarse invalid family status: {metrics}")
for family, value in family_feedback.items():
    if not isinstance(value, dict) or value.get("status") not in {
        "passed", "failed", "not_reached"
    }:
        raise SystemExit(f"{name} has invalid {family} family status: {metrics}")
    if "warmed_latency_band_ms" in value:
        raise SystemExit(f"{name} invalid result leaked a family latency band")
expected_version = {"invalid-source": "v999", "incorrect-output": "v998"}[name]
if metrics.get("selected_final_version") != expected_version:
    raise SystemExit(f"{name} selected version feedback is wrong: {metrics}")
summary = report.get("summary")
if not isinstance(summary, str) or f"FINAL_VERSION={expected_version}" not in summary:
    raise SystemExit(f"{name} summary omits FINAL_VERSION: {summary!r}")
if (
    "restore FINAL_VERSION" not in summary
    or "simplest locally correct version" not in summary
):
    raise SystemExit(f"{name} summary omits actionable rollback: {summary!r}")
passed = sum(item.get("status") == "PASSED" for item in details)
if not math.isclose(
    float(report.get("pass_rate")),
    passed / len(details),
    rel_tol=0.0,
    abs_tol=1e-12,
):
    raise SystemExit(f"{name} pass_rate conflicts with detail counts: {report}")
serialized = json.dumps(report, sort_keys=True)
for private_case in (
    "hidden_smoke_assign",
    "hidden_smoke_update",
    "hidden_smoke_full",
    "hidden_smoke_tie_gate",
    "hidden_smoke_cold_1m_1k_d128",
):
    if private_case in serialized:
        raise SystemExit(f"{name} leaked private case name {private_case}")
if name == "incorrect-output":
    correctness = next(
        (item for item in details if item.get("name") == "correctness"),
        None,
    )
    if (
        not isinstance(correctness, dict)
        or correctness.get("status") != "FAILED"
        or not any(
            stage in str(correctness.get("message"))
            for stage in ("assignment", "centroid update", "combined iteration")
        )
    ):
        raise SystemExit(f"{name} lacks stage-level correctness feedback: {details}")
print(
    f"{name}: PASS (valid=false, score=null, "
    f"feedback={metrics['selected_final_version']})"
)
PY
}

# Exercise independent source-policy and numerical-correctness fail-closed paths.
assert_invalid_fixture "invalid-source" "invalid-source-smoke"
assert_invalid_fixture "incorrect-output" "incorrect-output-smoke"
