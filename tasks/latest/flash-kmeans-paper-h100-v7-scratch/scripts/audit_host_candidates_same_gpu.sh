#!/usr/bin/env bash
set -euo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_command docker
require_command flock
require_command install
require_command jq
require_command nvidia-smi
require_command python3
require_command sha256sum
require_command tar

assert_built_images_current

audit_stamp="${AUDIT_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
[[ "${audit_stamp}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] \
  || die "unsafe AUDIT_STAMP: ${audit_stamp}"
audit_root="${LOGS_DIR}/host_audit/candidate-only-same-gpu-${audit_stamp}"
[[ ! -e "${audit_root}" ]] \
  || die "audit directory already exists: ${audit_root}"

judge_gpu="$(canonical_gpu_index "${JUDGE_GPU:-6}")"
judge_uuid="$(gpu_uuid "${judge_gpu}")"
judge_name="$(gpu_name "${judge_gpu}")"
judge_image_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
source_digest="$(image_source_digest)"

exec 9>"$(gpu_lock_path "${judge_uuid}")"
flock -n 9 || die "judge GPU ${judge_gpu} lock is already held"
assert_gpu_idle "${judge_gpu}" "same-GPU audit"
assert_container_gpu "${JUDGE_IMAGE}" "${judge_uuid}" "same-GPU audit"

mkdir -p "${audit_root}"
chmod 0755 "${audit_root}"

run_one() {
  local label="$1"
  local archive="$2"
  local output_dir="${audit_root}/${label}"
  local preserved_archive="${output_dir}/submission.tar.gz"
  local container="flash-kmeans-v7-scratch-audit-${label}-${audit_stamp}"
  local started_at
  local finished_at
  local archive_sha
  local source_before_sha
  local source_after_sha
  local wrapper_rc

  [[ -f "${archive}" && ! -L "${archive}" ]] \
    || die "missing regular archive: ${archive}"
  mkdir -p "${output_dir}/private"
  chmod 0755 "${output_dir}" "${output_dir}/private"
  install -m 0444 "${archive}" "${preserved_archive}"
  archive_sha="$(sha256sum "${preserved_archive}" | awk '{print $1}')"
  started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  docker create \
    --name "${container}" \
    --cpus 8 \
    --memory 32g \
    -e "NVIDIA_VISIBLE_DEVICES=${judge_uuid}" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e FLASH_KMEANS_TARGET=H100 \
    "${JUDGE_IMAGE}" \
    tail -f /dev/null >/dev/null
  cleanup_container() {
    docker rm -f "${container}" >/dev/null 2>&1 || true
  }
  trap cleanup_container RETURN

  docker start "${container}" >/dev/null
  docker inspect "${container}" >"${output_dir}/container_inspect.json"
  docker cp "${preserved_archive}" "${container}:/tmp/submission.tar.gz"
  docker exec \
    -u root \
    -w /app/workdir/flash_kmeans_lab \
    "${container}" \
    tar xzf /tmp/submission.tar.gz
  docker cp \
    "${container}:/app/workdir/flash_kmeans_lab/kernels/flash_kmeans_v001.py" \
    "${output_dir}/evaluated_candidate_before.py"
  source_before_sha="$(
    sha256sum "${output_dir}/evaluated_candidate_before.py" | awk '{print $1}'
  )"

  set +e
  docker exec \
    -u root \
    -w /app/workdir/flash_kmeans_lab \
    "${container}" \
    python3 /opt/flash_kmeans_judge/run_judge.py --app-dir /app \
    >"${output_dir}/test_output.txt" 2>&1
  wrapper_rc=$?
  set -e

  docker cp \
    "${container}:/tmp/flash-kmeans-verifier-private/." \
    "${output_dir}/private/"
  docker cp \
    "${container}:/app/workdir/flash_kmeans_lab/kernels/flash_kmeans_v001.py" \
    "${output_dir}/evaluated_candidate_after.py"
  source_after_sha="$(
    sha256sum "${output_dir}/evaluated_candidate_after.py" | awk '{print $1}'
  )"
  [[ "${source_before_sha}" == "${source_after_sha}" ]] \
    || die "${label} candidate source changed during evaluation"
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  jq -n \
    --arg schema_version "1" \
    --arg label "${label}" \
    --arg started_at "${started_at}" \
    --arg finished_at "${finished_at}" \
    --arg archive_path "${preserved_archive}" \
    --arg archive_sha256 "${archive_sha}" \
    --arg candidate_sha256 "${source_before_sha}" \
    --arg judge_image "${JUDGE_IMAGE}" \
    --arg judge_image_id "${judge_image_id}" \
    --arg source_digest "${source_digest}" \
    --arg gpu_index "${judge_gpu}" \
    --arg gpu_uuid "${judge_uuid}" \
    --arg gpu_name "${judge_name}" \
    --arg wrapper_rc "${wrapper_rc}" \
    '{
      schema_version: ($schema_version | tonumber),
      label: $label,
      started_at: $started_at,
      finished_at: $finished_at,
      execution: "sequential_same_gpu",
      archive_path: $archive_path,
      archive_sha256: $archive_sha256,
      evaluated_candidate_sha256: $candidate_sha256,
      judge_image: $judge_image,
      judge_image_id: $judge_image_id,
      image_source_digest: $source_digest,
      gpu: {
        physical_index: ($gpu_index | tonumber),
        uuid: $gpu_uuid,
        name: $gpu_name
      },
      container_workspace: "/app/workdir/flash_kmeans_lab",
      judge_command: [
        "python3",
        "/opt/flash_kmeans_judge/run_judge.py",
        "--app-dir",
        "/app"
      ],
      wrapper_returncode: ($wrapper_rc | tonumber)
    }' >"${output_dir}/invocation_manifest.json"

  cleanup_container
  trap - RETURN
}

baseline_archive="${BASELINE_ARCHIVE:-${ARTIFACTS_DIR}/torch-fallback-baseline.tar.gz}"
run_one "baseline" "${baseline_archive}"
run_one "official" "${ARTIFACTS_DIR}/official-oracle.tar.gz"

python3 - "${audit_root}" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


root = Path(sys.argv[1]).resolve()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


details = {
    label: load_json(root / label / "private/flash_kmeans_results.json")
    for label in ("baseline", "official")
}
manifests = {
    label: load_json(root / label / "invocation_manifest.json")
    for label in ("baseline", "official")
}
for label in ("baseline", "official"):
    archive = root / label / "submission.tar.gz"
    if sha256(archive) != manifests[label]["archive_sha256"]:
        raise SystemExit(f"{label} archive digest mismatch")
    source = root / label / "evaluated_candidate_after.py"
    if sha256(source) != manifests[label]["evaluated_candidate_sha256"]:
        raise SystemExit(f"{label} candidate digest mismatch")
if manifests["baseline"]["judge_image_id"] != manifests["official"]["judge_image_id"]:
    raise SystemExit("candidates used different Judge images")
if manifests["baseline"]["gpu"] != manifests["official"]["gpu"]:
    raise SystemExit("candidates used different GPUs")
if details["official"].get("all_gates_ok") != 1.0:
    raise SystemExit("official candidate did not pass every gate")
if details["baseline"].get("all_gates_ok") != 0.0:
    raise SystemExit("fallback unexpectedly passed every gate")

baseline_rows = {
    row["name"]: row
    for row in details["baseline"].get("cases", [])
    if isinstance(row, dict) and isinstance(row.get("name"), str)
}
official_rows = {
    row["name"]: row
    for row in details["official"].get("cases", [])
    if isinstance(row, dict) and isinstance(row.get("name"), str)
}
shared: list[dict[str, Any]] = []
for name, baseline in baseline_rows.items():
    official = official_rows.get(name)
    if official is None or float(baseline.get("weight", 0.0)) <= 0.0:
        continue
    baseline_ms = baseline.get("candidate_ms")
    official_ms = official.get("candidate_ms")
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
        for value in (baseline_ms, official_ms)
    ):
        continue
    if (
        baseline.get("non_memory_gates_ok") is not True
        or official.get("non_memory_gates_ok") is not True
    ):
        continue
    shared.append(
        {
            "stage": baseline.get("component", baseline.get("stage")),
            "shape_b_n_k_d": baseline.get("shape"),
            "task_weight": float(baseline["weight"]),
            "baseline_ms": float(baseline_ms),
            "official_ms": float(official_ms),
            "official_over_baseline_speedup": (
                float(baseline_ms) / float(official_ms)
            ),
            "baseline_all_gates_ok": baseline.get("all_gates_ok") is True,
            "baseline_memory_ok": baseline.get("memory_ok") is True,
            "official_all_gates_ok": official.get("all_gates_ok") is True,
        }
    )
shared_weight = sum(row["task_weight"] for row in shared)
if shared_weight <= 0.0:
    raise SystemExit("no shared non-memory-valid timed rows")
baseline_gm = math.exp(
    sum(
        row["task_weight"] / shared_weight * math.log(row["baseline_ms"])
        for row in shared
    )
)
official_gm = math.exp(
    sum(
        row["task_weight"] / shared_weight * math.log(row["official_ms"])
        for row in shared
    )
)
summary = {
    "schema_version": 1,
    "comparison_kind": "host_post_hoc_shared_subset",
    "authoritative_task_score": False,
    "score_effect": "none",
    "inclusion_rule": (
        "Both candidates produced positive finite latency and passed every "
        "non-memory gate. Memory failures remain explicit and invalid."
    ),
    "provenance": manifests,
    "candidate_status": {
        "baseline": {
            "valid": False,
            "accepted_score_ms": None,
            "verifier_internal_score_ms": details["baseline"].get("score"),
        },
        "official": {
            "valid": True,
            "accepted_score_ms": details["official"].get("score"),
        },
    },
    "shared_positive_weight": shared_weight,
    "rows": shared,
    "shared_subset_weight_renormalized_geomean": {
        "baseline_ms": baseline_gm,
        "official_ms": official_gm,
        "official_over_baseline_speedup": baseline_gm / official_gm,
    },
    "excluded_from_latency_comparison": {
        "assignment_and_combined_iteration": (
            "The fallback produced no positive semantically valid timed row."
        ),
        "fresh_process_deployment": (
            "The fallback produced no valid isolated deployment measurement."
        ),
    },
}
(root / "comparison_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

echo "Same-GPU host audit: ${audit_root}"
