#!/usr/bin/env bash
set -euo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_sforge
require_command docker
require_command jq
require_command sha256sum

"${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}"
mkdir -p "${LOGS_DIR}/build"

# The final holdout (per-cell context-length slot and 63-bit seed) and the
# private edge cases are frozen, not drawn per build: both are committed under
# judge/private/ (DEFAULT_*_CASES_PATH) and are the default build inputs.
# FINAL_CASES_PATH / EDGE_CASES_PATH still override them.  A new instance is a
# new suite identity and must be minted deliberately
# (scripts/generate_final_manifest.py --redraw, scripts/generate_edge_manifest.py).
# Both are staged 0600 into the Judge-only build context; the public edge cases
# (images/work/benchmark/public_edge_cases.json) ship in the Work context.
stage_private_manifest() {
  local source="$1" expected_suite="$2" expected_count="$3" staged="$4"
  [[ -f "${source}" && ! -L "${source}" ]] \
    || die "private manifest must name a regular non-symlink file: ${source}"
  source="$(realpath "${source}")"
  local digest suite count
  digest="$(sha256sum "${source}" | awk '{print $1}')"
  suite="$(jq -er '.suite_id' "${source}")"
  count="$(jq -er '.cases | length' "${source}")"
  [[ "${suite}" == "${expected_suite}" ]] \
    || die "private manifest ${source} has unexpected suite_id: ${suite}"
  [[ "${count}" =~ ^[0-9]+$ && "${count}" -eq "${expected_count}" ]] \
    || die "private manifest ${source} must contain exactly ${expected_count} cases"
  install -m 0600 "${source}" "${staged}"
  # Fail-closed checks on the staged copy: nothing but a 0600 file owned by the
  # building user may enter the Judge build context.
  [[ "$(stat -c '%a' "${staged}")" == "600" ]] || die "staged ${staged} must have mode 600"
  [[ "$(stat -c '%u' "${staged}")" == "$(id -u)" ]] \
    || die "staged ${staged} must be owned by the building user"
  [[ "$(sha256sum "${staged}" | awk '{print $1}')" == "${digest}" ]] \
    || die "build-context copy of ${source} is not byte-identical"
  printf '%s %s %s\n' "${source}" "${digest}" "${count}"
}

final_cases_context="${LOGS_DIR}/build/final-cases-context"
rm -rf "${final_cases_context}"
mkdir -p "${final_cases_context}"
read -r final_cases_source final_cases_sha256 final_cases_count < <(
  stage_private_manifest "${FINAL_CASES_PATH:-${DEFAULT_FINAL_CASES_PATH}}" \
    "${FINAL_SUITE_ID}" "${FINAL_CASE_COUNT}" "${final_cases_context}/final_cases.json"
)
read -r edge_cases_source edge_cases_sha256 edge_cases_count < <(
  stage_private_manifest "${EDGE_CASES_PATH:-${DEFAULT_EDGE_CASES_PATH}}" \
    "${EDGE_SUITE_ID}" "${EDGE_CASE_COUNT}" "${final_cases_context}/edge_cases.json"
)
[[ -n "${final_cases_sha256}" && -n "${edge_cases_sha256}" ]] \
  || die "private manifests were not staged"
final_cases_suite_id="${FINAL_SUITE_ID}"
public_edge_cases="${MIGRATION_ROOT}/images/work/benchmark/public_edge_cases.json"
[[ -f "${public_edge_cases}" && ! -L "${public_edge_cases}" ]] \
  || die "public edge-case manifest must be a regular non-symlink file"
[[ "$(jq -er '.suite_id' "${public_edge_cases}")" == "${PUBLIC_EDGE_SUITE_ID}" \
    && "$(jq -er '.cases | length' "${public_edge_cases}")" -eq "${PUBLIC_EDGE_CASE_COUNT}" ]] \
  || die "public edge-case manifest identity or case count differs"

source "${REPOSITORY_ROOT}/scripts/infrathon/images.sh"
source_digest_before="$(image_source_digest)"
if infrathon_pull_published_images "${MIGRATION_ROOT}" "${WORK_IMAGE}" "${JUDGE_PUBLIC_IMAGE}" \
    "${source_digest_before}"; then
  echo "Pulled published ${INFRATHON_HARDWARE} images as ${WORK_IMAGE} and ${JUDGE_PUBLIC_IMAGE}"
else
  echo "Building ${WORK_IMAGE}"
  docker build \
    --tag "${WORK_IMAGE}" \
    --network host \
    --build-arg HTTP_PROXY \
    --build-arg HTTPS_PROXY \
    --build-arg NO_PROXY \
    --build-arg http_proxy \
    --build-arg https_proxy \
    --build-arg no_proxy \
    --file "${MIGRATION_ROOT}/images/work/Dockerfile" \
    "${MIGRATION_ROOT}/images/work" \
    2>&1 | tee "${LOGS_DIR}/build/work-image.log"

  echo "Building ${JUDGE_PUBLIC_IMAGE}"
  docker build \
    --tag "${JUDGE_PUBLIC_IMAGE}" \
    --network host \
    --build-arg HTTP_PROXY \
    --build-arg HTTPS_PROXY \
    --build-arg NO_PROXY \
    --build-arg http_proxy \
    --build-arg https_proxy \
    --build-arg no_proxy \
    --build-context "task_runtime=${MIGRATION_ROOT}/runtime" \
    --file "${MIGRATION_ROOT}/images/judge/Dockerfile" \
    "${MIGRATION_ROOT}/images/judge" \
    2>&1 | tee "${LOGS_DIR}/build/judge-public-image.log"
fi

# The holdout layer is built here in every mode and never published.
echo "Building ${JUDGE_IMAGE}"
docker build \
  --tag "${JUDGE_IMAGE}" \
  --build-arg "JUDGE_PUBLIC_IMAGE=${JUDGE_PUBLIC_IMAGE}" \
  --build-context "final_cases=${final_cases_context}" \
  --file "${MIGRATION_ROOT}/images/judge/Dockerfile.private" \
  "${MIGRATION_ROOT}/images/judge" \
  2>&1 | tee "${LOGS_DIR}/build/judge-image.log"

source_digest_after="$(image_source_digest)"
[[ "${source_digest_before}" == "${source_digest_after}" ]] \
  || die "image-driving source changed during build"
[[ "$(sha256sum "${final_cases_source}" | awk '{print $1}')" \
    == "${final_cases_sha256}" ]] \
  || die "final holdout manifest changed during build"
[[ "$(sha256sum "${edge_cases_source}" | awk '{print $1}')" \
    == "${edge_cases_sha256}" ]] \
  || die "edge-case manifest changed during build"
work_id="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")"
judge_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
judge_public_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_PUBLIC_IMAGE}")"

manifest="${LOGS_DIR}/build/image-manifest.json"
temporary_manifest="${manifest}.tmp.$$"
jq -n \
  --arg created_at "$(date -u +%FT%TZ)" \
  --arg source_digest "${source_digest_after}" \
  --arg work_name "${WORK_IMAGE}" \
  --arg work_id "${work_id}" \
  --arg judge_name "${JUDGE_IMAGE}" \
  --arg judge_id "${judge_id}" \
  --arg judge_public_name "${JUDGE_PUBLIC_IMAGE}" \
  --arg judge_public_id "${judge_public_id}" \
  --arg final_cases_sha256 "${final_cases_sha256}" \
  --arg final_cases_suite_id "${final_cases_suite_id}" \
  --argjson final_cases_count "${final_cases_count}" \
  --arg edge_cases_sha256 "${edge_cases_sha256}" \
  --arg edge_cases_suite_id "${EDGE_SUITE_ID}" \
  --argjson edge_cases_count "${edge_cases_count}" \
  '{
    schema_version: 3,
    created_at: $created_at,
    source_digest: $source_digest,
    work: {name: $work_name, id: $work_id},
    judge: {name: $judge_name, id: $judge_id},
    judge_public: {name: $judge_public_name, id: $judge_public_id},
    final_cases: {
      sha256: $final_cases_sha256,
      suite_id: $final_cases_suite_id,
      case_count: $final_cases_count
    },
    edge_cases: {
      sha256: $edge_cases_sha256,
      suite_id: $edge_cases_suite_id,
      case_count: $edge_cases_count
    }
  }' >"${temporary_manifest}"
mv -f -- "${temporary_manifest}" "${manifest}"

echo "Built work=${work_id}"
echo "Built judge=${judge_id} (public judge=${judge_public_id})"
echo "Final holdout=${final_cases_sha256} (${final_cases_count} cases)"
echo "Edge holdout=${edge_cases_sha256} (${edge_cases_count} cases)"
