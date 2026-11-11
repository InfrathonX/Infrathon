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

# The final holdout is frozen, not drawn per build: the qualification draw is
# committed at judge/private/final_cases.json (DEFAULT_FINAL_CASES_PATH) and is
# the default build input.  FINAL_CASES_PATH still overrides it.  A new instance
# is a new suite identity and must be minted deliberately with
# scripts/generate_final_manifest.py --redraw --seed <int>.
final_cases_source="${FINAL_CASES_PATH:-${DEFAULT_FINAL_CASES_PATH}}"
[[ -f "${final_cases_source}" && ! -L "${final_cases_source}" ]] \
  || die "final holdout manifest must name a regular non-symlink file: ${final_cases_source}"
final_cases_source="$(realpath "${final_cases_source}")"
final_cases_sha256="$(sha256sum "${final_cases_source}" | awk '{print $1}')"
final_cases_suite_id="$(jq -er '.suite_id' "${final_cases_source}")"
final_cases_count="$(jq -er '.cases | length' "${final_cases_source}")"
[[ "${final_cases_suite_id}" == "tilelang-ssd-final-v3" ]] \
  || die "final holdout manifest has unexpected suite_id: ${final_cases_suite_id}"
[[ "${final_cases_count}" =~ ^[0-9]+$ && "${final_cases_count}" -eq 138 ]] \
  || die "final manifest must contain exactly 138 cases"

# --- captured layers (fixtures/recipe/real_layers/build_real_layers.sh) -----------------
# REAL_LAYERS_DIR holds the Judge-only layers of the real cases and goes into the host-built
# Judge holdout layer; PUBLIC_LAYER_DIR holds the public layer and goes into the Work image.
real_recipe_sha256="$(sha256sum "${MIGRATION_ROOT}/fixtures/recipe/real_layers/REAL_LAYERS.json" | awk '{print $1}')"
for layer_dir in "${REAL_LAYERS_DIR}" "${PUBLIC_LAYER_DIR}"; do
  layer_manifest="${layer_dir}/real_layers.manifest.json"
  [[ -f "${layer_manifest}" && ! -L "${layer_manifest}" ]] \
    || die "captured layers are missing: ${layer_manifest} (run fixtures/recipe/real_layers/build_real_layers.sh)"
  [[ "$(jq -er '.recipe.sha256' "${layer_manifest}")" == "${real_recipe_sha256}" ]] \
    || die "captured layers in ${layer_dir} were not produced by the shipped REAL_LAYERS.json"
done
[[ "$(jq -er '.files | length' "${REAL_LAYERS_DIR}/real_layers.manifest.json")" -eq 24 ]] \
  || die "the Judge capture must hold 24 layers"
[[ "$(jq -er '.files | keys | join(",")' "${PUBLIC_LAYER_DIR}/real_layers.manifest.json")" == "public-real-layer16" ]] \
  || die "the public capture must hold exactly public-real-layer16"

final_cases_context="${LOGS_DIR}/build/final-cases-context"
mkdir -p "${final_cases_context}"
install -m 0600 "${final_cases_source}" "${final_cases_context}/final_cases.json"
# Fail-closed checks on the staged copy: the tracked source is a normal
# repository file, but nothing but a 0600 file owned by the building user may
# enter the Judge build context.
[[ "$(stat -c '%a' "${final_cases_context}/final_cases.json")" == "600" ]] \
  || die "staged final holdout must have mode 600"
[[ "$(stat -c '%u' "${final_cases_context}/final_cases.json")" == "$(id -u)" ]] \
  || die "staged final holdout must be owned by the building user"
[[ "$(sha256sum "${final_cases_context}/final_cases.json" | awk '{print $1}')" \
    == "${final_cases_sha256}" ]] \
  || die "final holdout build-context copy is not byte-identical"

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
    --build-context "public_layer=${PUBLIC_LAYER_DIR}" \
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
  --build-context "real_layers=${REAL_LAYERS_DIR}" \
  --file "${MIGRATION_ROOT}/images/judge/Dockerfile.private" \
  "${MIGRATION_ROOT}/images/judge" \
  2>&1 | tee "${LOGS_DIR}/build/judge-image.log"

source_digest_after="$(image_source_digest)"
[[ "${source_digest_before}" == "${source_digest_after}" ]] \
  || die "image-driving source changed during build"
[[ "$(sha256sum "${final_cases_source}" | awk '{print $1}')" \
    == "${final_cases_sha256}" ]] \
  || die "final holdout manifest changed during build"
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
  '{
    schema_version: 2,
    created_at: $created_at,
    source_digest: $source_digest,
    work: {name: $work_name, id: $work_id},
    judge: {name: $judge_name, id: $judge_id},
    judge_public: {name: $judge_public_name, id: $judge_public_id},
    final_cases: {
      sha256: $final_cases_sha256,
      suite_id: $final_cases_suite_id,
      case_count: $final_cases_count
    }
  }' >"${temporary_manifest}"
mv -f -- "${temporary_manifest}" "${manifest}"

echo "Built work=${work_id}"
echo "Built judge=${judge_id} (public judge=${judge_public_id})"
echo "Final holdout=${final_cases_sha256} (${final_cases_count} cases)"
