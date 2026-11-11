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
[[ "${final_cases_suite_id}" == "final-holdout-v3" ]] \
  || die "final holdout manifest has unexpected suite_id: ${final_cases_suite_id}"
[[ "${final_cases_count}" =~ ^[0-9]+$ && "${final_cases_count}" -eq 287 ]] \
  || die "final holdout manifest must contain exactly 287 cases"

# --- real captured layers: recipe-bound, captured with fixtures/recipe/real_layers/ ----
# REAL_LAYERS_DIR holds the Judge-only scored layers, mounted read-only at run time by
# scripts/sforge_judge_entry.py and never baked into an image.  PUBLIC_LAYER_DIR holds the
# public layer, which is part of the Work image.
real_layers_dir="${REAL_LAYERS_DIR}"
public_layer_dir="${PUBLIC_LAYER_DIR:-${DEFAULT_PUBLIC_LAYER_DIR}}"
real_recipe_sha256="$(sha256sum "${MIGRATION_ROOT}/fixtures/recipe/real_layers/REAL_LAYERS.json" | awk '{print $1}')"
real_manifest="${real_layers_dir}/real_layers.manifest.json"
[[ -f "${real_manifest}" && ! -L "${real_manifest}" ]] \
  || die "real-layer capture is missing: ${real_manifest} (run fixtures/recipe/real_layers/build_real_layers.sh)"
[[ "$(jq -er '.recipe.sha256' "${real_manifest}")" == "${real_recipe_sha256}" ]] \
  || die "real-layer capture was not produced by the shipped REAL_LAYERS.json"
real_layer_count="$(jq -er '.files | length' "${real_manifest}")"
[[ "${real_layer_count}" -eq 32 ]] || die "real-layer capture must hold 32 layers, found ${real_layer_count}"
real_v2_recipe_sha256="$(sha256sum "${MIGRATION_ROOT}/fixtures/recipe/real_layers/REAL_LAYERS_V2.json" | awk '{print $1}')"
real_v2_manifest="${real_layers_dir}/real_layers_v2.manifest.json"
[[ -f "${real_v2_manifest}" && ! -L "${real_v2_manifest}" ]] \
  || die "second real-layer capture is missing: ${real_v2_manifest} (build_real_layers.sh cogvideox2b, qwen3-14b, qwen35-4b)"
[[ "$(jq -er '.recipe.sha256' "${real_v2_manifest}")" == "${real_v2_recipe_sha256}" ]] \
  || die "second real-layer capture was not produced by the shipped REAL_LAYERS_V2.json"
real_v2_count="$(jq -er '.files | length' "${real_v2_manifest}")"
[[ "${real_v2_count}" -eq 46 ]] || die "second real-layer capture must hold 46 layers, found ${real_v2_count}"
[[ -f "${public_layer_dir}/public-video-layer.pt" && -f "${public_layer_dir}/capture.source.json" ]] \
  || die "public layer is missing under ${public_layer_dir} (run build_real_layers.sh public)"
[[ -f "${public_layer_dir}/public-llm-layer.pt" && -f "${public_layer_dir}/capture.llm.source.json" ]] \
  || die "public LLM layer is missing under ${public_layer_dir} (run build_real_layers.sh public_llm)"

# --- the e7 edge fixture: recipe-bound, captured once per deployment -----------------
# The capture is not shipped and is not pinned by digest.  If this deployment has no
# capture yet, take one now from `fixtures/recipe/` (which needs FIXTURE_GPU); a
# deployment that captured elsewhere passes FIXTURE_PATH and FIXTURE_MANIFEST_PATH,
# mirroring FINAL_CASES_PATH.
recipe_dir="${MIGRATION_ROOT}/fixtures/recipe"
recipe_path="${recipe_dir}/RECIPE.json"
[[ -f "${recipe_path}" && ! -L "${recipe_path}" ]] || die "missing fixture recipe: ${recipe_path}"
recipe_sha256="$(sha256sum "${recipe_path}" | awk '{print $1}')"
recipe_id="$(jq -er '.recipe_id' "${recipe_path}")"
fixture_name="$(jq -er '.fixture_name' "${recipe_path}")"
fixture_manifest_name="${fixture_name%.pt}.manifest.json"
fixture_destination="${MIGRATION_ROOT}/fixtures/${fixture_name}"
fixture_manifest_destination="${MIGRATION_ROOT}/fixtures/${fixture_manifest_name}"

if [[ -n "${FIXTURE_PATH:-}" || -n "${FIXTURE_MANIFEST_PATH:-}" ]]; then
  [[ -n "${FIXTURE_PATH:-}" && -n "${FIXTURE_MANIFEST_PATH:-}" ]] \
    || die "FIXTURE_PATH and FIXTURE_MANIFEST_PATH must be set together"
  [[ -f "${FIXTURE_PATH}" && ! -L "${FIXTURE_PATH}" ]] \
    || die "FIXTURE_PATH must name a regular non-symlink file"
  [[ -f "${FIXTURE_MANIFEST_PATH}" && ! -L "${FIXTURE_MANIFEST_PATH}" ]] \
    || die "FIXTURE_MANIFEST_PATH must name a regular non-symlink file"
  # Judge the supplied pair where it lies, by the same rule the staged pair is judged
  # by, so a capture that does not belong to this recipe never reaches the task tree.
  "${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}" \
    --fixture "${FIXTURE_PATH}" --fixture-manifest "${FIXTURE_MANIFEST_PATH}"
  install -m 0644 "${FIXTURE_PATH}" "${fixture_destination}"
  install -m 0644 "${FIXTURE_MANIFEST_PATH}" "${fixture_manifest_destination}"
elif [[ ! -f "${fixture_destination}" || ! -f "${fixture_manifest_destination}" ]]; then
  echo "Capturing ${fixture_name} from ${recipe_path}"
  "${recipe_dir}/build_fixture.sh"
fi
[[ -f "${fixture_destination}" && ! -L "${fixture_destination}" ]] \
  || die "e7 capture is missing: ${fixture_destination}"
[[ -f "${fixture_manifest_destination}" && ! -L "${fixture_manifest_destination}" ]] \
  || die "e7 capture manifest is missing: ${fixture_manifest_destination}"
fixture_sha256="$(sha256sum "${fixture_destination}" | awk '{print $1}')"
[[ "$(jq -er '.identity' "${fixture_manifest_destination}")" == "per-deployment-capture" ]] \
  || die "e7 capture manifest declares a different identity"
[[ "$(jq -er '.fixture.name' "${fixture_manifest_destination}")" == "${fixture_name}" ]] \
  || die "e7 capture manifest names a different fixture"
[[ "$(jq -er '.fixture.sha256' "${fixture_manifest_destination}")" == "${fixture_sha256}" ]] \
  || die "e7 capture manifest does not describe the captured bytes"
[[ "$(jq -er '.recipe.id' "${fixture_manifest_destination}")" == "${recipe_id}" ]] \
  || die "e7 capture was produced by a different recipe id"
[[ "$(jq -er '.recipe.sha256' "${fixture_manifest_destination}")" == "${recipe_sha256}" ]] \
  || die "e7 capture was not produced by the shipped RECIPE.json"
echo "e7 capture=${fixture_sha256} recipe=${recipe_sha256} (${recipe_id})"

# The first layout pass ran on a tree that legitimately carried no capture; run it
# again now that one is in place, so the pair that is about to be baked into both
# images is validated by the same rule a reviewer would apply to a shipped tree.
"${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}"

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
    --build-context "public_layer=${public_layer_dir}" \
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
    --build-context "recipe=${MIGRATION_ROOT}/fixtures/recipe" \
    --build-context "fixtures=${MIGRATION_ROOT}/fixtures" \
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
[[ "$(sha256sum "${fixture_destination}" | awk '{print $1}')" == "${fixture_sha256}" ]] \
  || die "e7 capture changed during build"
[[ "$(sha256sum "${recipe_path}" | awk '{print $1}')" == "${recipe_sha256}" ]] \
  || die "e7 recipe changed during build"
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
  --arg fixture_name "${fixture_name}" \
  --arg fixture_sha256 "${fixture_sha256}" \
  --arg recipe_id "${recipe_id}" \
  --arg recipe_sha256 "${recipe_sha256}" \
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
    },
    edge_fixture: {
      name: $fixture_name,
      sha256: $fixture_sha256,
      identity: "per-deployment-capture",
      recipe_id: $recipe_id,
      recipe_sha256: $recipe_sha256
    }
  }' >"${temporary_manifest}"
mv -f -- "${temporary_manifest}" "${manifest}"

echo "Built work=${work_id}"
echo "Built judge=${judge_id} (public judge=${judge_public_id})"
echo "Final holdout=${final_cases_sha256} (${final_cases_count} cases)"
echo "Edge fixture=${fixture_sha256} recipe=${recipe_sha256}"
