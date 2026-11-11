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

# Layout validation runs after the inputs outside the repository (the final
# holdout, the e8 capture and the real-layer captures) have been located,
# because a fresh checkout carries none of the captures.  It still gates every
# Docker build below and re-checks the capture/manifest/recipe binding of the
# e8 capture this script is about to bake.
#
#   scripts/build_images.sh            build both publishable images from source
#   INFRATHON_PULL=1 scripts/build_images.sh pull the published Work and public Judge
#
# Either way the two host layers (images/work/Dockerfile.local,
# images/judge/Dockerfile.private) are built here and never published.
mkdir -p "${LOGS_DIR}/build"

# The final holdout is frozen, not drawn per build: the qualification draw is
# the Judge-only asset judge/final_cases.json (a development checkout keeps it at
# judge/private/final_cases.json, DEFAULT_FINAL_CASES_PATH; lib.sh picks the
# asset root otherwise).  FINAL_CASES_PATH still overrides it.  A new instance
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
[[ "${final_cases_count}" =~ ^[0-9]+$ && "${final_cases_count}" -eq 774 ]] \
  || die "final holdout manifest must contain exactly 774 cases"

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

# The e8 edge input is a real Wan2.1-T2V-1.3B activation slice bound by
# fixtures/recipe/RECIPE.json, not by one host's bytes.  Each deployment
# captures its own copy; a capture from another deployment can be reused
# through FIXTURE_PATH / FIXTURE_MANIFEST_PATH, mirroring FINAL_CASES_PATH.
# (The final holdout, by contrast, is frozen in-tree and not drawn per build.)
recipe_source="${MIGRATION_ROOT}/fixtures/recipe/RECIPE.json"
[[ -f "${recipe_source}" && ! -L "${recipe_source}" ]] \
  || die "missing fixture recipe: ${recipe_source}"
recipe_sha256="$(sha256sum "${recipe_source}" | awk '{print $1}')"
recipe_id="$(jq -er '.recipe_id' "${recipe_source}")"
fixture_name="$(jq -er '.fixture.name' "${recipe_source}")"
fixture_manifest_name="$(jq -er '.fixture.manifest' "${recipe_source}")"

# The captures come from this task's Infrathon assets by default
# (assets/fetch.sh: the reference captures, public/ and judge/ under
# <INFRATHON_ASSETS_ROOT>/<task directory>/).  A capture made on this host with the
# recipe scripts (fixtures/, gitignored) takes precedence when present.
task_assets="${INFRATHON_TASK_ASSETS:-${INFRATHON_ASSETS_ROOT:-${HOME}/.cache/infrathon/assets}/$(basename "${MIGRATION_ROOT}")}"
fixture_source="${FIXTURE_PATH:-${MIGRATION_ROOT}/fixtures/${fixture_name}}"
fixture_manifest_source="${FIXTURE_MANIFEST_PATH:-$(dirname "${fixture_source}")/${fixture_manifest_name}}"
if [[ -z "${FIXTURE_PATH:-}" && -z "${FIXTURE_MANIFEST_PATH:-}" \
      && ! -e "${fixture_source}" && -e "${task_assets}/public/${fixture_name}" ]]; then
  fixture_source="${task_assets}/public/${fixture_name}"
  fixture_manifest_source="${task_assets}/public/${fixture_manifest_name}"
fi
if [[ ! -e "${fixture_source}" || ! -e "${fixture_manifest_source}" ]]; then
  [[ -z "${FIXTURE_PATH:-}" && -z "${FIXTURE_MANIFEST_PATH:-}" ]] \
    || die "FIXTURE_PATH/FIXTURE_MANIFEST_PATH must name existing regular files"
  [[ -n "${FIXTURE_GPU:-}" ]] \
    || die "e8 capture is absent; run assets/fetch.sh (or set FIXTURE_GPU to a GPU UUID so fixtures/recipe/build_fixture.sh captures it)"
  FIXTURE_FORCE=1 "${MIGRATION_ROOT}/fixtures/recipe/build_fixture.sh"
fi
for candidate in "${fixture_source}" "${fixture_manifest_source}"; do
  [[ -f "${candidate}" && ! -L "${candidate}" ]] \
    || die "e8 capture input must be a regular non-symlink file: ${candidate}"
done
fixture_sha256="$(sha256sum "${fixture_source}" | awk '{print $1}')"
[[ "$(jq -er '.fixture' "${fixture_manifest_source}")" == "${fixture_name}" ]] \
  || die "e8 capture manifest names a different fixture"
[[ "$(jq -er '.recipe_id' "${fixture_manifest_source}")" == "${recipe_id}" ]] \
  || die "e8 capture manifest was produced by a different recipe id"
[[ "$(jq -er '.recipe_sha256' "${fixture_manifest_source}")" == "${recipe_sha256}" ]] \
  || die "e8 capture manifest does not match ${recipe_source}"
[[ "$(jq -er '.sha256' "${fixture_manifest_source}")" == "${fixture_sha256}" ]] \
  || die "e8 capture manifest does not describe the capture file"

fixtures_context="${LOGS_DIR}/build/fixtures-context"
mkdir -p "${fixtures_context}/recipe"
install -m 0644 "${fixture_source}" "${fixtures_context}/${fixture_name}"
install -m 0644 "${fixture_manifest_source}" "${fixtures_context}/${fixture_manifest_name}"
install -m 0644 "${recipe_source}" "${fixtures_context}/recipe/RECIPE.json"
[[ "$(sha256sum "${fixtures_context}/${fixture_name}" | awk '{print $1}')" \
    == "${fixture_sha256}" ]] \
  || die "e8 capture build-context copy is not byte-identical"
[[ "$(sha256sum "${fixtures_context}/recipe/RECIPE.json" | awk '{print $1}')" \
    == "${recipe_sha256}" ]] \
  || die "recipe build-context copy is not byte-identical"

# v2 real-model activation gates: every layer of CogVideoX-2B and Llama-3.1-8B,
# bound to the recipes by capture.manifest.json.  By default they are the
# reference captures of this task's assets (judge/real_layers/ for the Judge,
# public/real_layers/ for the two Work layers).  REAL_LAYERS_DIR may point at
# another set of captures (as FIXTURE_PATH does for e8); a set captured on this
# host by fixtures/recipe/real_layers/build_real_layers.sh (fixtures/real_layers,
# gitignored) takes precedence over the assets, and with REAL_LAYER_GPU the
# script captures what is missing.
real_recipe_dir="${MIGRATION_ROOT}/fixtures/recipe/real_layers"
complete_real_set() {  # DIR: every recipe's capture manifest is present
  local recipe
  for recipe in "${real_recipe_dir}"/RECIPE_*.json; do
    [[ -e "$1/$(jq -er '.capture_dir' "${recipe}")/capture.manifest.json" ]] || return 1
  done
}
if [[ -n "${REAL_LAYERS_DIR:-}" ]]; then
  real_layers_dir="${REAL_LAYERS_DIR}"
elif ! complete_real_set "${MIGRATION_ROOT}/fixtures/real_layers" \
    && complete_real_set "${task_assets}/judge/real_layers"; then
  real_layers_dir="${task_assets}/judge/real_layers"
else
  real_layers_dir="${MIGRATION_ROOT}/fixtures/real_layers"
fi
real_layer_records=()
missing_real=()
for real_recipe in "${real_recipe_dir}"/RECIPE_*.json; do
  [[ -e "${real_layers_dir}/$(jq -er '.capture_dir' "${real_recipe}")/capture.manifest.json" ]] || missing_real+=("${real_recipe}")
done
if (( ${#missing_real[@]} )); then
  [[ -z "${REAL_LAYERS_DIR:-}" ]] || die "REAL_LAYERS_DIR lacks captures for: ${missing_real[*]}"
  [[ -n "${REAL_LAYER_GPU:-}" ]] \
    || die "real-layer captures are absent; run assets/fetch.sh (or set REAL_LAYER_GPU so fixtures/recipe/real_layers/build_real_layers.sh captures them)"
  REAL_LAYERS_DIR="${real_layers_dir}" "${real_recipe_dir}/build_real_layers.sh" "${missing_real[@]}"
fi
for real_recipe in "${real_recipe_dir}"/RECIPE_*.json; do
  capture_dir="$(jq -er '.capture_dir' "${real_recipe}")"
  capture_manifest="${real_layers_dir}/${capture_dir}/capture.manifest.json"
  [[ -f "${capture_manifest}" && ! -L "${capture_manifest}" ]] || die "missing real-layer capture manifest: ${capture_manifest}"
  [[ "$(jq -er '.recipe_sha256' "${capture_manifest}")" == "$(sha256sum "${real_recipe}" | awk '{print $1}')" ]] \
    || die "real-layer capture ${capture_dir} was produced by a different recipe"
  [[ "$(jq -er '.files | length' "${capture_manifest}")" == "$(jq -er 'if (.capture.layers | type) == "array" then (.capture.layers | length) else .attention.layers end' "${real_recipe}")" ]] \
    || die "real-layer capture ${capture_dir} does not list every captured layer"
  while read -r name digest; do
    [[ -f "${real_layers_dir}/${capture_dir}/${name}" && ! -L "${real_layers_dir}/${capture_dir}/${name}" ]] \
      || die "real layer ${capture_dir}/${name} is missing"
    [[ "$(sha256sum "${real_layers_dir}/${capture_dir}/${name}" | awk '{print $1}')" == "${digest}" ]] \
      || die "real layer ${capture_dir}/${name} does not match its manifest"
  done < <(jq -r '.files | to_entries[] | "\(.key) \(.value.sha256)"' "${capture_manifest}")
  real_layer_records+=("$(jq -c --arg dir "${capture_dir}" --arg manifest_sha256 "$(sha256sum "${capture_manifest}" | awk '{print $1}')" \
    '{capture_dir: $dir, recipe_id: .recipe_id, recipe_sha256: .recipe_sha256, manifest_sha256: $manifest_sha256, layers: (.files | length), gpu_name: .gpu_name}' "${capture_manifest}")")
done
# Both images carry the recipes next to the runtime (fixtures build context).
mkdir -p "${fixtures_context}/recipe/real_layers"
for real_recipe in "${real_recipe_dir}"/RECIPE_*.json; do
  install -m 0644 "${real_recipe}" "${fixtures_context}/recipe/real_layers/$(basename "${real_recipe}")"
  cmp -s "${real_recipe}" "${fixtures_context}/recipe/real_layers/$(basename "${real_recipe}")" \
    || die "real-layer recipe build-context copy is not byte-identical"
done
# Work has its own two real-model layers (public assets, or a capture made here by
# fixtures/recipe/public_layers/build_public_layers.sh with PUBLIC_LAYERS_DIR).  The
# context is staged with the modes the Work image gives them (directories 0555,
# files 0444).
real_public_context="${LOGS_DIR}/build/real-layers-public-context"
if [[ -e "${real_public_context}" ]]; then
  chmod -R u+w -- "${real_public_context}"
  rm -rf -- "${real_public_context}"
fi
public_layers_dir="${PUBLIC_LAYERS_DIR:-${task_assets}/public/public_layers}"
public_layer_records=()
for capture_dir in public-wan22 public-lm; do
  mkdir -p "${real_public_context}/real_layers/${capture_dir}"
  for name in layer.pt capture.manifest.json; do
    [[ -f "${public_layers_dir}/${capture_dir}/${name}" && ! -L "${public_layers_dir}/${capture_dir}/${name}" ]] \
      || die "public real layer is missing: ${public_layers_dir}/${capture_dir}/${name} (run assets/fetch.sh)"
    install -m 0444 "${public_layers_dir}/${capture_dir}/${name}" "${real_public_context}/real_layers/${capture_dir}/${name}"
  done
  [[ "$(jq -er '.sha256' "${real_public_context}/real_layers/${capture_dir}/capture.manifest.json")" \
      == "$(sha256sum "${real_public_context}/real_layers/${capture_dir}/layer.pt" | awk '{print $1}')" ]] \
    || die "public real layer ${capture_dir} does not match its manifest"
  public_layer_records+=("$(jq -cn --arg dir "${capture_dir}" --arg sha "$(sha256sum "${real_public_context}/real_layers/${capture_dir}/capture.manifest.json" | awk '{print $1}')" '{capture_dir: $dir, manifest_sha256: $sha}')")
done
public_layers_json="$(printf '%s\n' "${public_layer_records[@]}" | jq -s '.')"
find "${real_public_context}" -type d -exec chmod 0555 {} +
real_layers_json="$(printf '%s\n' "${real_layer_records[@]}" | jq -s '.')"

# Validate the pair that is actually about to be baked, which may be another
# deployment's capture reached through FIXTURE_PATH/FIXTURE_MANIFEST_PATH.
"${PYTHON_BIN}" "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}" \
  --fixture "${fixture_source}" \
  --fixture-manifest "${fixture_manifest_source}"

source "${REPOSITORY_ROOT}/scripts/infrathon/images.sh"
source_digest_before="$(image_source_digest)"
if infrathon_pull_published_images "${MIGRATION_ROOT}" "${WORK_PUBLIC_IMAGE}" "${JUDGE_PUBLIC_IMAGE}" \
    "${source_digest_before}"; then
  echo "Pulled published ${INFRATHON_HARDWARE} images as ${WORK_PUBLIC_IMAGE} and ${JUDGE_PUBLIC_IMAGE}"
else
  echo "Building ${WORK_PUBLIC_IMAGE}"
  docker build \
    --tag "${WORK_PUBLIC_IMAGE}" \
    --network host \
    --build-arg HTTP_PROXY \
    --build-arg HTTPS_PROXY \
    --build-arg NO_PROXY \
    --build-arg http_proxy \
    --build-arg https_proxy \
    --build-arg no_proxy \
    --file "${MIGRATION_ROOT}/images/work/Dockerfile" \
    "${MIGRATION_ROOT}/images/work" \
    2>&1 | tee "${LOGS_DIR}/build/work-public-image.log"

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
    --build-context "fixtures=${fixtures_context}" \
    --file "${MIGRATION_ROOT}/images/judge/Dockerfile" \
    "${MIGRATION_ROOT}/images/judge" \
    2>&1 | tee "${LOGS_DIR}/build/judge-public-image.log"
fi

# The host layers are built here in every mode and never published: the two
# public real layers over the Work image, the final holdout and the 62 real
# layers over the public Judge.
echo "Building ${WORK_IMAGE}"
docker build \
  --tag "${WORK_IMAGE}" \
  --build-arg "WORK_PUBLIC_IMAGE=${WORK_PUBLIC_IMAGE}" \
  --build-context "real_layers_public=${real_public_context}" \
  --file "${MIGRATION_ROOT}/images/work/Dockerfile.local" \
  "${MIGRATION_ROOT}/images/work" \
  2>&1 | tee "${LOGS_DIR}/build/work-image.log"

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
[[ "$(sha256sum "${recipe_source}" | awk '{print $1}')" == "${recipe_sha256}" ]] \
  || die "fixture recipe changed during build"
work_id="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")"
judge_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
work_public_id="$(docker image inspect --format '{{.Id}}' "${WORK_PUBLIC_IMAGE}")"
judge_public_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_PUBLIC_IMAGE}")"

manifest="${LOGS_DIR}/build/image-manifest.json"
temporary_manifest="${manifest}.tmp.$$"
jq -n \
  --arg created_at "$(date -u +%FT%TZ)" \
  --arg source_digest "${source_digest_after}" \
  --arg work_name "${WORK_PUBLIC_IMAGE}" \
  --arg work_id "${work_public_id}" \
  --arg work_local_name "${WORK_IMAGE}" \
  --arg work_local_id "${work_id}" \
  --arg judge_name "${JUDGE_IMAGE}" \
  --arg judge_id "${judge_id}" \
  --arg judge_public_name "${JUDGE_PUBLIC_IMAGE}" \
  --arg judge_public_id "${judge_public_id}" \
  --arg final_cases_sha256 "${final_cases_sha256}" \
  --arg final_cases_suite_id "${final_cases_suite_id}" \
  --argjson final_cases_count "${final_cases_count}" \
  --arg fixture_name "${fixture_name}" \
  --arg fixture_sha256 "${fixture_sha256}" \
  --arg recipe_id "${recipe_id}" \
  --arg recipe_sha256 "${recipe_sha256}" \
  --argjson real_layers "${real_layers_json}" \
  --argjson public_layers "${public_layers_json}" \
  '{
    schema_version: 5,
    created_at: $created_at,
    source_digest: $source_digest,
    work: {name: $work_name, id: $work_id},
    work_local: {name: $work_local_name, id: $work_local_id},
    judge: {name: $judge_name, id: $judge_id},
    judge_public: {name: $judge_public_name, id: $judge_public_id},
    final_cases: {
      sha256: $final_cases_sha256,
      suite_id: $final_cases_suite_id,
      case_count: $final_cases_count
    },
    fixture: {
      name: $fixture_name,
      sha256: $fixture_sha256,
      identity: "per-deployment-capture",
      recipe: "fixtures/recipe/RECIPE.json",
      recipe_id: $recipe_id,
      recipe_sha256: $recipe_sha256
    },
    real_layers: $real_layers,
    public_layers: $public_layers
  }' >"${temporary_manifest}"
mv -f -- "${temporary_manifest}" "${manifest}"

echo "Built work=${work_id} (published work=${work_public_id})"
echo "Built judge=${judge_id} (public judge=${judge_public_id})"
echo "Final holdout=${final_cases_sha256} (${final_cases_count} cases)"
echo "e8 capture=${fixture_sha256} (recipe ${recipe_sha256})"
echo "real layers=$(jq -c '[.[] | {capture_dir, layers, manifest_sha256}]' <<<"${real_layers_json}")"
