#!/usr/bin/env bash
set -euo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_command jq
require_command python3

mkdir -p "${LOGS_DIR}/build"

cp "${MIGRATION_ROOT}/judge/run_judge.py" \
   "${MIGRATION_ROOT}/images/judge/run_judge.py"
cp "${MIGRATION_ROOT}/images/judge/verifier/source_admission.py" \
   "${MIGRATION_ROOT}/images/work/workdir/decoder_lab/runtime/source_policy.py"
# The quality reference is only comparable within one measurement profile.
# The Judge carries every calibrated profile under reference_cache/ and selects
# the one for the GPU it observes, so one image serves NVL and SXM; refuse early
# when the Judge GPU this host will use has none.
if [[ -n "${JUDGE_GPU:-}" ]]; then
  python3 "${MIGRATION_ROOT}/scripts/select_reference_profile.py" \
    --judge-gpu "${JUDGE_GPU}" --check --allow-pending
fi
python3 "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}"
source_digest_before="$(image_source_digest)"

# private_data/ is Judge-only: it is staged into its own build context (from
# <INFRATHON_ASSETS_ROOT>/<task directory>/judge/private_data when INFRATHON_ASSETS_ROOT is
# set, else the committed copy, else the default asset root) and enters only the
# host-built holdout layer.
private_data_source="$(judge_private_data_dir --prefer-asset-root)"
private_data_context="${LOGS_DIR}/build/private-data-context"
python3 "${REPOSITORY_ROOT}/scripts/infrathon/assets.py" context "$(basename "${MIGRATION_ROOT}")" \
  judge/private_data "${private_data_source}" "${private_data_context}"

source "${REPOSITORY_ROOT}/scripts/infrathon/images.sh"
if infrathon_pull_published_images "${MIGRATION_ROOT}" "${WORK_IMAGE}" "${JUDGE_PUBLIC_IMAGE}" \
    "${source_digest_before}"; then
  echo "Pulled published ${INFRATHON_HARDWARE} images as ${WORK_IMAGE} and ${JUDGE_PUBLIC_IMAGE}"
else
  echo "Building ${WORK_IMAGE}"
  docker build \
    --tag "${WORK_IMAGE}" \
    --file "${MIGRATION_ROOT}/images/work/Dockerfile" \
    "${MIGRATION_ROOT}/images/work" \
    2>&1 | tee "${LOGS_DIR}/build/work-image.log"

  echo "Building ${JUDGE_PUBLIC_IMAGE}"
  docker build \
    --tag "${JUDGE_PUBLIC_IMAGE}" \
    --build-arg "WORK_IMAGE=${WORK_IMAGE}" \
    --file "${MIGRATION_ROOT}/images/judge/Dockerfile" \
    "${MIGRATION_ROOT}/images/judge" \
    2>&1 | tee "${LOGS_DIR}/build/judge-public-image.log"
fi

# The holdout layer is built here in every mode and never published.
echo "Building ${JUDGE_IMAGE}"
docker build \
  --tag "${JUDGE_IMAGE}" \
  --build-arg "JUDGE_PUBLIC_IMAGE=${JUDGE_PUBLIC_IMAGE}" \
  --build-context "private_data=${private_data_context}" \
  --file "${MIGRATION_ROOT}/images/judge/Dockerfile.private" \
  "${MIGRATION_ROOT}/images/judge" \
  2>&1 | tee "${LOGS_DIR}/build/judge-image.log"

source_digest_after="$(image_source_digest)"
[[ "${source_digest_before}" == "${source_digest_after}" ]] \
  || die "image sources changed during the build; discard and rebuild"
work_id="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")"
judge_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
judge_public_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_PUBLIC_IMAGE}")"
created_at="$(date -u +%FT%TZ)"
jq -n \
  --arg created_at "${created_at}" \
  --arg work_image "${WORK_IMAGE}" \
  --arg work_id "${work_id}" \
  --arg judge_image "${JUDGE_IMAGE}" \
  --arg judge_id "${judge_id}" \
  --arg judge_public_image "${JUDGE_PUBLIC_IMAGE}" \
  --arg judge_public_id "${judge_public_id}" \
  --arg source_digest "${source_digest_after}" \
  '{
    created_at: $created_at,
    source_digest: $source_digest,
    work: {name: $work_image, id: $work_id},
    judge: {name: $judge_image, id: $judge_id},
    judge_public: {name: $judge_public_image, id: $judge_public_id}
  }' > "${LOGS_DIR}/build/image-manifest.json"

echo "Built work=${work_id}"
echo "Built judge=${judge_id} (public judge=${judge_public_id})"
