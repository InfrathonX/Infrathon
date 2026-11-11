#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_command docker
require_command python3

python3 "${TASK_ROOT}/scripts/validate_layout.py" "${TASK_ROOT}"
# The protected-file manifest covers both the Work base image and the Judge
# context.  Do not rewrite it during a build: a maintainer must explicitly
# regenerate and review it after any trusted runtime/Judge change.
python3 "${TASK_ROOT}/scripts/generate_protected_hashes.py" --check
mkdir -p "${LOGS_DIR}/build"

# private_data/ is Judge-only: it is staged into its own build context (from the
# committed copy, or <INFRATHON_ASSETS_ROOT>/<task directory>/judge/private_data) and
# enters only the host-built holdout layer.
private_data_source="${TASK_ROOT}/images/judge/private_data"
# Unset: the default root of assets/fetch.sh, when it exists.
[[ -n "${INFRATHON_ASSETS_ROOT:-}" || ! -d "${HOME}/.cache/infrathon/assets" ]] \
  || INFRATHON_ASSETS_ROOT="${HOME}/.cache/infrathon/assets"
if [[ -n "${INFRATHON_ASSETS_ROOT:-}" ]]; then
  private_data_source="${INFRATHON_ASSETS_ROOT}/$(basename "${TASK_ROOT}")/judge/private_data"
fi
private_data_context="${LOGS_DIR}/build/private-data-context"
python3 "${REPOSITORY_ROOT}/scripts/infrathon/assets.py" context "$(basename "${TASK_ROOT}")" \
  judge/private_data "${private_data_source}" "${private_data_context}"

source_digest="$(image_source_digest)"
source "${REPOSITORY_ROOT}/scripts/infrathon/images.sh"
if infrathon_pull_published_images "${TASK_ROOT}" "${WORK_IMAGE}" "${JUDGE_PUBLIC_IMAGE}" "${source_digest}"; then
  echo "Pulled published ${INFRATHON_HARDWARE} images as ${WORK_IMAGE} and ${JUDGE_PUBLIC_IMAGE}"
else
  docker build \
    --tag "${WORK_IMAGE}" \
    --file "${TASK_ROOT}/images/work/Dockerfile" \
    "${TASK_ROOT}/images/work" \
    2>&1 | tee "${LOGS_DIR}/build/work-image.log"

  docker build \
    --tag "${JUDGE_PUBLIC_IMAGE}" \
    --build-arg "WORK_IMAGE=${WORK_IMAGE}" \
    --file "${TASK_ROOT}/images/judge/Dockerfile" \
    "${TASK_ROOT}/images/judge" \
    2>&1 | tee "${LOGS_DIR}/build/judge-public-image.log"
fi

# The holdout layer is built here in every mode and never published.
docker build \
  --tag "${JUDGE_IMAGE}" \
  --build-arg "JUDGE_PUBLIC_IMAGE=${JUDGE_PUBLIC_IMAGE}" \
  --build-context "private_data=${private_data_context}" \
  --file "${TASK_ROOT}/images/judge/Dockerfile.private" \
  "${TASK_ROOT}/images/judge" \
  2>&1 | tee "${LOGS_DIR}/build/judge-image.log"
[[ "$(image_source_digest)" == "${source_digest}" ]] || die "image sources changed during the build"

docker image inspect --format '{{.Id}}' "${WORK_IMAGE}" > "${LOGS_DIR}/build/work-image.id"
docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}" > "${LOGS_DIR}/build/judge-image.id"
docker image inspect --format '{{.Id}}' "${JUDGE_PUBLIC_IMAGE}" > "${LOGS_DIR}/build/judge-public-image.id"
jq -n \
  --arg created_at "$(date -u +%FT%TZ)" \
  --arg source_digest "${source_digest}" \
  --arg work_name "${WORK_IMAGE}" --arg work_id "$(cat "${LOGS_DIR}/build/work-image.id")" \
  --arg judge_name "${JUDGE_IMAGE}" --arg judge_id "$(cat "${LOGS_DIR}/build/judge-image.id")" \
  --arg judge_public_name "${JUDGE_PUBLIC_IMAGE}" \
  --arg judge_public_id "$(cat "${LOGS_DIR}/build/judge-public-image.id")" \
  '{created_at: $created_at, source_digest: $source_digest,
    work: {name: $work_name, id: $work_id}, judge: {name: $judge_name, id: $judge_id},
    judge_public: {name: $judge_public_name, id: $judge_public_id}}' \
  > "${LOGS_DIR}/build/image-manifest.json"
