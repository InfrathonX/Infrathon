#!/usr/bin/env bash
# Build and immutably record the Work/Judge images. Helper C2 only syntax-
# checks this script; helper D performs the actual build and qualification.
set -Eeuo pipefail

main() {
SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
(( $# == 0 )) || die "usage: build_images.sh"

ensure_docker_group "$@"
for required_command in docker jq sha256sum; do
  require_command "${required_command}"
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/validate_layout.py" "${MIGRATION_ROOT}" >/dev/null
"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_protected_hashes.py" --check \
  --private-data "${EAGLE3_PRIVATE_DATA}"
mkdir -p "${BUILD_DIR}"

source_digest="$(image_source_digest)"
# The default arch list covers the whole capability envelope (sm_80 through
# sm_100). Narrow it only when the fleet is known to be single-architecture.
work_build_args=()
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  work_build_args+=(--build-arg "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}")
fi
# private_data/ is Judge-only: it is staged into its own build context (from
# EAGLE3_PRIVATE_DATA, resolved by lib.sh) and enters only the host-built
# holdout layer.
private_data_source="${EAGLE3_PRIVATE_DATA}"
private_data_context="${BUILD_DIR}/private-data-context"
"${PYTHON_BIN}" "${REPOSITORY_ROOT}/scripts/infrathon/assets.py" context "$(basename "${MIGRATION_ROOT}")" \
  judge/private_data "${private_data_source}" "${private_data_context}"

source "${REPOSITORY_ROOT}/scripts/infrathon/images.sh"
if infrathon_pull_published_images "${MIGRATION_ROOT}" "${WORK_IMAGE}" "${JUDGE_PUBLIC_IMAGE}" "${source_digest}"; then
  echo "Pulled published ${INFRATHON_HARDWARE} images as ${WORK_IMAGE} and ${JUDGE_PUBLIC_IMAGE}"
else
  docker build \
    --pull=false \
    "${work_build_args[@]+"${work_build_args[@]}"}" \
    --tag "${WORK_IMAGE}" \
    "${MIGRATION_ROOT}/images/work"
  docker build \
    --pull=false \
    --build-arg "WORK_IMAGE=${WORK_IMAGE}" \
    --tag "${JUDGE_PUBLIC_IMAGE}" \
    "${MIGRATION_ROOT}/images/judge"
fi
# The holdout layer is built here in every mode and never published.
docker build \
  --pull=false \
  --build-arg "JUDGE_PUBLIC_IMAGE=${JUDGE_PUBLIC_IMAGE}" \
  --build-context "private_data=${private_data_context}" \
  --file "${MIGRATION_ROOT}/images/judge/Dockerfile.private" \
  --tag "${JUDGE_IMAGE}" \
  "${MIGRATION_ROOT}/images/judge"

work_id="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")"
judge_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
judge_public_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_PUBLIC_IMAGE}")"
[[ "${work_id}" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || die "Work image has an invalid immutable ID"
[[ "${judge_id}" =~ ^sha256:[0-9a-f]{64}$ ]] \
  || die "Judge image has an invalid immutable ID"

manifest="${BUILD_DIR}/image-manifest.json"
jq -n \
  --arg built_at "$(date -u +%FT%TZ)" \
  --arg source_digest "${source_digest}" \
  --arg work_name "${WORK_IMAGE}" \
  --arg work_id "${work_id}" \
  --arg judge_name "${JUDGE_IMAGE}" \
  --arg judge_id "${judge_id}" \
  --arg judge_public_name "${JUDGE_PUBLIC_IMAGE}" \
  --arg judge_public_id "${judge_public_id}" \
  --arg protected_hashes_sha256 "$(sha256sum "${MIGRATION_ROOT}/images/judge/protected_hashes.txt" | awk '{print $1}')" \
  --arg policy_sha256 "$(sha256sum "${private_data_context}/policy.json" | awk '{print $1}')" \
  '{schema_version:1,built_at:$built_at,source_digest:$source_digest,
    work:{name:$work_name,id:$work_id},judge:{name:$judge_name,id:$judge_id},
    judge_public:{name:$judge_public_name,id:$judge_public_id},
    protected_hashes_sha256:$protected_hashes_sha256,policy_sha256:$policy_sha256}' \
  >"${manifest}.tmp"
mv -f -- "${manifest}.tmp" "${manifest}"
chmod 444 "${manifest}"

inspect_record="${BUILD_DIR}/image-inspect.json"
docker image inspect "${work_id}" "${judge_id}" \
  >"${inspect_record}.tmp"
mv -f -- "${inspect_record}.tmp" "${inspect_record}"
chmod 444 "${inspect_record}"
printf 'built %s (%s)\n' "${WORK_IMAGE}" "${work_id}"
printf 'built %s (%s)\n' "${JUDGE_IMAGE}" "${judge_id}"
printf 'built %s (%s)\n' "${JUDGE_PUBLIC_IMAGE}" "${judge_public_id}"
}

main "$@"
