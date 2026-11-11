#!/usr/bin/env bash
set -Eeuo pipefail
SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_command docker
require_command jq

# Only the base build reaches the network (the registry, PyPI and the pinned
# flash-attention release).  Deployments behind an egress proxy export the
# usual proxy variables; forward them as Docker's predefined build args so
# they reach the RUN steps without being baked into the image.
BUILD_PROXY_ARGS=()
for proxy_variable in \
  HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; do
  if [[ -n "${!proxy_variable:-}" ]]; then
    BUILD_PROXY_ARGS+=(--build-arg "${proxy_variable}=${!proxy_variable}")
  fi
done

# The Work base is built from this repository: images/work/base/Dockerfile
# starts at a digest-pinned public PyTorch image and installs the fully
# pinned closure in images/work/base/requirements.lock.  BASE_WORK_IMAGE may
# still name an externally supplied base, which is then only validated.
mkdir -p "${LOGS_DIR}/build"
# Recorded in the build manifest so the Judge-only launchers refuse an image
# built from older sources (the Judge bakes the controller and the release).
source_digest="$(image_source_digest)"
# INFRATHON_PULL=1: pull the published Work and public Judge for this host's hardware
# type (images/published.json) instead of building the base, Work and public Judge.
source "${REPOSITORY_ROOT}/scripts/infrathon/images.sh"
pulled=0
if infrathon_pull_published_images "${TASK_ROOT}" "${WORK_IMAGE}" "${JUDGE_PUBLIC_IMAGE}" "${source_digest}"; then
  pulled=1
  echo "Pulled published ${INFRATHON_HARDWARE} images as ${WORK_IMAGE} and ${JUDGE_PUBLIC_IMAGE}"
else
  if [[ -z "${BASE_WORK_IMAGE:-}" || "${BASE_WORK_IMAGE}" == "${DEFAULT_BASE_WORK_IMAGE}" ]]; then
    BASE_WORK_IMAGE="${DEFAULT_BASE_WORK_IMAGE}"
    docker build ${BUILD_PROXY_ARGS[@]+"${BUILD_PROXY_ARGS[@]}"} \
      -f "${TASK_ROOT}/images/work/base/Dockerfile" \
      -t "${BASE_WORK_IMAGE}" "${TASK_ROOT}" \
      2>&1 | tee "${LOGS_DIR}/build/base-image.log"
  fi
  docker image inspect "${BASE_WORK_IMAGE}" >/dev/null \
    || die "missing Work base image: ${BASE_WORK_IMAGE}"

  docker build --build-arg "BASE_WORK_IMAGE=${BASE_WORK_IMAGE}" \
    -f "${TASK_ROOT}/images/work/Dockerfile" -t "${WORK_IMAGE}" "${TASK_ROOT}" \
    2>&1 | tee "${LOGS_DIR}/build/work-image.log"
  docker build -f "${TASK_ROOT}/images/judge/Dockerfile" \
    --build-arg "WORK_IMAGE=${WORK_IMAGE}" -t "${JUDGE_PUBLIC_IMAGE}" "${TASK_ROOT}" \
    2>&1 | tee "${LOGS_DIR}/build/judge-public-image.log"
fi
# The release config is Judge-only: staged into its own build context from
# RELEASE_PATH (scripts/lib.sh: the committed copy, else
# <asset root>/<task directory>/judge/h100_scratch_v4_release.json) and added by
# the host-built holdout layer, which is never published.  This host's quality
# file (scripts/lib.sh quality_path: hw/<type>/judge/h100_scratch_v4_quality.json)
# joins it there; only a calibrating release may go without one.
release_source="${RELEASE_PATH}"
release_context="${LOGS_DIR}/build/release-context"
python3 "${REPOSITORY_ROOT}/scripts/infrathon/assets.py" context "$(basename "${TASK_ROOT}")" \
  judge/h100_scratch_v4_release.json "${release_source}" "${release_context}"
quality_source="$(quality_path)"
if [[ -f "${quality_source}" ]]; then
  quality_context="${LOGS_DIR}/build/quality-context"
  quality_asset="${quality_source#"${ASSET_TASK_ROOT}/"}"
  [[ "${quality_asset}" != "${quality_source}" ]] || quality_asset="hw/$(python3 -I \
    "${REPOSITORY_ROOT}/scripts/infrathon/assets.py" hardware)/judge/h100_scratch_v4_quality.json"
  python3 "${REPOSITORY_ROOT}/scripts/infrathon/assets.py" context "$(basename "${TASK_ROOT}")" \
    "${quality_asset}" "${quality_source}" "${quality_context}"
  install -m 0600 "${quality_context}/h100_scratch_v4_quality.json" "${release_context}/"
  echo "Judge quality reference: ${quality_asset}"
elif [[ "$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["status"])' "${RELEASE_PATH}")" == calibrated ]]; then
  die "the calibrated release needs this device class's quality file: ${quality_source} (run assets/fetch.sh)"
else
  echo "Judge quality reference: none (calibrating release)"
fi
docker build -f "${TASK_ROOT}/images/judge/Dockerfile.private" \
  --build-arg "JUDGE_PUBLIC_IMAGE=${JUDGE_PUBLIC_IMAGE}" --build-context "release=${release_context}" \
  -t "${JUDGE_IMAGE}" "${TASK_ROOT}/images/judge" \
  2>&1 | tee "${LOGS_DIR}/build/judge-image.log"
# Every image in the lineage must be agent-free: the agent CLI arrives at run
# time as a host-provided runtime bind-mounted read-only at /opt/agent-runtime.
# A pulled Work image comes without its base.
lineage=("${WORK_IMAGE}" "${JUDGE_PUBLIC_IMAGE}" "${JUDGE_IMAGE}")
(( pulled )) || lineage=("${BASE_WORK_IMAGE}" "${lineage[@]}")
for image in "${lineage[@]}"; do
  docker run --rm --network none --entrypoint bash "${image}" -c \
      'for tool in codex node npm; do
         if command -v "${tool}" >/dev/null; then
           echo "unexpected agent toolchain in image: ${tool}" >&2
           exit 1
         fi
       done' \
    || die "built image is not agent-free: ${image}"
done
if (( pulled )); then
  docker image inspect "${WORK_IMAGE}" "${JUDGE_IMAGE}" > "${LOGS_DIR}/build/images.json"
  base_tag="" base_id=""
else
  docker image inspect "${BASE_WORK_IMAGE}" "${WORK_IMAGE}" "${JUDGE_IMAGE}" \
    > "${LOGS_DIR}/build/images.json"
  base_tag="${BASE_WORK_IMAGE}"
  base_id="$(docker image inspect --format '{{.Id}}' "${BASE_WORK_IMAGE}")"
fi
[[ "$(image_source_digest)" == "${source_digest}" ]] \
  || die "image-driving source changed during the build; rebuild"
jq -n \
  --arg task_id "${TASK_ID}" \
  --arg source_digest "${source_digest}" \
  --arg base_tag "${base_tag}" \
  --arg base_id "${base_id}" \
  --arg work_tag "${WORK_IMAGE}" \
  --arg work_id "$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")" \
  --arg judge_tag "${JUDGE_IMAGE}" \
  --arg judge_id "$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")" \
  --arg judge_public_tag "${JUDGE_PUBLIC_IMAGE}" \
  --arg judge_public_id "$(docker image inspect --format '{{.Id}}' "${JUDGE_PUBLIC_IMAGE}")" \
  '{schema_version: 1, task_id: $task_id, source_digest: $source_digest,
    base: (if $base_id == "" then null else {tag: $base_tag, id: $base_id} end),
    work: {tag: $work_tag, id: $work_id},
    judge: {tag: $judge_tag, id: $judge_id},
    judge_public: {tag: $judge_public_tag, id: $judge_public_id}}' \
  > "${IMAGE_MANIFEST}.tmp"
mv -f -- "${IMAGE_MANIFEST}.tmp" "${IMAGE_MANIFEST}"
echo "Image manifest: ${IMAGE_MANIFEST} (source ${source_digest:0:12})"
