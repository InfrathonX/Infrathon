#!/usr/bin/env bash
set -euo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_command docker
require_command jq
require_command tee
require_command python3

DOCKER_BUILD_NETWORK="${DOCKER_BUILD_NETWORK:-default}"
docker_proxy_args=()
for proxy_variable in \
  HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
  if [[ -n "${!proxy_variable:-}" ]]; then
    # Docker's predefined proxy args are intentionally omitted from image
    # history and allow a host-local proxy to be reached with --network host.
    docker_proxy_args+=(--build-arg "${proxy_variable}")
  fi
done

mkdir -p "${LOGS_DIR}/build"
python3 "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}" \
  --output "${LOGS_DIR}/build/layout-validation.json"
source_digest_before="$(image_source_digest)"

# The Judge image holds no Judge-only data: the nonce source
# (scripts/lib.sh default_nonce_source) is bind-mounted into each Judge
# container at run time, never baked.  So there is no holdout layer and the
# published public Judge is the Judge itself.  INFRATHON_PULL=1 pulls the published
# Work and public Judge (scripts/infrathon/images.sh) as WORK_IMAGE and JUDGE_IMAGE
# instead of building them; the nested images below are always built here.
source "${REPOSITORY_ROOT}/scripts/infrathon/images.sh"
pulled_work_and_judge=0
if infrathon_pull_published_images "${MIGRATION_ROOT}" "${WORK_IMAGE}" "${JUDGE_IMAGE}" "${source_digest_before}"; then
  echo "Pulled published ${INFRATHON_HARDWARE} images as ${WORK_IMAGE} and ${JUDGE_IMAGE}"
  pulled_work_and_judge=1
else
  echo "Building ${WORK_IMAGE}"
  docker build \
    --tag "${WORK_IMAGE}" \
    --file "${MIGRATION_ROOT}/images/work/Dockerfile" \
    "${MIGRATION_ROOT}/images/work" \
    2>&1 | tee "${LOGS_DIR}/build/work-image.log"
fi

echo "Building ${MODEL_API_IMAGE}"
docker build \
  --tag "${MODEL_API_IMAGE}" \
  --file "${MODEL_API_DOCKERFILE}" \
  "${NESTED_ENVIRONMENT_ROOT}" \
  2>&1 | tee "${LOGS_DIR}/build/model-api-image.log"

echo "Building ${CONTROLLER_IMAGE}"
docker build \
  --tag "${CONTROLLER_IMAGE}" \
  --file "${CONTROLLER_DOCKERFILE}" \
  "${NESTED_ENVIRONMENT_ROOT}/bench-controller" \
  2>&1 | tee "${LOGS_DIR}/build/public-controller-image.log"

echo "Building ${MAIN_IMAGE}"
docker build \
  --tag "${MAIN_IMAGE}" \
  --file "${MAIN_DOCKERFILE}" \
  "${NESTED_ENVIRONMENT_ROOT}" \
  2>&1 | tee "${LOGS_DIR}/build/main-image.log"

echo "Building ${SANDBOX_WORKER_IMAGE}"
docker build \
  --tag "${SANDBOX_WORKER_IMAGE}" \
  --file "${SANDBOX_WORKER_DOCKERFILE}" \
  "${NESTED_ENVIRONMENT_ROOT}/sandbox-worker" \
  2>&1 | tee "${LOGS_DIR}/build/sandbox-worker-image.log"

if [[ "${SKIP_BACKEND_IMAGE:-0}" != 1 ]]; then
  echo "Building ${BACKEND_IMAGE}"
  docker build \
    --tag "${BACKEND_IMAGE}" \
    --file "${BACKEND_DOCKERFILE}" \
    "${MIGRATION_ROOT}/images/backend" \
    2>&1 | tee "${LOGS_DIR}/build/backend-image.log"
fi

if (( ! pulled_work_and_judge )); then
  echo "Building ${JUDGE_IMAGE}"
  docker build \
    --network "${DOCKER_BUILD_NETWORK}" \
    "${docker_proxy_args[@]}" \
    --build-arg "WORK_IMAGE=${WORK_IMAGE}" \
    --tag "${JUDGE_IMAGE}" \
    --file "${MIGRATION_ROOT}/images/judge/Dockerfile" \
    "${MIGRATION_ROOT}" \
    2>&1 | tee "${LOGS_DIR}/build/judge-image.log"
fi

source_digest_after="$(image_source_digest)"
[[ "${source_digest_before}" == "${source_digest_after}" ]] \
  || die "image sources changed during build; discard these tags and rebuild"

work_id="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")"
judge_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
public_controller_id="$(docker image inspect --format '{{.Id}}' "${PUBLIC_CONTROLLER_IMAGE}")"
model_api_id="$(docker image inspect --format '{{.Id}}' "${MODEL_API_IMAGE}")"
main_id="$(docker image inspect --format '{{.Id}}' "${MAIN_IMAGE}")"
sandbox_worker_id="$(docker image inspect --format '{{.Id}}' "${SANDBOX_WORKER_IMAGE}")"
backend_id="$(docker image inspect --format '{{.Id}}' "${BACKEND_IMAGE}" 2>/dev/null | head -n1)"
[[ -n "${backend_id}" ]] || backend_id="null"
judge_source_digest_value="$(judge_source_digest)"
manifest_tmp="${LOGS_DIR}/build/image-manifest.json.tmp.$$"
trap 'rm -f -- "${manifest_tmp}"' EXIT
jq -n \
  --arg schema_version "1" \
  --arg created_at "$(date -u +%FT%TZ)" \
  --arg task_id "${TASK_ID}" \
  --arg benchmark "${BENCHMARK_NAME}" \
  --arg source_digest "${source_digest_after}" \
  --arg work_image "${WORK_IMAGE}" \
  --arg work_id "${work_id}" \
  --arg judge_image "${JUDGE_IMAGE}" \
  --arg judge_id "${judge_id}" \
  --arg judge_source_digest "${judge_source_digest_value}" \
  --arg public_controller_image "${PUBLIC_CONTROLLER_IMAGE}" \
  --arg public_controller_id "${public_controller_id}" \
  --arg model_api_image "${MODEL_API_IMAGE}" \
  --arg model_api_id "${model_api_id}" \
  --arg main_image "${MAIN_IMAGE}" \
  --arg main_id "${main_id}" \
  --arg sandbox_worker_image "${SANDBOX_WORKER_IMAGE}" \
  --arg sandbox_worker_id "${sandbox_worker_id}" \
  --arg backend_image "${BACKEND_IMAGE}" \
  --arg backend_id "${backend_id}" \
  '{
    schema_version: ($schema_version | tonumber),
    created_at: $created_at,
    task_id: $task_id,
    benchmark: $benchmark,
    source_digest: $source_digest,
    work: {name: $work_image, id: $work_id},
    judge: {
      name: $judge_image,
      id: $judge_id,
      source_digest: $judge_source_digest
    },
    judge_public: {name: $judge_image, id: $judge_id},
    public_controller: {
      name: $public_controller_image,
      id: $public_controller_id,
      protocol: "300s unscored warmup + 600s scored window"
    },
    model_api: {
      name: $model_api_image,
      id: $model_api_id,
      transport: "configured-http-origin"
    },
    main: {name: $main_image, id: $main_id},
    sandbox_worker: {name: $sandbox_worker_image, id: $sandbox_worker_id},
    backend: {
      name: $backend_image,
      id: (if $backend_id == "null" then null else $backend_id end),
      identity: "recipe-bound (images/backend/Dockerfile); the id is deployment provenance"
    }
  }' >"${manifest_tmp}"
mv -f -- "${manifest_tmp}" "${LOGS_DIR}/build/image-manifest.json"
trap - EXIT

echo "Built work=${work_id}"
echo "Built judge=${judge_id}"
echo "Built public_controller=${public_controller_id}"
echo "Built model_api=${model_api_id}"
echo "Built main=${main_id}"
echo "Built sandbox_worker=${sandbox_worker_id}"
echo "Built backend=${backend_id}"
echo "Source digest=${source_digest_after}"
