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

# The final holdout is a frozen manifest (judge/private/final_manifest.json).
# Its items are rebuilt here from the pinned Hugging Face parquet files and
# every item's SHA-256 is checked against the manifest; a mismatch aborts the
# build.  The rebuilt file is staged 0600 into a build context only the Judge
# image reads.
[[ -f "${FINAL_MANIFEST_PATH}" && ! -L "${FINAL_MANIFEST_PATH}" ]] || die "missing final manifest: ${FINAL_MANIFEST_PATH}"
manifest_sha256="$(sha256sum "${FINAL_MANIFEST_PATH}" | awk '{print $1}')"
[[ "$(jq -er '.suite_id' "${FINAL_MANIFEST_PATH}")" == "final-holdout-v2" ]] || die "final manifest has an unexpected suite_id"
item_count="$(jq -er '.item_count' "${FINAL_MANIFEST_PATH}")"
[[ "${item_count}" == "5000" ]] || die "final manifest must contain 5000 items, found ${item_count}"

rm -rf -- "${FINAL_ITEMS_CONTEXT}"; mkdir -p "${FINAL_ITEMS_CONTEXT}"; chmod 0700 "${FINAL_ITEMS_CONTEXT}"
"$(data_python)" "${MIGRATION_ROOT}/scripts/prepare_items.py" materialize \
  --hf-hub "${KV_EVICTION_HF_HUB}" --manifest "${FINAL_MANIFEST_PATH}" --out "${FINAL_ITEMS_CONTEXT}/final_items.jsonl.gz"
install -m 0600 "${FINAL_MANIFEST_PATH}" "${FINAL_ITEMS_CONTEXT}/final_manifest.json"
chmod 0600 "${FINAL_ITEMS_CONTEXT}/final_items.jsonl.gz"
items_sha256="$(sha256sum "${FINAL_ITEMS_CONTEXT}/final_items.jsonl.gz" | awk '{print $1}')"
[[ "$(sha256sum "${FINAL_ITEMS_CONTEXT}/final_manifest.json" | awk '{print $1}')" == "${manifest_sha256}" ]] \
  || die "final manifest build-context copy is not byte-identical"

build_args=(--network host --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy)
source_digest_before="$(image_source_digest)"
source "${REPOSITORY_ROOT}/scripts/infrathon/images.sh"
if infrathon_pull_published_images "${MIGRATION_ROOT}" "${WORK_IMAGE}" "${JUDGE_PUBLIC_IMAGE}" \
    "${source_digest_before}"; then
  echo "Pulled published ${INFRATHON_HARDWARE} images as ${WORK_IMAGE} and ${JUDGE_PUBLIC_IMAGE}"
else
  echo "Building ${WORK_IMAGE}"
  docker build --tag "${WORK_IMAGE}" "${build_args[@]}" \
    --build-context "task_runtime=${MIGRATION_ROOT}/runtime" \
    --build-context "corpus_public=${MIGRATION_ROOT}/corpus/public" \
    --file "${MIGRATION_ROOT}/images/work/Dockerfile" \
    "${MIGRATION_ROOT}/images/work" 2>&1 | tee "${LOGS_DIR}/build/work-image.log"

  echo "Building ${JUDGE_PUBLIC_IMAGE}"
  docker build --tag "${JUDGE_PUBLIC_IMAGE}" "${build_args[@]}" \
    --build-context "task_runtime=${MIGRATION_ROOT}/runtime" \
    --build-context "corpus_public=${MIGRATION_ROOT}/corpus/public" \
    --file "${MIGRATION_ROOT}/images/judge/Dockerfile" \
    "${MIGRATION_ROOT}/images/judge" 2>&1 | tee "${LOGS_DIR}/build/judge-public-image.log"
fi

# The holdout layer is built here in every mode and never published.
echo "Building ${JUDGE_IMAGE}"
docker build --tag "${JUDGE_IMAGE}" \
  --build-arg "JUDGE_PUBLIC_IMAGE=${JUDGE_PUBLIC_IMAGE}" \
  --build-context "final_items=${FINAL_ITEMS_CONTEXT}" \
  --file "${MIGRATION_ROOT}/images/judge/Dockerfile.private" \
  "${MIGRATION_ROOT}/images/judge" 2>&1 | tee "${LOGS_DIR}/build/judge-image.log"

source_digest_after="$(image_source_digest)"
[[ "${source_digest_before}" == "${source_digest_after}" ]] || die "image-driving source changed during build"
[[ "$(sha256sum "${FINAL_MANIFEST_PATH}" | awk '{print $1}')" == "${manifest_sha256}" ]] || die "final manifest changed during build"
work_id="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")"
judge_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
judge_public_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_PUBLIC_IMAGE}")"

manifest="${LOGS_DIR}/build/image-manifest.json"
jq -n \
  --arg created_at "$(date -u +%FT%TZ)" --arg source_digest "${source_digest_after}" \
  --arg work_name "${WORK_IMAGE}" --arg work_id "${work_id}" --arg judge_name "${JUDGE_IMAGE}" --arg judge_id "${judge_id}" \
  --arg judge_public_name "${JUDGE_PUBLIC_IMAGE}" --arg judge_public_id "${judge_public_id}" \
  --arg manifest_sha256 "${manifest_sha256}" --arg items_sha256 "${items_sha256}" --argjson item_count "${item_count}" \
  '{schema_version: 3, created_at: $created_at, source_digest: $source_digest,
    work: {name: $work_name, id: $work_id}, judge: {name: $judge_name, id: $judge_id},
    judge_public: {name: $judge_public_name, id: $judge_public_id},
    final_suite: {suite_id: "final-holdout-v2", manifest_sha256: $manifest_sha256, items_sha256: $items_sha256, item_count: $item_count}}' \
  >"${manifest}.tmp.$$"
mv -f -- "${manifest}.tmp.$$" "${manifest}"

echo "Built work=${work_id}"
echo "Built judge=${judge_id} (public judge=${judge_public_id})"
echo "Final holdout: manifest ${manifest_sha256}, items ${items_sha256} (${item_count} items)"
