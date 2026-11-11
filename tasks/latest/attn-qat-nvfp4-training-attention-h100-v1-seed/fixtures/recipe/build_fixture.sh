#!/usr/bin/env bash
# Capture the e7 edge fixture for this deployment, from the recipe.
#
# The fixture is recipe-bound, not byte-bound: every deployment runs this once at
# build time on its own accelerator and keeps its own capture, exactly the way the
# final holdout is drawn per build.  `RECIPE.json` freezes the identity (model
# revision, upstream commit, prompt, seeds, geometry, block, sigma, slicing and the
# sanity bands); `capture_e7_wan_grad.py` is the executable part.
#
#   FIXTURE_GPU=GPU-xxxxxxxx fixtures/recipe/build_fixture.sh
#
# Environment:
#   FIXTURE_GPU            required, the accelerator UUID (or index) to capture on
#   HF_HOME                honoured for the model snapshot cache
#   FIXTURE_OUT_DIR        where the fixture and its manifest land (default fixtures/)
#   FIXTURE_BUILDER_IMAGE  tag for the throwaway builder image
#   KEEP_BUILDER_IMAGE=1   do not delete the builder image afterwards
#   HTTP_PROXY/HTTPS_PROXY forwarded to the image build and to `hf download`
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFORGE_ENTRY_SCRIPT="${SFORGE_ENTRY_SCRIPT:-$0}"
source "${SCRIPT_DIR}/../../scripts/lib.sh"
ensure_docker_group "$@"
require_command docker
require_command jq
require_command sha256sum
require_command hf

RECIPE_PATH="${SCRIPT_DIR}/RECIPE.json"
CAPTURE_PATH="${SCRIPT_DIR}/capture_e7_wan_grad.py"
[[ -f "${RECIPE_PATH}" && ! -L "${RECIPE_PATH}" ]] || die "missing recipe: ${RECIPE_PATH}"
[[ -f "${CAPTURE_PATH}" && ! -L "${CAPTURE_PATH}" ]] || die "missing capture: ${CAPTURE_PATH}"

fixture_gpu="${FIXTURE_GPU:-}"
[[ -n "${fixture_gpu}" ]] \
  || die "set FIXTURE_GPU to the accelerator UUID this deployment may capture on"

fixture_name="$(jq -er '.fixture_name' "${RECIPE_PATH}")"
recipe_id="$(jq -er '.recipe_id' "${RECIPE_PATH}")"
model_repo="$(jq -er '.model.repo_id' "${RECIPE_PATH}")"
model_revision="$(jq -er '.model.revision' "${RECIPE_PATH}")"
upstream_repo="$(jq -er '.upstream.repo' "${RECIPE_PATH}")"
upstream_commit="$(jq -er '.upstream.commit' "${RECIPE_PATH}")"
recipe_sha256="$(sha256sum "${RECIPE_PATH}" | awk '{print $1}')"

out_dir="${FIXTURE_OUT_DIR:-${MIGRATION_ROOT}/fixtures}"
mkdir -p "${out_dir}"
out_dir="$(cd "${out_dir}" && pwd)"
fixture_path="${out_dir}/${fixture_name}"
manifest_path="${out_dir}/${fixture_name%.pt}.manifest.json"

builder_image="${FIXTURE_BUILDER_IMAGE:-attn_qat_fixture_builder.${recipe_id}:local}"
base_image="$(awk '/^FROM /{print $2; exit}' "${MIGRATION_ROOT}/images/work/Dockerfile")"
[[ -n "${base_image}" ]] || die "cannot read the task base image from images/work/Dockerfile"

# ---------------------------------------------------------------- model snapshot ---
# `hf download` is a no-op when the pinned revision is already in the cache, so a
# deployment that already holds the weights never touches the network.
echo "Fetching ${model_repo}@${model_revision}"
# Older CLIs print the snapshot directory alone; 1.x prints `path=<directory>`.
snapshot="$(hf download "${model_repo}" --revision "${model_revision}" \
  | grep -v '^[[:space:]]*$' | tail -n 1 | sed 's/^path=//')"
[[ -n "${snapshot}" && -d "${snapshot}" ]] \
  || die "hf download did not produce a snapshot directory: ${snapshot:-<empty>}"
snapshot="$(cd "${snapshot}" && pwd)"
echo "Snapshot: ${snapshot}"

# In a Hugging Face cache every snapshot entry is a symlink into the sibling `blobs/`
# directory, so the container has to see the whole repository directory, not just the
# snapshot leaf.  A plain directory of weights is mounted as-is.
snapshot_repository="$(dirname "$(dirname "${snapshot}")")"
if [[ -d "${snapshot_repository}/blobs" ]]; then
  snapshot_mount="${snapshot_repository}:/models/repository:ro"
  snapshot_in_container="/models/repository/snapshots/$(basename "${snapshot}")"
else
  snapshot_mount="${snapshot}:/models/snapshot:ro"
  snapshot_in_container="/models/snapshot"
fi

while read -r relative expected; do
  [[ -n "${relative}" ]] || continue
  [[ -f "${snapshot}/${relative}" ]] || die "snapshot is missing ${relative}"
  actual="$(sha256sum "${snapshot}/${relative}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] \
    || die "snapshot file ${relative} has sha256 ${actual}, expected ${expected}"
done < <(jq -r '.model.files | to_entries[] | "\(.key) \(.value)"' "${RECIPE_PATH}")
echo "Snapshot digests: PASS"

# ---------------------------------------------------------------- builder image ---
context="$(mktemp -d)"
staging="${out_dir}/.capture.$$"
cleanup() {
  rm -rf -- "${context}" "${staging}"
  if [[ "${KEEP_BUILDER_IMAGE:-0}" != "1" ]]; then
    docker image rm -f "${builder_image}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

install -m 0644 "${CAPTURE_PATH}" "${context}/capture_e7_wan_grad.py"
install -m 0644 "${RECIPE_PATH}" "${context}/RECIPE.json"
cat >"${context}/Dockerfile" <<'BUILDER'
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG UPSTREAM_REPO
ARG UPSTREAM_COMMIT
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Pinned so the capture environment is reproducible.  diffusers supplies the
# ConfigMixin/ModelMixin the upstream transformer subclasses and the flow-matching
# Euler scheduler the training step reads sigma/timestep from; the rest are the
# upstream import requirements.  No fused attention package is installed.
RUN pip install --no-cache-dir \
      "numpy==1.26.4" \
      "diffusers==0.31.0" \
      "transformers==4.49.0" \
      "accelerate==1.1.1" \
      "safetensors==0.4.5" \
      "easydict==1.13" \
      "ftfy==6.3.1" \
      "regex==2024.11.6" \
      "einops==0.8.0" \
      "imageio==2.36.1" \
      "tqdm==4.67.1"

RUN git clone --no-checkout "${UPSTREAM_REPO}" /opt/upstream/Wan2.1 \
    && git -C /opt/upstream/Wan2.1 checkout --detach "${UPSTREAM_COMMIT}" \
    && git -C /opt/upstream/Wan2.1 rev-parse HEAD >/opt/upstream/Wan2.1/UPSTREAM_COMMIT \
    && rm -rf /opt/upstream/Wan2.1/.git

COPY capture_e7_wan_grad.py /opt/recipe/capture_e7_wan_grad.py
COPY RECIPE.json /opt/recipe/RECIPE.json
WORKDIR /opt/recipe
BUILDER

echo "Building ${builder_image}"
docker build \
  --tag "${builder_image}" \
  --network host \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "UPSTREAM_REPO=${upstream_repo}" \
  --build-arg "UPSTREAM_COMMIT=${upstream_commit}" \
  --build-arg HTTP_PROXY \
  --build-arg HTTPS_PROXY \
  --build-arg NO_PROXY \
  --build-arg http_proxy \
  --build-arg https_proxy \
  --build-arg no_proxy \
  "${context}"

# ---------------------------------------------------------------------- capture ---
rm -rf -- "${staging}"
mkdir -p "${staging}"
echo "Capturing on ${fixture_gpu}"
docker run --rm \
  -e "NVIDIA_VISIBLE_DEVICES=${fixture_gpu}" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e "PYTHONHASHSEED=0" \
  -e HOME=/tmp \
  -u "$(id -u):$(id -g)" \
  -v "${snapshot_mount}" \
  -v "${staging}:/out" \
  --entrypoint python \
  "${builder_image}" \
  /opt/recipe/capture_e7_wan_grad.py \
  --recipe /opt/recipe/RECIPE.json \
  --upstream /opt/upstream/Wan2.1 \
  --snapshot "${snapshot_in_container}" \
  --out "/out/${fixture_name}" \
  --manifest "/out/${fixture_name%.pt}.manifest.json"

[[ -f "${staging}/${fixture_name}" ]] || die "capture produced no fixture"
[[ -f "${staging}/${fixture_name%.pt}.manifest.json" ]] || die "capture produced no manifest"

captured_sha256="$(sha256sum "${staging}/${fixture_name}" | awk '{print $1}')"
declared_sha256="$(jq -er '.fixture.sha256' "${staging}/${fixture_name%.pt}.manifest.json")"
declared_recipe="$(jq -er '.recipe.sha256' "${staging}/${fixture_name%.pt}.manifest.json")"
declared_id="$(jq -er '.recipe.id' "${staging}/${fixture_name%.pt}.manifest.json")"
[[ "${captured_sha256}" == "${declared_sha256}" ]] \
  || die "manifest does not describe the captured bytes"
[[ "${declared_recipe}" == "${recipe_sha256}" ]] \
  || die "manifest recipe digest ${declared_recipe} != ${recipe_sha256}"
[[ "${declared_id}" == "${recipe_id}" ]] || die "manifest recipe id mismatch"

install -m 0644 "${staging}/${fixture_name}" "${fixture_path}"
install -m 0644 "${staging}/${fixture_name%.pt}.manifest.json" "${manifest_path}"
rm -rf -- "${staging}"

echo "Fixture=${fixture_path} sha256=${captured_sha256}"
echo "Manifest=${manifest_path} recipe=${recipe_sha256} (${recipe_id})"
