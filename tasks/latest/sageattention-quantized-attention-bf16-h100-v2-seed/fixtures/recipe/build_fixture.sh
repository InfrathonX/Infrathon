#!/usr/bin/env bash
# Capture this deployment's copy of the e8 edge fixture from fixtures/recipe/RECIPE.json.
#
# The fixture is a real Wan2.1-T2V-1.3B activation slice.  It is bound by this
# recipe, not by the SHA-256 of one host's bytes: every deployment captures its
# own copy, exactly as the final holdout is drawn per build.  Everything the
# capture needs is public and pinned -- the Hugging Face snapshot revision and
# the upstream Wan2.1 commit named in RECIPE.json.
#
#   FIXTURE_GPU=GPU-xxxxxxxx-... fixtures/recipe/build_fixture.sh
#
# Writes, next to each other in the task's fixtures/ directory:
#   e8_wan13b_p0_b15_s00_cond_8192.pt
#   e8_wan13b_p0_b15_s00_cond_8192.manifest.json
#
# Environment:
#   FIXTURE_GPU       (required) GPU UUID handed to the builder container.
#   HF_HOME           honoured for the snapshot download (default ~/.cache/huggingface).
#   HF_CLI            Hugging Face CLI to use (default: hf).
#   FIXTURE_WORK_DIR  scratch root for the upstream clone and builder context.
#   BUILDER_IMAGE     tag for the throwaway builder image.
#   FIXTURE_FORCE=1   replace an existing fixture/manifest pair.

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIXTURES_DIR="$(cd "${RECIPE_DIR}/.." && pwd)"
TASK_ROOT="$(cd "${FIXTURES_DIR}/.." && pwd)"
RECIPE="${RECIPE_DIR}/RECIPE.json"

die() { echo "ERROR: $*" >&2; exit 1; }
require_command() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

require_command docker
require_command jq
require_command sha256sum
require_command git

[[ -f "${RECIPE}" && ! -L "${RECIPE}" ]] || die "missing recipe: ${RECIPE}"
[[ -n "${FIXTURE_GPU:-}" ]] \
  || die "set FIXTURE_GPU to the GPU UUID the capture may use (nvidia-smi -L)"

HF_CLI="${HF_CLI:-hf}"
require_command "${HF_CLI}"

recipe_sha256="$(sha256sum "${RECIPE}" | awk '{print $1}')"
recipe_id="$(jq -er '.recipe_id' "${RECIPE}")"
fixture_name="$(jq -er '.fixture.name' "${RECIPE}")"
manifest_name="$(jq -er '.fixture.manifest' "${RECIPE}")"
model_repo="$(jq -er '.model.repo' "${RECIPE}")"
model_revision="$(jq -er '.model.revision' "${RECIPE}")"
upstream_repo="$(jq -er '.upstream.repo' "${RECIPE}")"
upstream_commit="$(jq -er '.upstream.commit' "${RECIPE}")"
base_image="$(jq -er '.builder.base_image' "${RECIPE}")"
attention_backend="$(jq -er '.builder.attention_backend' "${RECIPE}")"

fixture_path="${FIXTURES_DIR}/${fixture_name}"
manifest_path="${FIXTURES_DIR}/${manifest_name}"
if [[ -e "${fixture_path}" || -e "${manifest_path}" ]]; then
  [[ "${FIXTURE_FORCE:-0}" == "1" ]] \
    || die "fixture already present; set FIXTURE_FORCE=1 to recapture"
  rm -f -- "${fixture_path}" "${manifest_path}"
fi

WORK_DIR="${FIXTURE_WORK_DIR:-${TASK_ROOT}/build/fixture-capture}"
mkdir -p "${WORK_DIR}"
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
BUILDER_IMAGE="${BUILDER_IMAGE:-sageattention-e8-fixture-builder:${recipe_id}}"

echo "recipe        ${recipe_id} (sha256 ${recipe_sha256})"
echo "model         ${model_repo}@${model_revision}"
echo "upstream      ${upstream_repo}@${upstream_commit}"
echo "gpu           ${FIXTURE_GPU}"

# ---------------------------------------------------------------------------
# 1. Pinned upstream checkout (public source, resolved at build time).
# ---------------------------------------------------------------------------
upstream_dir="${WORK_DIR}/wan2.1"
if [[ ! -d "${upstream_dir}/.git" ]]; then
  rm -rf -- "${upstream_dir}"
  git clone --quiet "${upstream_repo}" "${upstream_dir}"
fi
git -C "${upstream_dir}" fetch --quiet origin "${upstream_commit}" 2>/dev/null \
  || git -C "${upstream_dir}" fetch --quiet origin
git -C "${upstream_dir}" -c advice.detachedHead=false checkout --quiet "${upstream_commit}"
git -C "${upstream_dir}" clean -qfdx
actual_commit="$(git -C "${upstream_dir}" rev-parse HEAD)"
[[ "${actual_commit}" == "${upstream_commit}" ]] \
  || die "upstream checkout is ${actual_commit}, recipe pins ${upstream_commit}"
echo "upstream checkout ${actual_commit}"

# ---------------------------------------------------------------------------
# 2. Pinned Hugging Face snapshot (only the files the recipe names).
# ---------------------------------------------------------------------------
mapfile -t model_files < <(jq -er '.model.files | keys[]' "${RECIPE}")
include_args=()
for name in "${model_files[@]}"; do
  include_args+=(--include "${name}")
done
snapshot_dir="$(
  "${HF_CLI}" download "${model_repo}" \
    --revision "${model_revision}" \
    "${include_args[@]}" \
    | tail -n 1
)"
# Older CLIs print the bare path; 1.x prints "path=<dir>".
snapshot_dir="${snapshot_dir#path=}"
[[ -d "${snapshot_dir}" ]] \
  || die "hf download did not report a snapshot directory (got ${snapshot_dir@Q})"
snapshot_dir="$(cd "${snapshot_dir}" && pwd -P)"
# The snapshot files are symlinks into ../../blobs, so the whole cached repo
# directory has to be visible inside the container.
repository_dir="$(cd "${snapshot_dir}/../.." && pwd -P)"
snapshot_rel="${snapshot_dir#"${repository_dir}/"}"
echo "snapshot      ${snapshot_dir}"

for name in "${model_files[@]}"; do
  expected="$(jq -er --arg key "${name}" '.model.files[$key]' "${RECIPE}")"
  actual="$(sha256sum "${snapshot_dir}/${name}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] \
    || die "snapshot file ${name} has SHA-256 ${actual}, recipe pins ${expected}"
done
echo "snapshot files verified (${#model_files[@]})"

# ---------------------------------------------------------------------------
# 3. Throwaway builder image: task base image plus the pinned upstream deps.
# ---------------------------------------------------------------------------
builder_context="${WORK_DIR}/builder"
mkdir -p "${builder_context}"
pip_lines="$(jq -r '.builder.pip_packages[] | "      " + . + " \\"' "${RECIPE}")"
cat >"${builder_context}/Dockerfile" <<BUILDER_DOCKERFILE
FROM ${base_image}

ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    CUDA_HOME=/usr/local/cuda \\
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Pinned upstream Wan2.1 dependency set (fixtures/recipe/RECIPE.json).
RUN pip install --no-cache-dir --root-user-action=ignore \\
${pip_lines}
    && pip install --no-cache-dir --root-user-action=ignore ${attention_backend}

# wan/modules/attention.py silently prefers FA3 when it is importable, which
# would change every block executed before the capture point.
RUN python - <<'VERIFY'
import importlib.util
import flash_attn
assert importlib.util.find_spec("flash_attn_interface") is None, "FA3 must be absent"
print("flash_attn", flash_attn.__version__)
VERIFY
BUILDER_DOCKERFILE

docker build \
  --tag "${BUILDER_IMAGE}" \
  --network host \
  --build-arg HTTP_PROXY \
  --build-arg HTTPS_PROXY \
  --build-arg NO_PROXY \
  --build-arg http_proxy \
  --build-arg https_proxy \
  --build-arg no_proxy \
  --file "${builder_context}/Dockerfile" \
  "${builder_context}"

# ---------------------------------------------------------------------------
# 4. Capture.
# ---------------------------------------------------------------------------
output_dir="${WORK_DIR}/out"
rm -rf -- "${output_dir}"
mkdir -p "${output_dir}"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e "NVIDIA_VISIBLE_DEVICES=${FIXTURE_GPU}" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HOME=/tmp \
  -e TORCH_HOME=/tmp/torch \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "${upstream_dir}:/opt/wan2.1:ro" \
  -v "${repository_dir}:/models/hf:ro" \
  -v "${RECIPE_DIR}:/recipe:ro" \
  -v "${output_dir}:/out" \
  --entrypoint python \
  "${BUILDER_IMAGE}" \
  /recipe/capture_e8_wan_qkv.py \
  --recipe /recipe/RECIPE.json \
  --wan-src /opt/wan2.1 \
  --checkpoint "/models/hf/${snapshot_rel}" \
  --output-dir /out

[[ -f "${output_dir}/${fixture_name}" ]] || die "capture produced no fixture"
[[ -f "${output_dir}/${manifest_name}" ]] || die "capture produced no manifest"

captured_sha256="$(sha256sum "${output_dir}/${fixture_name}" | awk '{print $1}')"
declared_sha256="$(jq -er '.sha256' "${output_dir}/${manifest_name}")"
declared_recipe="$(jq -er '.recipe_sha256' "${output_dir}/${manifest_name}")"
[[ "${captured_sha256}" == "${declared_sha256}" ]] \
  || die "capture manifest SHA-256 does not describe the captured file"
[[ "${declared_recipe}" == "${recipe_sha256}" ]] \
  || die "capture manifest recipe SHA-256 does not match ${RECIPE}"

install -m 0644 "${output_dir}/${fixture_name}" "${fixture_path}"
install -m 0644 "${output_dir}/${manifest_name}" "${manifest_path}"
rm -rf -- "${output_dir}"

echo "fixture       ${fixture_path}"
echo "manifest      ${manifest_path}"
echo "fixture sha256 ${captured_sha256}"
echo "recipe sha256  ${recipe_sha256}"
