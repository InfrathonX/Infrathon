#!/usr/bin/env bash
# Capture this deployment's real-layer gate inputs from the RECIPE_*.json files here.
#
# Each recipe captures every attention layer's Q/K/V of one real model
# (CogVideoX-2B, Llama-3.1-8B), following SageAttention's layer-wise accuracy
# protocol.  Like the e8 edge fixture, the captures are bound by their recipe,
# not by one host's bytes, and are never committed.  The task's assets carry a
# reference capture of both (assets/fetch.sh); this script is the optional way
# to capture them again on this host.
#
#   REAL_LAYER_GPU=GPU-xxxxxxxx-... fixtures/recipe/real_layers/build_real_layers.sh [recipe ...]
#
# Writes <REAL_LAYERS_DIR>/<capture_dir>/layerNN.pt and capture.manifest.json.
#
# Environment:
#   REAL_LAYER_GPU    (required) GPU UUID handed to the builder container; both host
#                     GPU locks (/tmp/{sforge,mlsysforge}-gpu-<uuid>.lock) are held.
#   HF_TOKEN          not needed: the Llama weights come from an ungated mirror whose
#                     files are byte-identical to the recipe's pinned snapshot (below);
#                     if set it is only read by `hf download`, never written anywhere.
#   REAL_LAYER_NO_MIRROR=1  download the recipe's own repository instead (gated for
#                     Llama: accept its license on Hugging Face and log in first).
#   HF_CLI            Hugging Face CLI (default: hf).
#   REAL_LAYERS_DIR   output root (default: <task>/fixtures/real_layers).
#   BUILDER_IMAGE     builder tag (default: sageattention-v2-real-layer-builder:<Dockerfile sha>).
#   REAL_LAYERS_FORCE=1  replace an existing capture directory.

set -euo pipefail

RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ROOT="$(cd "${RECIPE_DIR}/../../.." && pwd)"
OUT_ROOT="${REAL_LAYERS_DIR:-${TASK_ROOT}/fixtures/real_layers}"

die() { echo "ERROR: $*" >&2; exit 1; }
for c in docker jq sha256sum flock nvidia-smi; do command -v "$c" >/dev/null || die "missing command: $c"; done
[[ -n "${REAL_LAYER_GPU:-}" ]] || die "set REAL_LAYER_GPU to the GPU UUID the capture may use (nvidia-smi -L)"
HF_CLI="${HF_CLI:-hf}"
command -v "${HF_CLI}" >/dev/null || die "missing Hugging Face CLI: ${HF_CLI}"

if (( $# )); then recipes=("$@"); else recipes=("${RECIPE_DIR}"/RECIPE_*.json); fi

exec 8>"/tmp/sforge-gpu-${REAL_LAYER_GPU}.lock" 9>"/tmp/mlsysforge-gpu-${REAL_LAYER_GPU}.lock"
flock -n 8 && flock -n 9 || die "GPU ${REAL_LAYER_GPU} is locked by another job"

builder_sha="$(cat "${RECIPE_DIR}/Dockerfile.builder" "${RECIPE_DIR}/capture_real_layers.py" | sha256sum | cut -c1-16)"
BUILDER_IMAGE="${BUILDER_IMAGE:-sageattention-v2-real-layer-builder:${builder_sha}}"
docker build --quiet --tag "${BUILDER_IMAGE}" --network host \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
  --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy \
  --file "${RECIPE_DIR}/Dockerfile.builder" "${RECIPE_DIR}" >/dev/null
echo "builder       ${BUILDER_IMAGE}"

snapshot() {  # repo revision repo_type [include...] -> prints snapshot dir
  local repo=$1 rev=$2 type=$3; shift 3
  local args=() out
  for f in "$@"; do args+=(--include "$f"); done
  out="$("${HF_CLI}" download "${repo}" --revision "${rev}" --repo-type "${type}" "${args[@]}" | tail -n 1)"
  out="${out#path=}"
  [[ -d "${out}" ]] || die "hf download did not report a snapshot directory for ${repo} (got ${out@Q})"
  (cd "${out}" && pwd -P)
}

# Ungated mirrors of a pinned snapshot: every file the recipe pins has the same
# SHA-256 at the mirror's revision (capture_real_layers.py verifies each pin
# before it loads anything, so a mirror that drifted fails closed).
model_source() {  # repo revision -> "repo revision" to download from
  local repo=$1 rev=$2
  if [[ "${REAL_LAYER_NO_MIRROR:-0}" != "1" ]]; then
    case "${repo}@${rev}" in
      meta-llama/Llama-3.1-8B@d04e592bb4f6aa9cfee91e2e20afa771667e1d4b)
        repo=NousResearch/Meta-Llama-3.1-8B rev=1f47e50cdbe801ad8a5174156ec3a0655108fb9f ;;
    esac
  fi
  printf '%s %s\n' "${repo}" "${rev}"
}

for recipe in "${recipes[@]}"; do
  [[ -f "${recipe}" && ! -L "${recipe}" ]] || die "missing recipe: ${recipe}"
  recipe_sha="$(sha256sum "${recipe}" | awk '{print $1}')"
  recipe_id="$(jq -er '.recipe_id' "${recipe}")"
  capture_dir="$(jq -er '.capture_dir' "${recipe}")"
  out="${OUT_ROOT}/${capture_dir}"
  if [[ -e "${out}" ]]; then
    [[ "${REAL_LAYERS_FORCE:-0}" == "1" ]] || die "${out} exists; set REAL_LAYERS_FORCE=1 to recapture"
    rm -rf -- "${out}"
  fi
  mkdir -p "${out}"
  echo "recipe        ${recipe_id} (sha256 ${recipe_sha}) -> ${out}"

  mapfile -t model_files < <(jq -er '.model.files | keys[]' "${recipe}")
  read -r model_repo model_revision < <(model_source "$(jq -er '.model.repo' "${recipe}")" "$(jq -er '.model.revision' "${recipe}")")
  echo "model         ${model_repo}@${model_revision}"
  model_snapshot="$(snapshot "${model_repo}" "${model_revision}" model "${model_files[@]}")"
  # Snapshot files are symlinks into blobs/, and some caches deduplicate blobs
  # again through absolute links; mount the whole hub at its own real path and
  # require every pinned file to resolve inside it.
  hub_root="$(cd "${model_snapshot}/../../.." && pwd -P)"
  pinned=()
  for f in "${model_files[@]}"; do pinned+=("${model_snapshot}/${f}"); done
  mounts=(-v "${hub_root}:${hub_root}:ro")
  args=(--model-dir "${model_snapshot}")
  if jq -e '.dataset' "${recipe}" >/dev/null; then
    ds_file="$(jq -er '.dataset.file' "${recipe}")"
    ds_snapshot="$(snapshot "$(jq -er '.dataset.repo' "${recipe}")" "$(jq -er '.dataset.revision' "${recipe}")" dataset "${ds_file}")"
    [[ "$(cd "${ds_snapshot}/../../.." && pwd -P)" == "${hub_root}" ]] || die "dataset and model snapshots live in different caches"
    pinned+=("${ds_snapshot}/${ds_file}")
    args+=(--dataset-file "${ds_snapshot}/${ds_file}")
  fi
  for f in "${pinned[@]}"; do
    target="$(readlink -f "${f}")"
    [[ -f "${target}" && "${target}" == "${hub_root}/"* ]] || die "pinned file does not resolve inside ${hub_root}: ${f}"
  done

  docker run --rm --user "$(id -u):$(id -g)" --ipc=host \
    --gpus "device=${REAL_LAYER_GPU}" \
    -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/tmp/hf \
    "${mounts[@]}" -v "${RECIPE_DIR}:/recipe:ro" -v "${out}:/out" \
    --entrypoint python "${BUILDER_IMAGE}" /opt/real_layers/capture_real_layers.py \
    --recipe "/recipe/$(basename "${recipe}")" "${args[@]}" --out /out

  manifest="${out}/capture.manifest.json"
  [[ "$(jq -er '.recipe_sha256' "${manifest}")" == "${recipe_sha}" ]] || die "manifest recipe SHA-256 differs from ${recipe}"
  n_layers="$(jq -er '.attention.layers' "${recipe}")"
  [[ "$(jq -er '.files | length' "${manifest}")" == "${n_layers}" ]] || die "manifest does not list ${n_layers} layers"
  while read -r name digest; do
    [[ "$(sha256sum "${out}/${name}" | awk '{print $1}')" == "${digest}" ]] || die "${name} does not match its manifest SHA-256"
  done < <(jq -r '.files | to_entries[] | "\(.key) \(.value.sha256)"' "${manifest}")
  chmod 0444 "${out}"/*.pt "${manifest}"
  echo "captured      ${n_layers} layers, $(du -sh "${out}" | cut -f1)"
done
