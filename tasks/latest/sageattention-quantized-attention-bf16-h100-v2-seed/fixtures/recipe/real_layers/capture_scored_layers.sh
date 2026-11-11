#!/usr/bin/env bash
# Capture the scored real layers of final-holdout-v3 from the RECIPE_*.json files named
# on the command line (default: every recipe whose capture is missing), one GPU per recipe.
#
#   HF_HUB=~/.cache/huggingface/hub OUT_ROOT=<assets>/judge/real_layers GPUS="0 1 2 3" \
#     fixtures/recipe/real_layers/capture_scored_layers.sh [RECIPE_x.json ...]
#
# Models and the WikiText parquet files come from a local Hugging Face hub cache (HF_HUB),
# pinned by each recipe's revision and per-file SHA-256.  A recipe without sanity bands is a
# reference capture: the capture writes RECIPE.final.json (the recipe plus its bands) next to
# the layers, and that file replaces the recipe in the repository.
set -euo pipefail
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${HF_HUB:?set HF_HUB to a Hugging Face hub cache directory}"
: "${OUT_ROOT:?set OUT_ROOT to the real_layers output root}"
read -r -a gpus <<<"${GPUS:-0}"
# BUILDER_IMAGE: an image built from Dockerfile.builder (built here when unset).  The capture
# code is bind-mounted from this directory, so the image only has to carry the dependencies.
if [[ -z "${BUILDER_IMAGE:-}" ]]; then
  BUILDER_IMAGE="sageattention-v2-real-layer-builder:$(sha256sum "${RECIPE_DIR}/Dockerfile.builder" | cut -c1-16)"
  docker build --quiet --tag "${BUILDER_IMAGE}" --network host --file "${RECIPE_DIR}/Dockerfile.builder" "${RECIPE_DIR}" >/dev/null
fi
mkdir -p "${OUT_ROOT}"
if (( $# )); then recipes=("$@"); else recipes=(); for r in "${RECIPE_DIR}"/RECIPE_*.json; do recipes+=("$(basename "$r")"); done; fi
i=0
for name in "${recipes[@]}"; do
  recipe="${RECIPE_DIR}/${name}"
  cdir="$(jq -er .capture_dir "${recipe}")"
  [[ -e "${OUT_ROOT}/${cdir}/capture.manifest.json" ]] && { echo "skip ${cdir} (captured)"; continue; }
  rm -rf "${OUT_ROOT:?}/${cdir}"
  repo_dir="models--$(jq -er '.model.repo' "${recipe}" | sed -E 's/ .*//; s#/#--#g')"
  case "${name}" in *sd35m*) repo_dir=models--adamo1139--stable-diffusion-3.5-medium-ungated;; esac
  model="/hub/${repo_dir}/snapshots/$(jq -er .model.revision "${recipe}")"
  extra=()
  if jq -e '.dataset.files' "${recipe}" >/dev/null; then
    extra=(--dataset-dir "/hub/datasets--Salesforce--wikitext/snapshots/$(jq -er .dataset.revision "${recipe}")")
  elif jq -e '.dataset.file' "${recipe}" >/dev/null; then
    extra=(--dataset-file "/hub/datasets--Salesforce--wikitext/snapshots/$(jq -er .dataset.revision "${recipe}")/$(jq -er .dataset.file "${recipe}")")
  fi
  gpu="${gpus[$((i % ${#gpus[@]}))]}"; i=$((i + 1))
  mkdir -p "${OUT_ROOT}/${cdir}"
  echo "capture ${cdir} on GPU ${gpu}"
  docker run --rm --gpus "device=${gpu}" --ipc host --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -e HF_HUB_OFFLINE=1 -v "${HF_HUB}:/hub:ro" -v "${RECIPE_DIR}:/recipes:ro" -v "${OUT_ROOT}/${cdir}:/out" \
    -v "${RECIPE_DIR}/capture_real_layers.py:/opt/real_layers/capture_real_layers.py:ro" \
    "${BUILDER_IMAGE}" python /opt/real_layers/capture_real_layers.py --recipe "/recipes/${name}" \
    --model-dir "${model}" "${extra[@]}" --out /out > "${OUT_ROOT}/${cdir}.log" 2>&1 &
done
wait
for name in "${recipes[@]}"; do
  cdir="$(jq -er .capture_dir "${RECIPE_DIR}/${name}")"
  if [[ -e "${OUT_ROOT}/${cdir}/capture.manifest.json" ]]; then echo "ok   ${cdir}"; else echo "FAIL ${cdir}: $(tail -n 3 "${OUT_ROOT}/${cdir}.log" | tr '\n' ' ')"; fi
done
