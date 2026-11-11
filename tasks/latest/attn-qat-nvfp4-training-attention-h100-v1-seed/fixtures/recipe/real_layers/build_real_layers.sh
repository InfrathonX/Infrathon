#!/usr/bin/env bash
# Capture the real layers of one source of a recipe.
#
#   CAPTURE_GPU=GPU-xxxx OUT=<dir> HF_HUB_CACHE=<hub> fixtures/recipe/real_layers/build_real_layers.sh \
#     wan13b|wan14b                                     (Judge-only, recipe REAL_LAYERS.json)
#   ... build_real_layers.sh cogvideox2b|qwen3-14b|qwen35-4b   (Judge-only, REAL_LAYERS_V2.json)
#   ... build_real_layers.sh public                     (Work, recipe PUBLIC_LAYER.json)
#   ... build_real_layers.sh public_llm                 (Work, recipe PUBLIC_LLM_LAYER.json)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_id="${1:?name a source: wan13b, wan14b, cogvideox2b, qwen3-14b, qwen35-4b, public or public_llm}"
: "${CAPTURE_GPU:?set the GPU UUID}" "${OUT:?set the output directory}" "${HF_HUB_CACHE:?set the Hugging Face hub cache}"
export HF_HUB_CACHE
HF_CLI="${HF_CLI:-hf}"
recipe="REAL_LAYERS.json"
script="capture_real_layers.py"
case "${source_id}" in
  public) recipe="PUBLIC_LAYER.json" ;;
  public_llm) recipe="PUBLIC_LLM_LAYER.json"; script="capture_real_layers_v2.py" ;;
  cogvideox2b|qwen3-14b|qwen35-4b) recipe="REAL_LAYERS_V2.json"; script="capture_real_layers_v2.py" ;;
esac

exec 8>"/tmp/sforge-gpu-${CAPTURE_GPU}.lock" 9>"/tmp/mlsysforge-gpu-${CAPTURE_GPU}.lock"
flock -n 8 && flock -n 9 || { echo "GPU ${CAPTURE_GPU} is locked" >&2; exit 1; }

image="attn-qat-real-layer-builder:$(cat "${here}/Dockerfile.builder" "${here}/capture_real_layers.py" "${here}/capture_real_layers_v2.py" | sha256sum | cut -c1-12)"
docker build --quiet --tag "${image}" --network host \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
  --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy \
  --file "${here}/Dockerfile.builder" "${here}" >/dev/null

repo="$(jq -er --arg s "${source_id}" '.sources[$s].model.repo_id' "${here}/${recipe}")"
revision="$(jq -er --arg s "${source_id}" '.sources[$s].model.revision' "${here}/${recipe}")"
snapshot="$("${HF_CLI}" download "${repo}" --revision "${revision}" | grep -v '^[[:space:]]*$' | tail -n 1)"
snapshot="${snapshot#path=}"
[[ -d "${snapshot}" ]] || { echo "no snapshot for ${repo}@${revision}: ${snapshot}" >&2; exit 1; }
hub="$(cd "${snapshot}/../../.." && pwd -P)"
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd -P)"

docker run --rm --user "$(id -u):$(id -g)" --ipc=host --gpus "device=${CAPTURE_GPU}" \
  -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/tmp/hf -e "HF_HUB_CACHE=${hub}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "${hub}:${hub}:ro" -v "${here}:/tool:ro" -v "${OUT}:/out" \
  --entrypoint python "${image}" "/opt/real_layers/${script}" \
  --recipe "/tool/${recipe}" --source "${source_id}" --model-dir "${snapshot}" --out /out
