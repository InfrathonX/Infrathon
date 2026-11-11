#!/usr/bin/env bash
# Capture the public real-model layers.  They use different prompts, steps,
# layers, models and text than any other capture of this task.
#
#   PUBLIC_LAYER_GPU=GPU-xxxx OUT=<dir> HF_HUB_CACHE=<hub> fixtures/recipe/public_layers/build_public_layers.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PUBLIC_LAYER_GPU:?set the GPU UUID}" "${OUT:?set the output root}" "${HF_HUB_CACHE:?set the Hugging Face hub cache}"
export HF_HUB_CACHE
HF_CLI="${HF_CLI:-hf}"
exec 8>"/tmp/sforge-gpu-${PUBLIC_LAYER_GPU}.lock" 9>"/tmp/mlsysforge-gpu-${PUBLIC_LAYER_GPU}.lock"
flock -n 8 && flock -n 9 || { echo "GPU ${PUBLIC_LAYER_GPU} is locked" >&2; exit 1; }

image="sageattention-public-layer-builder:$(sha256sum "${here}/../real_layers/Dockerfile.builder" | cut -c1-12)"
docker build --quiet --tag "${image}" --network host \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
  --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy \
  --file "${here}/../real_layers/Dockerfile.builder" "${here}/../real_layers" >/dev/null

video="$("${HF_CLI}" download Wan-AI/Wan2.2-TI2V-5B-Diffusers --revision b8fff7315c768468a5333511427288870b2e9635 | tail -n 1)"
lm="$("${HF_CLI}" download Qwen/Qwen3-8B | tail -n 1)"
text="$("${HF_CLI}" download Salesforce/wikitext --repo-type dataset \
  --revision b08601e04326c79dfdd32d625aee71d232d685c3 \
  --include wikitext-2-raw-v1/validation-00000-of-00001.parquet | tail -n 1)"
video="${video#path=}"; lm="${lm#path=}"; text="${text#path=}"
hub="$(cd "${video}/../../.." && pwd -P)"
mkdir -p "${OUT}"

run() {
  docker run --rm --user "$(id -u):$(id -g)" --ipc=host --gpus "device=${PUBLIC_LAYER_GPU}" \
    -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/tmp/hf \
    -v "${hub}:${hub}:ro" -v "${here}:/tool:ro" -v "${OUT}:/out" \
    --entrypoint python "${image}" /tool/capture_public_layers.py "$@"
}
# The public video layer comes from a model no Judge case captures (Wan2.2-TI2V-5B).
[[ -e "${OUT}/public-wan22/capture.manifest.json" ]] || run --kind wan22 --model-dir "${video}" --out /out/public-wan22
run --kind lm --model-dir "${lm}" --dataset-file "${text}/wikitext-2-raw-v1/validation-00000-of-00001.parquet" \
  --out /out/public-lm
