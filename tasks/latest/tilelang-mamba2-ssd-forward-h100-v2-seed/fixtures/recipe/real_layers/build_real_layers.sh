#!/usr/bin/env bash
# Capture the real layers of REAL_LAYERS.json.
#
#   CAPTURE_GPU=GPU-xxxx OUT=<dir> HF_HUB_CACHE=<hub> fixtures/recipe/real_layers/build_real_layers.sh judge
#       Judge-only layers (granite_h_micro, bamba_9b_v2); OUT becomes the Judge holdout's real_layers/
#   ... build_real_layers.sh public
#       the public layer; OUT becomes the Work image's public_layer/
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_root="$(cd "${here}/../../.." && pwd)"
target="${1:?name a target: judge or public}"
: "${CAPTURE_GPU:?set the GPU UUID}" "${OUT:?set the output directory}" "${HF_HUB_CACHE:?set the Hugging Face hub cache}"
export HF_HUB_CACHE
HF_CLI="${HF_CLI:-hf}"
recipe="${here}/REAL_LAYERS.json"
case "${target}" in
  judge) sources=(granite_h_micro bamba_9b_v2) ;;
  public) sources=(public) ;;
  *) echo "unknown target ${target}" >&2; exit 1 ;;
esac

exec 8>"/tmp/sforge-gpu-${CAPTURE_GPU}.lock" 9>"/tmp/mlsysforge-gpu-${CAPTURE_GPU}.lock"
flock -n 8 && flock -n 9 || { echo "GPU ${CAPTURE_GPU} is locked" >&2; exit 1; }

image="tilelang-mamba2-real-layer-builder:$(cat "${here}/Dockerfile.builder" "${here}/capture_real_layers.py" | sha256sum | cut -c1-12)"
docker build --quiet --tag "${image}" --network host \
  --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY \
  --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy \
  --file "${here}/Dockerfile.builder" "${here}" >/dev/null

snapshot() {
  local repo="$1" revision="$2" kind="$3" path
  path="$("${HF_CLI}" download "${repo}" --revision "${revision}" --repo-type "${kind}" "${@:4}" | grep -v '^[[:space:]]*$' | tail -n 1)"
  path="${path#path=}"
  [[ -d "${path}" ]] || { echo "no snapshot for ${repo}@${revision}: ${path}" >&2; exit 1; }
  echo "${path}"
}

dataset_repo="$(jq -er '.text.dataset.repo_id' "${recipe}")"
dataset_revision="$(jq -er '.text.dataset.revision' "${recipe}")"
parquets=()
model_dirs=()
for source in "${sources[@]}"; do
  parquets+=("$(jq -er --arg s "${source}" '.sources[$s].parquet' "${recipe}")")
  repo="$(jq -er --arg s "${source}" '.sources[$s].model.repo_id' "${recipe}")"
  revision="$(jq -er --arg s "${source}" '.sources[$s].model.revision' "${recipe}")"
  model_dirs+=("$(snapshot "${repo}" "${revision}" model)")
done
"${HF_CLI}" download "${dataset_repo}" --revision "${dataset_revision}" --repo-type dataset "${parquets[@]}" >/dev/null
dataset_dir="${HF_HUB_CACHE}/datasets--${dataset_repo//\//--}/snapshots/${dataset_revision}"
for parquet in "${parquets[@]}"; do
  [[ -f "${dataset_dir}/${parquet}" ]] || { echo "missing ${dataset_dir}/${parquet}" >&2; exit 1; }
done
hub="$(cd "${HF_HUB_CACHE}" && pwd -P)"
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd -P)"

docker run --rm --user "$(id -u):$(id -g)" --ipc=host --gpus "device=${CAPTURE_GPU}" \
  -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HOME=/tmp/hf \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "${hub}:${hub}:ro" -v "${here}:/tool:ro" -v "${task_root}/runtime:/runtime:ro" -v "${OUT}:/out" \
  --entrypoint python "${image}" /opt/real_layers/capture_real_layers.py \
  --recipe /tool/REAL_LAYERS.json --runtime /runtime/mamba2_ssd_runtime.py \
  --sources "${sources[@]}" --model-dirs "${model_dirs[@]}" --dataset-dir "${dataset_dir}" --out /out
