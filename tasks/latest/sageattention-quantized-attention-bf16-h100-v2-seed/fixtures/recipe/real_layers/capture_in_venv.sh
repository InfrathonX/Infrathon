#!/usr/bin/env bash
# Same as capture_scored_layers.sh, without Docker: run each recipe's capture in a Python
# environment that has Dockerfile.builder's pinned packages (CAPTURE_PYTHON), one GPU each.
#   CAPTURE_PYTHON=<venv>/bin/python HF_HUB=<hub> OUT_ROOT=<dir> GPUS="0 1" capture_in_venv.sh RECIPE_x.json ...
set -euo pipefail
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CAPTURE_PYTHON:?}" "${HF_HUB:?}" "${OUT_ROOT:?}"
read -r -a gpus <<<"${GPUS:-0}"
i=0
for name in "$@"; do
  recipe="${RECIPE_DIR}/${name}"
  cdir="$(jq -er .capture_dir "${recipe}")"
  [[ -e "${OUT_ROOT}/${cdir}/capture.manifest.json" ]] && { echo "skip ${cdir} (captured)"; continue; }
  rm -rf "${OUT_ROOT:?}/${cdir}"; mkdir -p "${OUT_ROOT}/${cdir}"
  repo_dir="models--$(jq -er '.model.repo' "${recipe}" | sed -E 's/ .*//; s#/#--#g')"
  case "${name}" in *sd35m*) repo_dir=models--adamo1139--stable-diffusion-3.5-medium-ungated;; esac
  extra=()
  if jq -e '.dataset.files' "${recipe}" >/dev/null; then
    extra=(--dataset-dir "${HF_HUB}/datasets--Salesforce--wikitext/snapshots/$(jq -er .dataset.revision "${recipe}")")
  elif jq -e '.dataset.file' "${recipe}" >/dev/null; then
    extra=(--dataset-file "${HF_HUB}/datasets--Salesforce--wikitext/snapshots/$(jq -er .dataset.revision "${recipe}")/$(jq -er .dataset.file "${recipe}")")
  fi
  gpu="${gpus[$((i % ${#gpus[@]}))]}"; i=$((i + 1))
  echo "capture ${cdir} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${CAPTURE_PYTHON}" "${RECIPE_DIR}/capture_real_layers.py" \
    --recipe "${recipe}" --model-dir "${HF_HUB}/${repo_dir}/snapshots/$(jq -er .model.revision "${recipe}")" \
    "${extra[@]}" --out "${OUT_ROOT}/${cdir}" > "${OUT_ROOT}/${cdir}.log" 2>&1 &
done
wait || true
for name in "$@"; do
  cdir="$(jq -er .capture_dir "${RECIPE_DIR}/${name}")"
  if [[ -e "${OUT_ROOT}/${cdir}/capture.manifest.json" ]]; then echo "ok   ${cdir}"; else echo "FAIL ${cdir}: $(tail -n 3 "${OUT_ROOT}/${cdir}.log" | tr '\n' ' ')"; fi
done
