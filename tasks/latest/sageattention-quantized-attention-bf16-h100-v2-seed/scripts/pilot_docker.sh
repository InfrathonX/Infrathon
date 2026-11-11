#!/usr/bin/env bash
# Run scripts/pilot_suite.py in the local Judge image on one GPU (host-side calibration only).
#   scripts/pilot_docker.sh GPU CANDIDATE.py MANIFEST.json OUT.json [pilot_suite args...]
# Environment: PILOT_IMAGE (Judge image), E8_DIR (dir with the e8 capture and its manifest),
# REAL_LAYERS_DIR (real-layer captures, optional).
set -euo pipefail
TASK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu=$1 candidate=$(realpath "$2") manifest=$(realpath "$3") out=$(realpath -m "$4"); shift 4
image="${PILOT_IMAGE:-sageattention_quantized_attention.judge.sageattention_quantized_attention_bf16_h100_seed_v2:local}"
mkdir -p "$(dirname "${out}")"
mounts=(-v "${TASK}:/task:ro" -v "${candidate}:/pilot/candidate.py:ro" -v "${manifest}:/pilot/manifest.json:ro"
        -v "$(dirname "${out}"):/pilot/out")
for f in "${E8_DIR:?set E8_DIR}"/e8_*; do mounts+=(-v "${f}:/fix/$(basename "${f}"):ro"); done
[[ -n "${REAL_LAYERS_DIR:-}" ]] && mounts+=(-v "${REAL_LAYERS_DIR}:/fix/real_layers:ro")
exec docker run --rm --gpus "device=${gpu}" --ipc host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -e SAGE_QATTN_FIXTURE_DIR=/fix -e TRITON_CACHE_DIR=/tmp/triton "${mounts[@]}" --entrypoint python "${image}" \
  /task/scripts/pilot_suite.py /pilot/candidate.py /pilot/manifest.json "/pilot/out/$(basename "${out}")" "$@"
