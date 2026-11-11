#!/usr/bin/env bash
# Build the software parent from images/parent/Dockerfile and verify it against the
# recorded clean parent (scripts/verify_software_parent.py): official Torch base
# layers, then package files identical to references/software-parent's manifest.
#
#   scripts/build_software_parent.sh [TAG]     (default mlsysforge/minference-prefill-parent:v3-rebuilt)
#
# Needs the official base pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel@sha256:cf5aa3f7...
# and network access to PyPI and the flash-attention GitHub release.  Writes the
# receipt to build/software-parent/<image id>.json and prints the two variables
# scripts/build_images.sh takes.
set -euo pipefail
umask 077
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TAG="${1:-mlsysforge/minference-prefill-parent:v3-rebuilt}"
docker build --pull=false --tag "${TAG}" --file "${TASK_ROOT}/images/parent/Dockerfile" "${TASK_ROOT}/images/parent"
image_id="$(docker image inspect --format '{{.Id}}' "${TAG}")"
out="${TASK_ROOT}/build/software-parent"
mkdir -p "${out}"
receipt="${out}/${image_id#sha256:}.json"
[[ -e "${receipt}" ]] || "${PYTHON_BIN}" "${TASK_ROOT}/scripts/verify_software_parent.py" "${TAG}" --receipt "${receipt}"
printf 'ON_DEMAND_SOFTWARE_IMAGE=%s\nSOFTWARE_PARENT_VERIFICATION=%s\n' "${TAG}" "${receipt}"
