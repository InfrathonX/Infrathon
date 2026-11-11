#!/usr/bin/env bash
set -euo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_command docker
require_command flock
require_command nvidia-smi

candidate="${1:-}"
case_name="${2:-}"
output="${3:-}"
[[ -f "${candidate}" && -n "${case_name}" && -n "${output}" ]] \
  || die "usage: $0 CANDIDATE.py CASE_NAME OUTPUT.json"
candidate="$(realpath "${candidate}")"
output="$(realpath -m "${output}")"
assert_built_images_current
judge_gpu="$(canonical_gpu_index "${JUDGE_GPU:-3}")"
judge_uuid="$(gpu_uuid "${judge_gpu}")"

exec 9>"$(gpu_lock_path "${judge_uuid}")"
flock -n 9 || die "judge GPU ${judge_gpu} lock is already held"
assert_gpu_idle "${judge_gpu}" "host diagnostic"
assert_container_gpu "${JUDGE_IMAGE}" "${judge_uuid}" "host diagnostic"

mkdir -p "$(dirname "${output}")"
docker run --rm \
  --cpus 8 \
  --memory 32g \
  -e "NVIDIA_VISIBLE_DEVICES=${judge_uuid}" \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e FLASH_KMEANS_TARGET=H100 \
  --volume \
    "${SCRIPT_DIR}/benchmark_host_candidate_case.py:/tmp/benchmark_host_candidate_case.py:ro" \
  --volume "${candidate}:/tmp/candidate.py:ro" \
  --entrypoint python3 \
  "${JUDGE_IMAGE}" \
  /tmp/benchmark_host_candidate_case.py \
  --candidate /tmp/candidate.py \
  --case "${case_name}" \
  | tee "${output}"
