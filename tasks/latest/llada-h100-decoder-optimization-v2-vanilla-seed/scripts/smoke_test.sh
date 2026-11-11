#!/usr/bin/env bash
set -Eeuo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
for command in docker flock jq nvidia-smi python3 realpath sha256sum; do
  require_command "${command}"
done

[[ -v WORK_GPU && -n "${WORK_GPU}" ]] \
  || die "smoke requires an explicit numeric WORK_GPU"
[[ -v JUDGE_GPU && -n "${JUDGE_GPU}" ]] \
  || die "smoke requires an explicit numeric JUDGE_GPU"
WORK_GPU="$(canonical_gpu_index "${WORK_GPU}")"
JUDGE_GPU="$(canonical_gpu_index "${JUDGE_GPU}")"
[[ "${WORK_GPU}" != "${JUDGE_GPU}" ]] \
  || die "Work and Judge GPU indices must differ"
work_uuid="$(gpu_uuid "${WORK_GPU}")"
judge_uuid="$(gpu_uuid "${JUDGE_GPU}")"
[[ "${work_uuid}" != "${judge_uuid}" ]] \
  || die "Work and Judge GPU UUIDs must differ"

# Before memory qualification, smoke is explicitly non-release evidence and
# uses only a caller-overridable nonzero liveness floor. Once the policy is
# frozen, the exact release admission formula is exercised here too.
if jq -e '
  .qualification_status == "QUALIFIED"
  and .release_status == "FROZEN"
' "${LLADA_MEMORY_POLICY_PATH:-${MIGRATION_ROOT}/memory_policy.json}" \
    >/dev/null 2>&1; then
  load_frozen_memory_policy
  smoke_min_free_mib="${GPU_ADMISSION_MIN_FREE_MIB}"
  smoke_policy_mode="frozen-release-policy"
else
  smoke_min_free_mib="${GPU_SMOKE_MIN_FREE_MIB:-1}"
  [[ "${smoke_min_free_mib}" =~ ^[0-9]+$ ]] \
    || die "GPU_SMOKE_MIN_FREE_MIB must be a positive integer"
  (( 10#${smoke_min_free_mib} > 0 )) \
    || die "GPU_SMOKE_MIN_FREE_MIB must be positive"
  smoke_policy_mode="bootstrap-unqualified-memory"
fi

acquire_dual_gpu_locks "${work_uuid}" "${judge_uuid}"
assert_distinct_gpu_pair \
  "${WORK_GPU}" "${JUDGE_GPU}" "${smoke_min_free_mib}"

LLADA_MODEL_REPO="${LLADA_MODEL_REPO:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--GSAI-ML--LLaDA-8B-Instruct}"
export LLADA_MODEL_REPO
python3 "${MIGRATION_ROOT}/scripts/validate_layout.py" "${MIGRATION_ROOT}"
python3 "${MIGRATION_ROOT}/scripts/validate_model_artifact.py" \
  --repo "${LLADA_MODEL_REPO}" --full
assert_built_images_current
assert_container_gpu "${WORK_IMAGE}" "${work_uuid}" "Work"
assert_container_gpu "${JUDGE_IMAGE}" "${judge_uuid}" "Judge"

docker run --rm --network none \
  --entrypoint bash \
  "${WORK_IMAGE}" \
  /workspace/public_tests/run_public_tests.sh --cpu-only

if docker run --rm --network none --user agent --entrypoint bash "${WORK_IMAGE}" -lc \
    'test ! -e /opt/decoder_judge && test ! -e /opt/decoder_judge/private_data'; then
  echo "Work contains no Judge assets: PASS"
else
  die "Work confidentiality probe failed"
fi

model_path="/opt/models/llada-hf/snapshots/6059b30c9531767ce47e0ca6a8161a6bb0c67693"
docker run --rm --network none \
  --gpus "device=${work_uuid}" \
  --volume "${LLADA_MODEL_REPO}:/opt/models/llada-hf:ro" \
  --env "NVIDIA_VISIBLE_DEVICES=${work_uuid}" \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env LLADA_TARGET=dual-cuda-sm80-34g \
  --env LLADA_GPU_ROLE=work \
  --env LLADA_MODEL_PATH="${model_path}" \
  --entrypoint bash \
  "${WORK_IMAGE}" \
  /workspace/public_tests/run_public_tests.sh --quick

if docker run --rm --network none --user agent --entrypoint bash "${JUDGE_IMAGE}" -lc \
    'test ! -r /opt/decoder_judge/private_data/hidden_cases.json \
      && test ! -r /opt/decoder_judge/verifier/verify_decoder.py'; then
  echo "Judge private-data permissions: PASS"
else
  die "unprivileged Judge user can read private assets"
fi

assert_gpu_released "${work_uuid}" "${WORK_GPU}" "Work"
assert_gpu_released "${judge_uuid}" "${JUDGE_GPU}" "Judge"
assert_dual_gpu_locks_held
echo "LLaDA dual-H100 decoder smoke PASS (${smoke_policy_mode})"
