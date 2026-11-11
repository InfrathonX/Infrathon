#!/usr/bin/env bash
set -euo pipefail
umask 077
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SOURCE_ROOT="${SOURCE_ROOT:-${TASK_ROOT}}"
BUILD_ROOT="${BUILD_ROOT:-${SOURCE_ROOT}/build/kernel-only-thin-contexts}"
WORK_IMAGE="${WORK_IMAGE:-mlsysforge/minference-prefill-work:v3-kernel-only-thin}"
JUDGE_IMAGE="${JUDGE_IMAGE:-mlsysforge/minference-prefill-judge:v3-kernel-only-thin}"
BUILD_BUILDER="${BUILD_BUILDER:-default}"
EXPECTED_SOFTWARE_IMAGE_ID="sha256:603d5163cef94886eade0d7b55e8b39ecc29f6ff1ad68c76556d573e16613189"
prequalification=0
validate_only=0
build_role=both
for argument in "$@"; do
  case "${argument}" in
    --prequalification) prequalification=1 ;;
    --validate-only) validate_only=1 ;;
    --work-only) [[ "${build_role}" != judge ]] || exit 2; build_role=work ;;
    --judge-only) [[ "${build_role}" != work ]] || exit 2; build_role=judge ;;
    *) echo 'usage: build_images.sh [--prequalification] [--validate-only] [--work-only|--judge-only]' >&2; exit 2 ;;
  esac
done
: "${FINAL_PERFORMANCE_PLAN:?set the private on-demand metadata plan}"
: "${JUDGE_ASSET_CONTRACT:?set the READY two-role private asset descriptor}"
NUMERICAL_POLICY="${NUMERICAL_POLICY:-${SOURCE_ROOT}/qualification/numerical_policy.json}"
PROTECTED_HASHES="${PROTECTED_HASHES:-}"
stage_args=(--source-root "${SOURCE_ROOT}" --build-root "${BUILD_ROOT}"
 --final-plan "${FINAL_PERFORMANCE_PLAN}"
 --judge-asset-contract "${JUDGE_ASSET_CONTRACT}"
 --numerical-policy "${NUMERICAL_POLICY}")
if [[ -n "${PROTECTED_HASHES}" ]]; then stage_args+=(--protected-hashes "${PROTECTED_HASHES}"); fi
if (( prequalification )); then stage_args+=(--prequalification); fi
stage_record="$("${PYTHON_BIN}" "${TASK_ROOT}/scripts/stage_thin_images_v3.py" "${stage_args[@]}")"
printf '%s\n' "${stage_record}"
export STAGE_RECORD="${stage_record}"
stage="$("${PYTHON_BIN}" -c 'import json,os;print(json.loads(os.environ["STAGE_RECORD"])["stage"])')"
if (( validate_only )); then exit 0; fi
: "${ON_DEMAND_SOFTWARE_IMAGE:?set an OWNED existing local tag resolving to the verified clean603d parent or a verified rebuild of it (scripts/build_software_parent.sh); raw IDs are not FROM names}"
[[ "${ON_DEMAND_SOFTWARE_IMAGE}" != sha256:* && "${ON_DEMAND_SOFTWARE_IMAGE}" != *@* ]] || { echo 'use the verified owned local software tag, not a raw ID or synthetic registry digest' >&2; exit 2; }
# Prequalification is deliberately a separate local image epoch. It is never
# labelled release-qualified and cannot score until policies are truly frozen.
actual_base="$(docker image inspect --format '{{.Id}}' "${ON_DEMAND_SOFTWARE_IMAGE}")"
policy_args=(--source-root "${SOURCE_ROOT}" --stage-receipt "${stage}/staging-receipt.json" --image-only)
if [[ "${actual_base}" == "${EXPECTED_SOFTWARE_IMAGE_ID}" ]]; then
  # The recorded clean parent, bound by its original build receipts.
  : "${SOFTWARE_PARENT_RECEIPT:?set the original actual clean software build receipt}"
  : "${SOFTWARE_EQUIVALENCE_RECEIPT:?set the original actual CPU software equality receipt}"
  "${PYTHON_BIN}" -c 'import sys;sys.path.insert(0,sys.argv[1]);from built_image_policy import validate_software_parent_evidence;validate_software_parent_evidence(sys.argv[2],sys.argv[3])' "${TASK_ROOT}/scripts" "${SOFTWARE_PARENT_RECEIPT}" "${SOFTWARE_EQUIVALENCE_RECEIPT}"
  policy_args+=(--software-parent-receipt "${SOFTWARE_PARENT_RECEIPT}" --software-equivalence-receipt "${SOFTWARE_EQUIVALENCE_RECEIPT}")
else
  # A parent rebuilt from images/parent/Dockerfile: the official Torch base's
  # layers, then package files identical to the recorded clean parent's
  # (scripts/verify_software_parent.py --receipt, written by build_software_parent.sh).
  : "${SOFTWARE_PARENT_VERIFICATION:?the parent is not the recorded clean603d image; build and verify one with scripts/build_software_parent.sh and set SOFTWARE_PARENT_VERIFICATION to its receipt}"
  verified="$("${PYTHON_BIN}" -c 'import sys;sys.path.insert(0,sys.argv[1]);from built_image_policy import validate_rebuilt_software_parent;print(validate_rebuilt_software_parent(sys.argv[2])["clean_image_id"])' "${TASK_ROOT}/scripts" "${SOFTWARE_PARENT_VERIFICATION}")"
  [[ "${verified}" == "${actual_base}" ]] || { echo "SOFTWARE_PARENT_VERIFICATION verifies ${verified}, not ${ON_DEMAND_SOFTWARE_IMAGE} (${actual_base})" >&2; exit 1; }
  policy_args+=(--software-parent-verification "${SOFTWARE_PARENT_VERIFICATION}")
fi
if (( prequalification )); then policy_args+=(--prequalification); fi
build_one() {
  local role="$1" tag="$2"
  [[ "$(docker image inspect --format '{{.Id}}' "${ON_DEMAND_SOFTWARE_IMAGE}")" == "${actual_base}" ]] || return 1
  local args=()
  if [[ "${role}" == judge ]]; then args=(--build-context "installed_sources=${stage}/installed"); fi
  docker buildx build --builder "${BUILD_BUILDER}" --pull=false --network none \
    --output type=docker,compression=gzip,compression-level=1 --tag "${tag}" \
    --build-arg "ON_DEMAND_SOFTWARE_IMAGE=${ON_DEMAND_SOFTWARE_IMAGE}" \
    "${args[@]}" --file "${stage}/${role}/Dockerfile" "${stage}/${role}"
  local actual_image
  actual_image="$(docker image inspect --format '{{.Id}}' "${tag}")"
  [[ "$(docker image inspect --format '{{.Id}}' "${ON_DEMAND_SOFTWARE_IMAGE}")" == "${actual_base}" ]] || return 1
  "${PYTHON_BIN}" "${TASK_ROOT}/scripts/validate_built_image_policy.py" \
    "${policy_args[@]}" "--${role}-image" "${actual_image}" --base-image "${actual_base}" \
    --output "${stage}/${role}-image-policy.json"
}
if [[ "${build_role}" != judge ]]; then build_one work "${WORK_IMAGE}"; fi
if [[ "${build_role}" != work ]]; then build_one judge "${JUDGE_IMAGE}"; fi
# The build record (images/release.json, scripts/infrathon/image_record.py) for publishing; the default tags only.
if [[ "${build_role}" == both && "${WORK_IMAGE}" == mlsysforge/minference-prefill-work:v3-kernel-only-thin \
      && "${JUDGE_IMAGE}" == mlsysforge/minference-prefill-judge:v3-kernel-only-thin ]]; then
  "${PYTHON_BIN}" "${TASK_ROOT}/../../../scripts/infrathon/image_record.py" write "$(basename "${TASK_ROOT}")"
fi
