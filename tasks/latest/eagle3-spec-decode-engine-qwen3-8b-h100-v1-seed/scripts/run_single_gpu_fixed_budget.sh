#!/usr/bin/env bash
# Portable single-allocation entrypoint for the EAGLE-3 task. This wrapper
# performs no Docker or GPU operation until the production SForge handoff API
# passes its fail-closed probe. The historical two-GPU mode remains available
# through run_fixed_budget.sh (launch/eagle3-spec-decode/run.sh --legacy-dual-gpu).
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

single_allocation_preflight() {
  # Validate the scheduler-issued whole-GPU contract, derive the canonical
  # allocation digest and a never-reused lease ID, then run the production
  # API probe before the fixed-budget body may touch Docker or GPU state.
  local device_uuid="${SFORGE_ACCELERATOR_DEVICE_UUID:-}"
  local canonical_allocation_key allocation_key trusted_lease_id

  [[ "${device_uuid}" =~ ^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]] \
    || die "SFORGE_ACCELERATOR_DEVICE_UUID must be one qualified canonical whole-GPU UUID, never an index, alias, or MIG identity"
  [[ ! -v WORK_GPU && ! -v JUDGE_GPU ]] \
    || die "single-allocation mode rejects WORK_GPU/JUDGE_GPU; use the scheduler-issued UUID contract or launch --legacy-dual-gpu"

  require_sforge

  canonical_allocation_key="$(
    "${PYTHON_BIN}" - "${device_uuid}" <<'PY'
import sys
from sforge.harness.backend import AcceleratorAllocation
from sforge.harness.resource_handoff import canonical_allocation_key

print(canonical_allocation_key(AcceleratorAllocation("nvidia", (sys.argv[1],))))
PY
  )"
  allocation_key="${SFORGE_ACCELERATOR_ALLOCATION_KEY:-${canonical_allocation_key}}"
  [[ "${allocation_key}" =~ ^allocation-sha256:[0-9a-f]{64}$ ]] \
    || die "SFORGE_ACCELERATOR_ALLOCATION_KEY must be allocation-sha256:<64 lowercase hex>"
  [[ "${allocation_key}" == "${canonical_allocation_key}" ]] \
    || die "SFORGE_ACCELERATOR_ALLOCATION_KEY does not match the canonical device-set digest"

  trusted_lease_id="${SFORGE_TRUSTED_LEASE_ID:-}"
  if [[ -z "${trusted_lease_id}" ]]; then
    trusted_lease_id="$(
      "${PYTHON_BIN}" -c \
        'import secrets; print("lease-eagle3-" + secrets.token_hex(20))'
    )"
  fi
  [[ "${trusted_lease_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$ ]] \
    || die "SFORGE_TRUSTED_LEASE_ID must be a label-safe never-reused identity"

  export SFORGE_SINGLE_GPU_HANDOFF=1
  export SFORGE_ACCELERATOR_ALLOCATION_KEY="${allocation_key}"
  export SFORGE_ACCELERATOR_OBSERVER=nvml
  export SFORGE_TRUSTED_LEASE_ID="${trusted_lease_id}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/single_gpu_runtime_preflight.py"
}

main() {
  # The formal fixed-budget body owns the policy/image preflight, the Judge
  # server, the restricted proxy, the immutable final capture and artifact
  # validation. It takes its single-allocation branch only after the
  # production capability probe above succeeds; there is no legacy fallback.
  single_allocation_preflight
  exec "${SCRIPT_DIR}/run_fixed_budget.sh" "$@"
}

main "$@"
