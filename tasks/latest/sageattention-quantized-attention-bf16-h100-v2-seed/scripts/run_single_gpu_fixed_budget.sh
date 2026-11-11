#!/usr/bin/env bash
set -euo pipefail

# Portable single-allocation entrypoint. This wrapper performs no Docker or
# GPU operation until the production SForge handoff API passes its fail-closed
# probe. The historical two-GPU mode remains available through run_fixed_budget.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

DEVICE_UUID="${SFORGE_ACCELERATOR_DEVICE_UUID:-}"

[[ "${DEVICE_UUID}" =~ ^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]] \
  || die "SFORGE_ACCELERATOR_DEVICE_UUID must be one qualified canonical whole-GPU UUID, never an index, alias, or MIG identity"
[[ ! -v WORK_GPU && ! -v JUDGE_GPU ]] \
  || die "single-allocation mode rejects WORK_GPU/JUDGE_GPU; use the scheduler-issued UUID contract or launch --legacy-dual-gpu"

require_sforge

CANONICAL_ALLOCATION_KEY="$(
  "${PYTHON_BIN}" - "${DEVICE_UUID}" <<'PY'
import sys
from sforge.harness.backend import AcceleratorAllocation
from sforge.harness.resource_handoff import canonical_allocation_key

print(canonical_allocation_key(AcceleratorAllocation("nvidia", (sys.argv[1],))))
PY
)"
ALLOCATION_KEY="${SFORGE_ACCELERATOR_ALLOCATION_KEY:-${CANONICAL_ALLOCATION_KEY}}"
[[ "${ALLOCATION_KEY}" =~ ^allocation-sha256:[0-9a-f]{64}$ ]] \
  || die "SFORGE_ACCELERATOR_ALLOCATION_KEY must be allocation-sha256:<64 lowercase hex>"
[[ "${ALLOCATION_KEY}" == "${CANONICAL_ALLOCATION_KEY}" ]] \
  || die "SFORGE_ACCELERATOR_ALLOCATION_KEY does not match the canonical device-set digest"

TRUSTED_LEASE_ID="${SFORGE_TRUSTED_LEASE_ID:-}"
if [[ -z "${TRUSTED_LEASE_ID}" ]]; then
  TRUSTED_LEASE_ID="$(
    "${PYTHON_BIN}" -c \
      'import secrets; print("lease-sage-qattn-" + secrets.token_hex(20))'
  )"
fi
[[ "${TRUSTED_LEASE_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$ ]] \
  || die "SFORGE_TRUSTED_LEASE_ID must be a label-safe never-reused identity"

export SFORGE_SINGLE_GPU_HANDOFF=1
export SFORGE_ACCELERATOR_ALLOCATION_KEY="${ALLOCATION_KEY}"
export SFORGE_ACCELERATOR_OBSERVER=nvml
export SFORGE_TRUSTED_LEASE_ID="${TRUSTED_LEASE_ID}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/single_gpu_runtime_preflight.py"

# The common fixed-budget body owns the Judge server, immutable final capture
# and artifact validation. It takes its single-allocation branch only after
# the production capability probe above succeeds; there is no legacy fallback.
exec "${SCRIPT_DIR}/run_fixed_budget.sh" "$@"
