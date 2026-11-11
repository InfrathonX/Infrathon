#!/usr/bin/env bash
# Host-side calibration queue: run pilots one after another so no two measure at once.
#   PILOT_PYTHON=<python with torch/triton> FIXTURE_DIR=<dir> GPU=N scripts/pilot_queue.sh OUT_DIR MANIFEST \
#       name:candidate.py[:pilot args] ...
set -euo pipefail
TASK="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out=$1 manifest=$2; shift 2
: "${PILOT_PYTHON:?}" "${FIXTURE_DIR:?}" "${GPU:?}"
# Wait for any pilot already running on this host (a Python process whose command line runs
# pilot_suite.py; shells that merely mention it in their command text do not count).
while pgrep -f '^[^ ]*python[0-9.]* [^ ]*scripts/pilot_suite\.py' >/dev/null; do sleep 20; done
for spec in "$@"; do
  IFS=: read -r name candidate extra <<<"${spec}"
  read -r -a args <<<"${extra//,/ }"
  CUDA_VISIBLE_DEVICES="${GPU}" SAGE_QATTN_FIXTURE_DIR="${FIXTURE_DIR}" TRITON_CACHE_DIR="/tmp/sage-triton-${name}" \
    PYTHONDONTWRITEBYTECODE=1 "${PILOT_PYTHON}" "${TASK}/scripts/pilot_suite.py" "${TASK}/${candidate}" "${manifest}" \
    "${out}/${name}.json" "${args[@]}" > "${out}/${name}.log" 2>&1 || echo "FAILED ${name}"
  grep -v '^\[' "${out}/${name}.log" | head -20
done
