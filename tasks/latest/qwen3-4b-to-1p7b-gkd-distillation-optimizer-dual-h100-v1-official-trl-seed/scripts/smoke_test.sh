#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

export PYTHONDONTWRITEBYTECODE=1

# Host-side prerequisites for the CPU public tests.
#
# `run_public_tests.sh --cpu-only` builds the official-TRL reference, so the
# interpreter that runs it needs torch, numpy, transformers and the *exact*
# pinned trl wheel that images/work/Dockerfile installs.  Without them the eval
# prints `PackageNotFoundError: No package metadata was found for trl`, exits 1,
# and `set -euo pipefail` ends the smoke before pytest ever runs.  Check first,
# and say what to install.
OFFICIAL_TRL_VERSION="0.12.1"
if ! missing="$(python3 - "${OFFICIAL_TRL_VERSION}" <<'PY'
import importlib.metadata as metadata
import importlib.util
import sys

pin = sys.argv[1]
missing = []
for module in ("torch", "numpy", "transformers"):
    if importlib.util.find_spec(module) is None:
        missing.append(module)
try:
    found = metadata.version("trl")
except metadata.PackageNotFoundError:
    missing.append(f"trl=={pin} (not installed)")
else:
    if found != pin:
        missing.append(f"trl=={pin} (found {found})")
if missing:
    print(", ".join(missing))
    raise SystemExit(1)
PY
)"; then
  die "the host interpreter that runs this smoke ($(command -v python3), $(python3 -V 2>&1)) is missing: ${missing}
The CPU public tests import them, so the smoke would fail before pytest. Install into that same interpreter:
  uv pip install --python \"$(command -v python3)\" --index-url https://download.pytorch.org/whl/cpu torch numpy
  uv pip install --python \"$(command -v python3)\" transformers 'trl==${OFFICIAL_TRL_VERSION}'
trl must be exactly ${OFFICIAL_TRL_VERSION}: runtime/hf_reference.py binds the official baseline to that version and to the SHA-256 of its trainer source, the same wheel images/work/Dockerfile installs."
fi
python3 "${TASK_ROOT}/scripts/validate_layout.py" "${TASK_ROOT}"
bash "${TASK_ROOT}/images/work/public_tests/run_public_tests.sh" --cpu-only
python3 -m pytest -q "${TASK_ROOT}/tests"
