#!/usr/bin/env bash
set -euo pipefail
suite="full"; version=""
while (($#)); do
  case "$1" in
    --quick) suite="quick"; shift ;;
    --candidate-version) version="${2:-}"; shift 2 ;;
    *) echo "usage: $0 [--quick] --candidate-version N" >&2; exit 2 ;;
  esac
done
[[ "$version" =~ ^[0-9]{1,3}$ ]] || { echo "candidate version must be 1-999" >&2; exit 2; }
status=0
python3 /workspace/benchmark/public_runner.py --lab-root /app/workdir/kv_scoring_lab --suite "$suite" --candidate-version "$((10#$version))" || status=$?
if ! /app/workdir/kv_scoring_lab/tools/check_submission_size.sh; then
  echo "WARNING: submission exceeds an SForge archive size cap" >&2
fi
exit "$status"
