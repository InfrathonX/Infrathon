#!/usr/bin/env bash
set -euo pipefail

suite="full"
version=""
extra=()
while (($#)); do
  case "$1" in
    --quick)
      suite="quick"
      shift
      ;;
    --candidate-version)
      version="${2:-}"
      shift 2
      ;;
    --case)
      extra+=(--case "${2:-}")
      shift 2
      ;;
    --perf-only)
      extra+=(--perf-only)
      shift
      ;;
    *)
      echo "usage: $0 [--quick] [--case ID] [--perf-only] --candidate-version N" >&2
      exit 2
      ;;
  esac
done
[[ "${version}" =~ ^[0-9]{1,3}$ ]] || { echo "candidate version must be 1-999" >&2; exit 2; }
version="$((10#${version}))"
(( version >= 1 && version <= 999 )) || { echo "candidate version must be 1-999" >&2; exit 2; }

exec python3 /workspace/benchmark/public_runner.py \
  --lab-root /app/workdir/attn_qat_lab \
  --suite "${suite}" \
  --candidate-version "${version}" \
  "${extra[@]}"
