#!/usr/bin/env bash
set -euo pipefail

suite="full"
version=""
family=""
while (($#)); do
  case "$1" in
    --quick)
      suite="quick"
      shift
      ;;
    --edge)
      suite="edge"
      shift
      ;;
    --sanitize)
      suite="sanitize"
      shift
      ;;
    --family)
      suite="family"
      family="${2:-}"
      shift 2
      ;;
    --candidate-version)
      version="${2:-}"
      shift 2
      ;;
    *)
      echo "usage: $0 [--quick | --family NAME | --edge | --sanitize] --candidate-version N" >&2
      exit 2
      ;;
  esac
done
[[ "${version}" =~ ^[0-9]{1,3}$ ]] || { echo "candidate version must be 1-999" >&2; exit 2; }
version="$((10#${version}))"
(( version >= 1 && version <= 999 )) || { echo "candidate version must be 1-999" >&2; exit 2; }

family_args=()
if [[ "${suite}" == "family" ]]; then
  case "${family}" in
    prefill|decode|latency|sequence) family_args=(--family "${family}") ;;
    *) echo "--family must be prefill, decode, latency or sequence" >&2; exit 2 ;;
  esac
fi

exec python3 /workspace/benchmark/public_runner.py \
  --lab-root /app/workdir/grouped_fp8_lab \
  --suite "${suite}" \
  "${family_args[@]}" \
  --candidate-version "${version}"
