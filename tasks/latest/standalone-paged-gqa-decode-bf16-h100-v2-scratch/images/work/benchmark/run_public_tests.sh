#!/usr/bin/env bash
set -euo pipefail

suite="full"
version=""
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
      case "${2:-}" in
        decode|spec|long|stream) suite="$2" ;;
        *) echo "--family must be decode, spec, long or stream" >&2; exit 2 ;;
      esac
      shift 2
      ;;
    --candidate-version)
      version="${2:-}"
      shift 2
      ;;
    *)
      echo "usage: $0 [--quick | --family F | --edge | --sanitize] --candidate-version N" >&2
      exit 2
      ;;
  esac
done
[[ "${version}" =~ ^[0-9]{1,3}$ ]] || { echo "candidate version must be 1-999" >&2; exit 2; }
version="$((10#${version}))"
(( version >= 1 && version <= 999 )) || { echo "candidate version must be 1-999" >&2; exit 2; }

exec python3 /workspace/benchmark/public_runner.py \
  --lab-root /app/workdir/paged_gqa_decode_lab \
  --suite "${suite}" \
  --candidate-version "${version}"
