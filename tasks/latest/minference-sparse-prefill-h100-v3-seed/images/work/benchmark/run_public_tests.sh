#!/usr/bin/env bash
set -euo pipefail
version=""
suite=full
while (($#)); do
  case "$1" in
    --quick) suite=quick; shift ;;
    --boundaries-only) suite=boundaries; shift ;;
    --candidate-version)
      [[ $# -ge 2 && -z "${version}" ]] || { echo 'provide --candidate-version once with N' >&2; exit 2; }
      version="$2"; shift 2 ;;
    *) echo "usage: $0 [--quick | --boundaries-only] --candidate-version N" >&2; exit 2 ;;
  esac
done
[[ "${version}" =~ ^[1-9][0-9]{0,2}$ ]] || { echo 'candidate version must be 1-999' >&2; exit 2; }
exec python3 /workspace/benchmark/public_runner.py --candidate-version "${version}" --suite "${suite}"
