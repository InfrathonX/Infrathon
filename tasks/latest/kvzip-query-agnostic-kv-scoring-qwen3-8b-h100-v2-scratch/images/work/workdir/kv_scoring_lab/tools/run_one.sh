#!/usr/bin/env bash
set -euo pipefail
version=""
while (($#)); do case "$1" in --version) version="${2:-}"; shift 2;; *) echo "usage: $0 --version N" >&2; exit 2;; esac; done
[[ "$version" =~ ^[0-9]{1,3}$ ]] || { echo "version must be 1-999" >&2; exit 2; }
exec python3 /workspace/benchmark/public_runner.py --lab-root /app/workdir/kv_scoring_lab --suite one --candidate-version "$((10#$version))"

