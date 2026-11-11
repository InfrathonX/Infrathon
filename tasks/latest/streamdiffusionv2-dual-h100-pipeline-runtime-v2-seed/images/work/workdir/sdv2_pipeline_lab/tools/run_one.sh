#!/usr/bin/env bash
set -euo pipefail
version=1
case_id=""
while (( $# )); do
  case "$1" in
    --version) version="${2:?missing version}"; shift 2 ;;
    --case) case_id="${2:?missing case id}"; shift 2 ;;
    *) echo "usage: $0 --version N --case CASE_ID" >&2; exit 2 ;;
  esac
done
[[ -n "${case_id}" ]] || { echo "usage: $0 --version N --case CASE_ID" >&2; exit 2; }
exec python3 /workspace/benchmark/public_runner.py --workspace "$PWD" --candidate-version "$version" --case "${case_id}"
