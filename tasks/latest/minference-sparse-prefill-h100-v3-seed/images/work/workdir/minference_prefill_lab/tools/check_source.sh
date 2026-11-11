#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 || "$1" != --candidate-version || ! "$2" =~ ^[1-9][0-9]{0,2}$ ]]; then
  echo "usage: $0 --candidate-version N (CPU source admission only)" >&2
  exit 2
fi
lab_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
printf -v version '%03d' "$2"
exec python3 /workspace/benchmark/source_admission.py "${lab_root}/candidates/kernel_v${version}"
