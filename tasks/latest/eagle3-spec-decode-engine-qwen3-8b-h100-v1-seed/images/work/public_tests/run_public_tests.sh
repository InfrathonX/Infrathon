#!/usr/bin/env bash
set -Eeuo pipefail

main() {
  local script_dir
  script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  exec python3 "${script_dir}/public_eval.py" "$@"
}

main "$@"
