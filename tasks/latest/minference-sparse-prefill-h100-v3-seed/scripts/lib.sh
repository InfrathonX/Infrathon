#!/usr/bin/env bash
set -euo pipefail
MIGRATION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOSITORY_ROOT="$(cd "${MIGRATION_ROOT}/../../.." && pwd)"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${MIGRATION_ROOT}/artifacts}"
die() { echo "error: $*" >&2; exit 1; }
sha256_file() { sha256sum "$1" | awk '{print $1}'; }
