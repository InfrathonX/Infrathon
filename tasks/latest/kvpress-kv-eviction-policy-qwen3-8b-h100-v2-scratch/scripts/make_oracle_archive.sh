#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; source "${SCRIPT_DIR}/lib.sh"
version="$(tr -d '[:space:]' < "${MIGRATION_ROOT}/oracle/FINAL_VERSION")"; [[ "$version" =~ ^[0-9]{1,3}$ ]] || die "bad Oracle version"
printf -v selected 'candidates/policy_v%03d/submission.py' "$((10#$version))"; [[ -f "${MIGRATION_ROOT}/oracle/$selected" ]] || die "Oracle source missing"
mkdir -p "$ARTIFACTS_DIR"; archive="$ARTIFACTS_DIR/oracle.tar.gz"; temporary="$archive.tmp.$$"
tar -C "${MIGRATION_ROOT}/oracle" --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner --no-recursion -cf - FINAL_VERSION notes.md "$selected" | gzip -n > "$temporary"
mv -- "$temporary" "$archive"; echo "$archive"; echo "sha256=$(sha256sum "$archive" | awk '{print $1}')" >&2

