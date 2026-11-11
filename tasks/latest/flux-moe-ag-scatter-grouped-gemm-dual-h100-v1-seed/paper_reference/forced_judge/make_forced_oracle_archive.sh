#!/usr/bin/env bash
# Pack an oracle archive selecting one candidate (default: oracle/FINAL_VERSION).
# Same deterministic layout as scripts/make_oracle_archive.sh, but the packed
# FINAL_VERSION is the requested version.
# usage: make_forced_oracle_archive.sh OUT.tar.gz [VERSION]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
oracle_root="$(cd "${here}/../../oracle" && pwd)"
out="${1:?usage: make_forced_oracle_archive.sh OUT.tar.gz [VERSION]}"
version="$((10#${2:-$(tr -d '[:space:]' < "${oracle_root}/FINAL_VERSION")}))"
printf -v selected 'candidates/kernel_v%03d/submission.py' "${version}"
[[ -f "${oracle_root}/${selected}" ]] || { echo "missing ${selected}" >&2; exit 1; }
stage="$(mktemp -d)"; trap 'rm -rf "${stage}"' EXIT
mkdir -p "${stage}/$(dirname "${selected}")"
cp -- "${oracle_root}/${selected}" "${stage}/${selected}"
cp -- "${oracle_root}/notes.md" "${stage}/notes.md"
printf '%s\n' "${version}" > "${stage}/FINAL_VERSION"
tar -C "${stage}" --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner \
  --no-recursion -cf - FINAL_VERSION notes.md "${selected}" | gzip -n > "${out}"
sha256sum "${out}"
