#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_command gzip
require_command sha256sum
require_command tar

oracle_root="${MIGRATION_ROOT}/oracle"
for required in candidates FINAL_VERSION notes.md; do
  [[ -e "${oracle_root}/${required}" ]] \
    || die "oracle is missing ${required}"
done
version="$(tr -d '[:space:]' <"${oracle_root}/FINAL_VERSION")"
[[ "${version}" =~ ^[0-9]{1,3}$ && "$((10#${version}))" -ge 1 ]] \
  || die "oracle FINAL_VERSION must be an integer from 1 to 999"
printf -v selected 'candidates/kernel_v%03d/submission.py' "$((10#${version}))"
[[ -f "${oracle_root}/${selected}" && ! -L "${oracle_root}/${selected}" ]] \
  || die "oracle selected source is missing: ${selected}"

mkdir -p "${ARTIFACTS_DIR}"
archive="${ARTIFACTS_DIR}/oracle.tar.gz"
temporary_archive="${archive}.tmp.$$"
tar -C "${oracle_root}" \
  --sort=name \
  --mtime='UTC 1970-01-01' \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  -cf - candidates FINAL_VERSION notes.md \
  | gzip -n >"${temporary_archive}"
mv -f -- "${temporary_archive}" "${archive}"
echo "${archive}"
echo "sha256=$(sha256sum "${archive}" | awk '{print $1}')" >&2
