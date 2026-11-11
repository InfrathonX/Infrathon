#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_command gzip
require_command sha256sum
require_command tar

baseline_root="${MIGRATION_ROOT}/baseline"
for required in candidates FINAL_VERSION notes.md; do
  [[ -e "${baseline_root}/${required}" ]] \
    || die "baseline is missing ${required}"
done
version="$(tr -d '[:space:]' <"${baseline_root}/FINAL_VERSION")"
[[ "${version}" =~ ^[0-9]{1,3}$ && "$((10#${version}))" -ge 1 ]] \
  || die "baseline FINAL_VERSION must be an integer from 1 to 999"
printf -v selected 'candidates/kernel_v%03d/submission.py' "$((10#${version}))"
[[ -f "${baseline_root}/${selected}" && ! -L "${baseline_root}/${selected}" ]] \
  || die "baseline selected source is missing: ${selected}"

mkdir -p "${ARTIFACTS_DIR}"
archive="${ARTIFACTS_DIR}/baseline.tar.gz"
temporary_archive="${archive}.tmp.$$"
tar -C "${baseline_root}" \
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
