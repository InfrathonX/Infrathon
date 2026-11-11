#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_command gzip
require_command sha256sum
require_command tar

oracle_root="${MIGRATION_ROOT}/oracle"
# The archive must pass the normal (admission-enforcing) Judge, so it packs the
# admissible Oracle: oracle/ADMISSIBLE_VERSION when present, else
# oracle/FINAL_VERSION.  oracle/FINAL_VERSION may select a stronger upstream
# candidate that only the local forced-oracle Judge can score
# (oracle/paper_reference/forced_judge/).  The packed FINAL_VERSION member
# names the packed candidate.
selector="${oracle_root}/ADMISSIBLE_VERSION"
[[ -e "${selector}" ]] || selector="${oracle_root}/FINAL_VERSION"
for required in "${selector}" "${oracle_root}/notes.md"; do
  [[ -e "${required}" ]] \
    || die "oracle is missing ${required#"${oracle_root}/"}"
done
version="$(tr -d '[:space:]' <"${selector}")"
[[ "${version}" =~ ^[0-9]{1,3}$ && "$((10#${version}))" -ge 1 ]] \
  || die "oracle ${selector##*/} must be an integer from 1 to 999"
printf -v selected 'candidates/kernel_v%03d/submission.py' "$((10#${version}))"
[[ -f "${oracle_root}/${selected}" && ! -L "${oracle_root}/${selected}" ]] \
  || die "oracle selected source is missing: ${selected}"
for member in "${selector##*/}" notes.md "${selected}"; do
  [[ -f "${oracle_root}/${member}" && ! -L "${oracle_root}/${member}" ]] \
    || die "oracle archive member must be a regular non-symlink file: ${member}"
done

stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT
mkdir -p "${stage}/$(dirname "${selected}")"
cp -- "${selector}" "${stage}/FINAL_VERSION"
cp -- "${oracle_root}/notes.md" "${stage}/notes.md"
cp -- "${oracle_root}/${selected}" "${stage}/${selected}"
chmod 0644 "${stage}/FINAL_VERSION" "${stage}/notes.md" "${stage}/${selected}"

mkdir -p "${ARTIFACTS_DIR}"
archive="${ARTIFACTS_DIR}/oracle.tar.gz"
temporary_archive="${archive}.tmp.$$"
tar -C "${stage}" \
  --sort=name \
  --mtime='UTC 1970-01-01' \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  --no-recursion \
  -cf - FINAL_VERSION notes.md "${selected}" \
  | gzip -n >"${temporary_archive}"
mv -f -- "${temporary_archive}" "${archive}"
echo "${archive}"
echo "sha256=$(sha256sum "${archive}" | awk '{print $1}')" >&2
