#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_command gzip
require_command sha256sum
require_command tar

oracle_root="${MIGRATION_ROOT}/oracle"
# ORACLE_SELECTOR=ADMISSIBLE_VERSION packs the best admissible candidate (for
# anything that must pass the normal, admission-enforcing Judge) as
# artifacts/oracle-admissible.tar.gz; the default packs FINAL_VERSION.
selector="${ORACLE_SELECTOR:-FINAL_VERSION}"
[[ "${selector}" == FINAL_VERSION || "${selector}" == ADMISSIBLE_VERSION ]] \
  || die "ORACLE_SELECTOR must be FINAL_VERSION or ADMISSIBLE_VERSION"
for required in "${selector}" notes.md; do
  [[ -e "${oracle_root}/${required}" ]] \
    || die "oracle is missing ${required}"
done
version="$(tr -d '[:space:]' <"${oracle_root}/${selector}")"
[[ "${version}" =~ ^[0-9]{1,3}$ && "$((10#${version}))" -ge 1 ]] \
  || die "oracle FINAL_VERSION must be an integer from 1 to 999"
printf -v selected 'candidates/kernel_v%03d/submission.py' "$((10#${version}))"
[[ -f "${oracle_root}/${selected}" && ! -L "${oracle_root}/${selected}" ]] \
  || die "oracle selected source is missing: ${selected}"
for member in "${selector}" notes.md "${selected}"; do
  [[ -f "${oracle_root}/${member}" && ! -L "${oracle_root}/${member}" ]] \
    || die "oracle archive member must be a regular non-symlink file: ${member}"
done

mkdir -p "${ARTIFACTS_DIR}"
archive="${ARTIFACTS_DIR}/oracle.tar.gz"
selector_root="${oracle_root}"
if [[ "${selector}" == ADMISSIBLE_VERSION ]]; then
  archive="${ARTIFACTS_DIR}/oracle-admissible.tar.gz"
  selector_root="$(mktemp -d)"
  trap 'rm -rf "${selector_root}"' EXIT
  printf '%d\n' "$((10#${version}))" >"${selector_root}/FINAL_VERSION"
fi
temporary_archive="${archive}.tmp.$$"
tar --sort=name \
  --mtime='UTC 1970-01-01' \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  --no-recursion \
  -cf - -C "${selector_root}" FINAL_VERSION -C "${oracle_root}" notes.md "${selected}" \
  | gzip -n >"${temporary_archive}"
mv -f -- "${temporary_archive}" "${archive}"
echo "${archive}"
echo "sha256=$(sha256sum "${archive}" | awk '{print $1}')" >&2
