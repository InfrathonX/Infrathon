#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_command gzip
require_command sha256sum
require_command tar

# Default: pack oracle/FINAL_VERSION and the candidate it selects (the strongest
# oracle, possibly a paper solution that only a local forced-oracle Judge admits;
# see oracle/paper_reference/forced_judge/).  --admissible: pack the candidate
# oracle/ADMISSIBLE_VERSION names, with a FINAL_VERSION that selects it, for
# anything that runs the normal, admission-enforcing Judge (scripts/smoke_test.sh).
selector=FINAL_VERSION
case "${1:-}" in
  "") ;;
  --admissible) selector=ADMISSIBLE_VERSION ;;
  *) die "usage: $0 [--admissible]" ;;
esac

oracle_root="${MIGRATION_ROOT}/oracle"
for required in "${selector}" notes.md; do
  [[ -e "${oracle_root}/${required}" ]] \
    || die "oracle is missing ${required}"
done
version="$(tr -d '[:space:]' <"${oracle_root}/${selector}")"
[[ "${version}" =~ ^[0-9]{1,3}$ && "$((10#${version}))" -ge 1 ]] \
  || die "oracle ${selector} must be an integer from 1 to 999"
printf -v selected 'candidates/kernel_v%03d/submission.py' "$((10#${version}))"
[[ -f "${oracle_root}/${selected}" && ! -L "${oracle_root}/${selected}" ]] \
  || die "oracle selected source is missing: ${selected}"
for member in FINAL_VERSION notes.md "${selected}"; do
  [[ -f "${oracle_root}/${member}" && ! -L "${oracle_root}/${member}" ]] \
    || die "oracle archive member must be a regular non-symlink file: ${member}"
done

pack_root="${oracle_root}"
if [[ "${selector}" != FINAL_VERSION ]]; then
  pack_root="$(mktemp -d)"
  trap 'rm -rf "${pack_root}"' EXIT
  mkdir -p "${pack_root}/$(dirname "${selected}")"
  cp -p "${oracle_root}/${selected}" "${pack_root}/${selected}"
  cp -p "${oracle_root}/notes.md" "${pack_root}/notes.md"
  cp -p "${oracle_root}/${selector}" "${pack_root}/FINAL_VERSION"
fi

mkdir -p "${ARTIFACTS_DIR}"
archive="${ARTIFACTS_DIR}/oracle.tar.gz"
temporary_archive="${archive}.tmp.$$"
tar -C "${pack_root}" \
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
