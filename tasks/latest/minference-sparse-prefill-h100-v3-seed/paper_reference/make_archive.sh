#!/usr/bin/env bash
# Packs the paper-derived oracle candidate as a submission archive:
#   FINAL_VERSION 2, candidates/kernel_v001 = oracle/ seed control (unchanged),
#   candidates/kernel_v002 = paper_reference/candidates/kernel_v002 (MInference + guard).
# Usage: paper_reference/make_archive.sh [OUTPUT]   (default artifacts/minference-prefill-v3-paper-oracle.tar.gz)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "${HERE}/.." && pwd)"
output="${1:-${ROOT}/artifacts/minference-prefill-v3-paper-oracle.tar.gz}"
[[ ! -e "${output}" ]] || { echo "refusing to replace archive: ${output}" >&2; exit 1; }
mkdir -p "$(dirname "${output}")"
staging="$(mktemp -d)"; trap 'rm -rf -- "${staging}"' EXIT
mkdir -p "${staging}/candidates/kernel_v001" "${staging}/candidates/kernel_v002"
printf '2\n' > "${staging}/FINAL_VERSION"
install -m 0644 "${HERE}/README.md" "${staging}/notes.md"
install -m 0644 "${ROOT}/oracle/candidates/kernel_v001/submission.py" "${staging}/candidates/kernel_v001/submission.py"
install -m 0644 "${HERE}/candidates/kernel_v002/submission.py" "${staging}/candidates/kernel_v002/submission.py"
chmod 0644 "${staging}/FINAL_VERSION"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner --mode=u=rwX,go=rX \
  -C "${staging}" -czf "${output}" FINAL_VERSION notes.md candidates
echo "${output}"; sha256sum "${output}" | awk '{print $1}'
