#!/usr/bin/env bash
# Pack the oracle archive: the Work seed engine_v001 (admissible control) plus
# oracle/solutions/engine_v002 (forced SGLang EAGLE-3 glue), selected by
# oracle/FINAL_VERSION. Deterministic tar (sorted names, fixed mtime, uid 0).
# usage: make_forced_oracle_archive.sh OUT.tar.gz [VERSION]
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task="$(cd "${here}/../../.." && pwd)"
out="${1:?usage: make_forced_oracle_archive.sh OUT.tar.gz [VERSION]}"
version="${2:-$(tr -d '[:space:]' < "${task}/oracle/FINAL_VERSION")}"
stage="$(mktemp -d)"
trap 'rm -rf "${stage}"' EXIT
mkdir -p "${stage}/solutions"
cp -a "${task}/images/work/workdir/spec_lab/solutions/engine_v001" "${stage}/solutions/"
for dir in "${task}"/oracle/solutions/engine_v*; do
  cp -a "${dir}" "${stage}/solutions/"
done
find "${stage}" -name __pycache__ -prune -exec rm -rf {} +
printf '%s\n' "${version}" > "${stage}/FINAL_VERSION"
printf '%s\n' "Oracle archive: engine_v001 = Work seed (admissible); engine_v002 = forced-oracle SGLang EAGLE-3 glue (oracle/paper_reference/forced_judge/README.md)." > "${stage}/notes.md"
chmod -R u=rwX,go=rX "${stage}"
tar --sort=name --mtime='2026-09-01 00:00:00Z' --owner=0 --group=0 --numeric-owner \
    -C "${stage}" -czf "${out}" solutions FINAL_VERSION notes.md
sha256sum "${out}"
