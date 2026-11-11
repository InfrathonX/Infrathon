#!/usr/bin/env bash
# Build the strongest admissible upstream-derived host oracle archives.
# See host_reference/official_oracle_fa3epi_h100/README.md.
#
#   artifacts/official-oracle-fa3epi.tar.gz      official HEAD adapter + the
#                                                 upstream fa3-branch epilogue
#   artifacts/official-oracle-fa3epi-h100.tar.gz  the same + an H100 NVL retune
#                                                 of the H100 heuristic table
#
# Both are packaged from the committed generated_task_adapter.py, so no
# upstream checkout is needed.  To re-derive the adapter from upstream as well,
# point FLASH_KMEANS_UPSTREAM_HEAD at a flash-kmeans checkout pinned to
# b2c1c83b49301dda4807990d0a90ae4bf9cdd3b0:
#
#   git clone https://github.com/svg-project/flash-kmeans /path/to/fk-head
#   git -C /path/to/fk-head checkout b2c1c83b49301dda4807990d0a90ae4bf9cdd3b0
#   FLASH_KMEANS_UPSTREAM_HEAD=/path/to/fk-head scripts/make_fa3epi_h100_oracle_archive.sh
#
# The script then builds the HEAD adapter with build_official_repo_head_archive.py,
# applies fa3_epilogue.patch and h100_retune.patch, and fails unless the result
# is byte-identical to the committed adapter.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_command patch

reference="${MIGRATION_ROOT}/host_reference/official_oracle_fa3epi_h100"
adapter="${reference}/generated_task_adapter.py"
# SHA-256 of the kernel sources measured on the Judge (see the README).
HEAD_SHA256=df209764750a95f5d77bd356abc7b95d7ad949aecc9f35e66e9f147c1663ea54
FA3EPI_SHA256=f1af3b86ed34eed9f69d7545d6fc85a85b48cdb14be6ad16f2b4197f57631394
FA3EPI_H100_SHA256=849f8a1dc585c13442ddd62f14c5fc50d9ffb3576945569281443add08424dbe

sha_of() { sha256sum "$1" | awk '{print $1}'; }
[[ -f "${adapter}" && ! -L "${adapter}" ]] || die "missing ${adapter}"
[[ "$(sha_of "${adapter}")" == "${FA3EPI_H100_SHA256}" ]] \
  || die "${adapter} drifted from the measured source"

scratch="$(mktemp -d)"
trap 'rm -rf "${scratch}"' EXIT
mkdir -p "${scratch}/fa3epi" "${scratch}/fa3epi_h100"

upstream_head="${FLASH_KMEANS_UPSTREAM_HEAD:-}"
if [[ -n "${upstream_head}" ]]; then
  [[ -f "${upstream_head}/flash_kmeans/assign_euclid_triton.py" ]] \
    || die "${upstream_head} is not a flash-kmeans checkout"
  python3 "${SCRIPT_DIR}/build_official_repo_head_archive.py" \
    --vendor-root "${upstream_head}" \
    --source-output "${scratch}/head/generated_task_adapter.py" \
    --output "${scratch}/head.tar.gz" >/dev/null
  [[ "$(sha_of "${scratch}/head/generated_task_adapter.py")" == "${HEAD_SHA256}" ]] \
    || die "HEAD adapter differs from b2c1c83 (is the checkout pinned?)"
  cp "${scratch}/head/generated_task_adapter.py" "${scratch}/fa3epi/"
  patch -s -p1 -d "${scratch}/fa3epi" <"${reference}/fa3_epilogue.patch"
  cp "${scratch}/fa3epi/generated_task_adapter.py" "${scratch}/fa3epi_h100/"
  patch -s -p1 -d "${scratch}/fa3epi_h100" <"${reference}/h100_retune.patch"
  cmp -s "${scratch}/fa3epi_h100/generated_task_adapter.py" "${adapter}" \
    || die "re-derived adapter differs from ${adapter}"
  echo "re-derived from upstream b2c1c83: byte-identical" >&2
else
  cp "${adapter}" "${scratch}/fa3epi_h100/generated_task_adapter.py"
  cp "${adapter}" "${scratch}/fa3epi/generated_task_adapter.py"
  patch -s -R -p1 -d "${scratch}/fa3epi" <"${reference}/h100_retune.patch"
fi
[[ "$(sha_of "${scratch}/fa3epi/generated_task_adapter.py")" == "${FA3EPI_SHA256}" ]] \
  || die "fa3-epilogue adapter differs from the measured source"

mkdir -p "${ARTIFACTS_DIR}"
package() {  # $1 source, $2 output, $3 kind
  python3 - "$1" "$2" "$3" <<'PY'
import gzip
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

source, output, kind = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
data = source.read_bytes()
notes = {
    "kind": kind,
    "base_commit": "b2c1c83b49301dda4807990d0a90ae4bf9cdd3b0",
    "epilogue_from": "svg-project/flash-kmeans fa3 branch 39be82a52e66964aaea816c7740db5526bb9fc90 (unmerged)",
    "h100_retune": kind.endswith("h100_retune"),
    "source_sha256": hashlib.sha256(data).hexdigest(),
}
text = (
    "# Official HEAD adapter + upstream fa3-branch epilogue (host-only oracle)\n\n"
    "```json\n" + json.dumps(notes, indent=2, sort_keys=True) + "\n```\n"
)


def add(archive, name, payload):
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o644
    info.mtime = 0
    archive.addfile(info, io.BytesIO(payload))


temporary = output.with_name(output.name + ".tmp")
with temporary.open("wb") as raw:
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            add(archive, "kernels/flash_kmeans_v001.py", data)
            add(archive, "FINAL_VERSION", b"1\n")
            add(archive, "notes.md", text.encode("utf-8"))
temporary.replace(output)
print(f"{output} sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")
PY
}
package "${scratch}/fa3epi/generated_task_adapter.py" \
  "${ARTIFACTS_DIR}/official-oracle-fa3epi.tar.gz" \
  official_head_adapter_plus_fa3_noxsq_epilogue
package "${scratch}/fa3epi_h100/generated_task_adapter.py" \
  "${ARTIFACTS_DIR}/official-oracle-fa3epi-h100.tar.gz" \
  official_head_adapter_plus_fa3_noxsq_epilogue_h100_retune
