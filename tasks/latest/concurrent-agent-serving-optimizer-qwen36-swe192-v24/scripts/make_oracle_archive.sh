#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_command cp
require_command tar
require_command sha256sum
[[ -x "${PYTHON_BIN}" ]] || die "missing MLSys Forge Python: ${PYTHON_BIN}"

source_root="${MIGRATION_ROOT}/oracle/reference_gateway"
[[ -d "${source_root}" && ! -L "${source_root}" ]] \
  || die "missing Oracle reference gateway: ${source_root}"
[[ -f "${source_root}/launch.sh" && ! -L "${source_root}/launch.sh" \
   && -x "${source_root}/launch.sh" ]] \
  || die "Oracle launch.sh must be a regular executable file"
special_entry="$(find "${source_root}" -mindepth 1 ! -type f ! -type d -print -quit)"
[[ -z "${special_entry}" ]] \
  || die "Oracle tree contains a symlink or special entry: ${special_entry}"

mkdir -p "${ARTIFACTS_DIR}"
output="${1:-${ARTIFACTS_DIR}/oracle-submission.tar.gz}"
[[ ! -L "${output}" ]] || die "refusing symlink archive target: ${output}"
staging="$(mktemp -d "${ARTIFACTS_DIR}/.oracle-stage.XXXXXX")"
temporary="$(mktemp "${ARTIFACTS_DIR}/.oracle-archive.XXXXXX.tar.gz")"
cleanup() {
  rm -f -- "${temporary}"
  if [[ "${staging}" == "${ARTIFACTS_DIR}/.oracle-stage."* ]]; then
    rm -rf -- "${staging}"
  fi
}
trap cleanup EXIT

mkdir "${staging}/submission"
cp -a "${source_root}/." "${staging}/submission/"
chmod 0755 "${staging}/submission/launch.sh"

# Stable ownership, ordering, and mtimes make the archive digest reproducible.
tar \
  --sort=name \
  --mtime='UTC 1970-01-01' \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  -C "${staging}" \
  -czf "${temporary}" \
  submission

"${PYTHON_BIN}" - "${source_root}" "${temporary}" <<'PY'
from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import sys
import tarfile

source = Path(sys.argv[1])
archive = Path(sys.argv[2])

expected = {
    p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in source.rglob("*")
    if p.is_file() and not p.is_symlink()
}
observed: dict[str, str] = {}
launch_mode = 0
with tarfile.open(archive, "r:gz") as bundle:
    for member in bundle.getmembers():
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise SystemExit(f"unsafe Oracle archive member: {member.name!r}")
        if path.parts[0] != "submission" or member.issym() or member.islnk():
            raise SystemExit(f"unexpected Oracle archive member: {member.name!r}")
        if member.isdir():
            continue
        if not member.isfile() or len(path.parts) < 2:
            raise SystemExit(f"unexpected Oracle archive entry type: {member.name!r}")
        stream = bundle.extractfile(member)
        if stream is None:
            raise SystemExit(f"cannot read Oracle archive member: {member.name!r}")
        relative = PurePosixPath(*path.parts[1:]).as_posix()
        observed[relative] = hashlib.sha256(stream.read()).hexdigest()
        if relative == "launch.sh":
            launch_mode = member.mode

if observed != expected:
    raise SystemExit("Oracle archive is not an exact submission/ copy of reference_gateway")
if launch_mode & 0o111 == 0:
    raise SystemExit("Oracle archive launch.sh lost its executable bit")
PY

mkdir -p "$(dirname "${output}")"
mv -f -- "${temporary}" "${output}"
trap - EXIT
rm -rf -- "${staging}"
echo "${output}"
echo "sha256=$(sha256sum "${output}" | awk '{print $1}')" >&2
