#!/usr/bin/env bash
set -euo pipefail

compressed_cap=$((128 * 1024 * 1024))
uncompressed_cap=$((512 * 1024 * 1024))
lab_root="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[[ -d "$lab_root" ]] || { echo "submission root is not a directory: $lab_root" >&2; exit 2; }

temporary_dir="$(mktemp -d)"
trap 'rm -rf -- "$temporary_dir"' EXIT
initial_archive="$temporary_dir/initial.tar.gz"
normalized_archive="$temporary_dir/submission.tar.gz"

# Keep this list synchronized with submit_paths and submit_exclude in the task JSON.
tar -C "$lab_root" -czf "$initial_archive" \
  --exclude=.git --exclude=.codex --exclude=auth.json \
  --exclude='*.auth.json' --exclude=build --exclude=tools --exclude=tests \
  --exclude=judge --exclude=__pycache__ \
  candidates FINAL_VERSION notes.md

# SForge canonicalizes GNU tar output, then emits gzip level 9 with mtime zero.
python3 - "$initial_archive" "$normalized_archive" <<'PY'
import copy
import gzip
import io
import sys
import tarfile

source_path, output_path = sys.argv[1:]
raw = io.BytesIO()
with tarfile.open(source_path, "r:*") as source, tarfile.open(fileobj=raw, mode="w") as output:
    for member in source:
        name = member.name
        while name.startswith("./"):
            name = name[2:]
        name = name[:-1] if name.endswith("/") else name
        if not name:
            continue
        info = copy.copy(member)
        info.name = name
        info.pax_headers = {key: value for key, value in member.pax_headers.items() if key not in {"path", "linkpath"}}
        payload = source.extractfile(member) if member.isfile() else None
        output.addfile(info, payload)
with open(output_path, "wb") as handle:
    handle.write(gzip.compress(raw.getvalue(), compresslevel=9, mtime=0))
PY

compressed_bytes="$(stat -c %s "$normalized_archive")"
uncompressed_bytes="$(python3 - "$normalized_archive" <<'PY'
import sys
import tarfile
with tarfile.open(sys.argv[1], "r:gz") as bundle:
    print(sum(member.size for member in bundle if member.isfile()))
PY
)"

printf 'submission compressed bytes: %s / %s\n' "$compressed_bytes" "$compressed_cap"
printf 'submission uncompressed bytes: %s / %s\n' "$uncompressed_bytes" "$uncompressed_cap"
(( compressed_bytes <= compressed_cap && uncompressed_bytes <= uncompressed_cap ))
