#!/usr/bin/env bash
set -Eeuo pipefail

main() {
(( $# == 0 )) || {
  printf 'usage: freeze_head_artifact.sh\n' >&2
  exit 2
}
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# The converted EAGLE-3 draft head and its identity record are multi-GB host
# artifacts produced in the internal calibration lab; they are not part of this
# repository.  Point EAGLE3_HEAD_SOURCE_DIR at the head directory (holding
# model.safetensors, config.json, MANIFEST.json, SOURCE_README.md) and
# EAGLE3_HEADS_JSON at the heads.json identity record.
SOURCE="${EAGLE3_HEAD_SOURCE_DIR:-}"
HEADS_JSON="${EAGLE3_HEADS_JSON:-}"
[[ -n "${SOURCE}" ]] || {
  printf 'set EAGLE3_HEAD_SOURCE_DIR to the converted draft-head directory\n' >&2
  exit 2
}
[[ -n "${HEADS_JSON}" ]] || {
  printf 'set EAGLE3_HEADS_JSON to the heads.json identity record\n' >&2
  exit 2
}
DESTINATION="${EAGLE3_FROZEN_HEAD_DIR:-${HOME}/mlsysforge-artifacts/eagle3-spec-decode/heads/eagle3-head}"

[[ -d "${SOURCE}" && ! -L "${SOURCE}" ]] || {
  printf 'source head directory is missing or unsafe: %s\n' "${SOURCE}" >&2
  exit 2
}
[[ -f "${HEADS_JSON}" && ! -L "${HEADS_JSON}" ]] || {
  printf 'head identity record is missing or unsafe: %s\n' "${HEADS_JSON}" >&2
  exit 2
}
[[ ! -L "${DESTINATION}" ]] || {
  printf 'destination may not be a symlink: %s\n' "${DESTINATION}" >&2
  exit 2
}
mkdir -p "${DESTINATION}"
for name in model.safetensors config.json MANIFEST.json SOURCE_README.md; do
  if [[ ! -f "${SOURCE}/${name}" ]]; then
    printf 'missing source %s\n' "${name}" >&2
    exit 2
  fi
  cp -- "${SOURCE}/${name}" "${DESTINATION}/${name}"
done

python3 - "${DESTINATION}" "${HEADS_JSON}" <<'PY'
"""Validate the frozen head bytes against the lab conversion record."""

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
heads = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
# The frozen head's record, found by its upstream repository id (the key is the
# conversion's output directory name).
expected = next(
    record
    for record in heads["heads"].values()
    if record["repo_id"] == "Tengyunw/qwen3_8b_eagle3"
)


def sha256(path: Path) -> str:
    """Hash one frozen artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


if sha256(root / "model.safetensors") != expected["converted"]["sha256"]:
    raise SystemExit("frozen head model.safetensors hash mismatch")
manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
if (
    manifest["revision"] != expected["revision"]
    or manifest["converted"]["sha256"] != expected["converted"]["sha256"]
):
    raise SystemExit("frozen head manifest identity mismatch")
print(
    json.dumps(
        {
            "status": "PASS",
            "revision": manifest["revision"],
            "model_sha256": sha256(root / "model.safetensors"),
        },
        indent=2,
    )
)
PY
for name in model.safetensors config.json MANIFEST.json SOURCE_README.md; do
  chmod a-w "${DESTINATION}/${name}"
done
}

main "$@"
