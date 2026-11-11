#!/usr/bin/env python3
"""Rewrite images/judge/protected_hashes.txt from the build-context sources.

The manifest pins the Judge's own sources so a submission cannot swap them at
evaluation time. Every legitimate edit to a protected file therefore has to be
followed by a manifest update, and doing that by hand — reading a validator
failure, running sha256sum, pasting the digest — is pure friction that has no
bearing on what the manifest protects against.

The runtime-path to source mapping is imported from validate_layout rather than
restated, so this script and the check that guards it can never disagree.

    scripts/refresh_protected_hashes.py            # rewrite in place
    scripts/refresh_protected_hashes.py --check    # report drift, change nothing

Rebuild the Judge image afterwards: the manifest is baked in beside the files
it covers.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_layout import (  # noqa: E402
    PROTECTED_EXACT_SOURCES,
    PROTECTED_RUNTIME_PREFIXES,
)

TASK_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = TASK_ROOT / "images" / "judge" / "protected_hashes.txt"


def source_for(runtime_path: str) -> Path:
    exact = PROTECTED_EXACT_SOURCES.get(runtime_path)
    if exact is not None:
        return TASK_ROOT / exact
    for prefix, source_root in PROTECTED_RUNTIME_PREFIXES.items():
        if runtime_path.startswith(prefix):
            return TASK_ROOT / source_root / runtime_path[len(prefix) :]
    raise SystemExit(f"protected path has no build-context source: {runtime_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero on drift instead of rewriting the manifest",
    )
    args = parser.parse_args()

    lines: list[str] = []
    drifted: list[str] = []
    for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            lines.append(raw)
            continue
        parts = raw.split()
        if len(parts) != 2:
            raise SystemExit(f"malformed manifest entry: {raw!r}")
        recorded, runtime_path = parts
        source = source_for(runtime_path)
        if not source.is_file() or source.is_symlink():
            raise SystemExit(f"protected source is missing: {source}")
        actual = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual != recorded:
            drifted.append(f"{runtime_path}: {recorded[:12]} -> {actual[:12]}")
        lines.append(f"{actual}  {runtime_path}")

    if args.check:
        for entry in drifted:
            print(entry)
        return 1 if drifted else 0
    if not drifted:
        print("protected hashes already current")
        return 0
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for entry in drifted:
        print(f"updated {entry}")
    print(f"{len(drifted)} entr{'y' if len(drifted) == 1 else 'ies'} rewritten")
    print("rebuild the Judge image: scripts/build_images.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
