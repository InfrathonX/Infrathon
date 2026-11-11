#!/usr/bin/env python3
"""Compute a deterministic identity for trusted runtime Python and JSON files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    """Hash one trusted runtime file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(root: Path) -> dict[str, Any]:
    """Bind both the selected file set and each file's contents."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("runtime root must be a real directory")
    root = root.resolve(strict=True)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"runtime symlink is forbidden: {path}")
        if path.is_file() and path.suffix in {".py", ".json"}:
            files[path.relative_to(root).as_posix()] = sha256(path)
    if not files:
        raise ValueError("runtime identity file set is empty")
    canonical = "".join(f"{name}\0{digest}\n" for name, digest in files.items())
    return {
        "schema_version": 1,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "file_count": len(files),
        "files": files,
    }


def main(argv: list[str] | None = None) -> int:
    """Print a deterministic runtime identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?")
    args = parser.parse_args(argv)
    root = args.root or (
        Path(__file__).parents[1] / "images/work/workdir/spec_lab/runtime"
    )
    print(json.dumps(identity(root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
