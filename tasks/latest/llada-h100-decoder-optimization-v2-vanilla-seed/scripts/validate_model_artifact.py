#!/usr/bin/env python3
"""Validate the whole-repository HF cache mount for the pinned checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(
            os.environ.get("LLADA_MODEL_REPO")
            or Path.home()
            / ".cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs/model_manifest.json",
    )
    parser.add_argument("--full", action="store_true", help="hash all six 16 GB weight shards")
    parser.add_argument("--output", type=Path, help="also write the JSON attestation to this file")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    repo = args.repo.resolve(strict=True)
    if not repo.is_dir():
        raise SystemExit("model repo is not a directory")
    revision = manifest["revision"]
    snapshot = repo / "snapshots" / revision
    if not snapshot.is_dir() or snapshot.is_symlink():
        raise SystemExit(f"missing real snapshot directory: {snapshot}")
    logical_size = 0
    checked_hashes = 0
    for relative, item in manifest["files"].items():
        expected_size, expected_hash = item
        path = snapshot / relative
        if not path.exists():
            raise SystemExit(f"missing model file: {relative}")
        resolved = path.resolve(strict=True)
        if not contained(resolved, repo):
            raise SystemExit(f"model symlink escapes repository: {relative}")
        size = resolved.stat().st_size
        if size != expected_size:
            raise SystemExit(f"size mismatch for {relative}: {size} != {expected_size}")
        logical_size += size
        is_weight = relative.endswith(".safetensors")
        if args.full or not is_weight:
            actual_hash = sha256(resolved)
            checked_hashes += 1
            if actual_hash != expected_hash:
                raise SystemExit(f"SHA-256 mismatch for {relative}")
    if logical_size != manifest["logical_size_bytes"]:
        raise SystemExit("logical snapshot size mismatch")
    payload = {
        "status": "PASS",
        "repo": str(repo),
        "revision": revision,
        "logical_size_bytes": logical_size,
        "hashes_checked": checked_hashes,
        "full": args.full,
    }
    rendered = json.dumps(payload, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
        args.output.chmod(0o444)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
