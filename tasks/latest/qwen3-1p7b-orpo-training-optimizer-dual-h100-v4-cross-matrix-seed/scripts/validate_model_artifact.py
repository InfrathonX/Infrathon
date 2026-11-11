#!/usr/bin/env python3
"""Validate the pinned local Qwen3-1.7B snapshot without loading weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TOTAL_SIZE = 4_063_479_808


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--skip-full-hash", action="store_true")
    args = parser.parse_args()
    root = args.model_path.resolve()
    required = {
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise SystemExit(f"missing model files: {missing}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["Qwen3ForCausalLM"]:
        raise SystemExit("unexpected architecture")
    if index.get("metadata", {}).get("total_size") != TOTAL_SIZE:
        raise SystemExit("safetensors logical size differs")
    actual = sum((root / name).stat().st_size for name in required if name.endswith(".safetensors"))
    if actual < TOTAL_SIZE:
        raise SystemExit("safetensors files appear truncated")
    if not args.skip_full_hash:
        manifest_path = Path(__file__).resolve().parents[1] / "docs/model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest["files"]:
            path = root / record["path"]
            if not path.is_file() or path.stat().st_size != record["size"]:
                raise SystemExit(f"model file size differs: {record['path']}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise SystemExit(f"model file hash differs: {record['path']}")
    print(json.dumps({"status": "PASS", "revision": REVISION, "root": str(root), "total_size": TOTAL_SIZE}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
