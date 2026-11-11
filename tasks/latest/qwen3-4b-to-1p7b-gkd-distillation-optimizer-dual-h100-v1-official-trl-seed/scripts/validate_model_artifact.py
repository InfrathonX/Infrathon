#!/usr/bin/env python3
"""Validate a pinned local Qwen3 snapshot (student or teacher) without loading weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROLES = {
    "student": {
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "total_size": 4_063_479_808,
        "required": {
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        },
    },
    "teacher": {
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "total_size": 8_044_936_192,
        "required": {
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        },
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--role", choices=sorted(ROLES), default="student")
    parser.add_argument("--skip-full-hash", action="store_true")
    args = parser.parse_args()
    role = ROLES[args.role]
    root = args.model_path.resolve()
    required = set(role["required"])
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise SystemExit(f"missing model files: {missing}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    if config.get("architectures") != ["Qwen3ForCausalLM"]:
        raise SystemExit("unexpected architecture")
    if index.get("metadata", {}).get("total_size") != role["total_size"]:
        raise SystemExit("safetensors logical size differs")
    actual = sum(
        (root / name).stat().st_size for name in required if name.endswith(".safetensors")
    )
    if actual < role["total_size"]:
        raise SystemExit("safetensors files appear truncated")
    if not args.skip_full_hash:
        manifest_path = Path(__file__).resolve().parents[1] / "docs/model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest[args.role]["files"]:
            path = root / record["path"]
            if not path.is_file() or path.stat().st_size != record["size"]:
                raise SystemExit(f"model file size differs: {record['path']}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise SystemExit(f"model file hash differs: {record['path']}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "role": args.role,
                "revision": role["revision"],
                "root": str(root),
                "total_size": role["total_size"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
