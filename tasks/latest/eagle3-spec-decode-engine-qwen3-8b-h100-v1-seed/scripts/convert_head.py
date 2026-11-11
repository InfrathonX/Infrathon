#!/usr/bin/env python3
"""Download, audit, validate, and convert the two frozen EAGLE-3 heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors.torch import save_file


REPOS = ("AngelSlim/Qwen3-8B_eagle3", "Tengyunw/qwen3_8b_eagle3")
# Output directory and heads.json key of each audited head; the frozen one is
# eagle3-head (the name of its task asset and container mount).
HEAD_NAMES = {"AngelSlim/Qwen3-8B_eagle3": "angelslim", "Tengyunw/qwen3_8b_eagle3": "eagle3-head"}


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one artifact file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """Download, validate, convert, and describe each frozen head source."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache = args.data_root / "hf_cache"
    heads_root = args.data_root / "heads"
    cache.mkdir(parents=True, exist_ok=True)
    heads_root.mkdir(parents=True, exist_ok=True)

    # Import only after the caller has placed vendor/EAGLE on PYTHONPATH.
    from eagle.model.cnets import Model
    from eagle.model.configs import EConfig

    result = {"schema_version": 1, "heads": {}}
    api = HfApi()
    for repo in REPOS:
        info = api.model_info(repo, files_metadata=True)
        revision = info.sha
        snapshot = Path(snapshot_download(repo, revision=revision, cache_dir=cache))
        source_files = []
        for path in sorted(p for p in snapshot.rglob("*") if p.is_file()):
            source_files.append({
                "path": str(path.relative_to(snapshot)),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            })
        bin_path = snapshot / "pytorch_model.bin"
        state = torch.load(bin_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(
            isinstance(value, torch.Tensor) for value in state.values()
        ):
            raise TypeError(f"{repo}: checkpoint is not a plain tensor state dict")
        tensors = []
        for name, tensor in sorted(state.items()):
            item = {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            tensors.append(item)
            print(json.dumps({"repo": repo, **item}, sort_keys=True))

        config_path = snapshot / "config.json"
        config = EConfig.from_pretrained(str(config_path))
        with torch.device("meta"):
            expected = Model(config, load_emb=False)
        incompatible = expected.load_state_dict(state, strict=False)
        required_maps = {
            "draft_vocab_size": int(config.draft_vocab_size),
            "d2t_present": "d2t" in state,
            "t2d_present": "t2d" in state,
            "d2t_shape": list(state["d2t"].shape) if "d2t" in state else None,
            "t2d_shape": list(state["t2d"].shape) if "t2d" in state else None,
        }
        if (
            required_maps["draft_vocab_size"] != 32000
            or not required_maps["d2t_present"]
            or not required_maps["t2d_present"]
        ):
            raise RuntimeError(f"{repo}: required 32k d2t/t2d maps missing")

        short = HEAD_NAMES[repo]
        destination = heads_root / short
        destination.mkdir(parents=True, exist_ok=True)
        converted = destination / "model.safetensors"
        save_file({k: v.contiguous() for k, v in state.items()}, converted)
        shutil.copy2(config_path, destination / "config.json")
        if (snapshot / "README.md").exists():
            shutil.copy2(snapshot / "README.md", destination / "SOURCE_README.md")
        for candidate in snapshot.iterdir():
            if candidate.is_file() and "license" in candidate.name.lower():
                shutil.copy2(candidate, destination / candidate.name)
        manifest = {
            "repo_id": repo,
            "revision": revision,
            "source_snapshot": str(snapshot),
            "source_files": source_files,
            "source_checkpoint": {"path": "pytorch_model.bin", "sha256": sha256(bin_path)},
            "converted": {
                "path": "model.safetensors",
                "size": converted.stat().st_size,
                "sha256": sha256(converted),
            },
            "config": json.loads(config_path.read_text()),
            "tensors": tensors,
            "state_dict_check": {
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys),
                **required_maps,
            },
            "weights_only_load": True,
        }
        (destination / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result["heads"][short] = manifest
        del state, expected
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
