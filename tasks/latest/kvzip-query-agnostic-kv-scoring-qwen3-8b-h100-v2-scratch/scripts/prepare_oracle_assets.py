#!/usr/bin/env python3
"""Reproduce the ignored host-only Fast KVzip oracle asset deterministically."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

REPO_ID = "Jang-Hyun/Fast-KVzip"
REVISION = "67ce1265f510a946bf22486a30269f99066b3cbb"
FILENAME = "qwen3-8b/q4_dim16_sink16.pt"
UPSTREAM_SHA256 = "8e0fd73396e487f5b7d01556ed9c119a686d0326a3517ae30168f48a8c92ed2a"
OUTPUT_SHA256 = "722aa9641a2ef8ce51655d4c26188c69e97d9e4679154ad882f9fe64e2e9ff39"
TRANSPORT_SHA256 = "300de4f36e3f09c31117fe66a02512db40f9c981eda365d2d1e8213fc147d20e"
TENSOR_NAMES = ("q_proj.weight", "q_proj.bias", "k_proj.weight", "q_norm.weight", "k_norm.weight", "k_base", "b")
TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = TASK_ROOT / "oracle/candidates/scorer_v001/assets.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def repack_state_dict(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    states = checkpoint.get("module")
    if not isinstance(states, list) or not states: raise ValueError("checkpoint module must be a non-empty layer list")
    assets: dict[str, Any] = {}
    for layer_index, state in enumerate(states):
        if not isinstance(state, Mapping) or set(TENSOR_NAMES) - set(state): raise ValueError(f"layer {layer_index} gate state differs")
        for name in TENSOR_NAMES:
            tensor = state[name]
            if not isinstance(tensor, torch.Tensor): raise TypeError(f"layer {layer_index}/{name} is not a tensor")
            assets[f"fastkvzip/layer{layer_index}/{name}"] = tensor.detach().cpu().contiguous()
    return assets


def save_repacked(assets: Mapping[str, Any], output: Path, expected_sha256: str | None = None) -> str:
    import torch
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".oracle-assets-", dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        torch.save(dict(assets), staged)
        digest = sha256_file(staged)
        if expected_sha256 is not None and digest != expected_sha256:
            raise RuntimeError(f"repacked asset SHA-256 differs: {digest}")
        os.replace(staged, output)
    return digest


def save_transport_repacked(source: Path, output: Path) -> str:
    """Write a bounded gzip transport of byte-shuffled, exact tensors."""
    import torch
    assets = torch.load(source, map_location="cpu", weights_only=True)
    packed = {
        f"packed/{name}": tensor.contiguous().view(torch.uint8)
        .reshape(*tensor.shape, tensor.element_size()).movedim(-1, 0).contiguous()
        for name, tensor in assets.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".oracle-transport-", dir=output.parent) as temporary:
        temporary_root = Path(temporary)
        inner = temporary_root / "inner" / "assets.pt"
        save_repacked(packed, inner)
        staged = temporary_root / "assets.pt"
        staged.write_bytes(gzip.compress(inner.read_bytes(), compresslevel=9, mtime=0))
        digest = sha256_file(staged)
        if staged.stat().st_size > 128 * 1024 * 1024:
            raise RuntimeError("oracle transport asset exceeds 128 MiB")
        os.replace(staged, output)
    return digest


def prepare(output: Path = DEFAULT_OUTPUT) -> str:
    if output.is_file() and not output.is_symlink():
        digest = sha256_file(output)
        if digest == OUTPUT_SHA256: return digest
        raise RuntimeError(f"existing oracle asset SHA-256 differs: {digest}")
    from huggingface_hub import hf_hub_download
    source = Path(hf_hub_download(REPO_ID, filename=FILENAME, revision=REVISION))
    upstream_digest = sha256_file(source)
    if upstream_digest != UPSTREAM_SHA256: raise RuntimeError(f"upstream asset SHA-256 differs: {upstream_digest}")
    import torch
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    assets = repack_state_dict(checkpoint)
    if len(checkpoint["module"]) != 36 or len(assets) != 252: raise RuntimeError("Qwen3-8B gate layer/tensor count differs")
    return save_repacked(assets, output, OUTPUT_SHA256)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--transport-output", type=Path)
    args = parser.parse_args(); digest = prepare(args.output)
    if args.transport_output is not None:
        transport_digest = save_transport_repacked(args.output, args.transport_output)
        if digest == OUTPUT_SHA256 and transport_digest != TRANSPORT_SHA256:
            raise RuntimeError(f"oracle transport SHA-256 differs: {transport_digest}")
        print(f"{args.transport_output} sha256={transport_digest}")
    print(f"{args.output} sha256={digest}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
