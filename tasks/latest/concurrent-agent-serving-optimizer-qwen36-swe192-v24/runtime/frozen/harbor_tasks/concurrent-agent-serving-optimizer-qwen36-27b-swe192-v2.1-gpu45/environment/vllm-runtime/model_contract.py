#!/usr/bin/env python3
"""Shared host/container validation for the immutable model profile."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class ContractError(RuntimeError):
    pass


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_profile(path: Path) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read model profile {path}: {exc}") from exc
    if profile.get("schema_version") != 1:
        raise ContractError("unsupported model profile schema")
    kv_bytes = profile.get("runtime", {}).get("kv_cache_memory_bytes")
    if kv_bytes is not None and (
        not isinstance(kv_bytes, int)
        or isinstance(kv_bytes, bool)
        or not 1024**3 <= kv_bytes <= 80 * 1024**3
    ):
        raise ContractError("kv_cache_memory_bytes must be null or an integer between 1 GiB and 80 GiB per GPU")
    runtime = profile.get("runtime", {})
    gpu_contract = profile.get("gpu_contract", {})
    gpu_count = gpu_contract.get("count")
    if not isinstance(gpu_count, int) or isinstance(gpu_count, bool) or gpu_count < 1:
        raise ContractError("GPU contract count must be a positive integer")
    if runtime.get("tensor_parallel_size") != gpu_contract.get("count"):
        raise ContractError("tensor_parallel_size must equal the visible GPU contract count")
    if not isinstance(runtime.get("max_num_seqs"), int) or runtime["max_num_seqs"] < 1:
        raise ContractError("max_num_seqs must be a positive integer")
    if not isinstance(runtime.get("max_model_len"), int) or runtime["max_model_len"] < 1024:
        raise ContractError("max_model_len must be an integer of at least 1024 tokens")
    if not isinstance(runtime.get("served_model_name"), str) or not runtime["served_model_name"]:
        raise ContractError("served_model_name must be a non-empty string")
    if runtime.get("server_dev_mode") is not True:
        raise ContractError("server_dev_mode must be exactly true for trusted prefix-cache reset")
    if "harmony" in runtime:
        raise ContractError("Harmony runtime assets are forbidden by the active model contract")
    return profile


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"expected a JSON object in {path}")
    return value


def _symlink_manifest(shards: list[Path]) -> str:
    rows = []
    for shard in shards:
        target = shard.readlink().name if shard.is_symlink() else None
        rows.append({"name": shard.name, "size": shard.stat().st_size, "target": target})
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_model(model_path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    model_path = model_path.resolve(strict=True)
    if not model_path.is_dir():
        raise ContractError(f"model path is not a directory: {model_path}")

    observed_hashes: dict[str, str] = {}
    for filename, expected in profile["required_files"].items():
        path = model_path / filename
        if not path.is_file():
            raise ContractError(f"required model file is missing: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ContractError(f"sha256 mismatch for {filename}: {observed} != {expected}")
        observed_hashes[filename] = observed

    config = _load_json(model_path / "config.json")
    expected_config = profile["config_contract"]
    quantization = config.get("quantization_config", {})
    quantization_group = quantization.get("config_groups", {}).get("group_0", {})
    weight_quantization = quantization_group.get("weights", {})
    activation_quantization = quantization_group.get("input_activations", {})
    text_config = config.get("text_config", {})
    if not isinstance(text_config, dict):
        raise ContractError("config.json text_config must be an object when present")
    layer_types = text_config.get("layer_types")
    checks = {
        "architectures": config.get("architectures"),
        "model_type": config.get("model_type"),
        "language_model_only": config.get("language_model_only"),
        "expert_dtype": config.get("expert_dtype"),
        "torch_dtype": config.get("torch_dtype"),
        "text_model_type": text_config.get("model_type"),
        "text_dtype": text_config.get("dtype"),
        "num_hidden_layers": text_config.get("num_hidden_layers", config.get("num_hidden_layers")),
        "num_attention_heads": text_config.get("num_attention_heads", config.get("num_attention_heads")),
        "num_key_value_heads": text_config.get("num_key_value_heads", config.get("num_key_value_heads")),
        "head_dim": text_config.get("head_dim", config.get("head_dim")),
        "full_attention_interval": text_config.get("full_attention_interval"),
        "full_attention_layer_count": (
            layer_types.count("full_attention") if isinstance(layer_types, list) else None
        ),
        "quantization_method": quantization.get("quant_method"),
        "quantization_format": quantization.get("format"),
        "quantization_status": quantization.get("quantization_status"),
        "weight_quantization_type": weight_quantization.get("type"),
        "weight_quantization_bits": weight_quantization.get("num_bits"),
        "weight_quantization_strategy": weight_quantization.get("strategy"),
        "weight_quantization_dynamic": weight_quantization.get("dynamic"),
        "activation_quantization_type": activation_quantization.get("type"),
        "activation_quantization_bits": activation_quantization.get("num_bits"),
        "activation_quantization_strategy": activation_quantization.get("strategy"),
        "activation_quantization_dynamic": activation_quantization.get("dynamic"),
        "max_position_embeddings": text_config.get(
            "max_position_embeddings", config.get("max_position_embeddings")
        ),
    }
    for key, expected in expected_config.items():
        if checks.get(key) != expected:
            raise ContractError(f"config contract mismatch for {key}: {checks.get(key)!r} != {expected!r}")

    index = _load_json(model_path / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ContractError("weight index has no object-valued weight_map")
    metadata = index.get("metadata")
    if not isinstance(metadata, dict):
        raise ContractError("weight index has no object-valued metadata")

    weights = profile["weights"]
    shard_names = sorted(set(weight_map.values()))
    if len(weight_map) != weights["index_tensor_count"]:
        raise ContractError(f"unexpected tensor count: {len(weight_map)}")
    if len(shard_names) != weights["shard_count"]:
        raise ContractError(f"unexpected shard count in index: {len(shard_names)}")
    if metadata.get("total_size") != weights["index_total_size"]:
        raise ContractError(f"unexpected index total_size: {metadata.get('total_size')}")

    shards = [model_path / name for name in shard_names]
    missing = [str(path) for path in shards if not path.is_file()]
    if missing:
        raise ContractError(f"missing or dangling weight shards: {missing[:3]}")
    stat_total = sum(path.stat().st_size for path in shards)
    if stat_total != weights["stat_total_size"]:
        raise ContractError(f"unexpected shard stat size: {stat_total}")
    symlink_manifest = _symlink_manifest(shards)
    if symlink_manifest != weights["hf_symlink_manifest_sha256"]:
        raise ContractError(
            "weight shard/symlink manifest mismatch; mount the pinned Hugging Face repository root, not an isolated snapshot"
        )

    return {
        "model_path": str(model_path),
        "profile_id": profile["profile_id"],
        "revision": profile["source"]["revision"],
        "file_hashes": observed_hashes,
        "shard_count": len(shards),
        "tensor_count": len(weight_map),
        "index_total_size": metadata["total_size"],
        "stat_total_size": stat_total,
        "hf_symlink_manifest_sha256": symlink_manifest,
        "first_shard_name": shard_names[0],
    }


def resolve_mount(model_path: Path, profile: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Return (source mount, relative model path, resolved model path).

    Hugging Face snapshots contain ../../blobs symlinks, so mounting just the
    snapshot breaks them in a container. In that layout we mount the model repo
    root and retain snapshots/<revision> as the in-container relative path.
    """

    if not model_path.is_absolute():
        raise ContractError("model path must be absolute")
    resolved = model_path.resolve(strict=True)
    expected_revision = profile["source"]["revision"]
    if resolved.name == expected_revision and resolved.parent.name == "snapshots":
        repo_root = resolved.parent.parent
        if not (repo_root / "blobs").is_dir():
            raise ContractError(f"Hugging Face snapshot has no sibling blobs directory: {repo_root}")
        return repo_root, resolved.relative_to(repo_root), resolved

    escaping = []
    for path in resolved.iterdir():
        if path.is_symlink():
            target = path.resolve(strict=True)
            try:
                target.relative_to(resolved)
            except ValueError:
                escaping.append(path.name)
    if escaping:
        raise ContractError(
            "model directory contains symlinks that escape the mount root; pass the pinned Hugging Face snapshot path "
            f"so its repository root can be mounted safely (examples: {escaping[:3]})"
        )
    return resolved, Path("."), resolved


def mount_is_read_only(path: Path) -> bool:
    """Check the effective Linux mount containing path."""

    resolved = path.resolve(strict=True)
    candidates: list[tuple[int, set[str]]] = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        separator = fields.index("-")
        mountpoint = Path(fields[4].replace("\\040", " "))
        try:
            resolved.relative_to(mountpoint)
        except ValueError:
            continue
        options = set(fields[5].split(",")) | set(fields[separator + 3].split(","))
        candidates.append((len(str(mountpoint)), options))
    if not candidates:
        raise ContractError(f"cannot find effective mount for {resolved}")
    return "ro" in max(candidates, key=lambda item: item[0])[1]
