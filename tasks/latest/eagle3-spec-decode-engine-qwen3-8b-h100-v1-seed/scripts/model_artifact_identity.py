#!/usr/bin/env python3
"""Attest the frozen Qwen3 target snapshot and EAGLE-3 draft head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TARGET_CONFIG_SHA256 = "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30"
HEAD_REVISION = "2a1059d51f622b8cad7d7d72840153ffea5488a0"
HEAD_MODEL_SHA256 = "596f0d62006793a8fa70cbd0ea5e7d907504a1b30ec349c841bf357af529b4b0"
HEAD_CONFIG_SHA256 = "d5816b5d7c70e68e07a493942f4a2905a14e403bd2fce3cf92750478cf101685"


def sha256(path: Path) -> str:
    """Hash a model artifact incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def real_directory(path: Path, label: str) -> Path:
    """Resolve an existing non-symlink artifact directory."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be a real directory: {path}")
    return path.resolve(strict=True)


def regular_file(
    path: Path,
    label: str,
    *,
    repository_root: Path | None = None,
) -> Path:
    """Require a real file or a repository-confined HF blob symlink."""
    if path.is_symlink():
        if repository_root is None:
            raise ValueError(f"{label} may not be a symlink: {path}")
        resolved = path.resolve(strict=True)
        repository = repository_root.resolve(strict=True)
        if (
            not resolved.is_file()
            or (resolved != repository and repository not in resolved.parents)
        ):
            raise ValueError(f"{label} symlink escapes its repository: {path}")
        return resolved
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file: {path}")
    return path.resolve(strict=True)


def compute(target_repo: Path, head: Path) -> dict[str, Any]:
    """Compute and validate both read-only mount identities."""
    target_repo = real_directory(target_repo, "target repository")
    head = real_directory(head, "draft head")
    snapshot = target_repo / "snapshots" / TARGET_REVISION
    real_directory(snapshot, "target snapshot")
    target_config = regular_file(
        snapshot / "config.json",
        "target config",
        repository_root=target_repo,
    )
    target_config_digest = sha256(target_config)
    if target_config_digest != TARGET_CONFIG_SHA256:
        raise ValueError("target config identity mismatch")
    target_index = regular_file(
        snapshot / "model.safetensors.index.json",
        "target safetensors index",
        repository_root=target_repo,
    )
    index = json.loads(target_index.read_text(encoding="utf-8"))
    weight_files = sorted(set(index.get("weight_map", {}).values()))
    if not weight_files:
        raise ValueError("target safetensors index has no weights")
    for relative in weight_files:
        regular_file(
            snapshot / relative,
            f"target weight {relative}",
            repository_root=target_repo,
        )

    manifest_path = regular_file(head / "MANIFEST.json", "head manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("revision") != HEAD_REVISION:
        raise ValueError("head revision mismatch")
    head_model = regular_file(head / "model.safetensors", "head model")
    head_config = regular_file(head / "config.json", "head config")
    if sha256(head_model) != HEAD_MODEL_SHA256:
        raise ValueError("head model identity mismatch")
    if sha256(head_config) != HEAD_CONFIG_SHA256:
        raise ValueError("head config identity mismatch")
    return {
        "target_revision": TARGET_REVISION,
        "target_config_sha256": target_config_digest,
        "target_index_sha256": sha256(target_index),
        "target_weight_file_count": len(weight_files),
        "head_revision": HEAD_REVISION,
        "head_model_sha256": HEAD_MODEL_SHA256,
        "head_config_sha256": HEAD_CONFIG_SHA256,
        "head_manifest_sha256": sha256(manifest_path),
    }


def main(argv: list[str] | None = None) -> int:
    """Validate command-line mount paths and print their identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-repo", type=Path, required=True)
    parser.add_argument("--head-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = {
        "schema_version": 1,
        "status": "PASS",
        **compute(args.target_repo, args.head_dir),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
