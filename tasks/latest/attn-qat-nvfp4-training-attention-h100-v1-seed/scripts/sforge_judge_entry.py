#!/usr/bin/env python3
"""Run SForge with the real captured layers mounted read-only into every Judge container.

The 78 real layers are about 24 GB, so they are not part of any image.  This entry wraps the Docker
backend's container creation for the Judge commands (`serve`, `eval`, `replay-*`): a Judge
container gets `REAL_LAYERS_DIR` at the path the runtime reads, and nothing else changes.  The
runtime re-checks every file against the capture manifest and the manifest against the recipe
inside the image, so a wrong or edited directory fails the evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sforge.harness.backend.docker_backend import DockerBackend

TASK_ID = "attn_qat_nvfp4_training_attention_h100_seed_v1"
REAL_LAYERS_ENV = "REAL_LAYERS_DIR"
REAL_LAYERS_MOUNT = "/opt/attn_qat_judge/fixtures/real_layers"
_RECIPES = Path(__file__).resolve().parents[1] / "fixtures/recipe/real_layers"
# (recipe, manifest it binds, layers it holds): the Wan captures and the second recipe's
# CogVideoX and Qwen3 captures live side by side in one directory.
CAPTURES = (
    (_RECIPES / "REAL_LAYERS.json", "real_layers.manifest.json", 32),
    (_RECIPES / "REAL_LAYERS_V2.json", "real_layers_v2.manifest.json", 46),
)
EXPECTED_LAYERS = sum(count for _, _, count in CAPTURES)


def real_layers_directory() -> str:
    """The checked host directory of the real layers, or an error naming what is wrong."""
    raw = os.environ.get(REAL_LAYERS_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{REAL_LAYERS_ENV} is required")
    directory = Path(raw).resolve(strict=True)
    for recipe, manifest_name, count in CAPTURES:
        manifest = json.loads((directory / manifest_name).read_text(encoding="utf-8"))
        if manifest["recipe"]["sha256"] != hashlib.sha256(recipe.read_bytes()).hexdigest():
            raise RuntimeError(f"{manifest_name} was not produced by the shipped {recipe.name}")
        names = manifest["files"]
        if len(names) != count or any(not (directory / f"{n}.pt").is_file() for n in names):
            raise RuntimeError(f"{manifest_name} must hold {count} layers")
    return str(directory)


def is_judge(image: str, name: str, image_tags: tuple[str, ...] = ()) -> bool:
    """True for this task's Judge, including a replay that pins the image to its sha256 ID."""
    probe = " ".join(image_tags) if image.startswith("sha256:") else image
    return f".judge.{TASK_ID}" in probe or f".judge.{TASK_ID}." in name


_create_container = DockerBackend.create_container


def create_container(self: DockerBackend, image: str, name: str, **kwargs: Any) -> Any:
    """Create the container, adding the read-only real-layer mount to a Judge."""
    tags: tuple[str, ...] = ()
    if image.startswith("sha256:"):
        tags = tuple(self.client.images.get(image).tags)
    if is_judge(image, name, tags):
        mounts = list(kwargs.get("mounts") or [])
        mounts.append((real_layers_directory(), REAL_LAYERS_MOUNT, "ro"))
        kwargs["mounts"] = mounts
    return _create_container(self, image, name, **kwargs)


DockerBackend.create_container = create_container

if __name__ == "__main__":
    real_layers_directory()
    from sforge.cli import main

    raise SystemExit(main())
