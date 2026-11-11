#!/usr/bin/env python3
"""Run SForge with the scored real layers mounted read-only into every Judge container.

The 139 real layers of final-holdout-v3 are about 60 GB, so they are not part of any image.
This entry wraps the Docker backend's container creation for the Judge commands (`serve`,
`eval`, `replay-*`): a Judge container gets `REAL_LAYERS_DIR` at the path the runtime reads,
and nothing else changes.  The runtime re-checks every file against its capture manifest and
every manifest against the recipe inside the image, so a wrong or edited directory fails the
evaluation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sforge.harness.backend.docker_backend import DockerBackend

TASK_ID = "sageattention_quantized_attention_bf16_h100_seed_v2"
REAL_LAYERS_ENV = "REAL_LAYERS_DIR"
REAL_LAYERS_MOUNT = "/opt/sageattention_quant_attention_judge/fixtures/real_layers"
RECIPES = Path(__file__).resolve().parents[1] / "fixtures/recipe/real_layers"


def _captured(recipe: dict) -> int:
    listed = recipe.get("capture", {}).get("layers")
    return len(listed) if listed is not None else int(recipe["attention"]["layers"])


def real_layers_directory() -> str:
    """The checked host directory of the real layers, or an error naming what is wrong."""
    raw = os.environ.get(REAL_LAYERS_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"{REAL_LAYERS_ENV} is required")
    directory = Path(raw).resolve(strict=True)
    for path in sorted(RECIPES.glob("RECIPE_*.json")):
        recipe = json.loads(path.read_text(encoding="utf-8"))
        capture = directory / recipe["capture_dir"]
        manifest = json.loads((capture / "capture.manifest.json").read_text(encoding="utf-8"))
        if manifest["recipe_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError(f"{capture.name} was not produced by the shipped {path.name}")
        files = manifest["files"]
        if len(files) != _captured(recipe) or any(not (capture / name).is_file() for name in files):
            raise RuntimeError(f"{capture.name} must hold every layer its recipe captures")
        # The Judge worker runs as an unprivileged container user, so the mounted tree must be
        # world-readable; otherwise every evaluation fails in the real stage with no visible cause.
        paths = [capture, *(capture / name for name in ("capture.manifest.json", *files))]
        unreadable = [p.name for p in paths if p.stat().st_mode & 0o004 == 0 or (p.is_dir() and p.stat().st_mode & 0o001 == 0)]
        if unreadable:
            raise RuntimeError(f"{capture.name} is not world-readable ({unreadable[0]}); run chmod -R a+rX {directory}")
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
