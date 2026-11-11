#!/usr/bin/env python3
"""Fail the Work base build if the pinned dependency closure is not exact.

Run inside the base image right after ``pip install -r requirements.lock``.
It proves the closure imports and that the score-affecting versions are the
ones the validated v2 Work image carried, so a drifting index cannot silently
produce a different runtime.
"""

from __future__ import annotations

import importlib
import sys


EXPECTED_VERSIONS = {
    "torch": "2.8.0+cu128",
    "torchvision": "0.23.0+cu128",
    "numpy": "1.26.4",
    "transformers": "4.56.0",
    "diffusers": "0.35.0",
    "safetensors": "0.4.5",
    "huggingface_hub": "0.34.0",
    "accelerate": "1.14.0",
    "einops": "0.8.2",
    "timm": "1.0.12",
    "scipy": "1.17.1",
}

# Imported for presence only: either they expose no stable __version__ or the
# task depends on the import succeeding rather than on an exact version.
REQUIRED_IMPORTS = (
    "av",
    "clip",
    "cv2",
    "decord",
    "easydict",
    "flash_attn",
    "ftfy",
    "imageio",
    "lmdb",
    "loguru",
    "lpips",
    "moviepy",
    "omegaconf",
    "openai",
    "PIL",
    "tabulate",
    "termcolor",
    "tokenizers",
    "triton",
)


def main() -> int:
    failures: list[str] = []

    for name, expected in EXPECTED_VERSIONS.items():
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - build-time guard
            failures.append(f"{name}: import failed: {exc!r}")
            continue
        found = getattr(module, "__version__", None)
        if found != expected:
            failures.append(f"{name}: expected {expected}, found {found}")

    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
        except Exception as exc:  # pragma: no cover - build-time guard
            failures.append(f"{name}: import failed: {exc!r}")

    import torch

    if torch.version.cuda != "12.8":
        failures.append(f"torch.version.cuda: expected 12.8, found {torch.version.cuda}")
    if sys.version_info[:2] != (3, 11):
        failures.append(f"python: expected 3.11, found {sys.version.split()[0]}")

    # pyiqa is installed without its own dependency closure and is loaded
    # through a minimal namespace shim by metrics/run_vbench.py, so only the
    # on-disk package is checked here.
    from pathlib import Path

    pyiqa_root = Path("/opt/conda/lib/python3.11/site-packages/pyiqa")
    if not (pyiqa_root / "archs").is_dir():
        failures.append(f"pyiqa package tree is missing under {pyiqa_root}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1

    print(
        "qvg base dependency closure OK: "
        f"python {sys.version.split()[0]}, torch {torch.__version__}, "
        f"cuda {torch.version.cuda}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
