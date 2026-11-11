# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict asset-rooted loader for the frozen causal DiT and stream VAE."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

import torch

from .dit import DITCore
from .vae import VAECore


UPSTREAM_COMMIT = "2f8ee1c5b6616b115662d190d5213677882d85d7"


@dataclass
class FrozenModel:
    dit: DITCore
    vae: VAECore
    text_encoder: torch.nn.Module | None

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        if self.text_encoder is None:
            raise RuntimeError("load_model(..., load_text_encoder=True) is required to encode prompts")
        return self.text_encoder(text_prompts=[prompt])["prompt_embeds"]


def _require_assets_root() -> Path:
    raw = os.environ.get("SDV2_ASSETS_ROOT")
    if not raw:
        raise RuntimeError("SDV2_ASSETS_ROOT must point to the lab asset directory")
    root = Path(raw).expanduser().resolve()
    required = (
        root / "wan_models" / "Wan2.1-T2V-1.3B",
        root / "ckpts" / "wan_causal_dmd_v2v" / "model.pt",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"SDV2 assets are incomplete: {missing}")
    return root


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint: Any = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("generator", "generator_ema", "state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint {path} did not contain a state dictionary")
    return {
        name if name.startswith("model.") else f"model.{name}": value
        for name, value in checkpoint.items()
    }


def load_model(
    device: torch.device | str,
    *,
    load_text_encoder: bool = False,
) -> FrozenModel:
    """Load pinned Wan assets strictly on ``device`` in BF16.

    No cwd lookup or non-strict retry is permitted.  The upstream support
    support package is the pinned copy shipped below ``sdv2_core/_upstream``.
    """
    root = _require_assets_root()
    source_root = Path(__file__).resolve().parent / "_upstream"
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    os.environ["STREAMDIFFUSIONV2_ROOT"] = str(root)

    # Import only after the explicit root has been installed: wan_wrapper's
    # PROJECT_ROOT is set at module import time.
    from models.wan.wan_wrapper import (
        CausalWanDiffusionWrapper,
        WanTextEncoder,
        WanVAEWrapper,
    )

    target = torch.device(device)
    generator = CausalWanDiffusionWrapper(model_type="T2V-1.3B")
    state = _checkpoint_state(root / "ckpts" / "wan_causal_dmd_v2v" / "model.pt")
    generator.load_state_dict(state, strict=True)
    generator.eval().requires_grad_(False)
    generator.to(device=target, dtype=torch.bfloat16)

    vae_wrapper = WanVAEWrapper(model_type="T2V-1.3B")
    vae_wrapper.eval().requires_grad_(False)
    vae_wrapper.to(device=target, dtype=torch.bfloat16)

    text_encoder = None
    if load_text_encoder:
        text_encoder = WanTextEncoder(model_type="T2V-1.3B")
        text_encoder.eval().requires_grad_(False)
        text_encoder.to(device=target, dtype=torch.bfloat16)

    return FrozenModel(
        dit=DITCore(generator),
        vae=VAECore(vae_wrapper),
        text_encoder=text_encoder,
    )
