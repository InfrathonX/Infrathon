# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateful frozen Wan stream-VAE facade."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch


def _clone_cache(value: Any, device: torch.device | None = None) -> Any:
    if isinstance(value, torch.Tensor):
        clone = value.detach().clone()
        return clone.to(device) if device is not None else clone
    if isinstance(value, list):
        return [_clone_cache(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_cache(item, device) for item in value)
    return deepcopy(value)


class VAECore:
    def __init__(self, wrapper: torch.nn.Module) -> None:
        self.wrapper = wrapper
        self.model = wrapper.model

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def reset(self) -> None:
        self.model.first_encode = True
        self.model.first_decode = True
        self.model.clear_cache()

    def stream_encode(self, frames_bf16: torch.Tensor) -> torch.Tensor:
        if frames_bf16.ndim != 5 or frames_bf16.shape[1] != 3:
            raise ValueError(f"frames must have shape [B,3,T,H,W], got {tuple(frames_bf16.shape)}")
        if frames_bf16.dtype != torch.bfloat16:
            raise TypeError(f"frames must be bfloat16, got {frames_bf16.dtype}")
        return self.wrapper.stream_encode(frames_bf16, is_scale=False)

    def stream_decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode ``[B,F,16,h,w]`` to float32 ``[B,T,3,H,W]`` in [0,1]."""
        if latent.ndim != 5 or latent.shape[2] != 16:
            raise ValueError(f"latent must have shape [B,F,16,h,w], got {tuple(latent.shape)}")
        output = self.wrapper.stream_decode_to_pixel(latent)
        return (output * 0.5 + 0.5).clamp(0, 1)

    def export_state(self) -> dict[str, Any]:
        return {
            "first_encode": bool(self.model.first_encode),
            "first_decode": bool(self.model.first_decode),
            "enc_feat_map": _clone_cache(getattr(self.model, "_enc_feat_map", [])),
            "feat_map": _clone_cache(getattr(self.model, "_feat_map", [])),
            "enc_conv_idx": deepcopy(getattr(self.model, "_enc_conv_idx", [0])),
            "conv_idx": deepcopy(getattr(self.model, "_conv_idx", [0])),
        }

    def import_state(self, state: dict[str, Any]) -> None:
        required = {
            "first_encode",
            "first_decode",
            "enc_feat_map",
            "feat_map",
            "enc_conv_idx",
            "conv_idx",
        }
        missing = required.difference(state)
        if missing:
            raise KeyError(f"VAE state is missing fields: {sorted(missing)}")
        self.model.first_encode = bool(state["first_encode"])
        self.model.first_decode = bool(state["first_decode"])
        self.model._enc_feat_map = _clone_cache(state["enc_feat_map"], self.device)
        self.model._feat_map = _clone_cache(state["feat_map"], self.device)
        self.model._enc_conv_idx = deepcopy(state["enc_conv_idx"])
        self.model._conv_idx = deepcopy(state["conv_idx"])
