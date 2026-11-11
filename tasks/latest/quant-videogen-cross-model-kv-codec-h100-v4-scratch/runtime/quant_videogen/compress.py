"""Forward the legacy model hook to the participant-owned encoder."""

from __future__ import annotations

from typing import Any

import torch


def get_quantize_fn(quant_type: str, quant_config: Any) -> None:
    """Retain the model integration signature; the codec owns its algorithm."""
    return None


def get_quantize_type(quant_type: str) -> str:
    """Compatibility helper for integrations that inspect the configured label."""
    return quant_type


def compress_kv_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    quant_type: str | None = None,
    quant_config: Any = None,
    quantize_fn: Any = None,
) -> tuple[object, object]:
    """Return candidate payloads unchanged by the model compatibility layer."""
    from mlsysforge_codec import encode

    return encode(k, v)
