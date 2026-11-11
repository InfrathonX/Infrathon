"""Forward the legacy model hook to the participant-owned decoder."""

from __future__ import annotations

import torch


def uncompress_kv_cache(
    packed_k: object,
    packed_v: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    from mlsysforge_codec import decode

    return decode(packed_k, packed_v)
