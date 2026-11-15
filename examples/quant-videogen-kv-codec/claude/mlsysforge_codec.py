"""Persistent BF16 KV-cache codec: clustered PCA residuals + interleaved rANS."""

import torch as _torch

from codec import core as _core

_STEP_REL_K = 2.0
_STEP_REL_V = 2.1
_SHAPE_K = 0.2
_SHAPE_V = -0.1
_ROW_K = -0.25
_ROW_V = -0.75


def encode(k, v):
    with _torch.no_grad(), _torch.autocast("cuda", enabled=False):
        return _core.encode_pair(k, v, _STEP_REL_K, _STEP_REL_V, _SHAPE_K, _SHAPE_V, _ROW_K, _ROW_V)


def decode(packed_k, packed_v):
    with _torch.no_grad(), _torch.autocast("cuda", enabled=False):
        k = _core.decode_tensor(packed_k)
        # Decoding is deterministic: the same payload object yields the same tensor.
        v = k if packed_v is packed_k else _core.decode_tensor(packed_v)
        return k, v
