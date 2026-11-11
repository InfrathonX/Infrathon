"""Small scratch-ABI smoke codec; it is not a task starter or Oracle."""

import torch


def _encode_one(value):
    scale = value.abs().amax().float() / 127.0
    quantized = (value.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return {"quantized": quantized, "scale": scale}


def encode(k, v):
    return _encode_one(k), _encode_one(v)


def _decode_one(payload):
    return (payload["quantized"].float() * payload["scale"]).to(torch.bfloat16)


def decode(packed_k, packed_v):
    return _decode_one(packed_k), _decode_one(packed_v)
