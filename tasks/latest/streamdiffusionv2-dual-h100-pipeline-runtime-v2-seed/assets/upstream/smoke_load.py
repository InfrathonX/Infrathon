#!/usr/bin/env python3
"""One-GPU image smoke: load the causal DiT checkpoint and streaming Wan VAE."""

from __future__ import annotations

import json
import time

import flash_attn
import torch

from models.wan.wan_wrapper import CausalWanDiffusionWrapper, WanVAEWrapper


def _state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("generator", "generator_ema", "state_dict"):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("model.pt did not contain a state dictionary")
    return {key if key.startswith("model.") else f"model.{key}": value for key, value in checkpoint.items()}


def main() -> int:
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    generator = CausalWanDiffusionWrapper(model_type="T2V-1.3B")
    raw = torch.load("/opt/sdv2_weights/ckpts/wan_causal_dmd_v2v/model.pt", map_location="cpu", weights_only=False)
    state = _state_dict(raw)
    try:
        incompatible = generator.load_state_dict(state, strict=True)
        strict = True
    except RuntimeError:
        incompatible = generator.load_state_dict(state, strict=False)
        strict = False
    generator.to(device="cuda:0", dtype=torch.bfloat16).eval()
    del raw, state
    vae = WanVAEWrapper(model_type="T2V-1.3B").to(device="cuda:0", dtype=torch.bfloat16).eval()
    torch.cuda.synchronize()
    result = {
        "ok": True,
        "torch": torch.__version__,
        "flash_attn": flash_attn.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "generator_class": type(generator).__name__,
        "vae_class": type(vae).__name__,
        "strict_checkpoint_load": strict,
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
        "load_seconds": round(time.monotonic() - started, 3),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
