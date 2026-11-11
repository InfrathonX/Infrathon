"""Model-file digest verification shared by v3 capture and QA workers.

The retired synthetic v2 capture generator lives in tasks/archive. Real RULER
capture is implemented in scripts/capture_performance_v3.py.
"""
import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any
class RecipeError(RuntimeError):
    pass

def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size

def verify_model_files(model_path: Path, recipe: dict[str, Any]) -> None:
    """Every file the recipe pins must be present with the pinned digest."""

    files = recipe["model"].get("files")
    if not isinstance(files, dict) or not files:
        raise RecipeError("recipe pins no model files")
    for name, expected in sorted(files.items()):
        path = model_path / name
        if not path.is_file():
            raise RecipeError(f"pinned model file is missing: {name}")
        digest = sha256_file(path)[0]
        if digest != expected:
            raise RecipeError(
                f"pinned model file {name} sha256 {digest} differs from the recipe's {expected}"
            )
    print(f"verified {len(files)} pinned model files", flush=True)

_DTYPE_NAMES = {"torch.bfloat16": "BF16"}

def write_safetensors(path: Path, tensors: dict[str, Any], metadata: dict[str, str]) -> None:
    import torch

    header: dict[str, Any] = {"__metadata__": dict(metadata)}
    offset = 0
    blobs = []
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or str(tensor.dtype) not in _DTYPE_NAMES:
            raise ValueError(f"unsupported tensor {name}")
        raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
        header[name] = {
            "dtype": _DTYPE_NAMES[str(tensor.dtype)],
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for raw in blobs:
            handle.write(raw)
    os.replace(tmp, path)

def _attention_inputs(module, args, kwargs):
    """Post-RoPE q/k and v exactly as the layer's attention would consume them."""

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    hidden_states = args[0] if args else kwargs["hidden_states"]
    hidden_shape = (*hidden_states.shape[:-1], -1, module.head_dim)
    query = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = kwargs["position_embeddings"]
    query, key = apply_rotary_pos_emb(query, key, cos, sin)
    return query.contiguous(), key.contiguous(), value.contiguous()
