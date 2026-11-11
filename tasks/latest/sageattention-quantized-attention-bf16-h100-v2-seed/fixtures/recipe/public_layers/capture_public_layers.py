#!/usr/bin/env python3
"""Capture the two real-model attention layers the public tests use.

Each capture writes ``<out>/<name>/layer.pt`` = {"q", "k", "v"} in the task ABI
(contiguous BF16 ``[B, L, H, D]``) and ``capture.manifest.json`` with its SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


class _Done(Exception):
    pass


def save(out: Path, tensors: dict, is_causal: bool, source: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tensors = {name: value.to(torch.bfloat16).contiguous().cpu() for name, value in tensors.items()}
    for name, value in tensors.items():
        if not bool(torch.isfinite(value.float()).all()):
            raise SystemExit(f"{name} has non-finite values")
    path = out / "layer.pt"
    torch.save(tensors, path)
    b, l, h_q, d = tensors["q"].shape
    manifest = {
        "file": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "b": b, "l": l, "h_q": h_q, "h_kv": tensors["k"].shape[2], "d": d,
        "is_causal": is_causal, "softmax_scale": 1 / math.sqrt(d),
        "source": source,
    }
    (out / "capture.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


PROMPT = (
    "A small wooden fishing boat drifts across a misty mountain lake at dawn. An old man in a "
    "yellow raincoat slowly pulls in a net while birds skim the still water around him."
)


def capture_video(model_dir: Path, out: Path, step: int, layer: int, tokens: int, seed: int) -> None:
    import torch.nn.functional as F
    from diffusers import CogVideoXPipeline

    pipe = CogVideoXPipeline.from_pretrained(str(model_dir), torch_dtype=torch.float16).to("cuda")
    n_layers = pipe.transformer.config.num_layers
    state = {"calls": 0, "inside": False}
    pipe.transformer.register_forward_pre_hook(lambda m, a: state.__setitem__("inside", True))
    pipe.transformer.register_forward_hook(lambda m, a, o: state.__setitem__("inside", False))
    original = F.scaled_dot_product_attention

    def hook(q, k, v, *args, **kw):
        if state["inside"]:
            s, index = divmod(state["calls"], n_layers)
            state["calls"] += 1
            if s == step and index == layer:
                tensors = {n: x.transpose(1, 2)[:, :tokens] for n, x in (("q", q), ("k", k), ("v", v))}
                save(out, tensors, False, {"model": "zai-org/CogVideoX-2b", "prompt": PROMPT,
                                           "seed": seed, "step": step, "layer": layer, "tokens": tokens})
                raise _Done
        return original(q, k, v, *args, **kw)

    F.scaled_dot_product_attention = hook
    try:
        pipe(prompt=PROMPT, num_videos_per_prompt=1, num_inference_steps=50, num_frames=49,
             guidance_scale=6, generator=torch.Generator(device="cuda").manual_seed(seed))
        raise SystemExit("pipeline finished before the capture step")
    except _Done:
        pass
    finally:
        F.scaled_dot_product_attention = original


WAN22_PROMPT = (
    "A hot-air balloon rises over terraced rice fields at sunrise, mist drifting through the "
    "valley and villagers waving from the paths below."
)


def capture_wan22(model_dir: Path, out: Path, step: int, block: int, seed: int) -> None:
    """One self-attention call (conditional pass) of Wan2.2-TI2V-5B text-to-video, 1280x704, 121 frames."""
    import torch.nn.functional as F
    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16).to("cuda")
    state = {"passes": 0, "inside": False, "block": 0}

    def pre(_m, _a, _k=None):
        state.update(inside=True, block=0)

    def post(_m, _a, _o, _k=None):
        state["inside"] = False
        state["passes"] += 1

    pipe.transformer.register_forward_pre_hook(pre)
    pipe.transformer.register_forward_hook(post)
    original = F.scaled_dot_product_attention

    def hook(*args, **kw):
        names = ("query", "key", "value")
        q, k, v = (args[i] if i < len(args) else kw.pop(names[i]) for i in range(3))
        args = args[3:]
        if state["inside"] and q.shape[-2] == k.shape[-2]:
            index = state["block"]
            state["block"] += 1
            if state["passes"] == 2 * step and index == block:
                tensors = {n: x.transpose(1, 2) for n, x in (("q", q), ("k", k), ("v", v))}
                save(out, tensors, False, {"model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "prompt": WAN22_PROMPT,
                                           "seed": seed, "step": step, "block": block,
                                           "height": 704, "width": 1280, "frames": 121})
                raise _Done
        return original(q, k, v, *args, **kw)

    F.scaled_dot_product_attention = hook
    try:
        pipe(prompt=WAN22_PROMPT, height=704, width=1280, num_frames=121, num_inference_steps=30,
             guidance_scale=5.0, generator=torch.Generator(device="cuda").manual_seed(seed), output_type="latent")
        raise SystemExit("pipeline finished before the capture step")
    except _Done:
        pass
    finally:
        F.scaled_dot_product_attention = original


def capture_lm(model_dir: Path, dataset_file: Path, out: Path, layer: int, tokens: int) -> None:
    import pyarrow.parquet as pq
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    text = "".join(pq.read_table(str(dataset_file)).column("text").to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[:, :tokens]
    if ids.shape[1] != tokens:
        raise SystemExit("tokenized text is shorter than requested")
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to("cuda").eval()

    def capture(module, query, key, value, attention_mask, **kw):
        if module.layer_idx == layer:
            tensors = {n: x.transpose(1, 2) for n, x in (("q", query), ("k", key), ("v", value))}
            save(out, tensors, True, {"model": "Qwen/Qwen3-8B", "text": dataset_file.name,
                                      "layer": layer, "tokens": tokens})
            raise _Done
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kw)

    AttentionInterface.register("public_layer_capture", capture)
    model.set_attn_implementation("public_layer_capture")
    with torch.no_grad():
        try:
            model(ids.cuda(), use_cache=False)
            raise SystemExit("forward finished before the capture layer")
        except _Done:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("video", "wan22", "lm"), required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.kind == "video":
        capture_video(args.model_dir, args.out, step=25, layer=7, tokens=4096, seed=7)
    elif args.kind == "wan22":
        capture_wan22(args.model_dir, args.out, step=10, block=15, seed=11)
    else:
        capture_lm(args.model_dir, args.dataset_file, args.out, layer=4, tokens=8192)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
