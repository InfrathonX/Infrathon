#!/usr/bin/env python3
"""Capture every attention layer's Q/K/V for one real-layer recipe.

Two recipe kinds, both following SageAttention's layer-wise accuracy protocol
(real Q/K/V across all layers of a real model):

* ``cogvideox`` -- CogVideoX-2B text-to-video, settings of upstream
  ``example/cogvideox_infer.py`` @ d1a57a5.  The hook sits on
  ``torch.nn.functional.scaled_dot_product_attention`` (the exact call upstream
  replaces with ``sageattn``) and only counts calls made inside the DiT
  transformer's forward; the T5 encoder's own SDPA calls are passed through.
* ``llama`` -- Llama-3.1-8B prefill of the first N WikiText-2 test tokens.  A
  registered attention function records the post-RoPE query and the key/value
  before ``repeat_kv`` and then calls the stock SDPA path.

Every pinned model/dataset file is re-hashed before anything is loaded.  Each
layer is written as ``layerNN.pt`` = {"q","k","v","meta"} in the task ABI
([B, L, H, D] BF16, contiguous), then hashed into ``capture.manifest.json``
together with the SHA-256 of the recipe that produced it.  The capture aborts as
soon as the last requested layer has been copied to the host.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


class _Done(Exception):
    pass


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_files(root: Path, files: dict[str, str], label: str) -> None:
    for rel, expected in sorted(files.items()):
        path = root / rel
        if not path.is_file():
            raise SystemExit(f"{label}: pinned file missing: {rel}")
        got = sha256(path)
        if got != expected:
            raise SystemExit(f"{label}: SHA-256 mismatch for {rel}: {got} != {expected}")
    print(f"{label}: {len(files)} pinned files verified", flush=True)


def capture_layers(recipe: dict) -> list[int]:
    """Layer files a recipe captures: capture.layers, else every attention layer."""
    listed = recipe.get("capture", {}).get("layers")
    return [int(x) for x in listed] if listed is not None else list(range(recipe["attention"]["layers"]))


def derive_band(stats: dict) -> dict:
    """The band a first (reference) capture writes: RMS widened 2x, abs-max 4x."""
    return {"rms_min": round(stats["rms"] / 2, 6), "rms_max": round(stats["rms"] * 2, 6),
            "abs_max_min": round(stats["abs_max"] / 4, 6), "abs_max_max": round(stats["abs_max"] * 4, 6)}


DERIVED: dict[str, dict] = {}


def check_bands(name: str, layer: int, tensors: dict[str, torch.Tensor], recipe: dict) -> None:
    if "sanity_bands" not in recipe:  # reference capture: record the stats, derive bands later
        entry = DERIVED.setdefault(f"{layer:02d}", {})
        for key, x in tensors.items():
            xf = x.float()
            if not bool(torch.isfinite(xf).all()):
                raise SystemExit(f"{name} layer {layer} {key}: non-finite values")
            entry[key] = derive_band({"rms": float(xf.square().mean().sqrt()), "abs_max": float(xf.abs().max())})
        return
    bands = recipe["sanity_bands"]
    layer_bands = bands["layers"][f"{layer:02d}"]
    for key, x in tensors.items():
        xf = x.float()
        if not bool(torch.isfinite(xf).all()):
            raise SystemExit(f"{name} layer {layer} {key}: non-finite values")
        b = layer_bands[key]
        rms = float(xf.square().mean().sqrt())
        amax = float(xf.abs().max())
        if not (b["rms_min"] <= rms <= b["rms_max"] and b["abs_max_min"] <= amax <= b["abs_max_max"]):
            raise SystemExit(f"{name} layer {layer} {key}: rms {rms} / abs-max {amax} outside recipe band {b}")


def save_layer(out: Path, recipe: dict, layer: int, tensors: dict[str, torch.Tensor], extra: dict) -> None:
    shapes = recipe["tensors"]
    for key in ("q", "k", "v"):
        if list(tensors[key].shape) != shapes[key]:
            raise SystemExit(f"layer {layer} {key} shape {list(tensors[key].shape)} != recipe {shapes[key]}")
        if tensors[key].dtype != torch.bfloat16 or not tensors[key].is_contiguous():
            raise SystemExit(f"layer {layer} {key} is not contiguous BF16")
    check_bands(recipe["recipe_id"], layer, tensors, recipe)
    meta = {"recipe_id": recipe["recipe_id"], "layer": layer, "layout": recipe["tensors"]["layout"],
            "is_causal": recipe["attention"]["is_causal"], "softmax_scale": recipe["attention"]["softmax_scale"], **extra}
    torch.save({**tensors, "meta": meta}, out / f"layer{layer:02d}.pt")


def capture_cogvideox(recipe: dict, model_dir: Path, out: Path) -> dict:
    import torch.nn.functional as F
    from diffusers import CogVideoXPipeline

    g = recipe["generation"]
    pipe = CogVideoXPipeline.from_pretrained(str(model_dir), torch_dtype=getattr(torch, g["dtype"])).to("cuda")
    n_layers = pipe.transformer.config.num_layers
    if n_layers != recipe["attention"]["layers"]:
        raise SystemExit(f"transformer has {n_layers} layers, recipe says {recipe['attention']['layers']}")
    step = recipe["capture"]["step_index"]
    state = {"calls": 0, "inside": False, "other": 0}
    pipe.transformer.register_forward_pre_hook(lambda m, a: state.__setitem__("inside", True))
    pipe.transformer.register_forward_hook(lambda m, a, o: state.__setitem__("inside", False))
    orig = F.scaled_dot_product_attention

    def hook(q, k, v, *args, **kw):
        if not state["inside"]:
            state["other"] += 1
            return orig(q, k, v, *args, **kw)
        s, layer = divmod(state["calls"], n_layers)
        state["calls"] += 1
        if s == step:
            if kw.get("attn_mask") is not None or kw.get("is_causal", False) or args:
                raise SystemExit("unexpected SDPA arguments at the capture point")
            tensors = {n: x.transpose(1, 2).to(torch.bfloat16).contiguous().cpu() for n, x in (("q", q), ("k", k), ("v", v))}
            save_layer(out, recipe, layer, tensors, {"src_dtype": str(q.dtype), "step_index": s})
        result = orig(q, k, v, *args, **kw)
        if s == step and layer == n_layers - 1:
            raise _Done
        return result

    F.scaled_dot_product_attention = hook
    try:
        pipe(prompt=g["prompt"], num_videos_per_prompt=1, num_inference_steps=g["num_inference_steps"],
             num_frames=g["num_frames"], guidance_scale=g["guidance_scale"],
             generator=torch.Generator(device="cuda").manual_seed(g["seed"]))
        raise SystemExit("pipeline finished before the capture step")
    except _Done:
        pass
    finally:
        F.scaled_dot_product_attention = orig
    return {"transformer_sdpa_calls": state["calls"], "non_transformer_sdpa_calls": state["other"]}


def capture_llama(recipe: dict, model_dir: Path, dataset_files: list[Path], out: Path) -> dict:
    import pyarrow.parquet as pq
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    t = recipe["text"]
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    text = t["joiner"].join(line for f in dataset_files for line in pq.read_table(str(f)).column(t["column"]).to_pylist())
    ids = tok(text, return_tensors="pt").input_ids[:, : t["tokens"]]
    if ids.shape[1] != t["tokens"] or ids[0, 0].item() != tok.bos_token_id:
        raise SystemExit("tokenized text is shorter than the recipe or lacks BOS")
    if "first_token_ids" in t and ids[0, : len(t["first_token_ids"])].tolist() != t["first_token_ids"]:
        raise SystemExit("tokenization differs from the recipe's first token ids")
    t.setdefault("first_token_ids", ids[0, :8].tolist())
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16).to("cuda").eval()
    n_layers = model.config.num_hidden_layers
    if n_layers != recipe["attention"]["layers"]:
        raise SystemExit(f"model has {n_layers} layers, recipe says {recipe['attention']['layers']}")
    wanted = set(capture_layers(recipe))
    last = max(wanted)

    def capture(module, query, key, value, attention_mask, **kw):
        layer = module.layer_idx
        if abs(float(module.scaling) - recipe["attention"]["softmax_scale"]) > 1e-12:
            raise SystemExit("attention scaling differs from the recipe")
        if layer in wanted:
            tensors = {n: x.transpose(1, 2).contiguous().cpu() for n, x in (("q", query), ("k", key), ("v", value))}
            save_layer(out, recipe, layer, tensors, {"attention_mask_is_none": attention_mask is None})
        result = sdpa_attention_forward(module, query, key, value, attention_mask, **kw)
        if layer == last:
            raise _Done
        return result

    AttentionInterface.register("sage_real_layer_capture", capture)
    model.set_attn_implementation("sage_real_layer_capture")
    with torch.no_grad():
        try:
            model(ids.cuda(), use_cache=False)
            raise SystemExit("forward finished without reaching the last layer")
        except _Done:
            pass
    return {"first_token_ids": ids[0, :8].tolist()}


def capture_diffusion(recipe: dict, model_dir: Path, out: Path) -> dict:
    """Self attention of a diffusers pipeline at chosen (denoising step, block) points.

    The hook sits on ``torch.nn.functional.scaled_dot_product_attention`` while the
    transformer runs.  A call is a self-attention call of the captured stream when both the
    query and the key length equal the recipe's L (this skips text cross attention and, for
    SD3.5's dual-attention blocks, the image-only second attention).  With separate CFG passes
    (``calls_per_step`` 2, Wan) only the conditional pass is captured; with batched CFG
    (CogVideoX, SD3.5) the batch of 2 is captured as is.  Layer file NN is the NN-th entry of
    ``capture.points``.
    """
    import torch.nn.functional as F
    import diffusers

    g, c = recipe["generation"], recipe["capture"]
    pipe = getattr(diffusers, g["pipeline"]).from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
    if g.get("vae_fp32"):
        pipe.vae.to(torch.float32)
    pipe.to("cuda")
    length = recipe["tensors"]["q"][1]
    points = {tuple(p): i for i, p in enumerate(c["points"])}
    per_step = int(c.get("calls_per_step", 1))
    state = {"passes": 0, "inside": False, "block": 0, "saved": 0, "other": 0}

    def pre(_m, _a, _k=None):
        state.update(inside=True, block=0)

    def post(_m, _a, _o, _k=None):
        state["inside"] = False
        state["passes"] += 1

    pipe.transformer.register_forward_pre_hook(pre)
    pipe.transformer.register_forward_hook(post)
    orig = F.scaled_dot_product_attention

    def hook(*args, **kw):
        # Callers pass Q/K/V positionally (CogVideoX, SD3) or as query=/key=/value= (Wan).
        names = ("query", "key", "value")
        q, k, v = (args[i] if i < len(args) else kw.pop(names[i]) for i in range(3))
        args = args[3:]
        if not (state["inside"] and q.shape[-2] == length and k.shape[-2] == length):
            state["other"] += 1
            return orig(q, k, v, *args, **kw)
        block = state["block"]
        state["block"] += 1
        step, phase = divmod(state["passes"], per_step)
        index = points.get((step, block))
        if phase == 0 and index is not None:
            if kw.get("attn_mask") is not None or kw.get("is_causal", False):
                raise SystemExit("unexpected SDPA arguments at a capture point")
            tensors = {n: x.transpose(1, 2).to(torch.bfloat16).contiguous().cpu() for n, x in (("q", q), ("k", k), ("v", v))}
            save_layer(out, recipe, index, tensors, {"step_index": step, "block": block, "src_dtype": str(q.dtype)})
            state["saved"] += 1
        result = orig(q, k, v, *args, **kw)
        if state["saved"] == len(points):
            raise _Done
        return result

    F.scaled_dot_product_attention = hook
    try:
        pipe(**g["call"], generator=torch.Generator(device="cuda").manual_seed(g["seed"]), output_type="latent")
        raise SystemExit(f"pipeline finished after {state['saved']} of {len(points)} capture points")
    except _Done:
        pass
    finally:
        F.scaled_dot_product_attention = orig
    return {"transformer_passes": state["passes"], "other_sdpa_calls": state["other"], "diffusers": diffusers.__version__}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--dataset-file", type=Path)
    ap.add_argument("--dataset-dir", type=Path, help="snapshot root for a recipe's dataset.files")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    recipe = json.loads(a.recipe.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    if any(a.out.iterdir()):
        raise SystemExit(f"output directory is not empty: {a.out}")
    verify_files(a.model_dir, recipe["model"]["files"], recipe["model"]["repo"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if recipe["kind"] == "cogvideox":
        info = capture_cogvideox(recipe, a.model_dir, a.out)
    elif recipe["kind"] == "llama":
        ds = recipe["dataset"]
        if "files" in ds:
            if a.dataset_dir is None:
                raise SystemExit("--dataset-dir is required for this recipe")
            verify_files(a.dataset_dir, ds["files"], ds["repo"])
            dataset_files = [a.dataset_dir / rel for rel in ds["files"]]
        else:
            if a.dataset_file is None or sha256(a.dataset_file) != ds["sha256"]:
                raise SystemExit("dataset file missing or SHA-256 differs from the recipe")
            dataset_files = [a.dataset_file]
        info = capture_llama(recipe, a.model_dir, dataset_files, a.out)
    elif recipe["kind"] == "diffusion":
        info = capture_diffusion(recipe, a.model_dir, a.out)
    else:
        raise SystemExit(f"unknown recipe kind {recipe['kind']!r}")
    if DERIVED:
        # Reference capture: the recipe gains its sanity bands (and, for text, its first
        # token ids); the finalized recipe is written next to the capture and is the one the
        # manifest is bound to.
        recipe["sanity_bands"] = {"note": "reference capture: RMS widened 2x, abs-max 4x", "layers": dict(sorted(DERIVED.items()))}
        final = a.out / "RECIPE.final.json"
        final.write_text(json.dumps(recipe, indent=1) + "\n")
        recipe_sha = sha256(final)
    else:
        recipe_sha = sha256(a.recipe)
    files = {}
    for layer in capture_layers(recipe):
        p = a.out / f"layer{layer:02d}.pt"
        if not p.is_file():
            raise SystemExit(f"layer file missing after capture: {p.name}")
        files[p.name] = {"sha256": sha256(p), "size_bytes": p.stat().st_size}
    props = torch.cuda.get_device_properties(0)
    manifest = {
        "schema_version": 1,
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": recipe_sha,
        "capture_dir": recipe["capture_dir"],
        "files": files,
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": f"{props.major}.{props.minor}",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "capture_info": info,
    }
    tmp = a.out / ".capture.manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, a.out / "capture.manifest.json")
    print(json.dumps({k: manifest[k] for k in ("recipe_id", "capture_dir", "gpu_name", "captured_utc")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
