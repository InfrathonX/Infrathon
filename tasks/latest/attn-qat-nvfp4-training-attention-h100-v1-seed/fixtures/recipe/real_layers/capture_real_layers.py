#!/usr/bin/env python3
"""Capture real attention layers from one flow-matching training step of a Wan video model.

For each sigma of the chosen source: sample a clean latent `x0` from the model itself,
interpolate `x_t`, run one training step with full activation checkpointing and record, on
the chosen blocks, the self-attention `q, k, v` and `do = dL/do`.  Tensors are stored in the
kernel's `[B, H, S, D]` layout, bf16, keeping the source's `heads_kept` leading heads (what one
rank holds under sequence parallelism).

Self attention runs in row chunks, each under its own checkpoint, because the dense
probability matrix of a long video sequence does not fit in device memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _attention_rows(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v)


class Capture:
    """Replaces the transformer's attention dispatch and records the target blocks."""

    def __init__(self, original, chunk_rows: int, heads_kept: int) -> None:
        self.original = original
        self.chunk_rows = chunk_rows
        self.heads_kept = heads_kept
        self.block: int | None = None
        self.targets: set[int] = set()
        self.records: dict[int, dict[str, torch.Tensor]] = {}

    def _record(self, block: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                out: torch.Tensor) -> None:
        kept = self.heads_kept
        record = {
            "q": q[:, :kept].detach().to(torch.bfloat16).cpu(),
            "k": k[:, :kept].detach().to(torch.bfloat16).cpu(),
            "v": v[:, :kept].detach().to(torch.bfloat16).cpu(),
        }
        self.records[block] = record

        def on_grad(grad: torch.Tensor) -> None:
            bhsd = grad.transpose(1, 2)[:, :kept]
            record["do"] = bhsd.detach().to(torch.bfloat16).cpu()

        out.register_hook(on_grad)

    def __call__(self, query, key, value, **kwargs):
        if query.shape[1] != key.shape[1]:
            return self.original(query, key, value, **kwargs)
        q, k, v = (x.transpose(1, 2) for x in (query, key, value))
        pieces = []
        for start in range(0, q.shape[2], self.chunk_rows):
            rows = q[:, :, start:start + self.chunk_rows]
            if torch.is_grad_enabled():
                pieces.append(checkpoint(_attention_rows, rows, k, v, use_reentrant=False))
            else:
                pieces.append(_attention_rows(rows, k, v))
        out = torch.cat(pieces, dim=2).transpose(1, 2)
        block = self.block
        if block in self.targets and out.requires_grad and block not in self.records:
            self._record(block, q, k, v, out)
        return out


def sample_latent(pipe, source: dict, prompt_embeds, negative_embeds) -> torch.Tensor:
    latent = source["latent"]
    geometry = source["geometry"]
    generator = torch.Generator(device="cuda").manual_seed(int(latent["seed"]))
    with torch.no_grad():
        out = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            height=int(geometry["height"]),
            width=int(geometry["width"]),
            num_frames=int(geometry["frames"]),
            num_inference_steps=int(latent["steps"]),
            guidance_scale=float(latent["guidance_scale"]),
            generator=generator,
            output_type="latent",
        )
    return out.frames.to(torch.float32)


def training_step(transformer, capture: Capture, x0: torch.Tensor, prompt_embeds, *, sigma: float,
                  noise_seed: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(noise_seed)
    eps = torch.randn(x0.shape, generator=generator, device="cuda", dtype=x0.dtype)
    x_t = ((1.0 - sigma) * x0 + sigma * eps).to(torch.bfloat16).requires_grad_(True)
    target = eps - x0
    timestep = torch.tensor([1000.0 * sigma], device="cuda", dtype=torch.float32)
    pred = transformer(
        hidden_states=x_t, timestep=timestep, encoder_hidden_states=prompt_embeds, return_dict=False
    )[0]
    loss = F.mse_loss(pred.float(), target)
    loss.backward()
    print(json.dumps({"sigma": sigma, "loss": float(loss.detach())}), flush=True)


def save_layer(path: Path, record: dict[str, torch.Tensor], meta: dict) -> dict:
    shapes = {tuple(tensor.shape) for name, tensor in record.items()}
    if set(record) != {"q", "k", "v", "do"} or len(shapes) != 1:
        raise SystemExit(f"incomplete capture for {path.name}: {sorted(record)} {shapes}")
    stats = {}
    for name, tensor in record.items():
        wide = tensor.float()
        if not bool(torch.isfinite(wide).all()):
            raise SystemExit(f"{path.name} {name} is not finite")
        stats[name] = {
            "rms": float(wide.pow(2).mean().sqrt()),
            "abs_max": float(wide.abs().max()),
        }
    torch.save({**{k: v.contiguous() for k, v in record.items()}, "meta": meta}, path)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape": list(next(iter(shapes))),
        "stats": stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    recipe_bytes = args.recipe.read_bytes()
    recipe = json.loads(recipe_bytes)
    source = recipe["sources"][args.source]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    import diffusers
    from diffusers import WanPipeline
    from diffusers.models.transformers import transformer_wan

    pipe = WanPipeline.from_pretrained(str(args.model_dir), torch_dtype=torch.bfloat16)
    pipe.vae.to(torch.float32)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)
    transformer = pipe.transformer
    transformer.requires_grad_(False)
    transformer.eval()

    capture = Capture(
        transformer_wan.dispatch_attention_fn,
        int(recipe["chunk_rows"]),
        int(source["heads_kept"]),
    )
    transformer_wan.dispatch_attention_fn = capture
    for index, block in enumerate(transformer.blocks):
        block.register_forward_pre_hook(
            lambda _m, _a, i=index: setattr(capture, "block", i)
        )

    with torch.no_grad():
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt=source["prompt"],
            negative_prompt=source["negative_prompt"],
            do_classifier_free_guidance=True,
            max_sequence_length=512,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    x0 = sample_latent(pipe, source, prompt_embeds, negative_embeds)
    expected = source["geometry"]["latent_shape"]
    if list(x0.shape[1:]) != expected:
        raise SystemExit(f"latent shape {list(x0.shape[1:])} differs from the recipe {expected}")

    transformer.enable_gradient_checkpointing()
    args.out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict] = {}
    capture.targets = {int(block) for block in source["blocks"]}
    for percent in source["sigma_percent"]:
        capture.records = {}
        training_step(
            transformer, capture, x0, prompt_embeds,
            sigma=int(percent) / 100.0,
            noise_seed=int(source["noise_seed"]) + int(percent),
        )
        for block in sorted(capture.targets):
            name = source["layer_format"].format(block=block, sigma=int(percent))
            files[name] = save_layer(
                args.out / f"{name}.pt",
                capture.records[block],
                {"recipe_id": recipe["recipe_id"], "source": args.source, "block": block,
                 "sigma_percent": int(percent)},
            )
            print(json.dumps({name: files[name]}), flush=True)

    manifest_path = args.out / recipe["manifest_name"]
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "identity": "per-deployment-capture",
        "recipe": {"id": recipe["recipe_id"], "sha256": hashlib.sha256(recipe_bytes).hexdigest()},
        "files": {},
    }
    if manifest["recipe"]["sha256"] != hashlib.sha256(recipe_bytes).hexdigest():
        raise SystemExit("existing capture manifest was produced by a different recipe")
    manifest["files"].update(files)
    manifest.setdefault("builder", {})["diffusers"] = diffusers.__version__
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
