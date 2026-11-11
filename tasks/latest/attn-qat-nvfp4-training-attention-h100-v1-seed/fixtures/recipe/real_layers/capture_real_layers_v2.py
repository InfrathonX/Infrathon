#!/usr/bin/env python3
"""Capture real attention layers from one training step of a CogVideoX or a Qwen3 model.

Video (`kind: cogvideox`): sample a clean latent `x0` from the model itself, noise it to the
recipe's timestep, run one v-prediction training step with activation checkpointing and record,
on the chosen blocks, the joint text+video self-attention `q, k, v` and `do = dL/do`.

LLM (`kind: llm`): pack real documents from the recipe's dataset rows into one sequence of the
recipe's length, run one next-token training step with activation checkpointing and record, on
the chosen layers, the causal grouped-query attention `q` (after the q/k norms and RoPE), the
un-repeated `k, v` and `do`.

Tensors are stored in the kernel's `[B, H, S, D]` layout, bf16 (`k`, `v` keep `H_kv` heads).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F


class Recorder:
    """Keeps the first grad-enabled attention call of every target block."""

    def __init__(self, targets) -> None:
        self.targets = set(targets)
        self.block: int | None = None
        self.records: dict[int, dict[str, torch.Tensor]] = {}

    def maybe_record(self, block, q, k, v, out_bhsd) -> None:
        if block not in self.targets or block in self.records or not out_bhsd.requires_grad:
            return
        record = {
            "q": q.detach().to(torch.bfloat16).cpu(),
            "k": k.detach().to(torch.bfloat16).cpu(),
            "v": v.detach().to(torch.bfloat16).cpu(),
        }
        self.records[block] = record

        def on_grad(grad: torch.Tensor) -> None:
            record["do"] = grad.detach().to(torch.bfloat16).cpu()

        out_bhsd.register_hook(on_grad)


def save_layer(path: Path, record: dict[str, torch.Tensor], meta: dict) -> dict:
    if set(record) != {"q", "k", "v", "do"}:
        raise SystemExit(f"incomplete capture for {path.name}: {sorted(record)}")
    if record["q"].shape != record["do"].shape or record["k"].shape != record["v"].shape:
        raise SystemExit(f"inconsistent shapes for {path.name}")
    stats = {}
    for name, tensor in record.items():
        wide = tensor.float()
        if not bool(torch.isfinite(wide).all()):
            raise SystemExit(f"{path.name} {name} is not finite")
        stats[name] = {"rms": float(wide.pow(2).mean().sqrt()), "abs_max": float(wide.abs().max())}
    torch.save({**{k: v.contiguous() for k, v in record.items()}, "meta": meta}, path)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape_q": list(record["q"].shape),
        "shape_kv": list(record["k"].shape),
        "stats": stats,
    }


# --------------------------------------------------------------------------------------------
# CogVideoX
# --------------------------------------------------------------------------------------------


def capture_cogvideox(source: dict, model_dir: Path, out: Path, recipe_id: str, name: str) -> dict:
    from diffusers import CogVideoXPipeline
    from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0
    from diffusers.models.embeddings import apply_rotary_emb

    geometry = source["geometry"]
    pipe = CogVideoXPipeline.from_pretrained(str(model_dir), torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe.vae.enable_tiling()
    transformer = pipe.transformer
    transformer.requires_grad_(False)
    layers = source["layers"]
    recorder = Recorder({int(x["block"]) for x in layers})

    class CaptureProcessor(CogVideoXAttnProcessor2_0):
        def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask=None,
                     image_rotary_emb=None):
            text_seq_length = encoder_hidden_states.size(1)
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            batch_size, _, _ = hidden_states.shape
            query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
            head_dim = key.shape[-1] // attn.heads
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)
            if image_rotary_emb is not None:
                query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
                key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)
            out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
            recorder.maybe_record(recorder.block, query, key, value, out)
            hidden_states = out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
            encoder_hidden_states, hidden_states = hidden_states.split(
                [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1)
            return hidden_states, encoder_hidden_states

    for index, block in enumerate(transformer.transformer_blocks):
        block.attn1.set_processor(CaptureProcessor())
        block.register_forward_pre_hook(lambda _m, _a, i=index: setattr(recorder, "block", i))

    with torch.no_grad():
        prompt_embeds, negative_embeds = pipe.encode_prompt(
            prompt=source["prompt"], negative_prompt=source["negative_prompt"],
            do_classifier_free_guidance=True, max_sequence_length=226, device=torch.device("cuda"),
            dtype=torch.bfloat16)
        latent = source["latent"]
        generator = torch.Generator(device="cuda").manual_seed(int(latent["seed"]))
        x0 = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
                  height=int(geometry["height"]), width=int(geometry["width"]),
                  num_frames=int(geometry["frames"]), num_inference_steps=int(latent["steps"]),
                  guidance_scale=float(latent["guidance_scale"]), generator=generator,
                  output_type="latent").frames
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()
    if list(x0.shape[1:]) != list(geometry["latent_shape"]):
        raise SystemExit(f"latent shape {list(x0.shape[1:])} differs from the recipe")
    rope = None
    if transformer.config.use_rotary_positional_embeddings:
        rope = pipe._prepare_rotary_positional_embeddings(
            int(geometry["height"]), int(geometry["width"]), x0.shape[1], torch.device("cuda"))
    transformer.enable_gradient_checkpointing()
    scheduler = pipe.scheduler
    files = {}
    for percent in sorted({int(x["timestep_percent"]) for x in layers}):
        recorder.records = {}
        t = int(round(percent / 100.0 * (scheduler.config.num_train_timesteps - 1)))
        timestep = torch.tensor([t], device="cuda", dtype=torch.long)
        generator = torch.Generator(device="cuda").manual_seed(int(source["noise_seed"]) + percent)
        eps = torch.randn(x0.shape, generator=generator, device="cuda", dtype=torch.float32)
        x_t = scheduler.add_noise(x0.float(), eps, timestep).to(torch.bfloat16).requires_grad_(True)
        target = scheduler.get_velocity(x0.float(), eps, timestep)
        pred = transformer(hidden_states=x_t, encoder_hidden_states=prompt_embeds, timestep=timestep,
                           image_rotary_emb=rope, return_dict=False)[0]
        loss = F.mse_loss(pred.float(), target)
        loss.backward()
        print(json.dumps({"timestep_percent": percent, "loss": float(loss.detach())}), flush=True)
        for entry in layers:
            if int(entry["timestep_percent"]) != percent:
                continue
            block = int(entry["block"])
            layer = source["layer_format"].format(block=block, sigma=percent)
            files[layer] = save_layer(out / f"{layer}.pt", recorder.records[block],
                                      {"recipe_id": recipe_id, "source": name, "block": block,
                                       "timestep_percent": percent})
            print(json.dumps({layer: files[layer]["stats"]}), flush=True)
    return files


# --------------------------------------------------------------------------------------------
# LLMs
# --------------------------------------------------------------------------------------------


def packed_tokens(tokenizer, text_spec: dict, hub: Path | None, length: int) -> torch.Tensor:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(text_spec["repo_id"], text_spec["file"], repo_type="dataset",
                           revision=text_spec["revision"])
    table = pq.read_table(path, columns=[text_spec["column"]]).to_pylist()
    ids: list[int] = []
    for row in text_spec["rows"]:
        ids.extend(tokenizer(table[int(row)][text_spec["column"]], add_special_tokens=False)["input_ids"])
        ids.append(int(tokenizer.eos_token_id))
        if len(ids) >= length:
            return torch.tensor([ids[:length]], dtype=torch.long)
    raise SystemExit(f"rows {text_spec['rows']} give only {len(ids)} tokens, need {length}")


def capture_llm(source: dict, model_dir: Path, out: Path, recipe_id: str, name: str) -> dict:
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer

    recorder = Recorder(int(x) for x in source["layers"])

    def capture_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scaling,
                                             enable_gqa=key.shape[1] != query.shape[1])
        recorder.maybe_record(int(module.layer_idx), query, key, value, out)
        return out.transpose(1, 2).contiguous(), None

    AttentionInterface.register("attn_qat_capture", capture_attention)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=torch.bfloat16,
                                                 attn_implementation="attn_qat_capture")
    model.to("cuda")
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    length = int(source["seq_len"])
    files = {}
    for text_id, text_spec in sorted(source["texts"].items()):
        recorder.records = {}
        ids = packed_tokens(tokenizer, text_spec, None, length).to("cuda")
        embeds = model.get_input_embeddings()(ids).detach().requires_grad_(True)
        logits = model(inputs_embeds=embeds, use_cache=False).logits
        loss = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:])
        loss.backward()
        print(json.dumps({"text": text_id, "loss": float(loss.detach())}), flush=True)
        del logits, loss, embeds
        for layer_index in source["layers"]:
            layer = source["layer_format"].format(layer=int(layer_index), text=int(text_id))
            record = recorder.records[int(layer_index)]
            heads_kept = int(source.get("heads_kept", record["q"].shape[1]))
            kv_kept = heads_kept * record["k"].shape[1] // record["q"].shape[1]
            record = {"q": record["q"][:, :heads_kept], "k": record["k"][:, :kv_kept],
                      "v": record["v"][:, :kv_kept], "do": record["do"][:, :heads_kept]}
            files[layer] = save_layer(out / f"{layer}.pt", record,
                                      {"recipe_id": recipe_id, "source": name, "layer": int(layer_index),
                                       "text": int(text_id)})
            print(json.dumps({layer: files[layer]["stats"]}), flush=True)
        torch.cuda.empty_cache()
    return files


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
    args.out.mkdir(parents=True, exist_ok=True)
    kind = source["kind"]
    if kind == "cogvideox":
        files = capture_cogvideox(source, args.model_dir, args.out, recipe["recipe_id"], args.source)
    elif kind == "llm":
        files = capture_llm(source, args.model_dir, args.out, recipe["recipe_id"], args.source)
    else:
        raise SystemExit(f"unknown source kind {kind}")
    manifest_path = args.out / recipe["manifest_name"]
    digest = hashlib.sha256(recipe_bytes).hexdigest()
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "identity": "per-deployment-capture",
        "recipe": {"id": recipe["recipe_id"], "sha256": digest},
        "files": {},
    }
    if manifest["recipe"]["sha256"] != digest:
        raise SystemExit("existing capture manifest was produced by a different recipe")
    manifest["files"].update(files)
    import diffusers
    import transformers
    manifest.setdefault("builder", {}).update(
        {"diffusers": diffusers.__version__, "transformers": transformers.__version__})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
