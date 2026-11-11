#!/usr/bin/env python3
"""Capture router top-k decisions of open MoE models on real text.

For each model: download (safetensors + config + tokenizer), run causal-LM forward
passes over a mixed real-text corpus (encyclopedia prose, chat, source code, math
word problems, reading-comprehension QA; the sources interleave so any contiguous
token window mixes them), record every MoE layer's top-k expert ids per token
(computed from the router's own weights on its input hidden states), save
<out>/<name>.npy (int16 [layers, tokens, topk]) and <name>.json (provenance and the
per-sequence source boundaries), then delete the weights.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

MODELS = {
    # name: (repo, experts, topk, device_map)
    "granite-3.1-3b-a800m": ("ibm-granite/granite-3.1-3b-a800m-instruct", 40, 8, "auto"),
    "gpt-oss-20b": ("openai/gpt-oss-20b", 32, 4, "single"),
    "olmoe-1b-7b": ("allenai/OLMoE-1B-7B-0924", 64, 8, "auto"),
    "qwen3-30b-a3b": ("Qwen/Qwen3-30B-A3B", 128, 8, "auto"),
    "phi-3.5-moe": ("microsoft/Phi-3.5-MoE-instruct", 16, 2, "auto"),
    "mixtral-8x7b": ("mistralai/Mixtral-8x7B-v0.1", 8, 2, "auto"),
}

# One slot per sequence; repeated until enough tokens are captured.
PATTERN = ("wiki", "chat", "code", "math", "wiki", "chat", "code", "qa")


def _wiki(limit: int) -> list[str]:
    from datasets import load_dataset

    out, buf = [], []
    for row in load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train"):
        t = row["text"]
        if t.strip().startswith("= ") and not t.strip().startswith("= =") and buf and sum(map(len, buf)) > 6000:
            out.append("".join(buf))
            buf = []
            if len(out) >= limit:
                break
        buf.append(t)
    return out


def _chat(limit: int) -> list[str]:
    from datasets import load_dataset

    out = []
    for row in load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft", streaming=True):
        turns = [f"<|{m['role']}|>\n{m['content']}\n" for m in row["messages"]]
        text = "".join(turns)
        if len(text) > 4000:
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _code(limit: int) -> list[str]:
    from datasets import load_dataset

    out, buf = [], []
    for row in load_dataset("codeparrot/codeparrot-clean-valid", split="train", streaming=True):
        content = row["content"]
        if len(content) < 1500:
            continue
        buf.append(f"# file: {row['repo_name']}/{row['path']}\n{content}\n")
        if sum(map(len, buf)) > 8000:
            out.append("".join(buf))
            buf = []
        if len(out) >= limit:
            break
    return out


def _math(limit: int) -> list[str]:
    from datasets import load_dataset

    out, block = [], []
    for row in load_dataset("openai/gsm8k", "main", split="train"):
        block.append(f"Question: {row['question']}\nAnswer: {row['answer']}\n\n")
        if len(block) == 40:
            out.append("".join(block))
            block = []
            if len(out) >= limit:
                break
    return out


def _qa(limit: int) -> list[str]:
    from datasets import load_dataset

    out, block, seen = [], [], set()
    for row in load_dataset("rajpurkar/squad", split="train"):
        if row["context"] in seen:
            continue
        seen.add(row["context"])
        block.append(f"{row['title']}: {row['context']}\nQ: {row['question']}\n\n")
        if len(block) == 20:
            out.append("".join(block))
            block = []
            if len(out) >= limit:
                break
    return out


LOADERS = {"wiki": _wiki, "chat": _chat, "code": _code, "math": _math, "qa": _qa}
DATASETS = {
    "wiki": "Salesforce/wikitext wikitext-103-raw-v1 train",
    "chat": "HuggingFaceH4/ultrachat_200k test_sft",
    "code": "codeparrot/codeparrot-clean-valid train",
    "math": "openai/gsm8k main train",
    "qa": "rajpurkar/squad train",
}


def corpus(sequences: int) -> tuple[list[tuple[str, str]], dict[str, str]]:
    per_source = {s: PATTERN.count(s) * (sequences // len(PATTERN) + 2) for s in set(PATTERN)}
    pools: dict[str, list[str]] = {}
    failed: dict[str, str] = {}
    for source, count in per_source.items():
        try:
            pools[source] = LOADERS[source](count)
        except Exception as exc:  # keep the other sources
            failed[source] = f"{type(exc).__name__}: {exc}"[:300]
            pools[source] = []
    out: list[tuple[str, str]] = []
    cursor = {s: 0 for s in pools}
    slot = 0
    while len(out) < sequences + 16 and any(cursor[s] < len(pools[s]) for s in pools):
        source = PATTERN[slot % len(PATTERN)]
        slot += 1
        if cursor[source] < len(pools[source]):
            out.append((source, pools[source][cursor[source]]))
            cursor[source] += 1
    return out, failed


def routers(model, experts: int, hidden: int):
    found = []
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.Tensor) and weight.dim() == 2 and tuple(weight.shape) == (experts, hidden):
            if any(k in name.lower() for k in ("gate", "router")) and "proj" not in name.lower():
                found.append((name, module))
    return found


def capture(name: str, repo: str, experts: int, topk: int, device_map: str, out: Path, cache: Path,
            texts: list[tuple[str, str]], failed: dict[str, str], tokens_needed: int, seq_len: int) -> None:
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    local = snapshot_download(repo, cache_dir=str(cache),
                              allow_patterns=["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt", "*.py"])
    revision = Path(local).name
    print(f"[{name}] downloaded in {time.time() - t0:.0f}s at {local}", flush=True)
    config = AutoConfig.from_pretrained(local, trust_remote_code=False)
    text_cfg = getattr(config, "text_config", config)
    hidden = int(text_cfg.hidden_size)
    tok = AutoTokenizer.from_pretrained(local)
    extra = {}
    if getattr(config, "quantization_config", None) and "mxfp4" in str(config.quantization_config).lower():
        from transformers import Mxfp4Config

        extra["quantization_config"] = Mxfp4Config(dequantize=True)  # BF16 weights, no hub kernels
    dmap = {"": 0} if device_map == "single" else device_map
    model = AutoModelForCausalLM.from_pretrained(local, dtype=torch.bfloat16, device_map=dmap, **extra)
    model.eval()
    found = routers(model, experts, hidden)
    if not found:
        raise RuntimeError(f"{name}: no router modules with weight [{experts}, {hidden}]")
    print(f"[{name}] {len(found)} routers, e.g. {found[0][0]}", flush=True)
    records: list[list[torch.Tensor]] = [[] for _ in found]

    def make_hook(index, module):
        def hook(_mod, args):
            x = args[0]
            x = x.reshape(-1, x.shape[-1]).float()
            logits = x @ module.weight.float().t()
            bias = getattr(module, "bias", None)
            if isinstance(bias, torch.Tensor):
                logits = logits + bias.float()
            records[index].append(torch.topk(logits, topk, dim=-1).indices.to(torch.int16).cpu())
        return hook

    handles = [m.register_forward_pre_hook(make_hook(i, m)) for i, (_n, m) in enumerate(found)]
    total = 0
    sequences = []
    first_device = next(model.parameters()).device
    with torch.no_grad():
        for source, text in texts:
            ids = tok(text, return_tensors="pt", add_special_tokens=True).input_ids[:, :seq_len]
            if ids.shape[1] < 256:
                continue
            model(input_ids=ids.to(first_device))
            sequences.append({"source": source, "start": total, "tokens": int(ids.shape[1])})
            total += int(ids.shape[1])
            if total >= tokens_needed:
                break
    for h in handles:
        h.remove()
    table = np.stack([torch.cat(r, 0).numpy() for r in records], 0)
    if table.shape[1] != total or table.min() < 0 or table.max() >= experts:
        raise RuntimeError(f"{name}: captured table {table.shape} inconsistent with {total} tokens")
    np.save(out / f"{name}.npy", table)
    counts = {}
    for row in sequences:
        counts[row["source"]] = counts.get(row["source"], 0) + row["tokens"]
    meta = {
        "name": name, "repo": repo, "revision": revision, "experts": experts, "topk": topk,
        "layers": int(table.shape[0]), "tokens": int(table.shape[1]), "router_modules": [n for n, _ in found],
        "source_tokens": counts, "datasets": {s: DATASETS[s] for s in counts}, "failed_sources": failed,
        "sequences": sequences, "seq_len": seq_len,
        "method": "router top-k computed from the router weight (and bias) on its input hidden states; "
                  "bf16 model, fp32 logits; one forward pass per sequence, no padding",
        "seconds": round(time.time() - t0, 1),
    }
    (out / f"{name}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[{name}] saved {table.shape} {counts} in {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    shutil.rmtree(Path(local).parents[1], ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--tokens", type=int, default=163840)
    p.add_argument("--seq-len", type=int, default=2048)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    texts, failed = corpus(a.tokens // 1024)
    print(f"corpus: {len(texts)} sequences, failed sources {failed}", flush=True)
    for name in a.models:
        if (a.out / f"{name}.npy").exists():
            print(f"[{name}] exists, skip", flush=True)
            continue
        repo, experts, topk, device_map = MODELS[name]
        try:
            capture(name, repo, experts, topk, device_map, a.out, a.cache, texts, failed, a.tokens, a.seq_len)
        except Exception as exc:  # keep going with the next model
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
