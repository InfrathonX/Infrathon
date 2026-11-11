#!/usr/bin/env python3
"""Build the disclosed public prompt set (eagle3-spec-decode-public-v2).

The public set is a smoke workload for the Work image. Its cells, batch sizes
and budgets differ from the Judge cells on purpose, and every prompt comes from
a split or dataset the Judge never draws from:

* chat, longctx: UltraChat-200k ``test_sft`` (MIT) at a pinned revision;
* math, think:   GSM8K ``main/train`` (MIT) at a pinned revision;
* code:          MBPP ``sanitized/test`` (CC-BY-4.0) at a pinned revision.

Selection is deterministic and public: rows are ordered by
``sha256(SEED:pool:source_id)`` and taken in that order subject to the length
window of each pool. Nothing here is secret. Pass ``--judge-private-dir`` to
additionally prove that no public message text appears anywhere in the Judge
data (final, anchor and warm-up).

Usage (needs pyarrow and transformers)::

    uv run --with pyarrow --with transformers==4.53.1 python scripts/prepare_public_prompts.py \
        --ultrachat-test-sft test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet \
        --gsm8k-train main/train-00000-of-00001.parquet \
        --mbpp-test sanitized/test-00000-of-00001.parquet \
        --tokenizer <Qwen/Qwen3-8B snapshot b968826d>
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TASK = Path(__file__).resolve().parents[1]
REL_TASK = "tasks/latest/eagle3-spec-decode-engine-qwen3-8b-h100-v1-seed"
PUBLIC_REL = "images/work/workdir/spec_lab/public_prompts"
SEED = "eagle3-spec-decode-public-v2"

SOURCES = {
    "ultrachat_test_sft": {
        "repo": "HuggingFaceH4/ultrachat_200k",
        "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
        "file": "data/test_sft-00000-of-00001-f7dfac4afe5b93f4.parquet",
        "sha256": "c18fd6e77395577652bdefbc5a87044ea799e989451ecfca5c6cf977ae5c6f70",
        "license": "MIT",
    },
    "gsm8k_train": {
        "repo": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "file": "main/train-00000-of-00001.parquet",
        "sha256": "ea82612ea9582142387730c793eb67d3b12849002bc0b7fa6f8efafa7351419d",
        "license": "MIT",
    },
    "mbpp_sanitized_test": {
        "repo": "google-research-datasets/mbpp",
        "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        "file": "sanitized/test-00000-of-00001.parquet",
        "sha256": "e9e9efa2c0d59ef5e55537a9d126b8f875d5ac010a8d75628d76824884e15850",
        "license": "CC-BY-4.0",
    },
}

# cell -> (pool, batch_size, max_new_tokens, enable_thinking, scored batches)
CELLS = {
    "chat_bs2": ("chat", 2, 384, False, 2),
    "math_bs4": ("math", 4, 256, False, 2),
    "code_bs3": ("code", 3, 384, False, 2),
    "longctx_bs2": ("longctx", 2, 128, False, 2),
    "think_bs2": ("think", 2, 768, True, 2),
    "mixed_bs6": ("mixed", 6, 320, False, 2),
}
MIXED_SLOTS = ("chat", "math", "code", "chat", "math", "code")
# Rendered (thinking-off) prompt length windows per pool.
WINDOWS = {
    "chat": (20, 300),
    "math": (40, 200),
    "think": (40, 200),
    "code": (40, 400),
    "longctx": (800, 1400),
}
MATH_INSTRUCTION = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of one file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order_key(pool: str, source_id: str) -> str:
    """Public deterministic ordering key."""
    return hashlib.sha256(f"{SEED}:{pool}:{source_id}".encode()).hexdigest()


def read_parquet(path: Path, expected: str) -> list[dict[str, Any]]:
    """Read one pinned parquet file after checking its digest."""
    import pyarrow.parquet as pq

    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{path}: sha256 {actual} != pinned {expected}")
    return pq.read_table(path).to_pylist()


class Renderer:
    """Count rendered prompt tokens with the frozen Qwen3 chat template."""

    def __init__(self, tokenizer_dir: Path):
        """Load the tokenizer from a local snapshot only."""
        from transformers import AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)

    def length(self, messages: list[dict[str, str]], thinking: bool) -> int:
        """Rendered prompt length in tokens."""
        return len(self.tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=thinking,
        ))


def candidates(args, renderer: Renderer) -> dict[str, list[dict[str, Any]]]:
    """Return every pool's candidate rows in public order (lengths unchecked)."""
    ultra = read_parquet(args.ultrachat_test_sft, SOURCES["ultrachat_test_sft"]["sha256"])
    gsm = read_parquet(args.gsm8k_train, SOURCES["gsm8k_train"]["sha256"])
    mbpp = read_parquet(args.mbpp_test, SOURCES["mbpp_sanitized_test"]["sha256"])
    pools: dict[str, list[dict[str, Any]]] = {}
    pools["chat"] = sorted(
        (
            {"source": "ultrachat_200k_test_sft", "source_id": row["prompt_id"],
             "messages": [{"role": "user", "content": row["prompt"].strip()}]}
            for row in ultra if row["prompt"].strip()
        ),
        key=lambda r: order_key("chat", r["source_id"]),
    )
    gsm_rows = [
        {"source": "gsm8k_train", "source_id": str(index),
         "messages": [{"role": "user", "content": item["question"] + MATH_INSTRUCTION}]}
        for index, item in enumerate(gsm)
    ]
    pools["math"] = sorted(gsm_rows, key=lambda r: order_key("math", r["source_id"]))
    pools["think"] = sorted(gsm_rows, key=lambda r: order_key("think", r["source_id"]))
    pools["code"] = sorted(
        (
            {"source": "mbpp_sanitized_test", "source_id": str(item["task_id"]),
             "messages": [{"role": "user", "content": (
                 item["prompt"].strip()
                 + "\n\nYour code should pass these tests:\n\n```python\n"
                 + "\n".join(item["test_list"]) + "\n```"
             )}]}
            for item in mbpp
        ),
        key=lambda r: order_key("code", r["source_id"]),
    )
    # Long context: concatenate whole UltraChat test_sft conversations (in
    # public order) and end on a user turn, until the length window is met.
    convs = sorted(ultra, key=lambda r: order_key("longctx", r["prompt_id"]))
    longctx, current, ids = [], [], []
    low, high = WINDOWS["longctx"]
    for conv in convs:
        turns = [{"role": m["role"], "content": m["content"]} for m in conv["messages"]]
        if len(turns) < 2 or turns[0]["role"] != "user":
            continue
        current += turns
        ids.append(conv["prompt_id"])
        trimmed = current[:-1] if current[-1]["role"] == "assistant" else current
        size = renderer.length(trimmed, False)
        if size > high:
            current, ids = [], []
        elif size >= low:
            longctx.append({"source": "ultrachat_200k_test_sft",
                            "source_id": "merged:" + ",".join(ids), "messages": trimmed})
            current, ids = [], []
        if len(longctx) >= 16:
            break
    pools["longctx"] = longctx
    return pools


class Allocator:
    """Hand out rows once per (source, source_id) across every public cell."""

    def __init__(self, pools, renderer: Renderer):
        """Copy the ordered pools; nothing is allocated yet."""
        self.pools = {name: list(rows) for name, rows in pools.items()}
        self.renderer = renderer
        self.used: set[tuple[str, str]] = set()

    def take(self, pool: str, count: int, thinking: bool) -> list[dict[str, Any]]:
        """Take the next unused in-window rows of one pool."""
        low, high = WINDOWS[pool]
        out = []
        while len(out) < count:
            if not self.pools[pool]:
                raise ValueError(f"pool {pool} exhausted")
            row = self.pools[pool].pop(0)
            parts = row["source_id"].removeprefix("merged:").split(",")
            keys = {(row["source"], part) for part in parts}
            if keys & self.used:
                continue
            size = self.renderer.length(row["messages"], False)
            if not low <= size <= high:
                continue
            self.used |= keys
            out.append(dict(row, token_len=self.renderer.length(row["messages"], thinking)))
        return out


def encoded(rows: list[dict[str, Any]]) -> bytes:
    """Canonical JSONL encoding (same as prepare_prompts.py)."""
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in rows
    )


def build(args) -> dict[str, Any]:
    """Write the public slices, warm-ups and manifest; return the manifest."""
    renderer = Renderer(args.tokenizer)
    alloc = Allocator(candidates(args, renderer), renderer)
    out_dir: Path = args.out_dir
    (out_dir / "warmup").mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "suite_id": SEED, "selection_seed": SEED,
                "sources": SOURCES, "cells": []}
    for cell, (pool, batch, budget, thinking, batches) in CELLS.items():
        def rows_for(count_batches: int, tag: str) -> list[dict[str, Any]]:
            """Allocate count_batches batches of this cell (tag '' = scored)."""
            if pool != "mixed":
                rows = alloc.take(pool, batch * count_batches, thinking)
                return [dict(r, prompt_id=f"{cell}_{tag}{i:02d}") for i, r in enumerate(rows)]
            rows = []
            for group in range(count_batches):
                for slot, name in enumerate(MIXED_SLOTS):
                    row = alloc.take(name, 1, thinking)[0]
                    rows.append(dict(
                        row, prompt_id=f"{cell}_{tag}g{group}_s{slot}_{name}",
                        mixed_group=group if tag == "" else -1, mixed_slot=slot,
                        mixed_pool=name,
                    ))
            return rows

        scored = rows_for(batches, "")
        warm = rows_for(1, "warm_")
        data = encoded(scored)
        (out_dir / f"{cell}.jsonl").write_bytes(data)
        (out_dir / "warmup" / f"{cell}.jsonl").write_bytes(encoded(warm))
        entry = {
            "cell": cell,
            "path": f"{REL_TASK}/{PUBLIC_REL}/{cell}.jsonl",
            "sha256": hashlib.sha256(data).hexdigest(),
            "count": batches,
            "batch_size": batch,
            "max_new_tokens": budget,
            "enable_thinking": thinking,
        }
        if pool == "mixed":
            entry["row_count"] = len(scored)
        manifest["cells"].append(entry)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def judge_overlap(public_dir: Path, private_dir: Path) -> int:
    """Count public message texts that also occur anywhere in the Judge data."""
    judge_texts = {
        msg["content"].strip()
        for path in private_dir.rglob("*.jsonl")
        for line in path.read_text().splitlines() if line.strip()
        for msg in json.loads(line)["messages"]
    }
    hits = 0
    for path in public_dir.rglob("*.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                for msg in json.loads(line)["messages"]:
                    hits += msg["content"].strip() in judge_texts
    return hits


def main() -> int:
    """Build the public set and optionally prove it is disjoint from the Judge."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ultrachat-test-sft", type=Path, required=True)
    parser.add_argument("--gsm8k-train", type=Path, required=True)
    parser.add_argument("--mbpp-test", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=TASK / PUBLIC_REL)
    parser.add_argument("--judge-private-dir", type=Path)
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps({c["cell"]: c["sha256"] for c in manifest["cells"]}, indent=2))
    if args.judge_private_dir:
        hits = judge_overlap(args.out_dir, args.judge_private_dir)
        print(f"public message texts found in Judge data: {hits}")
        if hits:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
