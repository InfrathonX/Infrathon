#!/usr/bin/env python3
"""Secret-split the ordered lab pools into public, anchor, final and warm-up data."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import stat
from pathlib import Path
from typing import Any


BS1 = {
    "chat_bs1": ("chat", 512, False),
    "math_bs1": ("math", 512, False),
    "code_bs1": ("code", 512, False),
    "longctx_bs1": ("longctx", 256, False),
    "think_bs1": ("think", 1024, True),
}
SPLITS = (("public", 12), ("anchor", 8), ("final", 20))


class UniquePrompts:
    """Allocate once by rendered token sequence across all cells and splits."""

    def __init__(self, tokenizer):
        """Bind the frozen renderer and global content census."""
        self.tokenizer = tokenizer
        self.seen = set()

    def take(self, rows, count, thinking):
        """Take the next distinct rendered prompts or fail on exhaustion."""
        selected = []
        for row in rows:
            key = tuple(self.tokenizer.apply_chat_template(
                row["messages"], tokenize=True, add_generation_prompt=True,
                enable_thinking=thinking,
            ))
            if key in self.seen:
                continue
            self.seen.add(key)
            selected.append(row)
            if len(selected) == count:
                return selected
        raise ValueError(f"insufficient unique rendered prompts: need {count}, found {len(selected)}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read nonblank JSONL rows from a trusted lab pool."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def encoded(rows: list[dict[str, Any]]) -> bytes:
    """Encode rows canonically for both storage and commitments."""
    return b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
        for row in rows
    )


def ranked(
    rows: list[dict[str, Any]],
    secret: bytes,
    domain: str,
) -> list[dict[str, Any]]:
    """Return a deterministic, domain-separated secret ranking."""
    return sorted(
        rows,
        key=lambda row: hmac.new(
            secret,
            (domain + ":" + row["prompt_id"]).encode(),
            hashlib.sha256,
        ).digest(),
    )


def extend_math(
    pool_dir: Path,
    rows: list[dict[str, Any]],
    target_snapshot: Path,
) -> list[dict[str, Any]]:
    """Add cached GSM8K records so C2 and C6 can each receive 40 unique questions."""
    if len(rows) >= 128:
        return rows
    from datasets import Dataset
    from transformers import AutoTokenizer

    cache = pool_dir.parent / "hf_cache" / "datasets"
    cached = sorted((cache / "openai___gsm8k/main").glob("*/*/gsm8k-test.arrow"))
    if len(cached) != 1:
        raise ValueError("expected one frozen cached GSM8K test dataset")
    dataset = Dataset.from_file(str(cached[0]))
    tokenizer = AutoTokenizer.from_pretrained(target_snapshot, local_files_only=True)
    used = {str(row["source_id"]) for row in rows}
    instruction = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    extended = list(rows)
    for index, item in enumerate(dataset):
        if str(index) in used:
            continue
        messages = [{"role": "user", "content": item["question"] + instruction}]
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        extended.append(
            {
                "prompt_id": f"math_extra_{index:04d}",
                "source": "gsm8k",
                "source_id": str(index),
                "messages": messages,
                "token_len": len(ids),
            }
        )
        if len(extended) >= 128:
            break
    if len(extended) < 128:
        raise RuntimeError("cached GSM8K does not provide enough unique questions")
    return extended


def write_slice(path: Path, rows: list[dict[str, Any]]) -> str:
    """Write one canonical slice and return its commitment digest."""
    data = encoded(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    """Create disjoint public/private slices and their manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--secret", type=Path, required=True)
    parser.add_argument("--public-dir", type=Path, required=True)
    parser.add_argument("--private-dir", type=Path, required=True)
    parser.add_argument("--dataset-doc", type=Path, required=True)
    parser.add_argument("--target-snapshot", type=Path, required=True)
    args = parser.parse_args()
    mode = stat.S_IMODE(args.secret.stat().st_mode)
    if mode != 0o600 or args.secret.is_symlink():
        raise SystemExit("secret split seed must be a real mode-0600 file")
    secret_text = args.secret.read_text(encoding="ascii").strip()
    if len(secret_text) != 64 or any(
        character not in "0123456789abcdef" for character in secret_text
    ):
        raise SystemExit("secret split seed must contain 32 lowercase hex bytes")
    secret = bytes.fromhex(secret_text)
    pools = {
        name: read_jsonl(args.pool_dir / f"{name}.jsonl")
        for name in ("chat", "math", "code", "longctx")
    }
    warms = {
        name: read_jsonl(args.pool_dir / f"{name}_warmup.jsonl")
        for name in pools
    }
    pools["math"] = extend_math(args.pool_dir, pools["math"], args.target_snapshot)

    from transformers import AutoTokenizer
    unique = UniquePrompts(AutoTokenizer.from_pretrained(args.target_snapshot, local_files_only=True))

    # C2 and C6 are allocated together from one secret ordering, proving no question overlap.
    math_ranked = ranked(pools["math"], secret, "math-c2-c6")
    cell_rows = {"math_bs1": unique.take(math_ranked, 40, False)}
    math_ids = {row["source_id"] for row in cell_rows["math_bs1"]}
    cell_rows["think_bs1"] = unique.take([row for row in math_ranked if row["source_id"] not in math_ids], 40, True)
    for cell, (pool, _budget, _thinking) in BS1.items():
        if cell not in cell_rows:
            cell_rows[cell] = unique.take(ranked(pools[pool], secret, cell), 40, _thinking)

    commitments: dict[str, str] = {}
    manifests: dict[str, list[dict[str, Any]]] = {
        "anchor": [],
        "final": [],
        "warmup": [],
    }
    public_manifest: list[dict[str, Any]] = []
    for cell, (pool, budget, thinking) in BS1.items():
        offset = 0
        for split, count in SPLITS:
            rows = cell_rows[cell][offset:offset + count]
            offset += count
            rel = f"{split}/{cell}.jsonl"
            target = (
                args.public_dir / f"{cell}.jsonl"
                if split == "public"
                else args.private_dir / rel
            )
            digest = write_slice(target, rows)
            commitments[f"{split}:{cell}"] = digest
            entry = {
                "cell": cell,
                "path": str(target),
                "sha256": digest,
                "count": count,
                "batch_size": 1,
                "max_new_tokens": budget,
                "enable_thinking": thinking,
            }
            (public_manifest if split == "public" else manifests[split]).append(entry)
        warm_pool = "math" if pool == "think" else pool
        warm = unique.take(ranked(warms[warm_pool], secret, "warm:" + cell), 1, thinking)
        target = args.private_dir / "warmup" / f"{cell}.jsonl"
        digest = write_slice(target, warm)
        write_slice(args.public_dir / "warmup" / f"{cell}.jsonl", warm)
        commitments[f"warmup:{cell}"] = digest
        manifests["warmup"].append(
            {
                "cell": cell,
                "path": str(target),
                "sha256": digest,
                "count": 1,
                "batch_size": 1,
                "max_new_tokens": budget,
                "enable_thinking": thinking,
            }
        )

    # Ten independently secret-ranked groups, each with 2-3 members from chat/math/code.
    compositions = (
        ("chat", "math", "code", "chat", "math", "code", "chat", "math"),
        ("math", "code", "chat", "math", "code", "chat", "math", "code"),
        ("code", "chat", "math", "code", "chat", "math", "code", "chat"),
    )
    ranked_mix = {
        name: ranked(pools[name], secret, "mixed:" + name)
        for name in ("chat", "math", "code")
    }
    offsets = {name: 0 for name in ranked_mix}
    groups = []
    for group_index in range(10):
        group = []
        for slot, name in enumerate(compositions[group_index % len(compositions)]):
            row = dict(unique.take(ranked_mix[name], 1, False)[0])
            row.update(
                prompt_id=(
                    f"mixed_g{group_index}_s{slot}_{row['prompt_id']}"
                ),
                mixed_group=group_index,
                mixed_slot=slot,
                mixed_pool=name,
            )
            group.append(row)
        groups.append(group)
    group_offset = 0
    for split, count in (("public", 3), ("anchor", 2), ("final", 5)):
        rows = [
            row
            for group in groups[group_offset:group_offset + count]
            for row in group
        ]
        group_offset += count
        target = (
            args.public_dir / "mixed_bs8.jsonl"
            if split == "public"
            else args.private_dir / split / "mixed_bs8.jsonl"
        )
        digest = write_slice(target, rows)
        commitments[f"{split}:mixed_bs8"] = digest
        entry = {
            "cell": "mixed_bs8",
            "path": str(target),
            "sha256": digest,
            "count": count,
            "row_count": count * 8,
            "batch_size": 8,
            "max_new_tokens": 512,
            "enable_thinking": False,
        }
        (public_manifest if split == "public" else manifests[split]).append(entry)
    mix_warm = []
    for slot, name in enumerate(compositions[0]):
        row = dict(
            unique.take(ranked(warms[name], secret, "mixed-warm:" + name), 1, False)[0]
        )
        row.update(
            prompt_id=f"mixed_warm_s{slot}_{row['prompt_id']}",
            mixed_group=-1,
            mixed_slot=slot,
            mixed_pool=name,
        )
        mix_warm.append(row)
    target = args.private_dir / "warmup" / "mixed_bs8.jsonl"
    digest = write_slice(target, mix_warm)
    write_slice(args.public_dir / "warmup" / "mixed_bs8.jsonl", mix_warm)
    commitments["warmup:mixed_bs8"] = digest
    manifests["warmup"].append(
        {
            "cell": "mixed_bs8",
            "path": str(target),
            "sha256": digest,
            "count": 1,
            "row_count": 8,
            "batch_size": 8,
            "max_new_tokens": 512,
            "enable_thinking": False,
        }
    )

    args.public_dir.mkdir(parents=True, exist_ok=True)
    (args.public_dir / "manifest.json").write_text(
        json.dumps(
            {"schema_version": 1, "cells": public_manifest},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for split in ("anchor", "final", "warmup"):
        (args.private_dir / f"{split}_manifest.json").write_text(
            json.dumps(
                {"schema_version": 1, "cells": manifests[split]},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    (args.private_dir / "commitments.json").write_text(
        json.dumps(commitments, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Dataset manifest",
        "",
        (
            "Secret split seed is retained outside the task. SHA-256 "
            "commitments bind canonical JSONL bytes."
        ),
    ]
    references = args.public_dir / "reference_outputs.json"
    if references.is_file():
        reference_digest = hashlib.sha256(references.read_bytes()).hexdigest()
        lines += [
            "",
            (
                "The seed-generated public reference stream file is "
                "`public_prompts/reference_outputs.json`, SHA-256 "
                f"`{reference_digest}`. It contains ordered full token streams "
                "for all 84 public prompts and is evidence, not an exact-match "
                "gate."
            ),
        ]
    lines += ["", "| slice | SHA-256 |", "| --- | --- |"]
    lines += [
        f"| `{key}` | `{value}` |"
        for key, value in sorted(commitments.items())
    ]
    args.dataset_doc.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
