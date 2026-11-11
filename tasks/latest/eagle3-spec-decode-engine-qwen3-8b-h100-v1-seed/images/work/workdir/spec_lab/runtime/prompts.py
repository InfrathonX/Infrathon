"""Prompt rendering and public/fallback pool loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


LAB_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = next(
    (
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "vendor/EAGLE/eagle/data/mt_bench/question.jsonl").is_file()
    ),
    LAB_ROOT,
)
MTBENCH = REPO_ROOT / "vendor/EAGLE/eagle/data/mt_bench/question.jsonl"


def render(messages: Sequence[Mapping[str, str]], tokenizer, enable_thinking: bool = False) -> torch.Tensor:
    ids = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return torch.as_tensor(ids, dtype=torch.int64)


def load_pool(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            required = {"prompt_id", "source", "source_id", "messages", "token_len"}
            missing = required - record.keys()
            if missing:
                raise ValueError(f"pool line {line_no} missing {sorted(missing)}")
            records.append(record)
    if not records:
        raise ValueError("prompt pool is empty")
    return records


def mtbench_pool() -> list[dict[str, Any]]:
    records = []
    with MTBENCH.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            first = row["turns"][0]
            records.append({
                "prompt_id": f"mtbench-{row['question_id']}",
                "source": "mt_bench",
                "source_id": str(row["question_id"]),
                "messages": [{"role": "user", "content": first}],
                "token_len": None,
            })
    return records
