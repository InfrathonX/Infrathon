#!/usr/bin/env python3
"""Generate final-cohort vanilla greedy streams inside the Judge runtime."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch


JUDGE_ROOT = Path("/opt/eagle3_judge")
SPEC_ROOT = Path("/app/workdir/spec_lab")


def atomic_json(path: Path, value: object) -> None:
    """Replace one output with a fully flushed canonical JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_rows(path: Path) -> list[dict]:
    """Load nonempty JSONL rows from one committed final-cohort file."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    """Generate all final-cohort streams and atomically publish the result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge-root", type=Path, default=JUDGE_ROOT)
    parser.add_argument("--spec-root", type=Path, default=SPEC_ROOT)
    args = parser.parse_args(argv)

    import sys

    sys.path.insert(0, str(args.spec_root))
    from runtime.loader import build_contract, default_head_dir, load_target, load_tokenizer
    from runtime.prompts import render

    private = args.judge_root / "private_data"
    manifest = json.loads((private / "final_manifest.json").read_text())
    head_manifest = json.loads((default_head_dir() / "MANIFEST.json").read_text())
    contract = build_contract(
        head_manifest["revision"],
        draft_model_id=head_manifest["repo_id"],
    )
    model = load_target(contract)
    tokenizer = load_tokenizer(contract)
    streams: dict[str, list[int]] = {}
    counts: dict[str, int] = {}
    with torch.inference_mode():
        for cell in manifest["cells"]:
            cell_name = str(cell["cell"])
            budget = int(cell["max_new_tokens"])
            thinking = bool(cell["enable_thinking"])
            rows = load_rows(private / "final" / f"{cell_name}.jsonl")
            counts[cell_name] = len(rows)
            for row in rows:
                prompt = render(row["messages"], tokenizer, thinking).to(contract.device)
                output = model.generate(
                    prompt.unsqueeze(0),
                    attention_mask=torch.ones_like(prompt).unsqueeze(0),
                    do_sample=False,
                    max_new_tokens=budget,
                    eos_token_id=list(contract.eos_token_ids),
                )[0]
                generated = output[prompt.numel() :].detach().cpu().tolist()
                streams[str(row["prompt_id"])] = [int(token) for token in generated]
                print(f"{cell_name} {row['prompt_id']} {len(generated)}", flush=True)
    # The label mirrors the frozen policy so the two private files cannot
    # disagree about the release state; the verifier never gates on it.
    policy_status = str(json.loads((private / "policy.json").read_text(encoding="utf-8"))["status"])
    atomic_json(
        args.output,
        {
            "schema_version": 1,
            "status": policy_status,
            "generator": "Qwen3-8B BF16 eager transformers.generate do_sample=False",
            "cell_prompt_counts": counts,
            "streams": streams,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
