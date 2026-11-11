#!/usr/bin/env python3
"""Public all-cell evaluator for the EAGLE-3 engine task."""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SPEC_ROOT = (
    Path("/app/workdir/spec_lab")
    if Path("/app/workdir/spec_lab").is_dir()
    else HERE.parent / "workdir/spec_lab"
)
sys.path.insert(0, str(SPEC_ROOT))
sys.path.insert(0, str(HERE))

from runtime.api import (  # noqa: E402
    GenerationResult,
    GenerationSpec,
    validate_prompts,
    validate_result,
)
from runtime.loader import (  # noqa: E402
    build_contract,
    default_head_dir,
    load_draft_head,
    load_target,
    load_tokenizer,
)
from runtime.evaluate import timed_generate, _WALL_CLOCK
from runtime.oracle import judge_outputs, teacher_forced_check  # noqa: E402
from runtime.prompts import load_pool, render  # noqa: E402
from source_admission import validate_submission  # noqa: E402

# Public smoke cells: (batch size, max new tokens, thinking).  These shapes and
# budgets deliberately differ from the six Judge cells; see public_prompts/manifest.json.
CELLS = {
    "chat_bs2": (2, 384, False),
    "math_bs4": (4, 256, False),
    "code_bs3": (3, 384, False),
    "longctx_bs2": (2, 128, False),
    "mixed_bs6": (6, 320, False),
    "think_bs2": (2, 768, True),
}


def geomean(values: list[float]) -> float:
    """Return the geometric mean used for the diagnostic throughput score."""
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def main(argv: list[str] | None = None) -> int:
    """Evaluate the selected candidate on the public corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cpu-only", action="store_true")
    modes.add_argument("--quick", action="store_true")
    modes.add_argument("--full", action="store_true")
    parser.add_argument("--candidate-version", type=int)
    parser.add_argument("--out", type=Path, default=HERE / "results/public_result.json")
    parser.add_argument("--write-reference-outputs", type=Path)
    args = parser.parse_args(argv)
    admission = validate_submission(SPEC_ROOT, args.candidate_version, workspace_mode=True)
    if not admission["ok"]:
        payload = {"status": "FAIL", "source_admission": admission}
        print(json.dumps(payload, indent=2))
        return 2
    if args.cpu_only:
        payload = {"status": "PASS", "mode": "cpu-only", "source_admission": admission}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload, indent=2))
        return 0

    head = default_head_dir()
    manifest = json.loads((head / "MANIFEST.json").read_text())
    contract = build_contract(manifest["revision"], draft_model_id=manifest["repo_id"])
    target = load_target(contract)
    tokenizer = load_tokenizer(contract)
    draft = load_draft_head(head, contract)
    version = admission["candidate_version"]
    module = importlib.import_module(f"solutions.engine_v{version:03d}.candidate")
    build_start = _WALL_CLOCK()
    engine = module.build(target, draft, tokenizer, contract)
    build_s = _WALL_CLOCK() - build_start
    if build_s > contract.build_timeout_s:
        raise TimeoutError("candidate build exceeded the public cap")
    policy = json.loads((SPEC_ROOT / "runtime/policy_public.json").read_text())
    reference_path = SPEC_ROOT / "public_prompts/reference_outputs.json"
    reference = json.loads(reference_path.read_text()) if reference_path.is_file() else {}
    generated_reference: dict[str, list[list[int]]] = {}
    cell_reports = {}
    all_records = []
    deterministic_pass = True
    for cell, (batch_size, budget, thinking) in CELLS.items():
        rows = load_pool(SPEC_ROOT / f"public_prompts/{cell}.jsonl")
        warm_rows = load_pool(SPEC_ROOT / f"public_prompts/warmup/{cell}.jsonl")
        limit = batch_size if args.quick else len(rows)
        rows = rows[:limit]
        prompts = [
            render(row["messages"], tokenizer, thinking).to(contract.device)
            for row in rows
        ]
        warm = [
            render(row["messages"], tokenizer, thinking).to(contract.device)
            for row in warm_rows[:batch_size]
        ]
        spec = GenerationSpec(budget)
        engine.generate(warm, spec)
        batches = [prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]
        walls, lengths, oracle_records, streams = [], [], [], []
        for batch in batches:
            validate_prompts(batch, contract, spec)
            result, generated_lengths, wall = timed_generate(engine, batch, spec, contract)
            walls.append(wall)
            lengths.extend(generated_lengths)
            for prompt, output in zip(batch, result.token_ids):
                rec = teacher_forced_check(target, output, prompt.numel(), policy["eps_hard"])
                rec.prompt_ids = [prompt]
                rec.result = GenerationResult([output])
                rec.spec = spec
                rec.contract = contract
                oracle_records.append(rec)
                streams.append(output.detach().cpu().tolist())
        # O1/O2 are meaningful per cell, but the frozen tier policy explicitly
        # has global-run scope.  Applying rare-event tiers to a three-prompt
        # quick cell would turn a single event into an artificial 33% rate.
        oracle = judge_outputs(
            oracle_records,
            policy["eps_hard"],
            policy["p_soft"],
        )
        all_records.extend(oracle_records)
        rerun = engine.generate([prompts[-1]], spec)
        deterministic = torch.equal(
            rerun.token_ids[0],
            torch.tensor(streams[-1], device=contract.device),
        )
        exact = None
        if cell in reference:
            expected = reference[cell][:len(streams)]
            exact = sum(left == right for left, right in zip(streams, expected))
        generated_reference[cell] = streams
        throughput = sum(lengths) / sum(walls)
        passed = oracle["passed"] and deterministic
        deterministic_pass &= deterministic
        cell_reports[cell] = {
            "passed": passed,
            "tokens_per_s": throughput,
            "tokens": sum(lengths),
            "wall_s": sum(walls),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "oracle": oracle,
            "deterministic": deterministic,
            "reference_exact": exact,
            "reference_count": len(streams),
        }
    aggregate = judge_outputs(
        all_records,
        policy["eps_hard"],
        policy["p_soft"],
        policy.get("tiers"),
    )
    all_pass = aggregate["passed"] and deterministic_pass
    payload = {
        "status": "PASS" if all_pass else "FAIL",
        "mode": "quick" if args.quick else "full",
        "build_s": build_s,
        "score_tokens_per_s": geomean(
            [item["tokens_per_s"] for item in cell_reports.values()]
        ),
        "cells": cell_reports,
        "oracle": aggregate,
        "source_admission": admission,
        "policy": policy,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.write_reference_outputs:
        args.write_reference_outputs.write_text(
            json.dumps(generated_reference, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, allow_nan=False))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
