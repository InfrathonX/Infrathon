#!/usr/bin/env python3
"""Generate source-pinned v3 partitions with at most four CPU workers.

The official templates, labels and generator parameters are unchanged. Seed
domains, immutable essay subset and generation-token reservation are disclosed
adaptations. Secret holdout/final seeds and manifests belong outside Work.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import secrets
import sys
from types import SimpleNamespace

import ruler_reference
from ruler_v3 import PROTOCOL, PARTITIONS, derive_partition_seed, load_prompts, performance_plan
from quality_v3 import TASKS, BUDGETS, file_sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--essay-checkout", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    parser.add_argument("--seed-file", type=Path, required=True,
                        help="private author-owned seed receipt; draw once if absent")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples-per-cell", type=int)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 4:
        parser.error("at most four CPU generator workers are allowed")
    defaults = {"public": 3, "final": 3, "calibration": 3, "author-pilot": 10, "author-holdout": 25}
    samples = args.samples_per_cell or defaults[args.partition]
    if samples < 1 or (args.partition in ("public", "final", "calibration") and samples != 3):
        parser.error("performance partitions require three independent prompts per cell")
    args.seed_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.seed_file.exists():
        receipt = json.loads(args.seed_file.read_text())
        if receipt.get("protocol") != PROTOCOL or receipt.get("partition") != args.partition:
            raise ValueError("seed receipt belongs to a different generator partition")
        seed = receipt["master_seed"]
    else:
        seed = secrets.randbits(128)
        fd = os.open(args.seed_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump({"protocol": PROTOCOL, "partition": args.partition, "master_seed": seed}, stream)
            stream.write("\n")
    os.chmod(args.seed_file, 0o600)
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700 if args.partition in ("final", "author-holdout") else 0o755)
    original = ruler_reference.derive_seed
    ruler_reference.derive_seed = derive_partition_seed
    try:
        ruler_reference.build_prompts(SimpleNamespace(upstream=args.upstream, tokenizer=args.tokenizer,
            essay_checkout=args.essay_checkout, output=str(args.output), tasks=",".join(TASKS),
            lengths=",".join(map(str, BUDGETS)), samples_per_task=samples, seed=seed,
            partition=args.partition, template="llama-3", budget_margin=16, workers=args.workers))
    finally:
        ruler_reference.derive_seed = original
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"v3_protocol": PROTOCOL,
        "v3_grid": {"tasks": list(TASKS), "budgets": list(BUDGETS), "samples_per_cell": samples},
        "v3_generator_sha256": file_sha256(Path(__file__)),
        "seed_domain_protocol": PROTOCOL,
        "scope": "four fixed task types, three context budgets and one model; author QA and kernel fixture protocol, not full published RULER"})
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    prompts, manifest = load_prompts(args.output, expected_partition=args.partition)
    if args.partition in ("public", "final", "calibration"):
        plan = performance_plan(prompts, args.partition, {
            "prompt_manifest_sha256": file_sha256(manifest_path), "source_commit": manifest["source_commit"],
            "tokenizer_sha256": manifest["tokenizer_sha256"], "status": "awaiting-dense-capture"})
        (args.output / "performance-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"partition": args.partition, "samples": len(prompts),
                     "prompts_sha256": manifest["prompts_sha256"], "v3_protocol": PROTOCOL}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
