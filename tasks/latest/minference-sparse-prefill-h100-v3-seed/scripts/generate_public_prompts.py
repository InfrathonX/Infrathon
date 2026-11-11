#!/usr/bin/env python3
"""Generate the public prompt asset with the pinned RULER generators (CPU only).

The output (manifest.json, prompts.jsonl) is the `public/public-prompts/` asset
in assets/manifest.json.  Its manifest also carries `public_suite`, so the
public layers, trials and boundary lengths change with the asset.  Needs nltk
(punkt_tab), wonderwords, pyyaml and transformers, a checkout of
microsoft/MInference at the pinned commit, a checkout of
gkamradt/LLMTest_NeedleInAHaystack and the public model's tokenizer.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "activations/recipe"))
import ruler_reference  # noqa: E402

TASKS = ("niah_multikey_2", "cwe", "fwe")
BUDGETS = (8192, 16384, 24576)
SEED = 20261105
SUITE = {"timed_layers": [8, 20], "warmup_layer": 5, "trials": 3,
         "boundary_lengths": [64, 67, 127, 128, 257, 1000, 1023, 1025, 3071, 3072, 6000]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--essay-checkout", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ruler_reference.TASK_NAMES = TASKS
    ruler_reference.SOURCE_FILES = (*ruler_reference.SOURCE_FILES, "data/synthetic/common_words_extraction.py")
    with tempfile.TemporaryDirectory(prefix="public-prompts-") as directory:
        build = Path(directory) / "build"
        ruler_reference.build_prompts(SimpleNamespace(
            upstream=args.upstream, tokenizer=args.tokenizer, essay_checkout=args.essay_checkout,
            output=str(build), tasks=",".join(TASKS), lengths=",".join(map(str, BUDGETS)),
            samples_per_task=1, seed=SEED, partition="public", template="llama-3",
            budget_margin=16, workers=4))
        manifest = json.loads((build / "manifest.json").read_text())
        manifest.update(description="RULER-derived public prompts", public_suite=SUITE,
                        scope="public diagnostic prompts for one model")
        args.output.mkdir(parents=True)
        shutil.copyfile(build / "prompts.jsonl", args.output / "prompts.jsonl")
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "prompts_sha256": manifest["prompts_sha256"]}))


if __name__ == "__main__":
    main()
