#!/usr/bin/env python3
"""Generate exact one-GPU reference ledgers (and one-GPU reference timing) for every case tier."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch


TASK_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = TASK_ROOT / "runtime"
sys.path.insert(0, str(RUNTIME_ROOT))

from cases import CASES_BY_TIER, manifest  # noqa: E402
from evaluation import atomic_json, ledger_timing, load_case_frames, load_prompt, save_golden, sha256_file  # noqa: E402
from sdv2_core.loader import load_model  # noqa: E402
from sdv2_core.plan import StreamPlan  # noqa: E402
from sdv2_core.reference import run_case  # noqa: E402


def _tier_output(tier: str, public_output: Path, judge_output: Path) -> Path:
    return public_output if tier == "public" else judge_output / tier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=TASK_ROOT.parent.parent / "assets")
    parser.add_argument("--public-output", type=Path, default=TASK_ROOT / "build" / "goldens_public")
    parser.add_argument("--judge-output", type=Path, default=TASK_ROOT / "build" / "goldens_judge")
    parser.add_argument("--tiers", nargs="+", choices=tuple(CASES_BY_TIER), default=list(CASES_BY_TIER))
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    # Goldens are a diagnostic reference now (the Judge parity oracle is the
    # seed reference measured in the same run), so any single leased CUDA
    # device may produce them.  Requiring an explicit selection keeps the lab
    # lock honest and records which device the payloads came from.
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise SystemExit("golden generation requires an explicit CUDA_VISIBLE_DEVICES lab lock")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    core = load_model(device, load_text_encoder=False)
    summaries: dict[str, list[dict[str, object]]] = {}
    for tier in args.tiers:
        output = _tier_output(tier, args.public_output, args.judge_output)
        output.mkdir(parents=True, exist_ok=True)
        summaries[tier] = []
        for case_spec in CASES_BY_TIER[tier]:
            if args.case and case_spec.case_id not in args.case:
                continue
            case = case_spec.to_manifest_row()
            frames = load_case_frames(case, args.assets / "clips")
            prompt_dir = "hidden" if tier == "final" else "public"
            prompt = load_prompt(args.assets / "embeddings" / prompt_dir / f"{case_spec.prompt_id}.pt")
            plan = StreamPlan.from_video(
                frames,
                steps=case_spec.steps,
                case_seed=case_spec.case_seed,
            )
            pushed_ns: dict[int, int] = {}
            ledger = run_case(plan, prompt, device, core=core, push_log=pushed_ns)
            ended_ns = time.monotonic_ns()
            started_ns = pushed_ns[0]
            execution = (ended_ns - started_ns) / 1e9
            # Single-GPU reference timing, recorded for the one-card comparison.
            timing = ledger_timing(
                ledger,
                pushed_ns,
                start_ns=started_ns,
                construction_seconds=0.0,
                execution_seconds=execution,
                construction_to_exit_seconds=execution,
            )
            golden = save_golden(
                output / case_spec.case_id,
                case=case,
                ledger=ledger,
                reference_timing=timing,
                include_full_frames=tier != "public",
            )
            summaries[tier].append(
                {
                    "id": case_spec.case_id,
                    "steady_fps": timing["steady_fps"],
                    "ttff_seconds": timing["ttff_seconds"],
                    "bytes": sum(item["bytes"] for item in golden["files"].values()),
                }
            )
            print(json.dumps(summaries[tier][-1], sort_keys=True), flush=True)
            del ledger, plan, frames, prompt
            gc.collect()
            torch.cuda.empty_cache()
        suite_manifest = {
            "schema_version": 1,
            "suite": manifest(tier)["suite"],
            "cases": [],
        }
        for spec in CASES_BY_TIER[tier]:
            case_manifest = output / spec.case_id / "manifest.json"
            saved = json.loads(case_manifest.read_text())
            row = {"id": spec.case_id, "steady_fps": saved["reference_timing"]["steady_fps"],
                   "ttff_seconds": saved["reference_timing"]["ttff_seconds"],
                   "bytes": sum(item["bytes"] for item in saved["files"].values())}
            suite_manifest["cases"].append(
                {
                    **row,
                    "manifest_sha256": sha256_file(case_manifest),
                }
            )
        atomic_json(output / "manifest.json", suite_manifest)
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

