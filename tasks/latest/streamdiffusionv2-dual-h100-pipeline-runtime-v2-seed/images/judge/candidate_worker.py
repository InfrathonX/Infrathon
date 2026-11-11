#!/usr/bin/env python3
"""Unprivileged two-GPU candidate worker; manifest enters only on stdin.

This controller, not the candidate, records when each chunk is pushed: the
stamp is taken immediately before ``start`` / ``push`` is called and written
to ``push_timestamps.json`` beside the sink's emission ledger. The runtime is
constructed with ``plan.meta()``, which carries no input frames.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

MAX_MANIFEST_BYTES = 2 * 1024 * 1024
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,127}\Z")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_runtime(candidate: Path):
    entry = candidate / "runtime.py"
    package_init = candidate / "__init__.py"
    if not entry.is_file() or not package_init.is_file():
        raise RuntimeError("candidate requires runtime.py and __init__.py")
    sys.path.insert(0, str(candidate.parent))
    module = importlib.import_module(f"{candidate.name}.runtime")
    if Path(module.__file__).resolve() != entry.resolve():
        raise RuntimeError("candidate runtime import escaped the admitted package")
    runtime_class = getattr(module, "Runtime", None)
    if not isinstance(runtime_class, type):
        raise RuntimeError("candidate runtime.py does not export Runtime")
    return runtime_class


def _execute_case(
    runtime_class,
    case: dict,
    output_root: Path,
    clip_root: Path,
    embedding_root: Path,
    master_port: int,
) -> dict:
    import torch
    from evaluation import ledger_timing, load_case_frames, load_prompt
    from sdv2_core import OutputSink, StreamPlan, load_model

    case_id = str(case["id"])
    if SAFE_ID.fullmatch(case_id) is None:
        raise RuntimeError(f"unsafe case id: {case_id}")
    frames = load_case_frames(case, clip_root)
    plan = StreamPlan.from_video(
        frames,
        steps=int(case["steps"]),
        case_seed=int(case["case_seed"]),
    )
    if len(plan) != int(case["chunks_expected"]):
        raise RuntimeError(f"case {case_id} chunk count disagrees with its manifest")
    prompt_embeds = load_prompt(embedding_root / f"{case['prompt_id']}.pt")
    sink = OutputSink()
    construction_started_ns = time.monotonic_ns()
    runtime = runtime_class(
        load_model,
        [torch.device("cuda:0"), torch.device("cuda:1")],
        plan.meta(),
        sink,
        master_port=master_port,
    )
    construction_ended_ns = time.monotonic_ns()
    pushed_ns: dict[int, int] = {}
    ledger = None
    try:
        started_ns = pushed_ns[0] = time.monotonic_ns()
        runtime.start(prompt_embeds, plan.frame_chunk(0))
        for chunk in plan.chunks[1:]:
            # Throughput mode: every input is already available and pushes are
            # issued back-to-back as soon as the candidate accepts the prior one.
            # The stamp is the moment this chunk is handed over.
            chunk_frames = plan.frame_chunk(chunk)
            pushed_ns[chunk.index] = time.monotonic_ns()
            runtime.push(chunk.index, chunk_frames)
        runtime.flush()
        ledger = runtime.close()
    finally:
        if ledger is None:
            runtime.close()
    ended_ns = time.monotonic_ns()
    if ledger is None:
        raise RuntimeError("Runtime.close did not return a merged ledger")
    timing = ledger_timing(
        ledger,
        pushed_ns,
        start_ns=started_ns,
        construction_seconds=(construction_ended_ns - construction_started_ns) / 1e9,
        execution_seconds=(ended_ns - started_ns) / 1e9,
        construction_to_exit_seconds=(ended_ns - construction_started_ns) / 1e9,
    )
    case_dir = output_root / case_id
    if case_dir.exists():
        raise RuntimeError(f"duplicate case artifact directory: {case_id}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{case_id}.", dir=output_root))
    try:
        sink.save(temporary)
        _atomic_json(temporary / "push_timestamps.json", {str(index): value for index, value in sorted(pushed_ns.items())})
        _atomic_json(temporary / "timing.json", timing)
        os.replace(temporary, case_dir)
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        sink.close()
    row = {
        "id": case_id,
        "chunks_expected": len(plan),
        "frame_chunks": len(ledger.frames),
        "latent_chunks": len(ledger.latents),
        "ledger_dir": case_id,
        **timing,
    }
    del ledger, runtime, plan, frames, prompt_embeds, sink
    gc.collect()
    torch.cuda.empty_cache()
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--clip-root", type=Path, required=True)
    parser.add_argument("--embedding-root", type=Path, required=True)
    parser.add_argument("--status", type=Path)
    args = parser.parse_args()
    # Match the pinned qualification CPU setup in both production images.
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    raw = sys.stdin.buffer.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise SystemExit("manifest exceeds 2 MiB")
    payload = {"schema_version": 1, "ok": False, "cases": [], "error": None}
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1":
            raise RuntimeError("worker requires exactly logical CUDA devices 0,1")
        import torch
        from hardware import hardware_inventory

        # Capability envelope, never a GPU model name: two CUDA devices,
        # compute capability >= 8.0, >= 26 GiB each, >= 64 GiB usable host RAM.
        # Device names and UUIDs are recorded as measurement provenance only.
        payload["hardware"] = hardware_inventory(torch)
        manifest = json.loads(raw)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
            raise RuntimeError("manifest must contain a cases list")
        runtime_class = _load_runtime(args.candidate.resolve())
        for case_index, case in enumerate(manifest["cases"]):
            if args.status:
                _atomic_json(args.status, {"case_id": case.get("id"), "phase": "running"})
            payload["cases"].append(
                _execute_case(
                    runtime_class,
                    case,
                    args.artifacts,
                    args.clip_root,
                    args.embedding_root,
                    29540 + (case_index % 21),
                )
            )
            if args.status:
                _atomic_json(args.status, {"case_id": case.get("id"), "phase": "complete"})
        payload["ok"] = True
    except BaseException as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        payload["traceback"] = traceback.format_exc(limit=40)
    _atomic_json(args.output, payload)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
