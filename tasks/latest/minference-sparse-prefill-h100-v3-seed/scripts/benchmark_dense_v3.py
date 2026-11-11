#!/usr/bin/env python3
"""Author-only paired seed/FA3 baseline on authenticated real v3 fixtures.

This diagnostic does not award a task score or replace terminal downstream QA.
Both calls include ABI output materialization and the trusted output snapshot.
Fixture I/O, FP32 references, input snapshots, and verification are untimed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time

TASK_ROOT = Path(__file__).resolve().parents[1]
FA3_COMMIT = "060c9188beec3a8b62b33a3bfa6d5d2d44975fab"
FA3_WHEEL_SHA256 = "39f00434791b257e59c171ce265220caf02ae98871432d89f0789d6d42800150"
FA3_INTERFACE_SHA256 = "69cfa291e4f21f1dc0c4281557fd2cb5fb349bfb68f12b2596d779ba473d6c4a"
FA3_EXTENSION_SHA256 = "bbfd19aaefb1f149b13db5f9cfd40ae4b1663b5ea0044788ac4ee5c34837701f"
SHA = re.compile(r"[0-9a-f]{64}\Z")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def load_runtime():
    """Import author helpers without importing Torch or initializing CUDA."""
    location = TASK_ROOT / "runtime"
    sys.path.insert(0, str(location))
    try:
        import activation_archive_v3
        spec = importlib.util.spec_from_file_location(
            "minference_dense_author_runtime_v3", location / "minference_prefill_runtime.py")
        minference_prefill_runtime = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = minference_prefill_runtime
        spec.loader.exec_module(minference_prefill_runtime)
    finally:
        sys.path.remove(str(location))
    return minference_prefill_runtime, activation_archive_v3


def fa3_entrypoint(function):
    """Adapt BHND views to the pinned BNHD native-GQA public FA3 function."""
    def invoke(q, k, v, softmax_scale, out):
        # Permute is a view: real N and the 8 KV heads remain unchanged.
        result = function(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
                          v.permute(0, 2, 1, 3), softmax_scale=softmax_scale,
                          causal=True, num_splits=1)
        out.copy_(result.permute(0, 2, 1, 3))
    return invoke


def geometric_mean(values):
    values = list(values)
    if not values or any(not math.isfinite(x) or x <= 0 for x in values):
        raise ValueError("geomean requires finite positive values")
    return math.exp(statistics.fmean(math.log(x) for x in values))


def summarize(cases):
    """Use each paired actual N before taking group medians and geomeans."""
    identities, ids = set(), set()
    for case in cases:
        identity = (case["task"], case["context_budget"], case["layer"])
        if identity in identities or case["case_id"] in ids or len(case["trials"]) != 3:
            raise ValueError("duplicate group or wrong trial count")
        identities.add(identity); ids.add(case["case_id"])
        for backend in ("seed", "fa3"):
            values = []
            for row in case["trials"]:
                n = row["actual_n"]
                t = row[backend]["seconds"]
                if type(n) is not int or not 2048 <= n < case["context_budget"]:
                    raise ValueError("invalid actual N")
                if isinstance(t, bool) or not math.isfinite(t) or t <= 0:
                    raise ValueError("invalid elapsed time")
                values.append(8192 * n * (n + 1) / t / 1e12)
            case[f"{backend}_median_tflops"] = statistics.median(values)
        case["seed_over_fa3_speedup"] = case["seed_median_tflops"] / case["fa3_median_tflops"]
    def aggregate(rows):
        return {"group_count": len(rows),
                "seed_geomean_tflops": geometric_mean(c["seed_median_tflops"] for c in rows),
                "fa3_geomean_tflops": geometric_mean(c["fa3_median_tflops"] for c in rows),
                "seed_over_fa3_speedup": geometric_mean(c["seed_over_fa3_speedup"] for c in rows)}
    return {"overall": aggregate(cases),
            "by_context_budget": {str(b): aggregate([c for c in cases if c["context_budget"] == b])
                                  for b in sorted({c["context_budget"] for c in cases})},
            "by_task": {t: aggregate([c for c in cases if c["task"] == t])
                        for t in sorted({c["task"] for c in cases})}}


def verify_fa3(torch, wheel):
    if torch.__version__ != "2.8.0+cu129" or importlib.metadata.version("triton") != "3.4.0":
        raise RuntimeError("baseline requires formal Torch 2.8.0+cu129 / Triton 3.4.0")
    if importlib.metadata.version("transformers") != "4.57.1":
        raise RuntimeError("baseline image must share the pinned HF 4.57.1 science environment")
    if sha256(wheel) != FA3_WHEEL_SHA256:
        raise RuntimeError("FA3 wheel differs from the retained CPU source-build receipt")
    import flash_attn_interface
    extension = importlib.util.find_spec("flash_attn_3._C")
    if extension is None or not extension.origin:
        raise RuntimeError("pinned FA3 extension unavailable")
    actual_interface = sha256(flash_attn_interface.__file__)
    actual_extension = sha256(extension.origin)
    if actual_interface != FA3_INTERFACE_SHA256 or actual_extension != FA3_EXTENSION_SHA256:
        raise RuntimeError("installed FA3 bytes differ from the pinned source-built wheel")
    return flash_attn_interface.flash_attn_func, {
        "repository": "https://github.com/Dao-AILab/flash-attention",
        "source_commit": FA3_COMMIT, "wheel_sha256": FA3_WHEEL_SHA256,
        "interface_sha256": actual_interface, "extension_sha256": actual_extension,
        "distribution_version": importlib.metadata.version("flash_attn_3"),
        "interface": "flash_attn_interface.flash_attn_func",
        "layout": "BNHD views, native 32:8 GQA, causal=True, num_splits=1",
        "input_layout_copy": False, "output_to_abi_copy_in_timing": True,
    }


def write_result(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--activation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, default=TASK_ROOT / "oracle/candidates/kernel_v001/submission.py")
    parser.add_argument("--fa3-wheel", type=Path)
    parser.add_argument("--image-id", help="Exact author-baseline Docker image sha256 ID")
    parser.add_argument("--work-image-id", help="Exact formal candidate Work image ID, for environment disclosure")
    parser.add_argument("--judge-image-id", help="Exact formal candidate Judge image ID, for environment disclosure")
    parser.add_argument("--probe-seed", type=int, default=20261004)
    parser.add_argument("--numerical-policy", type=Path, help="Optional policy; reports screening independently of speed")
    parser.add_argument("--validate-only", action="store_true", help="CPU metadata validation, without candidate import or CUDA")
    parser.add_argument("--verify-fixtures", action="store_true", help="Also authenticate all fixture bytes in CPU validation")
    args = parser.parse_args(argv)
    runtime, archive = load_runtime()
    manifest, manifest_sha = archive.read_json(args.manifest)
    archive.validate_manifest(manifest)
    if manifest["partition"] != "public":
        parser.error("this public baseline must not consume final or calibration fixtures")
    if not 0 <= args.probe_seed < 2**128:
        parser.error("probe seed must be in [0, 2**128)")
    base = {"schema_version": 3, "protocol": "minference-paired-dense-author-baseline-v3",
            "qualification": False, "manifest_sha256": manifest_sha,
            "script_sha256": sha256(__file__), "candidate_sha256": sha256(args.candidate),
            "runtime_sha256": sha256(runtime.__file__), "archive_reader_sha256": sha256(archive.__file__),
            "partition": manifest["partition"], "group_count": len(manifest["case_groups"]),
            "trial_count": 3 * len(manifest["case_groups"]), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "probe_protocol": "public reproducible diagnostic; separate from root private evaluation nonce",
            "probe_seed": args.probe_seed, "uniform_probe_rows": 512, "tail_probe_rows": 64,
            "timing_protocol": "trusted measure_call: max(host wall, CUDA events); whole ABI call + immediate output snapshot",
            "warmup_protocol": "same prompt and actual N, independent real layer 2; one warm-up per backend before each timed pair",
            "environment_disclosure": "FA3 extension exists only in author baseline image, outside candidate Work/Judge images",
            "author_image_id": args.image_id, "work_image_id": args.work_image_id,
            "judge_image_id": args.judge_image_id}
    if args.validate_only:
        if args.verify_fixtures:
            for fid, entry in manifest["fixtures"].items():
                handle = archive.open_handle(args.activation_root, fid, entry)
                handle.close()
        write_result(args.output, {**base, "status": "metadata-validated", "gpu_executed": False,
                                   "fixture_bytes_verified": args.verify_fixtures})
        return 0
    if args.fa3_wheel is None or not isinstance(args.image_id, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id):
        parser.error("GPU baseline requires --fa3-wheel and exact --image-id")
    import torch
    function, fa3_identity = verify_fa3(torch, args.fa3_wheel)
    inventory = runtime.hardware_inventory(torch)
    runtime._use_exact_fp32(torch)
    clock = runtime.TrustedClock.capture(torch)  # bind primitives before seed import
    _, seed, seed_sha = runtime.load_candidate(args.candidate, module_role="dense_baseline_seed")
    if seed_sha != base["candidate_sha256"]:
        raise RuntimeError("seed changed after baseline identity binding")
    backends = {"seed": seed, "fa3": fa3_entrypoint(function)}
    policy, policy_sha = (runtime.load_numerical_policy(args.numerical_policy, diagnostic=True)
                          if args.numerical_policy else (None, None))
    base.update(torch=torch.__version__, triton=importlib.metadata.version("triton"),
                transformers=importlib.metadata.version("transformers"),
                hardware=inventory, fa3=fa3_identity, numerical_policy_sha256=policy_sha)
    cases = []
    torch.cuda.reset_peak_memory_stats()
    try:
        for group_index, group in enumerate(manifest["case_groups"]):
            row = {key: group[key] for key in ("case_id", "task", "context_budget", "layer")}
            row["trials"] = []
            for trial_index, (wid, tid) in enumerate(zip(group["warmups"], group["trials"], strict=True)):
                handles = []
                try:
                    for fid in (wid, tid):
                        handles.append(archive.open_handle(args.activation_root, fid, manifest["fixtures"][fid]))
                    warm = archive.read_tensors(torch, handles[0], "cuda")
                    values = archive.read_tensors(torch, handles[1], "cuda")
                    entry = manifest["fixtures"][tid]
                    uniform, tail, probes = runtime._probe_rows(torch, entry["actual_n"], args.probe_seed, tid, values["q"].device)
                    reference = runtime.probe_dense_reference(torch, values["q"], values["k"], values["v"], runtime.SOFTMAX_SCALE, probes)
                    order = ("seed", "fa3") if (group_index + trial_index) % 2 == 0 else ("fa3", "seed")
                    # Both backends warm first; order alternates to avoid a fixed order effect.
                    for name in order:
                        discarded, _ = runtime._checked_call(torch, warm, backends[name], clock, measured=False)
                        del discarded
                    del warm
                    trial = {"fixture_id": tid, "fixture_sha256": entry["sha256"],
                             "input_token_sha256": entry["input_token_sha256"], "actual_n": entry["actual_n"],
                             "warmup_fixture_id": wid, "backend_order": list(order)}
                    split = int(uniform.numel())
                    for name in order:
                        out, measured = runtime._checked_call(torch, values, backends[name], clock, measured=True)
                        selected = out.index_select(2, probes)
                        statistics_uniform = runtime.row_error_stats(torch, selected[:, :, :split], reference[:, :, :split]).as_dict()
                        statistics_tail = runtime.row_error_stats(torch, selected[:, :, split:], reference[:, :, split:]).as_dict()
                        trial[name] = {"seconds": measured.seconds, "host_seconds": measured.host_seconds,
                                       "event_seconds": measured.event_seconds,
                                       "uniform_stats": statistics_uniform, "tail_stats": statistics_tail,
                                       "full_write_ok": True, "input_immutable_ok": True}
                        if policy is not None:
                            limits = runtime.case_limits(policy, group)
                            trial[name].update(limits=limits, numeric_passed=runtime.numeric_passes(statistics_uniform, statistics_tail, limits))
                        del selected, out
                    row["trials"].append(trial)
                    del values, reference, uniform, tail, probes
                finally:
                    for handle in handles:
                        handle.close()
            cases.append(row)
            write_result(args.output, {**base, "status": "running", "completed_groups": len(cases), "cases": cases})
            print(json.dumps({"case_id": row["case_id"], "completed_groups": len(cases)}), flush=True)
        summary = summarize(cases)
        write_result(args.output, {**base, "status": "completed", "gpu_executed": True,
                                   "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                                   "cases": cases, "summary": summary})
    except BaseException as error:
        write_result(args.output, {**base, "status": "failed", "completed_groups": len(cases),
                                   "error_type": type(error).__name__, "error": str(error), "cases": cases})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
