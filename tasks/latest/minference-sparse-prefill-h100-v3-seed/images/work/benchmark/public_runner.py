#!/usr/bin/env python3
"""Public kernel tests: real Q/K/V captured here, then checked and timed.

Suites:
  full        boundary lengths, then every public prompt and layer
  quick       boundary lengths, then the smallest prompts on one layer, one trial
  boundaries  boundary lengths only; does not load the model
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import tempfile
from source_admission import admit_workspace
from submission_loader import seal_approved_submission

BENCHMARK = Path("/workspace/benchmark")
LAB_ROOT = Path("/app/workdir/minference_prefill_lab")
PROMPTS = BENCHMARK / "public-prompts"
MODEL = BENCHMARK / "model"


def load_prompts(root):
    manifest = json.loads((root / "manifest.json").read_text())
    raw = (root / "prompts.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["prompts_sha256"]:
        raise ValueError("public prompts differ from their manifest")
    return [json.loads(line) for line in raw.splitlines()], manifest["public_suite"]


class _Captured(Exception):
    pass


def capture(torch, model, ids, layers):
    """Post-RoPE Q/K/V each selected layer's attention would consume."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    captured, handles = {}, []
    for layer in layers:
        def hook(module, args, kwargs, layer=layer):
            hidden = args[0] if args else kwargs["hidden_states"]
            shape = (*hidden.shape[:-1], -1, module.head_dim)
            q = module.q_proj(hidden).view(shape).transpose(1, 2)
            k = module.k_proj(hidden).view(shape).transpose(1, 2)
            v = module.v_proj(hidden).view(shape).transpose(1, 2)
            cos, sin = kwargs["position_embeddings"]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            captured[layer] = {"q": q.contiguous(), "k": k.contiguous(), "v": v.contiguous()}
            if layer == max(layers):
                raise _Captured
        handles.append(model.model.layers[layer].self_attn.register_forward_pre_hook(hook, with_kwargs=True))
    inputs = torch.tensor([ids], device="cuda", dtype=torch.long)
    try:
        with torch.inference_mode():
            model(input_ids=inputs, attention_mask=torch.ones_like(inputs), use_cache=False, logits_to_keep=1)
    except _Captured:
        pass
    finally:
        for handle in handles:
            handle.remove()
    return captured


def run_boundaries(torch, kernel, entrypoint, stream, lengths):
    rows = []
    for n in lengths:
        uniform, tail = kernel.boundary_case(torch, entrypoint, stream, n)
        ok = kernel.passes(uniform) and kernel.passes(tail)
        rows.append({"n": n, "ok": ok, "uniform": uniform, "tail": tail})
        print(f"BOUNDARY n={n:<6d} {'PASS' if ok else 'FAIL'}  "
              f"nrmse={max(uniform['nrmse'], tail['nrmse']):.4f}  "
              f"worst_head={max(uniform['head_max_nrmse'], tail['head_max_nrmse']):.4f}", flush=True)
    return rows


def run_cases(torch, kernel, entrypoint, stream, prompts, suite, model_path, *, quick):
    import transformers
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        device_map={"": 0}, low_cpu_mem_usage=True, local_files_only=True).eval()
    timed, warmup, trials = suite["timed_layers"], suite["warmup_layer"], suite["trials"]
    if quick:
        smallest = min(p["length"] for p in prompts)
        prompts, timed, trials = [p for p in prompts if p["length"] == smallest], timed[:1], 1
    print(f"{'CASE':<36s}{'N':>8s}{'LAYER':>7s}{'MEDIAN_MS':>11s}{'TFLOPS':>10s}{'NRMSE':>9s}{'WORST_HEAD':>12s}  RESULT", flush=True)
    rows = []
    for prompt in prompts:
        n = prompt["actual_input_length"]
        captured = capture(torch, model, prompt["input_ids"], sorted({warmup, *timed}))
        for layer in timed:
            discarded, _ = kernel.checked_call(torch, captured[warmup], entrypoint, stream, measured=False)
            del discarded
            tensors, seconds, stats = captured[layer], [], []
            for _ in range(trials):
                out, took = kernel.checked_call(torch, tensors, entrypoint, stream, measured=True)
                seconds.append(took)
                stats.extend(kernel.fidelity(torch, out, tensors, kernel.probe_rows(torch, n, out.device)))
                del out
            ok = all(kernel.passes(s) for s in stats)
            median = statistics.median(seconds)
            row = {"case": f"{prompt['id']}-L{layer}", "n": n, "layer": layer, "ok": ok, "median_seconds": median,
                   "tflops": kernel.effective_tflops(n, median),
                   "nrmse": max(s["nrmse"] for s in stats),
                   "worst_head_nrmse": max(s["head_max_nrmse"] for s in stats)}
            rows.append(row)
            print(f"{row['case']:<36s}{n:>8d}{layer:>7d}{median * 1e3:>11.2f}{row['tflops']:>10.1f}"
                  f"{row['nrmse']:>9.4f}{row['worst_head_nrmse']:>12.4f}  {'PASS' if ok else 'FAIL'}", flush=True)
        del captured
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-version", type=int, required=True)
    parser.add_argument("--suite", choices=("full", "quick", "boundaries"), default="full")
    parser.add_argument("--lab-root", type=Path, default=LAB_ROOT)
    parser.add_argument("--prompts", type=Path, default=PROMPTS)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.candidate_version <= 999:
        parser.error("candidate version must be 1-999")
    payload = {"suite": args.suite, "candidate_version": args.candidate_version, "hidden_cases_run": False}
    try:
        prompts, suite = load_prompts(args.prompts)
        approved = admit_workspace(args.lab_root / f"candidates/kernel_v{args.candidate_version:03d}")
        with tempfile.TemporaryDirectory(prefix="minference-public-") as directory:
            root = Path(directory)
            os.chmod(root, 0o700)
            sealed = seal_approved_submission(approved, root / "admitted")
            import torch
            import public_kernel as kernel
            kernel.use_exact_fp32(torch)
            stream = torch.cuda.Stream()
            entrypoint, digest = kernel.load_candidate(sealed.path)
            payload["candidate_sha256"] = digest
            boundaries = run_boundaries(torch, kernel, entrypoint, stream, suite["boundary_lengths"])
            payload["boundaries"] = boundaries
            ok = all(row["ok"] for row in boundaries)
            if ok and args.suite != "boundaries":
                cases = run_cases(torch, kernel, entrypoint, stream, prompts, suite, args.model, quick=args.suite == "quick")
                payload["cases"] = cases
                ok = all(row["ok"] for row in cases)
                payload["public_tflops"] = kernel.geometric_mean(row["tflops"] for row in cases)
                print(f"\nPUBLIC_TFLOPS  {payload['public_tflops']:.2f}")
        payload["ok"] = ok
        print("PUBLIC_RESULT ", "PASS" if ok else "FAIL")
    except Exception as error:
        payload.update(ok=False, error=f"{type(error).__name__}: {error}"[:1000])
        print(f"PUBLIC_RESULT  FAIL  {payload['error']}", file=sys.stderr)
    print("HIDDEN_CASES   NOT_RUN")
    if args.json_output:
        args.json_output.write_text(json.dumps(payload, allow_nan=False) + "\n")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
