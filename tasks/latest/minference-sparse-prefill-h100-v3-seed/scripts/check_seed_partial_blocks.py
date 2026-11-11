#!/usr/bin/env python3
"""Check the public seed ABI against FP32 attention on its exact sparse mask."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

TASK_ROOT = Path(__file__).resolve().parents[1]


def load_seed(path):
    spec = importlib.util.spec_from_file_location("variable_n_seed_check", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", default="2048,2049,4095,4096,4097,8193")
    args = parser.parse_args(argv)
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    seed_path = TASK_ROOT / "oracle/candidates/kernel_v001/submission.py"
    seed = load_seed(seed_path)
    cases = []
    lengths = [int(item) for item in args.lengths.split(",")]
    if not lengths or len(set(lengths)) != len(lengths) or min(lengths) < 2048:
        parser.error("lengths must be distinct integers >= 2048")
    for n in lengths:
        generator = torch.Generator(device="cuda").manual_seed(n)
        q = torch.randn((1, 32, n, 128), generator=generator, dtype=torch.bfloat16, device="cuda")
        k = torch.randn((1, 8, n, 128), generator=generator, dtype=torch.bfloat16, device="cuda")
        v = torch.randn((1, 8, n, 128), generator=generator, dtype=torch.bfloat16, device="cuda")
        out = torch.full_like(q, float("nan"))
        input_snapshots = [tensor.clone() for tensor in (q, k, v)]
        seed.minference_sparse_prefill_bf16(q, k, v, 128 ** -0.5, out)
        if not bool(torch.isfinite(out).all()):
            raise RuntimeError("public seed left nonfinite/unwritten values")
        if any(not torch.equal(tensor, snapshot) for tensor, snapshot in zip((q, k, v), input_snapshots)):
            raise RuntimeError("public seed mutated an input")
        vertical, slash = seed._estimate_last64_indices(q, k, 128 ** -0.5)
        vertical = torch.sort(vertical, dim=-1)[0]
        slash = torch.sort(slash, dim=-1, descending=True)[0]
        padded_n = (n + 63) // 64 * 64
        kernel_geometry = torch.empty((1, 32, padded_n, 128), dtype=q.dtype, device=q.device)
        counts, blocks, column_counts, columns = seed._get_scratch(
            kernel_geometry, seed._VERTICAL_BUDGET, seed._SLASH_BUDGET
        )
        rows = padded_n // 64
        seed._convert_vertical_slash_kernel[(rows * 32,)](
            vertical, slash, counts, blocks, column_counts, columns,
            NUM_ROWS=rows, NNZ_V=seed._VERTICAL_BUDGET, NNZ_S=seed._SLASH_BUDGET,
            BLOCK=64, num_warps=1, num_stages=1,
        )
        counts_cpu, blocks_cpu = counts.cpu(), blocks.cpu()
        column_counts_cpu, columns_cpu = column_counts.cpu(), columns.cpu()
        probes = sorted({0, 63, 64, n - 65, n - 64, n - 2, n - 1})
        max_error = 0.0
        for h in range(32):
            for row in probes:
                tile = row // 64
                mask = torch.zeros(n, dtype=torch.bool, device="cuda")
                for offset in blocks_cpu[0, h, tile, :int(counts_cpu[0, h, tile])].tolist():
                    end = min(n, row + 1, offset + 64)
                    if end > offset:
                        mask[offset:end] = True
                cols = columns_cpu[0, h, tile, :int(column_counts_cpu[0, h, tile])].tolist()
                if cols:
                    mask[cols] = True
                mask[row + 1:] = False
                if n == seed._SLASH_BUDGET and not bool(mask[:row + 1].all()):
                    raise RuntimeError("full-budget boundary did not cover dense causal attention")
                logits = (q[0, h, row].float() @ k[0, h // 4].float().T) * (128 ** -0.5)
                logits.masked_fill_(~mask, -float("inf"))
                reference = logits.softmax(-1) @ v[0, h // 4].float()
                error = float((out[0, h, row].float() - reference).abs().max())
                max_error = max(max_error, error)
                torch.testing.assert_close(out[0, h, row].float(), reference, atol=0.015, rtol=0.015)
        cases.append({"n": n, "probe_rows": probes, "max_abs_error": max_error,
                      "reference": "FP32 attention over the exact converted sparse mask",
                      "public_abi_called": True, "input_shapes_unchanged": True,
                      "inputs_unchanged": True, "all_output_values_finite": True,
                      "atol": 0.015, "rtol": 0.015, "passed": True})
        seed._SCRATCH.clear()
        print(cases[-1], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "protocol": "minference-seed-variable-n-kernel-check-v1",
        "status": "passed", "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "seed_sha256": hashlib.sha256(seed_path.read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cases": cases,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
