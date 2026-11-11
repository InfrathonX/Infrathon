#!/usr/bin/env python3
"""Numerical and timing helpers of the public kernel tests."""
from __future__ import annotations
import hashlib
import importlib.util
import io
import math
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import secrets
import statistics
import sys
import time

ENTRYPOINT = "minference_sparse_prefill_bf16"
QUERY_HEADS, KV_HEADS, HEAD_DIM = 32, 8, 128
GQA_GROUP = QUERY_HEADS // KV_HEADS
SOFTMAX_SCALE = HEAD_DIM ** -0.5
PROBE_ROWS = 512
TAIL_ROWS = 64
REFERENCE_Q_CHUNK = 2048
REFERENCE_K_CHUNK = 8192
ENERGY_FLOOR = 1e-12
NRMSE_LIMIT = 0.25
HEAD_NRMSE_LIMIT = 0.50


class KernelFailure(RuntimeError):
    """The candidate broke the contract in a way no later call can repair."""


def load_candidate(path):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    name = f"minference_prefill_candidate_{digest[:24]}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            spec.loader.exec_module(module)
    except BaseException as exc:
        raise KernelFailure(f"candidate import failed: {exc}") from exc
    entrypoint = getattr(module, ENTRYPOINT, None)
    if not callable(entrypoint):
        raise KernelFailure(f"candidate must define callable {ENTRYPOINT}")
    return entrypoint, digest


def use_exact_fp32(torch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def _invoke(entrypoint, arguments):
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            returned = entrypoint(*arguments)
    except BaseException as exc:
        raise KernelFailure(f"candidate call failed: {exc}") from exc
    if returned is not None:
        raise KernelFailure(f"{ENTRYPOINT} must return None")


def checked_call(torch, tensors, entrypoint, stream, *, measured):
    """One call on the caller stream; returns the output and, if measured, the seconds."""
    q, k, v = (tensors[name] for name in ("q", "k", "v"))
    snapshots = tuple(t.clone() for t in (q, k, v))
    out = torch.full_like(q, float("nan"))
    captured = torch.empty_like(q)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    wall_start = time.perf_counter()
    with torch.cuda.stream(stream):
        if measured:
            start.record(stream)
        _invoke(entrypoint, (q, k, v, SOFTMAX_SCALE, out))
        captured.copy_(out)
        if measured:
            end.record(stream)
    torch.cuda.synchronize()
    seconds = None
    if measured:
        seconds = max(time.perf_counter() - wall_start, float(start.elapsed_time(end)) / 1000.0)
    if any(not bool(torch.equal(t.view(torch.uint8), s.view(torch.uint8))) for t, s in zip((q, k, v), snapshots)):
        raise KernelFailure("candidate changed an input tensor")
    if not bool(torch.isfinite(captured).all()):
        raise KernelFailure("candidate left a nonfinite or unwritten output element")
    return captured, seconds


def dense_reference(torch, q, k, v, rows):
    """FP32 dense causal attention for the query rows ``rows``, with an online softmax."""
    probes = int(rows.numel())
    k32, v32 = k[0].float(), v[0].float()
    device = q.device
    out = []
    for begin in range(0, probes, REFERENCE_Q_CHUNK):
        chunk = rows[begin:begin + REFERENCE_Q_CHUNK]
        count = int(chunk.numel())
        q32 = q[0].index_select(1, chunk).float().reshape(KV_HEADS, GQA_GROUP * count, HEAD_DIM) * SOFTMAX_SCALE
        row_pos = chunk.repeat(GQA_GROUP)[:, None]
        m = torch.full((KV_HEADS, GQA_GROUP * count, 1), -float("inf"), dtype=torch.float32, device=device)
        total = torch.zeros((KV_HEADS, GQA_GROUP * count, 1), dtype=torch.float32, device=device)
        acc = torch.zeros((KV_HEADS, GQA_GROUP * count, HEAD_DIM), dtype=torch.float32, device=device)
        key_limit = int(chunk.max().item()) + 1
        for k_start in range(0, key_limit, REFERENCE_K_CHUNK):
            k_end = min(key_limit, k_start + REFERENCE_K_CHUNK)
            scores = torch.matmul(q32, k32[:, k_start:k_end].transpose(-1, -2))
            col_pos = torch.arange(k_start, k_end, device=device)[None, :]
            scores = scores.masked_fill(col_pos > row_pos, -float("inf"))
            new_m = torch.maximum(m, scores.amax(dim=-1, keepdim=True))
            probs = torch.exp(scores - new_m)
            alpha = torch.exp(m - new_m)
            total = total * alpha + probs.sum(dim=-1, keepdim=True)
            acc = acc * alpha + torch.matmul(probs, v32[:, k_start:k_end])
            m = new_m
        out.append((acc / total).reshape(QUERY_HEADS, count, HEAD_DIM))
    return torch.cat(out, dim=1)[None]


def energy_stats(torch, actual, reference):
    a, r = actual.float(), reference.float()
    err = (a - r).square().sum(dim=(0, 2, 3))
    ref = r.square().sum(dim=(0, 2, 3))
    heads = (err / ref.clamp_min(ENERGY_FLOOR)).sqrt()
    return {"nrmse": float((err.sum() / ref.sum().clamp_min(ENERGY_FLOOR)).sqrt()),
            "head_max_nrmse": float(heads.max())}


def passes(stats):
    return stats["nrmse"] <= NRMSE_LIMIT and stats["head_max_nrmse"] <= HEAD_NRMSE_LIMIT


def probe_rows(torch, n, device):
    generator = torch.Generator(device=device).manual_seed(secrets.randbits(63))
    uniform = torch.randperm(n, generator=generator, device=device)[:min(n, PROBE_ROWS)]
    tail = torch.arange(max(0, n - TAIL_ROWS), n, device=device)
    return uniform, tail


def fidelity(torch, out, tensors, rows):
    """(uniform stats, tail stats) of ``out`` against the dense reference."""
    uniform, tail = rows
    return tuple(energy_stats(torch, out.index_select(2, part),
                              dense_reference(torch, tensors["q"], tensors["k"], tensors["v"], part))
                 for part in (uniform, tail))


def effective_tflops(n, seconds):
    return 2 * QUERY_HEADS * HEAD_DIM * n * (n + 1) / seconds / 1.0e12


def geometric_mean(values):
    return math.exp(statistics.fmean(math.log(value) for value in values))


def boundary_case(torch, entrypoint, stream, n):
    """One random-input call at length ``n``, compared on every output row."""
    def inputs():
        return {name: torch.randn((1, heads, n, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
                for name, heads in (("q", QUERY_HEADS), ("k", KV_HEADS), ("v", KV_HEADS))}
    discarded, _ = checked_call(torch, inputs(), entrypoint, stream, measured=False)
    del discarded
    tensors = inputs()
    out, _ = checked_call(torch, tensors, entrypoint, stream, measured=False)
    rows = torch.arange(n, device="cuda")
    tail = torch.arange(max(0, n - TAIL_ROWS), n, device="cuda")
    return fidelity(torch, out, tensors, (rows, tail))
