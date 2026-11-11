"""Retained CPU padding/input/cache checks for the unchanged sparse core.

The public wrapper and dense guard are checked separately on GPU.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

TASK_ROOT = Path(__file__).resolve().parents[1]
ORACLE = TASK_ROOT / "oracle/candidates/kernel_v001/submission.py"
WORK = TASK_ROOT / "images/work/workdir/minference_prefill_lab/candidates/kernel_v001/submission.py"


class Kernel:
    def __init__(self, callback):
        self.callback = callback
        self.calls = []

    def __getitem__(self, grid):
        def invoke(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            return self.callback(*args, **kwargs)
        return invoke


def sparse_core():
    tree = ast.parse(ORACLE.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_sparse_prefill_core")
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)}
    calls = {}

    def estimate(q, k, scale):
        calls["estimate"] = (q, k, scale)
        vertical = torch.arange(min(q.shape[2], constants["_VERTICAL_BUDGET"]), dtype=torch.int32).expand(1, 32, -1).clone()
        slash = torch.arange(min(q.shape[2], constants["_SLASH_BUDGET"]), dtype=torch.int32).expand(1, 32, -1).clone()
        return vertical, slash

    def scratch(q, vertical, slash):
        calls["scratch"] = (q, vertical, slash)
        return ("count", "blocks", "column_count", "columns")

    def attention(q, k, v, counts, blocks, column_counts, columns, out, scale, *strides, **kwargs):
        out.copy_(q + 9)

    converter = Kernel(lambda *args, **kwargs: None)
    attention_kernel = Kernel(attention)
    namespace = {**constants, "torch": torch, "_estimate_last64_indices": estimate,
                 "_get_scratch": scratch, "_convert_vertical_slash_kernel": converter,
                 "_mixed_sparse_attention_kernel": attention_kernel}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ORACLE), "exec"), namespace)
    return namespace[function.name], calls, converter, attention_kernel


@pytest.mark.parametrize("n", [64, 65, 511, 1889, 2047, 2048, 2049, 4095, 4096, 4097, 8193])
def test_real_input_estimation_and_partial_tile_copy(n):
    entry, calls, converter, attention = sparse_core()
    q = torch.full((1, 32, n, 128), 1.0, dtype=torch.bfloat16)
    k = torch.full((1, 8, n, 128), 2.0, dtype=torch.bfloat16)
    v = torch.full_like(k, 3.0)
    out = torch.full_like(q, float("nan"))
    original = [tensor.clone() for tensor in (q, k, v)]
    assert entry(q, k, v, 0.125, out) is None
    estimated_q, estimated_k, scale = calls["estimate"]
    assert estimated_q is q and estimated_k is k and scale == 0.125
    assert q.shape == (1, 32, n, 128) and k.shape == v.shape == (1, 8, n, 128)
    for tensor, before in zip((q, k, v), original):
        assert torch.equal(tensor, before)
    assert out.shape == q.shape and bool(torch.all(out == 10))

    grid, args, kwargs = attention.calls[0]
    padded_n = ((n + 63) // 64) * 64
    assert grid == (padded_n // 64, 32)
    assert kwargs["N_CTX"] == n
    assert kwargs["NUM_ROWS"] == padded_n // 64
    assert calls["scratch"][0] is args[0]
    assert converter.calls[0][0] == (padded_n // 64 * 32,)
    constants = {node.targets[0].id: ast.literal_eval(node.value)
                 for node in ast.parse(ORACLE.read_text()).body
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
    nnz_v, nnz_s = min(n, constants["_VERTICAL_BUDGET"]), min(n, constants["_SLASH_BUDGET"])
    assert calls["scratch"][1:] == (nnz_v, nnz_s)
    for kernel in (converter, attention):
        assert kernel.calls[0][2]["NNZ_V"] == nnz_v
        assert kernel.calls[0][2]["NNZ_S"] == nnz_s
    for supplied, real in zip(args[:3], (q, k, v)):
        assert supplied.shape[2] == padded_n
        assert torch.equal(supplied[:, :, :n], real)
        if n % 64:
            assert supplied is not real
            assert bool(torch.all(supplied[:, :, n:] == 0))
        else:
            assert supplied is real
    assert (args[7] is out) == (n % 64 == 0)


def test_too_short_input_fails_before_estimation_or_kernel_launch():
    entry, calls, converter, attention = sparse_core()
    q = torch.empty((1, 32, 63, 128), dtype=torch.bfloat16)
    k = torch.empty((1, 8, 63, 128), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="at least 64"):
        entry(q, k, k, 0.125, torch.empty_like(q))
    assert not calls and not converter.calls and not attention.calls


def test_work_and_oracle_seed_have_same_bytes_and_are_admitted():
    assert ORACLE.read_bytes() == WORK.read_bytes()
    path = TASK_ROOT / "images/judge/source_admission.py"
    spec = importlib.util.spec_from_file_location("v3_seed_source_admission", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.validate_source_text(ORACLE.read_text(), str(ORACLE)) == []
    assert hashlib.sha256(ORACLE.read_bytes()).hexdigest() != "3d7ac66b41ed3ef10f7e8d266d50bdf1fa750d98cdf0c34143d05271f09dc32c"


def test_scratch_cache_reuses_one_shape_and_evicts_previous_shapes():
    tree = ast.parse(ORACLE.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_get_scratch")
    allocations = []

    def allocate(shape, **options):
        value = SimpleNamespace(shape=shape, **options)
        allocations.append(value)
        return value

    cache = {}
    namespace = {"_BLOCK_M":64,"_QUERY_HEADS":32,"_SCRATCH":cache,
                 "torch":SimpleNamespace(empty=allocate,int32="int32")}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(ORACLE),"exec"),namespace)
    scratch = namespace["_get_scratch"]
    first_q=SimpleNamespace(shape=(1,32,4096,128),device="cuda:0")
    first = scratch(first_q,512,2048)
    assert scratch(first_q,512,2048) is first and len(allocations)==4
    for n in (8192,131072,2048,4096):
        result=scratch(SimpleNamespace(shape=(1,32,n,128),device="cuda:0"),512,2048)
        assert len(cache)==1 and list(cache)[0][1]==(1,32,n,128)
        assert result[1].shape==(1,32,n//64,2048)
    assert len(allocations)==20
