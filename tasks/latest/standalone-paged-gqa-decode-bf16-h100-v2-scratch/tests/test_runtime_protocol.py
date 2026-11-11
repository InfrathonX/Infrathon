"""CPU checks of the v2 runtime's protocol pieces (no GPU required)."""

from __future__ import annotations

import math
import time

import pytest
import torch

from conftest import load_module


runtime = load_module("pgqa_runtime_protocol_runtime", "runtime/paged_gqa_decode_runtime.py")


def edge(**overrides):
    base = {
        "case_id": "x", "category": "c", "s_q": 3, "b": 5, "l_max": 339, "h_q": 6, "h_kv": 3,
        "page_size": 4, "lengths": [3, 4, 100, 338, 339], "table_mode": "perm",
        "pool_pages": 3 * 5 * 85, "table_width": 3 * 85, "fill": "nan", "values": "randn",
        "seed": (1 << 63) - 1, "sanitize": False, "native": True,
    }
    base.update(overrides)
    return base


def test_edge_domain_validation() -> None:
    runtime.validate_edge_case(edge())
    runtime.validate_edge_case(edge(h_q=7, h_kv=1, s_q=1, lengths=[1] * 5, l_max=1, table_width=1,
                                    pool_pages=5, page_size=1))
    runtime.validate_edge_case(edge(table_mode="random", pool_pages=1))
    for field, value in (
        ("s_q", 0), ("s_q", 17), ("b", 0), ("b", 257), ("h_q", 129), ("h_kv", 4),
        ("page_size", 3), ("page_size", 512), ("l_max", 32769), ("lengths", [3, 4, 100, 338]),
        ("lengths", [2, 4, 100, 338, 339]), ("lengths", [3, 4, 100, 338, 340]),
        ("table_width", 84), ("table_width", 4 * 85 + 1), ("pool_pages", 5 * 255 - 1),
        ("pool_pages", (1 << 18) + 1), ("table_mode", "dense"), ("fill", "zero"),
        ("values", "fp8"), ("seed", 1 << 63), ("sanitize", 1), ("native", False),
    ):
        with pytest.raises(ValueError):
            runtime.validate_edge_case(edge(**{field: value}))
    with pytest.raises(ValueError):
        runtime.validate_edge_case({**edge(), "extra": 1})
    # Sanitizer-only cases may hold a large pool; native ones may not.
    big = edge(h_q=8, h_kv=8, page_size=16, l_max=1000, lengths=[1000] * 5, table_width=63,
               pool_pages=135_000, table_mode="tail_perm", s_q=1)
    with pytest.raises(ValueError):
        runtime.validate_edge_case(big)
    runtime.validate_edge_case(dict(big, sanitize=True, native=False))


def _call(**overrides):
    case = {"case_id": "s", "family": "decode", "s_q": 1, "h_q": 32, "h_kv": 8, "page_size": 16,
            "block_tokens": 512, "lengths": [1200, 700, 40], "blocks": [[3], [3], []], "table_width": 80,
            "pool_pages": 0, "fill": "randn", "values": "randn", "seed": 1 << 40}
    case["pool_pages"] = runtime.call_page_demand(case["lengths"], case["blocks"], 16, 512)
    case.update(overrides)
    return case


def test_scored_cases_are_serving_calls_and_sequences() -> None:
    case = runtime.validate_scored_case(_call())
    # Block 3 is shared: 32 pages once, plus every request's private tail.
    assert runtime.call_page_demand([1200, 700, 40], [[3], [3], []], 16, 512) == 32 + 43 + 12 + 3
    # At 16-token blocks the same requests share 4 blocks of one page each.
    assert runtime.call_page_demand([1200, 700, 40], [[1, 2, 3, 4], [1, 2, 3, 4], []], 16, 16) == 4 + 71 + 40 + 3
    assert case["b"] == 3 and case["l_max"] == 1200
    assert case["total_pairs"] == runtime.pairs_for_lengths(1, [1200, 700, 40])
    spec = runtime.validate_scored_case(_call(family="spec", s_q=5))
    assert spec["total_pairs"] == sum(5 * length - 10 for length in (1200, 700, 40))
    for bad in ({"family": "q1"}, {"family": "spec"}, {"s_q": 2}, {"extra": 8}, {"cell": "M1"},
                {"lengths": [(1 << 20) + 1, 700, 40]}, {"blocks": [[3, 4, 5], [3], []]},
                {"pool_pages": 1}, {"table_width": 10}, {"fill": "inf"}, {"page_size": 3},
                {"block_tokens": 8}, {"block_tokens": 1024}, {"values": "large_logits"}):
        with pytest.raises(ValueError):
            runtime.validate_scored_case(_call(**bad))
    # Derived fields are recomputed, never trusted: validation is idempotent.
    tampered = runtime.validate_scored_case({**_call(), "b": 8, "total_pairs": 1})
    assert tampered["b"] == 3 and tampered["total_pairs"] == case["total_pairs"]
    assert runtime.validate_scored_case(tampered) == tampered

    stream = {"case_id": "q", "family": "stream", "s_q": 1, "h_q": 16, "h_kv": 2, "page_size": 32,
              "block_tokens": 512, "values": "randn", "steps": 4, "requests": [[0, 4, 600, [9]], [1, 3, 50, []], [2, 4, 530, [9]]],
              "table_width": 20, "pool_pages": 0, "seed": 1 << 41}
    plan = runtime.stream_plan(stream)
    assert [len(now) for now in plan.active] == [1, 2, 3, 2]
    assert plan.total_pairs == sum(600 + step for step in range(4)) + 50 + 51 + 530 + 531
    stream["pool_pages"] = plan.peak_pages
    checked = runtime.validate_scored_case(stream)
    assert checked["b"] == 3 and checked["total_pairs"] == plan.total_pairs
    with pytest.raises(ValueError):
        runtime.validate_scored_case({**stream, "pool_pages": plan.peak_pages - 1})
    with pytest.raises(ValueError):
        runtime.validate_scored_case({**stream, "requests": [[1, 4, 600, []]]})


def test_per_call_seeds_are_63_bit_and_role_separated() -> None:
    seeds = {runtime.derive_seed(7, role, trial) for role in ("decoy", "measured", "warmup") for trial in range(5)}
    assert len(seeds) == 15
    assert all(0 <= seed <= runtime.MAX_SEED for seed in seeds)
    assert runtime.derive_seed(7, "decoy", 0) == runtime.derive_seed(7, "decoy", 0)


def test_effective_tflops_counts_causal_pairs() -> None:
    assert runtime.pairs_for_lengths(1, [1, 5]) == 6
    assert runtime.pairs_for_lengths(16, [16, 17]) == 16 * 16 - 120 + 16 * 17 - 120
    assert math.isclose(runtime.effective_tflops(1000, 28, 1e-3), 4 * 128 * 28 * 1000 / 1e-3 / 1e12)
    with pytest.raises(RuntimeError):
        runtime.pairs_for_lengths(16, [15])


@pytest.mark.parametrize("mode", runtime.TABLE_MODES)
def test_table_layouts(mode: str) -> None:
    spec = runtime.validate_edge_case(edge(table_mode=mode))
    generator = torch.Generator()
    generator.manual_seed(3)
    table = runtime._build_table(spec, generator)
    assert table.shape == (5, 255) and table.dtype == torch.int32
    assert int(table.min()) >= 0 and int(table.max()) < spec["pool_pages"]
    flat = table.flatten().tolist()
    if mode in ("perm", "identity", "reversed", "tail_perm"):
        assert len(set(flat)) == len(flat)
    if mode == "identity":
        assert flat == list(range(len(flat)))
    if mode == "tail_perm":
        assert min(flat) == spec["pool_pages"] - len(flat)
    if mode == "shared":
        # Request 1 reuses the full pages of the shared prefix with request 0.
        shared = min(spec["lengths"][0], spec["lengths"][1]) // 2 // spec["page_size"]
        assert table[1, :shared].tolist() == table[0, :shared].tolist()
        shared = min(spec["lengths"][2], spec["lengths"][3]) // 2 // spec["page_size"]
        assert shared > 0 and table[3, :shared].tolist() == table[2, :shared].tolist()


def test_visible_slots_cover_exactly_the_attended_positions() -> None:
    table = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32)
    visible = runtime._visible_slots(table, [3, 5], pool=4, page_size=4)
    expected = torch.zeros((4, 4), dtype=torch.bool)
    expected[2, :3] = True
    expected[1, :4] = True
    expected[3, :1] = True
    assert torch.equal(visible, expected)


def cpu_invocation(spec: dict, seed: int = 5) -> object:
    """Build an Invocation on the CPU the way prepare_invocation does on CUDA."""
    spec = runtime.validate_edge_case(spec)
    generator = torch.Generator()
    generator.manual_seed(seed)
    table = runtime._build_table(spec, generator)
    pool, page_size, h_kv = spec["pool_pages"], spec["page_size"], spec["h_kv"]
    q = torch.randn((spec["b"], spec["s_q"], spec["h_q"], 128), generator=generator).to(torch.bfloat16)
    k = torch.randn((pool, page_size, h_kv, 128), generator=generator).to(torch.bfloat16)
    v = torch.randn((pool, page_size, h_kv, 128), generator=generator).to(torch.bfloat16)
    if spec["fill"] != "randn":
        hidden = ~runtime._visible_slots(table, spec["lengths"], pool, page_size)
        k[hidden] = float("nan")
        v[hidden] = float("nan")
    out = torch.full_like(q, float("nan"))
    return runtime.Invocation(
        case_id="cpu", s_q=spec["s_q"], b=spec["b"], l_max=spec["l_max"], h_q=spec["h_q"],
        h_kv=h_kv, page_size=page_size, lengths=list(spec["lengths"]),
        arena=runtime.GuardedArena(False), q=q, k_cache=k, v_cache=v, page_table=table,
        cache_seqlens=torch.tensor(spec["lengths"], dtype=torch.int32), out=out,
        total_pairs=runtime.pairs_for_lengths(spec["s_q"], spec["lengths"]),
    )


def naive_attention(prepared) -> torch.Tensor:
    rho = prepared.h_q // prepared.h_kv
    result = torch.empty((prepared.b, prepared.s_q, prepared.h_q, 128), dtype=torch.float64)
    for row in range(prepared.b):
        length = prepared.lengths[row]
        for i in range(prepared.s_q):
            visible = length - prepared.s_q + i + 1
            positions = torch.arange(visible)
            pages = prepared.page_table[row, positions // prepared.page_size].long()
            slots = positions % prepared.page_size
            for h in range(prepared.h_q):
                g = h // rho
                keys = prepared.k_cache[pages, slots, g].double()
                values = prepared.v_cache[pages, slots, g].double()
                logits = keys @ prepared.q[row, i, h].double() * runtime.SOFTMAX_SCALE
                result[row, i, h] = torch.softmax(logits, 0) @ values
    return result


@pytest.mark.parametrize("mode", ["shared", "random", "reversed"])
def test_reference_matches_naive_attention_and_ignores_hidden_nan(mode: str) -> None:
    spec = edge(table_mode=mode, b=4, lengths=[3, 9, 33, 40], l_max=40, table_width=10,
                pool_pages=4 * 10 if mode != "random" else 7)
    prepared = cpu_invocation(spec)
    reference = runtime.reference_outputs(prepared)
    assert bool(torch.isfinite(reference.golden).all())
    assert torch.allclose(reference.golden.double(), naive_attention(prepared), atol=1e-5)
    assert reference.max_abs_limit > 0.0 and reference.mean_abs_limit >= 0.0
    prepared.out.copy_(reference.golden.to(torch.bfloat16))
    runtime.check_output(prepared, reference, runtime.EdgeCorrectnessFailure)


def test_output_gate_rejects_nan_wrong_heads_and_accepts_exact_zero_limit() -> None:
    # A single visible key per row: BF16 and FP32 references agree exactly, the
    # mean limit is 0 and only the exact answer passes (upstream `<=` rule).
    spec = edge(s_q=1, lengths=[1] * 5, l_max=1, table_width=1, pool_pages=5, page_size=4, fill="randn")
    prepared = cpu_invocation(spec)
    reference = runtime.reference_outputs(prepared)
    assert reference.mean_abs_limit == 0.0
    prepared.out.copy_(reference.golden.to(torch.bfloat16))
    runtime.check_output(prepared, reference, runtime.EdgeCorrectnessFailure)
    prepared.out[0, 0, 0, 0] += 0.25
    with pytest.raises(runtime.EdgeCorrectnessFailure):
        runtime.check_output(prepared, reference, runtime.EdgeCorrectnessFailure)
    prepared.out.copy_(reference.golden.to(torch.bfloat16))
    prepared.out[4, 0, 5, 7] = float("nan")
    with pytest.raises(runtime.EdgeCorrectnessFailure, match="unwritten"):
        runtime.check_output(prepared, reference, runtime.EdgeCorrectnessFailure)


def test_wrong_kv_head_mapping_is_rejected() -> None:
    prepared = cpu_invocation(edge(fill="randn"))
    reference = runtime.reference_outputs(prepared)
    shuffled = reference.golden.clone()
    shuffled[:, :, [0, 1]] = shuffled[:, :, [2, 3]]
    prepared.out.copy_(shuffled.to(torch.bfloat16))
    with pytest.raises(runtime.EdgeCorrectnessFailure):
        runtime.check_output(prepared, reference, runtime.EdgeCorrectnessFailure)


def test_input_mutation_is_a_memory_safety_failure() -> None:
    prepared = cpu_invocation(edge(fill="randn"))
    prepared.pristine = {name: tensor.clone() for name, tensor in prepared.inputs.items()}
    runtime.check_memory(prepared, "call")
    prepared.page_table[0, 0] += 1
    with pytest.raises(runtime.MemorySafetyFailure, match="page_table"):
        runtime.check_memory(prepared, "call")


def test_exception_classification() -> None:
    assert runtime.classify_exception(RuntimeError("CUDA error: an illegal memory access was encountered")) == "memory_safety_ok"
    assert runtime.classify_exception(RuntimeError("Triton Error [CUDA]: operation not supported on global/shared address space")) == "memory_safety_ok"
    assert runtime.classify_exception(ValueError("arange's range must be a power of 2")) == "correctness_ok"
    assert runtime.classify_exception(runtime.MemorySafetyFailure("x")) == "memory_safety_ok"
    assert runtime.is_cuda_fault(RuntimeError("unspecified launch failure"))
    assert not runtime.is_cuda_fault(runtime.MemorySafetyFailure("guard"))


def test_sentinel_detects_replaced_clock_tensor_method_and_flags(monkeypatch) -> None:
    sentinel = runtime.HarnessSentinel()
    sentinel.check("baseline")
    monkeypatch.setattr(time, "perf_counter", lambda: 0.0)
    with pytest.raises(runtime.HarnessIntegrityFailure, match="perf_counter"):
        sentinel.check("after patch")
    monkeypatch.undo()
    monkeypatch.setattr(torch.Tensor, "masked_fill", lambda self, *args: self)
    with pytest.raises(runtime.HarnessIntegrityFailure, match="masked_fill"):
        sentinel.check("after patch")
    monkeypatch.undo()
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", not torch.backends.cuda.matmul.allow_tf32)
    with pytest.raises(runtime.HarnessIntegrityFailure, match="allow_tf32"):
        sentinel.check("after patch")
    monkeypatch.undo()
    sentinel.check("restored")


def test_trusted_references_were_captured_at_import() -> None:
    assert runtime._perf_counter is time.perf_counter
    assert runtime._einsum is torch.einsum and runtime._softmax is torch.softmax


def test_plausibility_bound_table() -> None:
    assert runtime.plausibility_bound([9, 0]) == 1_200.0
    assert runtime.plausibility_bound([8, 0]) == 400.0
    assert runtime.plausibility_bound([7, 5]) is None
    assert runtime.plausibility_bound(None) is None


def test_aggregation_is_a_four_family_geometric_mean() -> None:
    rows = []
    for index, family in enumerate(runtime.FAMILIES):
        for ordinal in range(3):
            rows.append({"scored": True, "family": family, "tflops": (index + 1) * 2.0 + ordinal})
    result = runtime.aggregate_case_results(rows)
    families = {family: math.exp(sum(math.log((index + 1) * 2.0 + o) for o in range(3)) / 3)
                for index, family in enumerate(runtime.FAMILIES)}
    for family, value in families.items():
        assert math.isclose(result["family_tflops"][family], value)
    assert math.isclose(result["primary_tflops"], math.exp(sum(map(math.log, families.values())) / 4))
    with pytest.raises(ValueError):
        runtime.aggregate_case_results([{"scored": True, "family": "q1", "tflops": 1.0}])


def test_soft_landing_fill_values_are_valid_metadata() -> None:
    assert runtime.SOFT_PAGE_ID == 0
    assert runtime.GUARD_BYTES % 4 == 0
    # 0xA5A5 as BF16 is a tiny finite negative number, harmless when read.
    value = torch.tensor([0xA5, 0xA5], dtype=torch.uint8).view(torch.bfloat16).float().item()
    assert math.isfinite(value) and abs(value) < 1e-15
