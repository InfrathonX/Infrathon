"""The trusted runtime's generator, reference, comparator and memory checks, on CPU.

These tests run the runtime's own code paths with ``DEVICE = "cpu"`` and a
reference-backed fake candidate, so they need no GPU; the GPU-only parts
(timing, CUDA faults, compute-sanitizer) are covered by the Judge-side tests
with fakes and by qualification on the GPU.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from conftest import TASK_ROOT, load_module


runtime = load_module("sparse_mla_v2_runtime_protocol", "runtime/sparse_mla_prefill_runtime.py")
PUBLIC = json.loads((TASK_ROOT / "images/work/benchmark/public_edge_cases.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def cpu_runtime(monkeypatch):
    monkeypatch.setattr(runtime, "DEVICE", "cpu")
    monkeypatch.setattr(runtime, "_cuda_synchronize", lambda: None)
    yield


def reference_candidate(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse):
    o, m, l = runtime.reference_outputs(q, kv, indices, sm_scale, attn_sink, topk_length)
    out.copy_(o.to(out.dtype))
    max_logits.copy_(m)
    lse.copy_(l)


def small_case(**overrides) -> dict:
    case = {
        "case_id": "unit-1", "category": "unit", "family": "v4_flash", "s_q": 9, "s_kv": 95,
        "topk": 128, "sm_scale": 0.5, "bias": "upstream", "index_mode": "upstream",
        "invalid_rows": "none", "duplicates": False, "scramble": False, "sink_mode": "none",
        "length_mode": "none", "seed": 12345, "sanitize": False, "native": True,
    }
    case.update(overrides)
    return runtime.validate_edge_case(case)


def prepare(case: dict, seed: int = 7, **kwargs):
    return runtime.prepare_invocation(case, seed, **kwargs)


def test_generated_inputs_follow_the_requested_semantics() -> None:
    case = small_case(index_mode="upstream", s_kv=95, topk=128)
    inv = prepare(case)
    idx = inv.indices.view(case["s_q"], case["topk"])
    valid = (idx >= 0) & (idx < case["s_kv"])
    # s_kv < topk: each row holds every key once and pads the rest with invalid values.
    assert bool((valid.sum(dim=1) == 95).all())
    for row in idx:
        values = row[(row >= 0) & (row < 95)].tolist()
        assert sorted(values) == list(range(95))
    padded = prepare(small_case(index_mode="valid_pad"))
    assert bool(((padded.indices >= 0) & (padded.indices < 95)).all())
    sprinkle = prepare(small_case(index_mode="sprinkle", s_kv=4096, s_q=64))
    invalid = (sprinkle.indices < 0) | (sprinkle.indices >= 4096)
    assert 0.05 < float(invalid.float().mean()) < 0.25
    rows = prepare(small_case(invalid_rows="some", s_q=64, s_kv=4096))
    empty = ((rows.indices < 0) | (rows.indices >= 4096)).view(64, -1).all(dim=1)
    assert bool(empty[0]) and int(empty.sum()) >= 2
    full = prepare(small_case(invalid_rows="all", s_kv=4096))
    assert bool(((full.indices < 0) | (full.indices >= 4096)).all())


def test_sink_and_length_modes() -> None:
    extreme = prepare(small_case(sink_mode="extreme"))
    sink = extreme.attn_sink
    assert sink[0] == float("-inf") and sink[1] == float("inf") and sink[2] == 100.0 and sink[3] == -100.0
    upstream = prepare(small_case(sink_mode="upstream", family="v4_pro"))
    infinite = torch.isinf(upstream.attn_sink).float().mean().item()
    assert 0.3 < infinite < 0.9
    assert prepare(small_case(sink_mode="none")).attn_sink is None
    edge = prepare(small_case(length_mode="edge", s_q=9, topk=256, s_kv=1000))
    assert edge.topk_length[:7].tolist() == [0, 1, 63, 64, 65, 255, 256]
    assert int(edge.topk_length.min()) >= 0 and int(edge.topk_length.max()) <= 256
    half = prepare(small_case(length_mode="half", topk=256, s_kv=1000))
    assert int(half.topk_length.min()) >= 128
    assert half.positions == int(half.topk_length.sum())
    assert prepare(small_case()).positions == 9 * 128


def test_zero_bias_and_upstream_bias() -> None:
    zero = prepare(small_case(bias="zero", s_q=256, s_kv=4096))
    assert abs(float(zero.q.float().mean())) < 2e-3
    assert float(zero.q.float().abs().max()) <= 10.0


def test_generation_is_seeded_and_distinct_per_seed() -> None:
    case = small_case(index_mode="sprinkle", duplicates=True, scramble=True, sink_mode="extreme",
                      length_mode="edge", invalid_rows="some", s_kv=4096)
    a, b, c = prepare(case, 1), prepare(case, 1), prepare(case, 2)
    for name in ("q", "kv", "indices", "attn_sink", "topk_length"):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    assert not torch.equal(a.q, c.q) and not torch.equal(a.indices, c.indices)


def test_reference_matches_a_naive_per_row_softmax() -> None:
    case = small_case(index_mode="sprinkle", duplicates=True, sink_mode="extreme",
                      length_mode="edge", invalid_rows="some", s_kv=300, s_q=11, family="v32")
    inv = prepare(case)
    out, max_logits, lse = runtime.reference_outputs(
        inv.q, inv.kv, inv.indices, inv.sm_scale, inv.attn_sink, inv.topk_length
    )
    q = inv.q.double()
    kv = inv.kv.view(inv.s_kv, inv.d_qk).double()
    for i in range(inv.s_q):
        length = int(inv.topk_length[i])
        slots = [j for j in range(inv.topk)
                 if j < length and 0 <= int(inv.indices[i, 0, j]) < inv.s_kv]
        for h in (0, 1, 5):
            if not slots:
                assert max_logits[i, h] == float("-inf") and lse[i, h] == float("inf")
                assert float(out[i, h].abs().max()) == 0.0
                continue
            keys = kv[[int(inv.indices[i, 0, j]) for j in slots]]
            logits = (keys @ q[i, h]) * inv.sm_scale
            raw = torch.logsumexp(logits, 0)
            assert math.isclose(float(max_logits[i, h]), float(logits.max()), rel_tol=1e-5, abs_tol=1e-6)
            assert math.isclose(float(lse[i, h]), float(raw), rel_tol=1e-5, abs_tol=1e-6)
            sink = float(inv.attn_sink[h])
            norm = float(torch.logaddexp(raw, torch.tensor(sink, dtype=torch.float64)))
            expected = torch.exp(logits - norm) @ keys[:, :512] if norm != float("inf") else torch.zeros(512)
            assert torch.allclose(out[i, h].double(), expected.double(), atol=1e-5, rtol=1e-4)


def test_streaming_comparison_matches_a_whole_tensor_comparison() -> None:
    torch.manual_seed(0)
    reference = torch.randn(3000, 40)
    answer = reference + 1e-5 * torch.randn_like(reference)
    whole = runtime.compare_tensors(answer, reference, label="x", abs_tol=8e-4, rel_tol=3.01 / 128, cos_tol=7e-6)
    stream = runtime.StreamingComparison("x", 8e-4, 3.01 / 128, 7e-6)
    for start in range(0, 3000, 700):
        stream.update(answer[start:start + 700], reference[start:start + 700])
    chunked = stream.result()
    assert whole.ok and chunked.ok
    assert math.isclose(whole.cos_diff, chunked.cos_diff, rel_tol=1e-9, abs_tol=1e-15)
    assert whole.max_abs == chunked.max_abs


def test_comparator_rejects_nan_inf_position_and_element_errors() -> None:
    reference = torch.ones(100)
    reference[3] = float("inf")
    good = reference.clone()
    assert runtime.compare_tensors(good, reference, label="t", abs_tol=1e-6, rel_tol=1e-6, cos_tol=1e-7).ok
    nan = good.clone()
    nan[5] = float("nan")
    result = runtime.compare_tensors(nan, reference, label="t", abs_tol=1e-6, rel_tol=1e-6, cos_tol=1e-7)
    assert not result.ok and result.nan_elements == 1 and "unwritten or NaN" in result.detail
    moved = good.clone()
    moved[3], moved[4] = 1.0, float("inf")
    assert not runtime.compare_tensors(moved, reference, label="t", abs_tol=1e-6, rel_tol=1e-6, cos_tol=1e-7).ok
    off = good.clone()
    off[7] = 1.1
    result = runtime.compare_tensors(off, reference, label="t", abs_tol=1e-6, rel_tol=1e-6, cos_tol=1e-7)
    assert not result.ok and result.fail_elements == 1


def test_reference_candidate_passes_and_semantic_shortcuts_fail() -> None:
    sentinel = runtime.HarnessSentinel()
    case = small_case(index_mode="sprinkle", duplicates=True, scramble=True, sink_mode="extreme",
                      length_mode="edge", invalid_rows="sparse", s_kv=2000, s_q=33, family="v32",
                      sm_scale=576 ** -0.55)
    row = runtime.run_edge_case(reference_candidate, case, sentinel=sentinel)
    assert row["ok"] and abs(row["out_cos_diff"]) <= runtime.OUT_COS_TOL
    assert set(runtime.ACCURACY_FIELDS) <= set(row)

    def fixed_scale(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse):
        reference_candidate(q, kv, indices, 0.5, attn_sink, topk_length, out, max_logits, lse)

    def no_length(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse):
        reference_candidate(q, kv, indices, sm_scale, attn_sink, None, out, max_logits, lse)

    def no_sink(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse):
        reference_candidate(q, kv, indices, sm_scale, None, topk_length, out, max_logits, lse)

    def clamp_invalid(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse):
        clamped = indices.clamp(0, kv.shape[0] - 1)
        reference_candidate(q, kv, clamped, sm_scale, attn_sink, topk_length, out, max_logits, lse)

    def sinked_lse(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse):
        reference_candidate(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse)
        lse.copy_(torch.logaddexp(lse, attn_sink.expand_as(lse)))

    for shortcut in (fixed_scale, no_length, no_sink, clamp_invalid, sinked_lse):
        with pytest.raises(runtime.EdgeCorrectnessFailure):
            runtime.run_edge_case(shortcut, case, sentinel=sentinel)


def test_guard_bands_catch_writes_outside_every_output_and_soft_landing_fills_are_valid() -> None:
    sentinel = runtime.HarnessSentinel()
    case = small_case(length_mode="upstream", sink_mode="upstream")
    for name in ("out", "max_logits", "lse"):
        def overrun(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse, _name=name):
            reference_candidate(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse)
            target = {"out": out, "max_logits": max_logits, "lse": lse}[_name].view(-1)
            torch.as_strided(target, (target.numel() + 1,), (1,))[-1] = 0
        with pytest.raises(runtime.MemorySafetyFailure, match=name):
            runtime.run_edge_case(overrun, case, sentinel=sentinel)

    def underrun(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse):
        reference_candidate(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse)
        torch.as_strided(out.view(-1), (1,), (1,), out.storage_offset() - 1)[0] = 0
    with pytest.raises(runtime.MemorySafetyFailure, match="before the tensor"):
        runtime.run_edge_case(underrun, case, sentinel=sentinel)

    inv = prepare(case)
    regions = {region.name: region for region in inv.arena.regions}
    index_band = regions["indices"].raw[: regions["indices"].guard].view(torch.int32)
    assert bool((index_band == runtime.SOFT_INDEX).all())
    length_band = regions["topk_length"].raw[-regions["topk_length"].guard:].view(torch.int32)
    assert bool((length_band == runtime.SOFT_LENGTH).all())
    q_band = regions["q"].raw[: regions["q"].guard].view(torch.bfloat16)
    assert bool(torch.isfinite(q_band.float()).all())
    assert regions["q"].guard == runtime.guard_bytes(2 * inv.h_q * inv.d_qk)
    assert runtime.guard_bytes(2 * 128 * 512) == 128 * 2 * 128 * 512  # 128 out rows = 16 MiB
    assert runtime.guard_bytes(4) == runtime.GUARD_MIN_BYTES


@pytest.mark.parametrize("name", ["q", "kv", "indices", "attn_sink", "topk_length"])
def test_any_input_mutation_is_a_memory_safety_failure(name: str) -> None:
    sentinel = runtime.HarnessSentinel()
    case = small_case(length_mode="upstream", sink_mode="finite")

    def mutate(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse):
        reference_candidate(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse)
        tensor = {"q": q, "kv": kv, "indices": indices, "attn_sink": attn_sink, "topk_length": topk_length}[name]
        flat = tensor.view(-1)
        flat[0] = flat[1] if name in ("indices", "topk_length") and flat[0] != flat[1] else flat[0] + 1

    with pytest.raises(runtime.MemorySafetyFailure, match="modified its inputs"):
        runtime.run_edge_case(mutate, case, sentinel=sentinel)


def test_second_edge_call_must_rewrite_every_output() -> None:
    sentinel = runtime.HarnessSentinel()
    calls = []

    def lazy(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse):
        if not calls:
            reference_candidate(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse)
        calls.append(1)

    with pytest.raises(runtime.EdgeCorrectnessFailure, match="unwritten or NaN"):
        runtime.run_edge_case(lazy, small_case(), sentinel=sentinel)
    assert len(calls) == 2


def test_exception_classification_and_fatal_faults() -> None:
    assert runtime.classify_exception(runtime.MemorySafetyFailure("x")) == "memory_safety_ok"
    assert runtime.classify_exception(RuntimeError("CUDA error: an illegal memory access was encountered")) == "memory_safety_ok"
    assert runtime.classify_exception(RuntimeError("shape mismatch")) == "correctness_ok"
    for text in ("CUDA error: invalid program counter", "CUDA error: an illegal instruction was encountered",
                 "unspecified launch failure", "device-side assert triggered"):
        assert runtime.is_cuda_fault(RuntimeError(text)), text
    assert not runtime.is_cuda_fault(RuntimeError("Triton Error [CUDA]: invalid argument"))
    assert not runtime.is_cuda_fault(runtime.EdgeCorrectnessFailure("illegal memory access"))


def test_sentinel_detects_replaced_clock_tensor_method_and_flags(monkeypatch) -> None:
    sentinel = runtime.HarnessSentinel()
    sentinel.check("baseline")
    import time as time_module

    monkeypatch.setattr(time_module, "perf_counter", lambda: 0.0)
    with pytest.raises(runtime.HarnessIntegrityFailure, match="perf_counter"):
        sentinel.check("patched clock")
    monkeypatch.undo()
    monkeypatch.setattr(runtime, "DEVICE", "cpu")
    monkeypatch.setattr(torch.Tensor, "__matmul__", lambda self, other: self)
    with pytest.raises(runtime.HarnessIntegrityFailure, match="__matmul__"):
        runtime.HarnessSentinel.check(sentinel, "patched matmul")
    monkeypatch.undo()
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = not previous
        with pytest.raises(runtime.HarnessIntegrityFailure, match="allow_tf32"):
            sentinel.check("flag")
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_trusted_references_were_captured_at_import() -> None:
    assert runtime._torch_equal is torch.equal
    assert runtime._logsumexp is torch.logsumexp
    assert runtime._Event is torch.cuda.Event


def test_aggregation_is_group_then_tier_then_family_then_primary() -> None:
    # Evidence groups at 2, 8, 2, 8, 4 (geomean 4), sweep groups at 1, 4, 16 (geomean 4):
    # every family scores sqrt(4 * 4) = 4 times its own factor.
    factor = {"v32": 1.0, "glm5": 2.0, "v4_flash": 4.0, "v4_pro": 8.0}
    group_value = {"anchor": 2.0, "trace": 8.0, "replay": 2.0, "multi": 8.0, "pattern": 4.0,
                   "sweep": 1.0, "heads": 4.0, "length": 16.0}
    assert set(group_value) == set(runtime.GROUPS)
    rows = []
    for family in runtime.FAMILIES:
        for group in runtime.GROUPS:
            # Two cases per group whose geometric mean is the group value.
            for scale in (0.5, 2.0):
                rows.append({"scored": True, "family": family, "group": group,
                             "tflops": group_value[group] * factor[family] * scale})
    rows.append({"scored": False, "family": "v32", "group": "anchor", "tflops": 1.0e6})
    aggregate = runtime.aggregate_case_results(rows)
    assert set(aggregate) == {"family_tflops", "tier_tflops", "group_tflops", "primary_tflops"}
    assert aggregate["family_tflops"] == pytest.approx({family: 4.0 * value for family, value in factor.items()})
    assert set(aggregate["tier_tflops"]) == {f"{f}:{t}" for f in runtime.FAMILIES for t in runtime.TIERS}
    assert aggregate["tier_tflops"]["glm5:evidence"] == pytest.approx(8.0)
    assert aggregate["tier_tflops"]["glm5:sweep"] == pytest.approx(8.0)
    assert aggregate["group_tflops"]["v4_pro:length"] == pytest.approx(128.0)
    assert set(aggregate["group_tflops"]) == {f"{f}:{g}" for f in runtime.FAMILIES for g in runtime.GROUPS}
    assert math.isclose(aggregate["primary_tflops"], 4.0 * 64 ** 0.25, rel_tol=1e-12)
    # A tier never wins by case count: repeating the sweep cases leaves the family score alone.
    more = rows + [dict(row) for row in rows if row["scored"] and runtime.GROUP_TIER[row["group"]] == "sweep"]
    assert runtime.aggregate_case_results(more)["family_tflops"] == pytest.approx(aggregate["family_tflops"])
    # A partial (public) run with one tier of one family scores that tier.
    partial = runtime.aggregate_case_results(
        [row for row in rows if row["family"] == "v32" and row["group"] == "sweep"]
    )
    assert partial["family_tflops"] == pytest.approx({"v32": 1.0})
    assert partial["primary_tflops"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        runtime.aggregate_case_results([{"scored": True, "family": "v32", "group": "bonus", "tflops": 1.0}])
    with pytest.raises(ValueError):
        runtime.aggregate_case_results([])


def _scored(**fields) -> dict:
    case = {"case_id": "c", "group": "sweep", "family": "v4_flash", "h_q": 64, "s_q": 48, "s_kv": 700,
            "sink_mode": "finite", "gated": True, "seed": 5}
    case.update(fields)
    return case


def _anchor(**fields) -> dict:
    return _scored(group="anchor", family="v32", h_q=128, s_q=4096, s_kv=8192, sink_mode="upstream", **fields)


def test_scored_cases_take_geometry_and_scale_from_the_family_and_group() -> None:
    anchor = runtime.validate_scored_case(_anchor())
    assert anchor["topk"] == 2048 and anchor["d_qk"] == 576
    assert anchor["sm_scale"] == runtime.ANCHOR_SM_SCALE == 0.5
    assert anchor["tier"] == "evidence" and anchor["stream"] is False and anchor["model"] == "DeepSeek-V3.2"
    with pytest.raises(ValueError):  # not one of the family's upstream s_kv values
        runtime.validate_scored_case(dict(_anchor(), s_kv=49152))
    qk_head_dim = {"v32": 192, "glm5": 256, "v4_flash": 512, "v4_pro": 512}
    for family in runtime.FAMILIES:
        heads = runtime.FAMILY_GEOMETRY[family][1]
        case = runtime.validate_scored_case(_scored(family=family, h_q=heads))
        assert case["sm_scale"] == runtime.FAMILY_SM_SCALE[family] == qk_head_dim[family] ** -0.5
        assert case["topk"] == runtime.FAMILY_TOPK[family] and case["tier"] == "sweep"
    # Derived fields are recomputed from the family, never trusted from the manifest.
    assert runtime.validate_scored_case(_scored(topk=128, sm_scale=4.0))["topk"] == 512
    assert runtime.validate_scored_case(_scored(topk=128, sm_scale=4.0))["sm_scale"] == 512 ** -0.5
    with pytest.raises(ValueError):
        runtime.validate_scored_case(_scored(b=1))
    stream = runtime.validate_scored_case(_scored(group="trace", s_q=64, s_kv=700, calls=[[32, 300], [64, 700]]))
    assert stream["stream"] is True and stream["tier"] == "evidence"


def test_anchor_inputs_are_the_upstream_performance_input(monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(runtime, "prepare_invocation", lambda spec, seed, **kwargs: seen.update(spec) or "inv")
    case = runtime.validate_scored_case(_anchor())
    assert runtime.prepare_scored_invocation(case, 4096, 8192, 9) == "inv"
    assert seen["s_q"] == 4096 and seen["s_kv"] == 8192 and seen["topk"] == 2048 and seen["sm_scale"] == 0.5
    assert {key: seen[key] for key in runtime.SCORED_CONTENT} == runtime.SCORED_CONTENT


def test_serving_inputs_are_causal_with_topk_length_and_a_stale_tail() -> None:
    case = runtime.validate_scored_case(_scored(s_q=48, s_kv=300))
    inv = runtime.prepare_scored_invocation(case, 48, 300, 11)
    topk = case["topk"]
    assert inv.sm_scale == runtime.FAMILY_SM_SCALE["v4_flash"] and inv.attn_sink is not None
    idx = inv.indices.view(48, topk)
    lengths = inv.topk_length
    # Row i sits at position s_kv - s_q + i and sees that many keys plus one, capped at topk.
    assert lengths.tolist() == [min(topk, 300 - 48 + i + 1) for i in range(48)]
    assert inv.positions == int(lengths.sum())
    live = torch.arange(topk).unsqueeze(0) < lengths.unsqueeze(1)
    position = 300 - 48 + torch.arange(48).unsqueeze(1).expand(48, topk)
    assert bool(((idx >= 0) & (idx <= position))[live].all())
    tail = idx[~live]
    assert tail.numel() > 1000
    # STALE_TAIL_FRACTION of the slots past topk_length look valid; the rest hold -1.
    assert 0.3 < (tail >= 0).float().mean().item() < 0.7
    assert bool(((tail == -1) | ((tail >= 0) & (tail < 300))).all())
    # Rows with fewer allowed keys than topk keep distinct keys.
    for row, length in zip(idx, lengths.tolist()):
        assert len(set(row[:length].tolist())) == length
    none_sink = runtime.prepare_scored_invocation(
        runtime.validate_scored_case(_scored(sink_mode="none")), 48, 700, 11
    )
    assert none_sink.attn_sink is None


def test_tensor_parallel_heads_shards_and_multi_request_chunks() -> None:
    heads = runtime.prepare_scored_invocation(
        runtime.validate_scored_case(_scored(group="heads", family="v4_pro", h_q=32, s_q=16, s_kv=3000)),
        16, 3000, 3,
    )
    assert heads.h_q == 32 and tuple(heads.q.shape) == (16, 32, 512) and tuple(heads.out.shape) == (16, 32, 512)
    with pytest.raises(ValueError):
        runtime.validate_scored_case(_scored(group="heads", family="v4_pro", h_q=128))
    shard = runtime.prepare_scored_invocation(
        runtime.validate_scored_case(_scored(group="length", length_divisor=4, ragged=False, s_q=32, s_kv=5000)),
        32, 5000, 3,
    )
    target = runtime.FAMILY_TOPK["v4_flash"] // 4
    assert all(abs(int(length) - target) <= max(1, target // 10) for length in shard.topk_length)
    # A ragged shard's rows hold anywhere from 1 to twice the target.
    ragged = runtime.prepare_scored_invocation(
        runtime.validate_scored_case(_scored(group="length", length_divisor=4, ragged=True, s_q=256, s_kv=5000)),
        256, 5000, 3,
    )
    ragged_lengths = [int(length) for length in ragged.topk_length]
    assert all(1 <= length <= 2 * target for length in ragged_lengths)
    assert max(ragged_lengths) - min(ragged_lengths) > 2 * max(1, target // 10)
    for bad in ({"ragged": 1}, {"ragged": None}, {}):
        with pytest.raises(ValueError):
            runtime.validate_scored_case(_scored(group="length", length_divisor=4, s_q=32, s_kv=5000, **bad))
    multi_case = runtime.validate_scored_case(_scored(group="multi", s_q=12, s_kv=900, request_lengths=[300, 600]))
    multi = runtime.prepare_scored_invocation(multi_case, 12, 900, 3)
    rows = runtime.multi_request_rows(12, [300, 600])
    assert multi.s_q == sum(rows) == runtime.scored_rows(multi_case)
    idx = multi.indices.view(multi.s_q, -1)
    live = torch.arange(idx.shape[1]).unsqueeze(0) < multi.topk_length.unsqueeze(1)
    # Each row selects inside its own request only.
    first, second = idx[: rows[0]], idx[rows[0]:]
    assert bool(((first >= 0) & (first < 300))[live[: rows[0]]].all())
    assert bool(((second >= 300) & (second < 900))[live[rows[0]:]].all())


@pytest.mark.parametrize("pattern", ["window", "sink_window", "blocks", "sorted"])
def test_pattern_cases_are_causal_checked_single_calls(pattern) -> None:
    assert runtime.INDEX_PATTERNS == ("window", "sink_window", "blocks", "sorted")
    s_q, s_kv = 160, 6000
    case = runtime.validate_scored_case(_scored(group="pattern", index_pattern=pattern, s_q=s_q, s_kv=s_kv))
    assert case["tier"] == "evidence" and case["stream"] is False
    inv = runtime.prepare_scored_invocation(case, s_q, s_kv, 13)
    topk = case["topk"]
    idx = inv.indices.view(s_q, topk)
    lengths = inv.topk_length
    assert lengths.tolist() == [min(topk, s_kv - s_q + i + 1) for i in range(s_q)]
    live = torch.arange(topk).unsqueeze(0) < lengths.unsqueeze(1)
    position = s_kv - s_q + torch.arange(s_q).unsqueeze(1).expand(s_q, topk)
    # Every live slot is a causal key; the tail stays a stale-or--1 tail.
    assert bool(((idx >= 0) & (idx <= position))[live].all())
    tail = idx[~live]
    assert bool(((tail == -1) | ((tail >= 0) & (tail < s_kv))).all())
    if pattern == "sorted":
        for row, length in zip(idx, lengths.tolist()):
            assert row[:length].tolist() == sorted(row[:length].tolist())
    elif pattern in ("window", "sink_window"):
        near = (idx > position - runtime.PATTERN_WINDOW_FACTOR * topk)[live].float().mean().item()
        assert near > 0.4
    with pytest.raises(ValueError):  # always checked
        runtime.validate_scored_case(_scored(group="pattern", index_pattern=pattern, gated=False))
    with pytest.raises(ValueError):  # a causal single call
        runtime.validate_scored_case(_scored(group="pattern", index_pattern=pattern, s_q=800, s_kv=700))
    with pytest.raises(ValueError):
        runtime.validate_scored_case(_scored(group="pattern", index_pattern="random"))
    with pytest.raises(ValueError):
        runtime.validate_scored_case(_scored(group="pattern"))


def test_sanitize_mode_stops_at_the_first_fatal_error_and_defers_numeric_failures(monkeypatch) -> None:
    judge_named = [
        {key: "upstream" if value == "default" else value for key, value in case.items()}
        for case in PUBLIC["cases"]
    ]
    flagged = [dict(case) for case in runtime.sanitize_order(judge_named)]
    seen: list[str] = []

    def fake_edge(entrypoint, case, *, sentinel, guards, calls):
        assert guards is False and calls == 1
        seen.append(case["case_id"])
        if len(seen) == 2:
            raise runtime.EdgeCorrectnessFailure("wrong under memcheck")
        if len(seen) == 3:
            raise RuntimeError("CUDA error: invalid program counter")
        return {"case_id": case["case_id"], "category": case["category"], "ok": True,
                **{field: 0.0 for field in runtime.ACCURACY_FIELDS}}

    monkeypatch.setattr(runtime, "run_edge_case", fake_edge)
    monkeypatch.setattr(runtime, "load_candidate", lambda path: (None, lambda *a: None, "s" * 64))
    monkeypatch.setattr(runtime, "context_alive", lambda: False)
    lines: list[str] = []
    result = runtime.run_sanitize_cases(TASK_ROOT / "x.py", flagged, progress=lines.append)
    assert result["status"] == "fatal"
    assert [row["status"] for row in result["cases"]] == ["passed", "deferred", "fatal"]
    assert seen == [case["case_id"] for case in flagged[:3]]  # never called again after the fault
    assert lines[0].startswith("SANITIZE_BEGIN 0 ") and lines[-1].endswith(" fatal")
    assert result["order"] == [case["case_id"] for case in flagged]

    seen.clear()

    def mutating(entrypoint, case, *, sentinel, guards, calls):
        seen.append(case["case_id"])
        raise runtime.MemorySafetyFailure("candidate modified its inputs")

    monkeypatch.setattr(runtime, "run_edge_case", mutating)
    result = runtime.run_sanitize_cases(TASK_ROOT / "x.py", flagged)
    assert result["status"] == "failed" and result["failed_gate"] == "memory_safety_ok" and len(seen) == 1
