from __future__ import annotations

import importlib.util
import math

import pytest

import kv_scoring_metric as metric
import kv_scoring_runtime as rt


def test_ruler_string_match() -> None:
    assert metric.ruler_match_all("The numbers are 123 and 456.", ["123", "456"]) == 1.0
    assert metric.ruler_match_all("Only 123.", ["123", "456"]) == 0.5
    assert metric.ruler_match_part("France", ["france", "Normandy"]) == 1.0
    assert metric.ruler_match_part("Spain", ["France"]) == 0.0


def test_score_item_dispatch() -> None:
    item = {"task": "ruler_vt", "answers": [["AAA", "BBB"]], "meta": {}}
    assert metric.score_item(["AAA BBB"], item, "ruler_all") == [1.0]
    squad = {"task": "squad", "answers": ["Paris"], "meta": {}}
    assert metric.score_item(["It is paris."], squad, "include") == [1.0]
    with pytest.raises(ValueError):
        metric.score_item(["a", "b"], squad, "include")


def test_repoqa_empty_answer_counts_as_a_miss() -> None:
    repo = {"lang": "python", "repo": "x/y", "func_names": ["f", "g"], "needles": ["def f():\n    return 1\n", "def g():\n    return 2\n"]}
    if importlib.util.find_spec("nltk") is None:
        assert metric.repoqa_scores(["", "   "], repo) == [0.0, 0.0]
        return
    assert metric.repoqa_scores(["", "def g():\n    return 2\n"], repo) == [0.0, 1.0]


def test_decode_chunks_depend_only_on_token_counts() -> None:
    per_token = 36 * 8 * 128 * 2 * 2
    assert per_token == 147456
    assert rt.decode_chunk_size(per_token, 200, 30, 256) == 16
    long = rt.decode_chunk_size(per_token, 24700, 51, 96)
    assert long == (40 << 30) // (per_token * (24700 + 51 + 96)) and 1 <= long < 16
    assert rt.decode_chunk_size(per_token, 31000, 400, 512) >= 1


def test_overhead_floor_scales_with_the_shortest_prefill() -> None:
    rows = [{"prefill_s": 0.02}, {"prefill_s": 0.5}, {"prefill_s": 0.0}]
    assert math.isclose(rt.overhead_item_floor_s(rows), 0.2)


def test_fresh_copy_has_new_storage_and_equal_values() -> None:
    torch = pytest.importorskip("torch")
    ops = rt.TrustedOps.capture(torch)
    q = torch.randn(1, 4, 8, 2).transpose(1, 2)  # non-contiguous like the captured queries
    pe = (torch.randn(1, 4, 2), torch.randn(1, 4, 2))
    copy_q = rt._fresh_copy(torch, ops, q); copy_pe = rt._fresh_copy(torch, ops, pe)
    assert copy_q.data_ptr() != q.data_ptr() and torch.equal(copy_q, q) and copy_q.stride() == q.stride()
    assert all(a.data_ptr() != b.data_ptr() and torch.equal(a, b) for a, b in zip(copy_pe, pe))


def test_handoff_estimate_is_fp32_per_layer_head_and_token() -> None:
    assert rt.handoff_bytes([{"meta": {"context_tokens": 1000}, "context": ""}]) == 36 * 8 * 4 * 1064 + (1 << 16)
    assert 2e10 < rt.handoff_bytes([{"meta": {"context_tokens": 13_000}, "context": ""}] * 1812) < 4e10


def test_warmup_tensors_match_shapes_and_strides_but_not_content() -> None:
    torch = pytest.importorskip("torch")
    ops = rt.TrustedOps.capture(torch)
    hidden = torch.randn(1, 40, 64, dtype=torch.bfloat16); queries = torch.randn(1, 40, 32, 16, dtype=torch.bfloat16).transpose(1, 2)
    keys = torch.randn(1, 8, 40, 16, dtype=torch.bfloat16); values = torch.randn(1, 8, 40, 16, dtype=torch.bfloat16)
    pe = (torch.randn(1, 40, 16), torch.randn(1, 40, 16)); weights = {"q_proj": torch.randn(4, 4)}
    abi = (hidden, queries, keys, values, pe, weights)
    a = rt._decoy_inputs(torch, ops, abi, torch.Generator().manual_seed(rt.decoy_seed("f0001")))
    b = rt._decoy_inputs(torch, ops, abi, torch.Generator().manual_seed(rt.decoy_seed("f0001")))
    for got, ref in zip(a[:4], abi[:4]):
        assert got.shape == ref.shape and got.stride() == ref.stride() and got.dtype == ref.dtype and not torch.equal(got, ref)
    assert all(torch.equal(x, y) for x, y in zip(a[:4], b[:4]))
    assert torch.equal(a[4][0], pe[0]) and a[4][0].data_ptr() != pe[0].data_ptr() and a[5] is weights
    assert rt.decoy_seed("f0001") != rt.decoy_seed("f0002")
