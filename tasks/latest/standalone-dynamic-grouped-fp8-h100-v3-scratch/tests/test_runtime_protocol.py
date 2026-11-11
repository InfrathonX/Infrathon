"""CPU checks of the v4 runtime's protocol pieces (no GPU required)."""

from __future__ import annotations

import math
import time

import pytest

from conftest import load_module


runtime = load_module("runtime_protocol_runtime", "runtime/grouped_fp8_runtime.py")


def test_prefix_sum_layout_handles_empty_and_full_groups() -> None:
    assert runtime.group_ends([0, 0, 5]) == [0, 0, 5]
    assert runtime.group_starts([0, 0, 5]) == [0, 0, 0]
    assert runtime.group_ends([1, 0, 128, 129]) == [1, 128, 256, 385]
    assert runtime.group_starts([1, 0, 128, 129]) == [0, 128, 128, 256]
    assert runtime.group_ends([128, 128]) == [128, 256]


SCORED = {
    "case_id": "s", "family": "decode", "layout": "masked", "g": 32, "m_max": 4096,
    "expected_m": 1024, "n": 4096, "k": 4096, "n2": None, "k2": None, "dist": "dirichlet",
    "alpha": 0.6, "hint_scale": 1.0, "rows_low": None, "rows_high": None, "seed": 10_000,
}


def test_domain_validation() -> None:
    base = {
        "case_id": "x", "layout": "psum", "category": "c", "g": 2, "m_max": 128,
        "n": 128, "k": 128, "hint": 1, "m_sizes": [128, 128], "seed": 1, "sanitize": False,
    }
    runtime.validate_edge_case(base)
    runtime.validate_edge_case(dict(base, m_max=131072, m_sizes=[131072, 1000]))
    for field, value in (("g", 65), ("g", 0), ("m_max", 200), ("n", 64), ("k", 8320),
                         ("hint", 0), ("hint", (1 << 20) + 1), ("m_sizes", [129, 0]),
                         ("m_sizes", [1]), ("m_max", 131072 + 128)):
        with pytest.raises(ValueError):
            runtime.validate_edge_case(dict(base, **{field: value}))
    with pytest.raises(ValueError):  # masked M_max stays at 4096
        runtime.validate_edge_case(dict(base, layout="masked", m_max=8192, m_sizes=[1, 1]))
    with pytest.raises(ValueError):  # G * M_max * (K + 2N) <= 2^34
        runtime.validate_edge_case(dict(base, g=64, m_max=131072, n=8192, k=8192, m_sizes=[0] * 64))
    runtime.validate_scored_case(SCORED)
    for field, value in (("expected_m", 5000), ("family", "masked"), ("layout", "x"), ("dist", "skew"),
                         ("alpha", None), ("hint_scale", 0.0), ("n2", 4096)):
        with pytest.raises(ValueError):
            runtime.validate_scored_case(dict(SCORED, **{field: value}))
    with pytest.raises(ValueError):  # prefill is prefix-sum only
        runtime.validate_scored_case(dict(SCORED, family="prefill"))
    runtime.validate_scored_case(dict(SCORED, family="prefill", layout="psum", g=8, m_max=65536))
    runtime.validate_scored_case(dict(SCORED, family="sequence", n2=2048, k2=4096))
    with pytest.raises(ValueError):
        runtime.validate_scored_case(dict(SCORED, family="sequence"))
    single = dict(SCORED, g=1, m_max=256, expected_m=96, dist="single", alpha=None, rows_low=64, rows_high=128)
    runtime.validate_scored_case(single)
    with pytest.raises(ValueError):
        runtime.validate_scored_case(dict(single, g=2))
    with pytest.raises(ValueError):
        runtime.validate_scored_case(dict(SCORED, extra=1))


def test_row_draws_are_pure_functions_of_the_seed() -> None:
    uniform = dict(SCORED, expected_m=20, dist="uniform", alpha=None)
    first = runtime.draw_m_sizes(uniform, 123)
    assert first == runtime.draw_m_sizes(uniform, 123)
    assert all(14 <= value <= 25 for value in first)
    skewed = dict(SCORED, expected_m=100, m_max=1280, alpha=0.4)
    rows = runtime.draw_m_sizes(skewed, 7)
    assert rows == runtime.draw_m_sizes(skewed, 7) and sum(rows) == 100 * 32
    assert max(rows) <= 1280 and max(rows) > 3 * 100 and min(rows) < 30
    single = dict(SCORED, g=1, m_max=256, expected_m=40, dist="single", alpha=None, rows_low=16, rows_high=64)
    assert 16 <= runtime.draw_m_sizes(single, 5)[0] <= 64
    assert runtime.case_hint(dict(SCORED, hint_scale=0.125)) == int(1.2 * 1024 * 0.125)


def test_sequence_schedule_is_seeded_and_complete() -> None:
    case = dict(SCORED, family="sequence", n2=2048, k2=4096, expected_m=64, m_max=768)
    calls = runtime.sequence_schedule(case, 99)
    assert calls == runtime.sequence_schedule(case, 99) != runtime.sequence_schedule(case, 100)
    assert len(calls) == runtime.SEQUENCE_LAYERS * 2
    assert [linear for linear, *_ in calls[:4]] == [0, 1, 0, 1]
    assert all(0 <= a < runtime.SEQUENCE_A_POOL and 0 <= out < runtime.SEQUENCE_OUT_POOL
               for *_rest, a, out in calls)
    hints = runtime.sequence_hints(case)
    assert len(hints) == runtime.SEQUENCE_WEIGHT_POOL and hints == runtime.sequence_hints(case)


def test_effective_tflops_counts_only_valid_rows_and_allows_empty_groups() -> None:
    value = runtime.effective_tflops([0, 10, 0], 128, 256, 1.0e-6)
    assert math.isclose(value, 2.0 * 10 * 128 * 256 / 1.0e-6 / 1.0e12)
    with pytest.raises(RuntimeError):
        runtime.effective_tflops([0, 0], 128, 128, 1.0e-6)


def test_normalized_error_and_limits() -> None:
    assert runtime.normalized_error(2.0, 4.0) == 0.0
    assert runtime.normalized_error(0.0, 0.0) == 0.0
    assert runtime.EXACT_ERROR_LIMIT == 5.0e-5
    assert runtime.SOURCE_ERROR_LIMIT == 1.0e-3
    assert runtime.SOURCE_CHECK_MIN_ELEMENTS == 16_384


def test_exception_classification() -> None:
    assert runtime.classify_exception(runtime.MemorySafetyFailure("x")) == "memory_safety_ok"
    assert runtime.classify_exception(runtime.HarnessIntegrityFailure("x")) == "harness_integrity_ok"
    assert runtime.classify_exception(RuntimeError("CUDA error: an illegal memory access was encountered")) == "memory_safety_ok"
    assert runtime.classify_exception(ValueError("shape mismatch")) == "correctness_ok"


def test_sentinel_detects_a_replaced_clock_and_restores(monkeypatch) -> None:
    sentinel = runtime.HarnessSentinel()
    sentinel.check("baseline")
    monkeypatch.setattr(time, "perf_counter", lambda: 0.0)
    with pytest.raises(runtime.HarnessIntegrityFailure, match="perf_counter"):
        sentinel.check("after patch")
    monkeypatch.undo()
    sentinel.check("after restore")


def test_sentinel_detects_a_replaced_tensor_method(monkeypatch) -> None:
    import torch

    sentinel = runtime.HarnessSentinel()
    monkeypatch.setattr(torch.Tensor, "fill_", lambda self, value: self)
    with pytest.raises(runtime.HarnessIntegrityFailure, match="fill_"):
        sentinel.check("after patch")


def test_sentinel_detects_global_numeric_configuration(monkeypatch) -> None:
    import torch

    sentinel = runtime.HarnessSentinel()
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", not torch.backends.cuda.matmul.allow_tf32)
    with pytest.raises(runtime.HarnessIntegrityFailure, match="allow_tf32"):
        sentinel.check("after flag change")


def test_trusted_references_were_captured_before_any_patch() -> None:
    assert runtime._perf_counter is time.perf_counter
    assert runtime._event_elapsed.__name__ == "elapsed_time"


def test_plausibility_bound_table() -> None:
    assert runtime.plausibility_bound([9, 0]) == 2500.0
    assert runtime.plausibility_bound([10, 0]) == 7000.0
    assert runtime.plausibility_bound([7, 5]) is None
    assert runtime.plausibility_bound(None) is None
