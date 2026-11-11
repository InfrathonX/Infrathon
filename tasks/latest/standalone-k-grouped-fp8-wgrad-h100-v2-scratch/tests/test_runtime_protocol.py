"""CPU checks of the v3 runtime's protocol pieces (no GPU required)."""

from __future__ import annotations

import math
import time

import pytest

from conftest import load_module


runtime = load_module("runtime_protocol_runtime", "runtime/k_grouped_wgrad_runtime.py")


def test_domain_validation() -> None:
    base = {
        "case_id": "x", "category": "c", "g": 3, "m": 7, "n": 9,
        "ks": [128, 0, 384], "hint": 1, "seed": (1 << 63) - 1, "sanitize": False,
    }
    runtime.validate_edge_case(base)
    runtime.validate_edge_case(dict(base, g=1, m=1, n=1, ks=[0]))
    runtime.validate_edge_case(dict(base, g=1, m=1, n=1, ks=[262_144]))
    for field, value in (("g", 129), ("g", 0), ("m", 0), ("n", 8193), ("ks", [64, 0, 0]),
                         ("ks", [128, 128]), ("ks", [-128, 0, 0]), ("ks", [262_272, 0, 0]),
                         ("hint", 0), ("hint", (1 << 20) + 1), ("seed", 1 << 63)):
        with pytest.raises(ValueError):
            runtime.validate_edge_case(dict(base, **{field: value}))
    scored = {"case_id": "s", "family": "large", "model": "m", "g": 16, "m": 4096, "n": 7168,
              "m2": None, "n2": None, "dist": "dirichlet", "total_k": 262_144, "alpha": 0.4,
              "hint_scale": 0.125, "seed": 1 << 40}
    runtime.validate_scored_case(scored)
    sequence = dict(scored, family="sequence", m2=7168, n2=2048)
    runtime.validate_scored_case(sequence)
    uniform = dict(scored, dist="uniform", alpha=None, total_k=16 * 2048)
    runtime.validate_scored_case(uniform)
    for field, value in (("total_k", 100), ("total_k", 262_272), ("family", "coarse"), ("n", 8193),
                         ("dist", "upstream"), ("alpha", None), ("hint_scale", 0.0), ("model", ""),
                         ("m2", 7168), ("g", 129)):
        with pytest.raises(ValueError):
            runtime.validate_scored_case(dict(scored, **{field: value}))
    for field, value in (("m2", None), ("dist", "uniform")):
        with pytest.raises(ValueError):
            runtime.validate_scored_case(dict(sequence, **{field: value}))
    with pytest.raises(ValueError):
        runtime.validate_scored_case(dict(scored, band="ep64"))


def test_group_size_draws_are_deterministic_and_in_range() -> None:
    case = {"g": 16, "total_k": 16 * 2048, "dist": "uniform"}
    first = runtime.draw_ks(case, 123)
    assert first == runtime.draw_ks(case, 123)
    assert all(value % 128 == 0 and 1408 <= value <= 2688 for value in first)
    tiny = runtime.draw_ks({"g": 8, "total_k": 1024, "dist": "uniform"}, 7)
    assert set(tiny) <= {128, 256}
    skewed = {"g": 48, "total_k": 3072, "dist": "dirichlet", "alpha": 0.1}
    ks = runtime.draw_ks(skewed, 99)
    assert ks == runtime.draw_ks(skewed, 99) and sum(ks) == 3072 and len(ks) == 48
    assert all(value % 128 == 0 for value in ks) and ks.count(0) > 24
    balanced = runtime.draw_ks({"g": 4, "total_k": 262_144, "dist": "dirichlet", "alpha": 4.0}, 5)
    assert sum(balanced) == 262_144 and max(balanced) < 2 * 262_144 // 4


def test_sequence_schedule_is_one_backward_pass() -> None:
    case = {"g": 8, "total_k": 4096, "dist": "dirichlet", "alpha": 1.0}
    calls = runtime.sequence_schedule(case, 17)
    assert calls == runtime.sequence_schedule(case, 17)
    assert len(calls) == 2 * runtime.SEQUENCE_LAYERS
    assert [layer for _linear, layer, _ks, _out in calls[:4]] == [57, 57, 56, 56]
    assert [linear for linear, *_rest in calls[:4]] == [1, 0, 1, 0]
    assert all(sum(ks) == 4096 for _linear, _layer, ks, _out in calls)
    assert len({tuple(ks) for _linear, _layer, ks, _out in calls}) > 100
    assert {out for *_rest, out in calls} == set(range(runtime.SEQUENCE_OUT_POOL))


def test_case_hint_follows_the_mean_group_k_and_the_scale() -> None:
    case = {"g": 8, "total_k": 32768, "hint_scale": 1.0}
    assert runtime.case_hint(case) == int(1.2 * 4096)
    assert runtime.case_hint(dict(case, hint_scale=0.05)) == int(1.2 * 4096 * 0.05)
    assert runtime.case_hint(dict(case, hint_scale=1e-9)) == 1


def test_per_call_seeds_are_63_bit_and_role_separated() -> None:
    seeds = {runtime.derive_seed(1 << 40, role, trial) for role in ("decoy", "measured", "ks") for trial in range(5)}
    assert len(seeds) == 15
    assert all(0 <= seed <= runtime.MAX_SEED for seed in seeds)
    assert runtime.derive_seed(5, "measured", 0) == runtime.derive_seed(5, "measured", 0)


def test_effective_tflops_counts_actual_k_and_allows_empty_groups() -> None:
    value = runtime.effective_tflops([0, 256, 0], 7, 9, 1.0e-6)
    assert math.isclose(value, 2.0 * 256 * 7 * 9 / 1.0e-6 / 1.0e12)
    with pytest.raises(RuntimeError):
        runtime.effective_tflops([0, 0], 128, 128, 1.0e-6)


def test_normalized_error_and_limits() -> None:
    assert runtime.normalized_error(2.0, 4.0) == 0.0
    assert runtime.normalized_error(0.0, 0.0) == 0.0
    assert runtime.ELEMENT_RATIO_LIMIT == 0.05
    assert runtime.ELEMENT_ADDEND_RELATIVE == 2.0**-20
    assert runtime.EXACT_ERROR_LIMIT == 2.0e-7
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
