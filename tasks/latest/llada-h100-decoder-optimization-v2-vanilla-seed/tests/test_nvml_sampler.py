from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
import time


def load_sampler(task_root: Path):
    path = task_root / "images/judge/verifier/nvml_sampler.py"
    spec = importlib.util.spec_from_file_location("nvml_sampler_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_concurrent_sampler_retains_transient_peak(task_root):
    module = load_sampler(task_root)
    values = iter([10, 20, 90, 30, 25, 25])
    last = [25]

    def probe():
        try:
            last[0] = next(values)
        except StopIteration:
            pass
        return last[0]

    sampler = module.PeakMemorySampler(probe, interval_seconds=0.001)
    sampler.start()
    time.sleep(0.008)
    assert sampler.stop() == 90


def test_fixed_shape_sentinel_ignores_cross_case_order_but_rejects_repeat_growth(task_root):
    module = load_sampler(task_root)
    baseline = {
        "torch_peak_allocated_bytes": 10,
        "torch_peak_reserved_bytes": 20,
        "nvml_peak_device_used_bytes": 30,
    }
    stable = {key: value + 1024 for key, value in baseline.items()}
    ok, _deltas = module.fixed_shape_memory_stable(baseline, stable)
    assert ok
    leaked = dict(stable)
    leaked["torch_peak_reserved_bytes"] += 17 * 1024 * 1024
    ok, deltas = module.fixed_shape_memory_stable(baseline, leaked)
    assert not ok
    assert deltas["torch_peak_reserved_bytes"] > 16 * 1024 * 1024


def test_qualification_guard_records_allocator_warmup_but_repeat_guard_stays_strict(
    task_root,
):
    module = load_sampler(task_root)
    start = {
        "torch_peak_allocated_bytes": 18 * 2**30,
        "torch_peak_reserved_bytes": 19 * 2**30,
        "nvml_peak_device_used_bytes": 20 * 2**30,
    }
    # Reproduce the largest planned-row H100 diagnostic (GSM8K 8-shot G1024):
    # the output was stable, allocated memory moved only 17,967,616 bytes, and
    # the caching allocator retained 10,190,061,568 bytes.
    end = {
        "torch_peak_allocated_bytes": start["torch_peak_allocated_bytes"]
        + 17_967_616,
        "torch_peak_reserved_bytes": start["torch_peak_reserved_bytes"]
        + 10_190_061_568,
        "nvml_peak_device_used_bytes": start["nvml_peak_device_used_bytes"]
        + 10_190_061_568,
    }
    old_ok, _old_deltas = module.fixed_shape_memory_stable(
        start,
        end,
        material_growth_bytes=4 * 2**30,
    )
    assert not old_ok
    ok, observed = module.fixed_shape_memory_stable(
        start,
        end,
        material_growth_bytes=12 * 2**30,
    )
    assert ok
    assert observed == {
        "torch_peak_allocated_bytes": 17_967_616,
        "torch_peak_reserved_bytes": 10_190_061_568,
        "nvml_peak_device_used_bytes": 10_190_061_568,
    }

    # Adjacent warmed repeats still use the independent 16 MiB leak sentinel;
    # raising the qualification observation ceiling does not weaken this gate.
    stable_repeat, repeat_deltas = module.fixed_shape_memory_stable(end, dict(end))
    assert stable_repeat
    assert set(repeat_deltas.values()) == {0}
    leaked_repeat = dict(end)
    leaked_repeat["torch_peak_reserved_bytes"] += 17 * 2**20
    leaked_repeat["nvml_peak_device_used_bytes"] += 17 * 2**20
    stable_repeat, repeat_deltas = module.fixed_shape_memory_stable(
        end, leaked_repeat
    )
    assert not stable_repeat
    assert repeat_deltas["torch_peak_reserved_bytes"] == 17 * 2**20
