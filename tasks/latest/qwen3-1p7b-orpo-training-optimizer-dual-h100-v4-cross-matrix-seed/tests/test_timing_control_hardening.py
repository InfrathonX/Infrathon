from __future__ import annotations

from conftest import LAB_ROOT
from runtime import distributed_probe, loader


def test_controller_caches_measurement_and_collective_handles_before_candidate_import(
    monkeypatch,
):
    trusted_perf_counter = distributed_probe._TRUSTED_PERF_COUNTER
    trusted_synchronize = distributed_probe._TRUSTED_CUDA_SYNCHRONIZE
    trusted_reset_peak = distributed_probe._TRUSTED_CUDA_RESET_PEAK_MEMORY_STATS
    trusted_max_reserved = distributed_probe._TRUSTED_CUDA_MAX_MEMORY_RESERVED
    trusted_barrier = distributed_probe._TRUSTED_DIST_BARRIER
    trusted_all_reduce = distributed_probe._TRUSTED_DIST_ALL_REDUCE
    trusted_adamw = distributed_probe._TRUSTED_ADAMW

    monkeypatch.setattr(distributed_probe.time, "perf_counter", lambda: -1.0)
    monkeypatch.setattr(distributed_probe.torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(
        distributed_probe.torch.cuda, "reset_peak_memory_stats", lambda *_: None
    )
    monkeypatch.setattr(distributed_probe.torch.cuda, "max_memory_reserved", lambda *_: 0)
    monkeypatch.setattr(distributed_probe.dist, "barrier", lambda: None)
    monkeypatch.setattr(distributed_probe.dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(distributed_probe.torch.optim, "AdamW", object)

    assert distributed_probe._TRUSTED_PERF_COUNTER is trusted_perf_counter
    assert distributed_probe._TRUSTED_CUDA_SYNCHRONIZE is trusted_synchronize
    assert distributed_probe._TRUSTED_CUDA_RESET_PEAK_MEMORY_STATS is trusted_reset_peak
    assert distributed_probe._TRUSTED_CUDA_MAX_MEMORY_RESERVED is trusted_max_reserved
    assert distributed_probe._TRUSTED_DIST_BARRIER is trusted_barrier
    assert distributed_probe._TRUSTED_DIST_ALL_REDUCE is trusted_all_reduce
    assert distributed_probe._TRUSTED_ADAMW is trusted_adamw


def test_loader_caches_import_and_module_type_handles(monkeypatch):
    trusted_spec = loader._TRUSTED_SPEC_FROM_FILE_LOCATION
    trusted_module_from_spec = loader._TRUSTED_MODULE_FROM_SPEC
    trusted_module_type = loader._TRUSTED_MODULE_TYPE
    trusted_patch_type = loader._TRUSTED_MODEL_PATCH_TYPE

    monkeypatch.setattr(loader.importlib.util, "spec_from_file_location", lambda *_: None)
    monkeypatch.setattr(loader.importlib.util, "module_from_spec", lambda *_: None)
    monkeypatch.setattr(loader.torch.nn, "Module", object)

    assert loader._TRUSTED_SPEC_FROM_FILE_LOCATION is trusted_spec
    assert loader._TRUSTED_MODULE_FROM_SPEC is trusted_module_from_spec
    assert loader._TRUSTED_MODULE_TYPE is trusted_module_type
    assert loader._TRUSTED_MODEL_PATCH_TYPE is trusted_patch_type


def test_model_is_constructed_before_candidate_objective_import():
    source = (LAB_ROOT / "runtime/distributed_probe.py").read_text(encoding="utf-8")
    model_load = source.index("model, artifact = load_pinned_model(args.model_path)")
    candidate_load = source.index(
        "objective, candidate_version, model_optimizer, model_patch = load_probe_objective("
    )
    optimizer_apply = source.index(
        "model_optimization = apply_model_optimizer("
    )
    fsdp_build = source.index("fsdp_module = _build_fsdp_module(step_module, device)")
    assert model_load < candidate_load
    assert candidate_load < optimizer_apply < fsdp_build


def test_controller_uses_cached_timing_and_peak_handles_at_measurement_sites():
    source = (LAB_ROOT / "runtime/distributed_probe.py").read_text(encoding="utf-8")
    assert "started = _TRUSTED_PERF_COUNTER()" in source
    assert "_TRUSTED_CUDA_SYNCHRONIZE(device)" in source
    assert "_TRUSTED_CUDA_RESET_PEAK_MEMORY_STATS(device)" in source
    assert "local_allocated = _TRUSTED_CUDA_MAX_MEMORY_ALLOCATED(device)" in source
    assert "local_reserved = _TRUSTED_CUDA_MAX_MEMORY_RESERVED(device)" in source
    assert "_TRUSTED_DIST_ALL_REDUCE(value" in source
    assert "optimizer = _TRUSTED_ADAMW(" in source
