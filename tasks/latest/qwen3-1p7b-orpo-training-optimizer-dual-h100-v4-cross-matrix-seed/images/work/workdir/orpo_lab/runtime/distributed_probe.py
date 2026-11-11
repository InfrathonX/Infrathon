#!/usr/bin/env python3
"""Two-rank FSDP1 worker for the diagnostic full-training public smoke.

Launch this file through ``torchrun --nproc-per-node=2``.  It intentionally
refuses single-GPU, extra-GPU, below-envelope, non-BF16, or partially
successful runs.  The emitted metrics are public-smoke telemetry, never a
formal score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
import time
import uuid
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .api import ObjectiveContract
from .train_step import (
    EXPECTED_CONFIG,
    TrainStepModule,
    apply_model_optimizer,
    load_frozen_preference_dataset,
    load_pinned_model,
    load_probe_objective,
    make_synthetic_preference_batch,
)


WORLD_SIZE_REQUIRED = 2

# Device name, UUID and total memory are recorded in the rank identity as
# provenance only; they never gate admission.  Keep these two constants
# identical in the Judge controller (tests assert the pair).
MIN_COMPUTE_CAPABILITY = (8, 0)
# The floor covers the largest scored workload with headroom.
MIN_DEVICE_MEMORY_BYTES = 60 * 1024**3

# Candidate code is imported later in ``run``.  Cache every timing, CUDA peak,
# optimizer-construction, and distributed control handle before that import so
# metric/control calls do not resolve through mutable module attributes.
_TRUSTED_PERF_COUNTER = time.perf_counter
_TRUSTED_ADAMW = torch.optim.AdamW
_TRUSTED_AUTOCAST = torch.autocast
_TRUSTED_ISFINITE = torch.isfinite
_TRUSTED_MANUAL_SEED = torch.manual_seed
_TRUSTED_CUDA_DEVICE_COUNT = torch.cuda.device_count
_TRUSTED_CUDA_IS_AVAILABLE = torch.cuda.is_available
_TRUSTED_CUDA_IS_BF16_SUPPORTED = torch.cuda.is_bf16_supported
_TRUSTED_CUDA_SET_DEVICE = torch.cuda.set_device
_TRUSTED_CUDA_GET_DEVICE_PROPERTIES = torch.cuda.get_device_properties
_TRUSTED_CUDA_MANUAL_SEED_ALL = torch.cuda.manual_seed_all
_TRUSTED_CUDA_SYNCHRONIZE = torch.cuda.synchronize
_TRUSTED_CUDA_EMPTY_CACHE = torch.cuda.empty_cache
_TRUSTED_CUDA_RESET_PEAK_MEMORY_STATS = torch.cuda.reset_peak_memory_stats
_TRUSTED_CUDA_MAX_MEMORY_ALLOCATED = torch.cuda.max_memory_allocated
_TRUSTED_CUDA_MAX_MEMORY_RESERVED = torch.cuda.max_memory_reserved
_TRUSTED_DIST_ALL_GATHER_OBJECT = dist.all_gather_object
_TRUSTED_DIST_ALL_REDUCE = dist.all_reduce
_TRUSTED_DIST_BARRIER = dist.barrier
_TRUSTED_DIST_DESTROY_PROCESS_GROUP = dist.destroy_process_group
_TRUSTED_DIST_GET_RANK = dist.get_rank
_TRUSTED_DIST_GET_WORLD_SIZE = dist.get_world_size
_TRUSTED_DIST_INIT_PROCESS_GROUP = dist.init_process_group
_TRUSTED_DIST_IS_INITIALIZED = dist.is_initialized
_TRUSTED_DIST_REDUCE_MAX = dist.ReduceOp.MAX
_TRUSTED_DIST_REDUCE_MIN = dist.ReduceOp.MIN
_TRUSTED_FSDP_CLIP_GRAD_NORM = FSDP.clip_grad_norm_

# This schema is deliberately emitted from the worker telemetry without
# changing the top-level worker schema.  Existing public tooling can therefore
# still consume schema_version=1, while paired controllers can fail closed on
# the richer, task-owned correctness witness.
CORRECTNESS_TELEMETRY_SCHEMA = 1
_AUX_NAMES = (
    "chosen_logps",
    "rejected_logps",
    "chosen_logits_mean",
    "rejected_logits_mean",
    "nll_loss",
    "chosen_rewards",
    "rejected_rewards",
    "logsigmoid_ratio_mean",
    "log_odds_mean",
)
_TRUSTED_DIST_REDUCE_SUM = dist.ReduceOp.SUM


def _env_int(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.isascii() or not raw.isdigit():
        raise RuntimeError(f"torchrun must provide integer {name}")
    return int(raw)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _all_gather_objects(value: Any, world_size: int) -> list[Any]:
    values: list[Any] = [None for _ in range(world_size)]
    _TRUSTED_DIST_ALL_GATHER_OBJECT(values, value)
    return values


def _validate_distributed_environment() -> tuple[int, int, torch.device, list[dict[str, Any]]]:
    requested_world = _env_int("WORLD_SIZE")
    rank = _env_int("RANK")
    local_rank = _env_int("LOCAL_RANK")
    if requested_world != WORLD_SIZE_REQUIRED:
        raise RuntimeError(f"WORLD_SIZE must equal {WORLD_SIZE_REQUIRED}, got {requested_world}")
    if _TRUSTED_CUDA_DEVICE_COUNT() != WORLD_SIZE_REQUIRED:
        raise RuntimeError(
            f"exactly {WORLD_SIZE_REQUIRED} CUDA devices must be visible, got "
            f"{_TRUSTED_CUDA_DEVICE_COUNT()}"
        )
    if not _TRUSTED_CUDA_IS_AVAILABLE() or not _TRUSTED_CUDA_IS_BF16_SUPPORTED():
        raise RuntimeError("CUDA with native BF16 support is required")
    if rank not in range(WORLD_SIZE_REQUIRED) or local_rank not in range(WORLD_SIZE_REQUIRED):
        raise RuntimeError("rank and local rank must both be in {0, 1}")

    _TRUSTED_CUDA_SET_DEVICE(local_rank)
    device = torch.device("cuda", local_rank)
    properties = _TRUSTED_CUDA_GET_DEVICE_PROPERTIES(device)
    capability = (properties.major, properties.minor)
    if capability < MIN_COMPUTE_CAPABILITY:
        raise RuntimeError(
            f"compute capability {capability[0]}.{capability[1]} is below the task minimum "
            f"{MIN_COMPUTE_CAPABILITY[0]}.{MIN_COMPUTE_CAPABILITY[1]}: {properties.name}"
        )
    if int(properties.total_memory) < MIN_DEVICE_MEMORY_BYTES:
        raise RuntimeError(
            f"GPU memory {int(properties.total_memory)} bytes is below the task minimum "
            f"{MIN_DEVICE_MEMORY_BYTES} bytes: {properties.name}"
        )

    _TRUSTED_DIST_INIT_PROCESS_GROUP(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=20),
    )
    if _TRUSTED_DIST_GET_WORLD_SIZE() != WORLD_SIZE_REQUIRED or _TRUSTED_DIST_GET_RANK() != rank:
        raise RuntimeError("initialized process group differs from torchrun environment")

    identity = {
        "rank": rank,
        "local_rank": local_rank,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "cuda_device_count": _TRUSTED_CUDA_DEVICE_COUNT(),
        "device_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": properties.total_memory,
    }
    identities = _all_gather_objects(identity, WORLD_SIZE_REQUIRED)
    if sorted(item["rank"] for item in identities) != [0, 1]:
        raise RuntimeError("all-rank handshake did not contain ranks 0 and 1")
    if sorted(item["local_rank"] for item in identities) != [0, 1]:
        raise RuntimeError("the two ranks are not mapped to distinct local CUDA devices")
    if len({item["pid"] for item in identities}) != WORLD_SIZE_REQUIRED:
        raise RuntimeError("the two distributed ranks do not have distinct processes")
    if any(item["cuda_device_count"] != WORLD_SIZE_REQUIRED for item in identities):
        raise RuntimeError("not every rank sees exactly two CUDA devices")
    return rank, local_rank, device, identities


def _build_fsdp_module(
    module: TrainStepModule, device: torch.device
) -> FSDP:
    decoder_layers = module.model.model.layers
    if len(decoder_layers) != EXPECTED_CONFIG["num_hidden_layers"]:
        raise RuntimeError("loaded model has an unexpected decoder layer count")
    decoder_type = type(decoder_layers[0])
    if any(type(layer) is not decoder_type for layer in decoder_layers):
        raise RuntimeError("Qwen3 decoder layers do not share one concrete type")

    module.to(device)
    if module.model.lm_head.weight is not module.model.model.embed_tokens.weight:
        raise RuntimeError("moving the model broke tied lm_head/embedding identity")
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )
    wrapped = FSDP(
        module,
        auto_wrap_policy=partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={decoder_type},
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=device,
        sync_module_states=True,
        forward_prefetch=False,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    if wrapped.module.model.lm_head.weight is not wrapped.module.model.model.embed_tokens.weight:
        raise RuntimeError("FSDP wrapping broke tied lm_head/embedding identity")
    return wrapped


def _global_integer(local_value: int, device: torch.device) -> int:
    value = torch.tensor(local_value, dtype=torch.int64, device=device)
    _TRUSTED_DIST_ALL_REDUCE(value, op=_TRUSTED_DIST_REDUCE_SUM)
    return int(value.item())


def _global_mean(local_value: float, device: torch.device) -> float:
    value = torch.tensor(local_value, dtype=torch.float64, device=device)
    _TRUSTED_DIST_ALL_REDUCE(value, op=_TRUSTED_DIST_REDUCE_SUM)
    return float(value.item() / _TRUSTED_DIST_GET_WORLD_SIZE())


def _global_max(local_value: float | int, device: torch.device, dtype: torch.dtype) -> float:
    value = torch.tensor(local_value, dtype=dtype, device=device)
    _TRUSTED_DIST_ALL_REDUCE(value, op=_TRUSTED_DIST_REDUCE_MAX)
    return float(value.item())


def _global_min(local_value: float | int, device: torch.device, dtype: torch.dtype) -> float:
    value = torch.tensor(local_value, dtype=dtype, device=device)
    _TRUSTED_DIST_ALL_REDUCE(value, op=_TRUSTED_DIST_REDUCE_MIN)
    return float(value.item())


def _global_tensor_sum(value: torch.Tensor) -> torch.Tensor:
    """Sum a compact trusted CUDA tensor across the two ranks in place."""

    _TRUSTED_DIST_ALL_REDUCE(value, op=_TRUSTED_DIST_REDUCE_SUM)
    return value


def _compact_tensor_statistics(value: torch.Tensor, device: torch.device) -> dict[str, Any]:
    """Return finite global summary statistics without serializing tensor data.

    This runs only after the last warm-up backward pass.  All reductions are
    task-owned and only six FP64 scalars (plus shape metadata) ever reach JSON.
    """

    if value.numel() <= 0:
        raise RuntimeError("correctness auxiliary tensors must be non-empty")
    detached = value.detach()
    if detached.device != device:
        raise RuntimeError("correctness auxiliary tensor is not on the local CUDA device")
    if not bool(_TRUSTED_ISFINITE(detached).all()):
        raise RuntimeError("correctness auxiliary tensor is non-finite")
    # Reductions promote internally; they do not make a full FP64 copy of a
    # model-sized tensor.  The min/max pair makes cancellation in a sum alone
    # insufficient to spoof the compact witness.
    local = torch.stack(
        (
            detached.sum(dtype=torch.float64),
            torch.linalg.vector_norm(detached, ord=2, dtype=torch.float64).square(),
            detached.amin().to(dtype=torch.float64),
            detached.amax().to(dtype=torch.float64),
        )
    )
    summed = local[:2].clone()
    _global_tensor_sum(summed)
    minimum = local[2].clone()
    _TRUSTED_DIST_ALL_REDUCE(minimum, op=_TRUSTED_DIST_REDUCE_MIN)
    maximum = local[3].clone()
    _TRUSTED_DIST_ALL_REDUCE(maximum, op=_TRUSTED_DIST_REDUCE_MAX)
    shapes = _all_gather_objects(tuple(detached.shape), WORLD_SIZE_REQUIRED)
    if any(shape != shapes[0] for shape in shapes):
        raise RuntimeError("correctness auxiliary shape differs across ranks")
    global_numel = _global_integer(int(detached.numel()), device)
    return {
        "shape_per_rank": list(shapes[0]),
        "dtype_per_rank": str(detached.dtype),
        "global_numel": global_numel,
        "global_sum": float(summed[0].item()),
        "global_l2_sq": float(summed[1].item()),
        "global_min": float(minimum.item()),
        "global_max": float(maximum.item()),
        "finite": True,
    }


def _parameter_signature(fsdp_module: FSDP, device: torch.device) -> dict[str, float | int]:
    """Compact global parameter witness; safe outside the timed window."""

    local_numel = 0
    local_sum = torch.zeros((), dtype=torch.float64, device=device)
    local_l2_sq = torch.zeros((), dtype=torch.float64, device=device)
    for parameter in fsdp_module.parameters():
        if not parameter.requires_grad or parameter.numel() == 0:
            continue
        value = parameter.detach()
        if not bool(_TRUSTED_ISFINITE(value).all()):
            raise RuntimeError("trainable FSDP parameter is non-finite")
        local_numel += int(value.numel())
        local_sum.add_(value.sum(dtype=torch.float64))
        local_l2_sq.add_(
            torch.linalg.vector_norm(value, ord=2, dtype=torch.float64).square()
        )
    if local_numel <= 0:
        raise RuntimeError("rank owns no trainable FSDP parameter elements")
    aggregate = torch.stack((local_sum, local_l2_sq))
    _global_tensor_sum(aggregate)
    signature = {
        "global_numel": _global_integer(local_numel, device),
        "global_sum": float(aggregate[0].item()),
        "global_l2_sq": float(aggregate[1].item()),
    }
    if (
        signature["global_numel"] <= 0
        or not math.isfinite(float(signature["global_sum"]))
        or not math.isfinite(float(signature["global_l2_sq"]))
        or float(signature["global_l2_sq"]) <= 0.0
    ):
        raise RuntimeError("compact FSDP parameter signature is invalid")
    return signature


def _capture_post_backward_correctness(
    *,
    fsdp_module: FSDP,
    optimizer: torch.optim.Optimizer,
    aux: tuple[torch.Tensor, ...],
    device: torch.device,
) -> dict[str, Any]:
    """Capture task-owned aux/gradient evidence after backward, before step."""

    if len(aux) != len(_AUX_NAMES):
        raise RuntimeError("candidate did not return the complete ORPO aux tuple")
    aux_statistics = []
    for name, value in zip(_AUX_NAMES, aux, strict=True):
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"correctness auxiliary {name} is not a tensor")
        aux_statistics.append({"name": name, **_compact_tensor_statistics(value, device)})

    local_tensors = 0
    local_elements = 0
    local_grad_tensors = 0
    local_grad_elements = 0
    for parameter in fsdp_module.parameters():
        if not parameter.requires_grad or parameter.numel() == 0:
            continue
        local_tensors += 1
        local_elements += int(parameter.numel())
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(_TRUSTED_ISFINITE(gradient.detach()).all()):
            raise RuntimeError("a trainable FSDP parameter has non-finite gradient")
        local_grad_tensors += 1
        local_grad_elements += int(gradient.numel())
    global_tensors = _global_integer(local_tensors, device)
    global_elements = _global_integer(local_elements, device)
    global_grad_tensors = _global_integer(local_grad_tensors, device)
    global_grad_elements = _global_integer(local_grad_elements, device)
    if (
        global_tensors <= 0
        or global_elements <= 0
        or global_grad_tensors != global_tensors
        or global_grad_elements != global_elements
    ):
        raise RuntimeError("not every trainable FSDP shard received a finite gradient")

    # This is FSDP's official global norm path.  max_norm=inf cannot alter the
    # gradients and is called after backward, outside every timed step.
    global_grad_norm = _TRUSTED_FSDP_CLIP_GRAD_NORM(
        fsdp_module, max_norm=float("inf"), norm_type=2.0
    )
    grad_norm = float(global_grad_norm.detach().float().item())
    if not math.isfinite(grad_norm) or grad_norm <= 0.0:
        raise RuntimeError("FSDP global gradient norm must be finite and non-zero")
    return {
        "schema_version": CORRECTNESS_TELEMETRY_SCHEMA,
        "capture_point": "warmup_after_backward_before_optimizer_step",
        "aux": aux_statistics,
        "gradients": {
            "method": "FSDP.clip_grad_norm_(max_norm=inf,norm_type=2)",
            "global_l2_norm": grad_norm,
            "global_trainable_parameter_tensors": global_tensors,
            "global_trainable_parameter_elements": global_elements,
            "global_gradient_tensors": global_grad_tensors,
            "global_gradient_elements": global_grad_elements,
            "finite": True,
        },
        "parameter_before_step": _parameter_signature(fsdp_module, device),
        # ``optimizer`` is intentionally an argument: it makes the capture
        # site unambiguously tied to the task-owned optimizer instance.
        "optimizer_class": type(optimizer).__name__,
    }


def _capture_post_step_update_evidence(
    *,
    fsdp_module: FSDP,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    before: Mapping[str, Any],
    expected_step: int,
) -> dict[str, Any]:
    """Prove the trusted AdamW step consumed gradients without tensor dumps."""

    after = _parameter_signature(fsdp_module, device)
    local_active = 0
    local_state_entries = 0
    local_expected_steps = 0
    local_exp_avg_l2_sq = torch.zeros((), dtype=torch.float64, device=device)
    local_min_step = math.inf
    local_max_step = -math.inf
    for parameter in fsdp_module.parameters():
        if not parameter.requires_grad or parameter.numel() == 0:
            continue
        local_active += 1
        state = optimizer.state.get(parameter)
        if not isinstance(state, dict):
            continue
        step = state.get("step")
        if isinstance(step, torch.Tensor):
            step_value = float(step.detach().item())
        elif isinstance(step, (int, float)) and not isinstance(step, bool):
            step_value = float(step)
        else:
            continue
        exp_avg = state.get("exp_avg")
        if not isinstance(exp_avg, torch.Tensor) or exp_avg.numel() != parameter.numel():
            continue
        if not bool(_TRUSTED_ISFINITE(exp_avg.detach()).all()):
            raise RuntimeError("AdamW exp_avg state is non-finite")
        local_state_entries += 1
        local_expected_steps += int(step_value == float(expected_step))
        local_min_step = min(local_min_step, step_value)
        local_max_step = max(local_max_step, step_value)
        local_exp_avg_l2_sq.add_(
            torch.linalg.vector_norm(exp_avg.detach(), ord=2, dtype=torch.float64).square()
        )
    global_active = _global_integer(local_active, device)
    global_entries = _global_integer(local_state_entries, device)
    global_expected = _global_integer(local_expected_steps, device)
    aggregate = local_exp_avg_l2_sq.clone()
    _global_tensor_sum(aggregate)
    minimum_step = _global_min(local_min_step, device, torch.float64)
    maximum_step = _global_max(local_max_step, device, torch.float64)
    exp_avg_l2 = math.sqrt(float(aggregate.item()))
    if (
        global_active <= 0
        or global_entries != global_active
        or global_expected != global_active
        or not math.isfinite(exp_avg_l2)
        or exp_avg_l2 <= 0.0
        or minimum_step != float(expected_step)
        or maximum_step != float(expected_step)
    ):
        raise RuntimeError("task-owned AdamW did not record one valid optimizer step per shard")
    before_sum = float(before["global_sum"])
    before_l2_sq = float(before["global_l2_sq"])
    sum_delta = float(after["global_sum"]) - before_sum
    l2_sq_delta = float(after["global_l2_sq"]) - before_l2_sq
    return {
        "method": "trusted_AdamW_state_and_compact_parameter_signature",
        "expected_step": expected_step,
        "global_active_parameter_tensors": global_active,
        "global_state_entries": global_entries,
        "global_entries_at_expected_step": global_expected,
        "minimum_step": minimum_step,
        "maximum_step": maximum_step,
        "global_exp_avg_l2_norm": exp_avg_l2,
        "parameter_before": before,
        "parameter_after": after,
        "parameter_global_sum_delta": sum_delta,
        "parameter_global_l2_sq_delta": l2_sq_delta,
        "parameter_signature_changed": bool(sum_delta != 0.0 or l2_sq_delta != 0.0),
        "finite": True,
    }


def _capture_final_optimizer_update_witness(
    *,
    fsdp_module: FSDP,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    capture_parameter_after: Mapping[str, Any],
    expected_step: int,
    expected_active_parameter_tensors: int,
) -> dict[str, Any]:
    """Witness the final trusted update after timing and peak reads are done.

    This intentionally runs only after all measured-step elapsed values and
    CUDA peak counters have been captured.  Its compact reductions cannot
    affect either scored TPS or reported peak allocator telemetry.
    """

    if expected_step <= 0 or expected_active_parameter_tensors <= 0:
        raise RuntimeError("final optimizer witness received invalid expected topology")
    capture_numel = capture_parameter_after.get("global_numel")
    capture_sum = capture_parameter_after.get("global_sum")
    capture_l2_sq = capture_parameter_after.get("global_l2_sq")
    if (
        isinstance(capture_numel, bool)
        or not isinstance(capture_numel, int)
        or capture_numel <= 0
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (capture_sum, capture_l2_sq)
        )
    ):
        raise RuntimeError("capture parameter signature is invalid for final witness")
    final_signature = _parameter_signature(fsdp_module, device)
    if final_signature["global_numel"] != capture_numel:
        raise RuntimeError("final optimizer witness changed parameter topology")

    local_active = 0
    local_state_entries = 0
    local_expected_steps = 0
    local_exp_avg_l2_sq = torch.zeros((), dtype=torch.float64, device=device)
    local_exp_avg_sq_l2_sq = torch.zeros((), dtype=torch.float64, device=device)
    local_min_step = math.inf
    local_max_step = -math.inf
    for parameter in fsdp_module.parameters():
        if not parameter.requires_grad or parameter.numel() == 0:
            continue
        local_active += 1
        state = optimizer.state.get(parameter)
        if not isinstance(state, dict):
            continue
        step = state.get("step")
        if isinstance(step, torch.Tensor):
            step_value = float(step.detach().item())
        elif isinstance(step, (int, float)) and not isinstance(step, bool):
            step_value = float(step)
        else:
            continue
        exp_avg = state.get("exp_avg")
        exp_avg_sq = state.get("exp_avg_sq")
        if (
            not isinstance(exp_avg, torch.Tensor)
            or not isinstance(exp_avg_sq, torch.Tensor)
            or exp_avg.numel() != parameter.numel()
            or exp_avg_sq.numel() != parameter.numel()
        ):
            continue
        if not bool(_TRUSTED_ISFINITE(exp_avg.detach()).all()) or not bool(
            _TRUSTED_ISFINITE(exp_avg_sq.detach()).all()
        ):
            raise RuntimeError("AdamW final state is non-finite")
        local_state_entries += 1
        local_expected_steps += int(step_value == float(expected_step))
        local_min_step = min(local_min_step, step_value)
        local_max_step = max(local_max_step, step_value)
        local_exp_avg_l2_sq.add_(
            torch.linalg.vector_norm(exp_avg.detach(), ord=2, dtype=torch.float64).square()
        )
        local_exp_avg_sq_l2_sq.add_(
            torch.linalg.vector_norm(exp_avg_sq.detach(), ord=2, dtype=torch.float64).square()
        )
    global_active = _global_integer(local_active, device)
    global_entries = _global_integer(local_state_entries, device)
    global_expected = _global_integer(local_expected_steps, device)
    aggregate = torch.stack((local_exp_avg_l2_sq, local_exp_avg_sq_l2_sq))
    _global_tensor_sum(aggregate)
    minimum_step = _global_min(local_min_step, device, torch.float64)
    maximum_step = _global_max(local_max_step, device, torch.float64)
    exp_avg_l2 = math.sqrt(float(aggregate[0].item()))
    exp_avg_sq_l2 = math.sqrt(float(aggregate[1].item()))
    cumulative_sum_delta = float(final_signature["global_sum"]) - float(capture_sum)
    cumulative_l2_sq_delta = float(final_signature["global_l2_sq"]) - float(capture_l2_sq)
    cumulative_changed = bool(
        cumulative_sum_delta != 0.0 or cumulative_l2_sq_delta != 0.0
    )
    if (
        global_active != expected_active_parameter_tensors
        or global_entries != global_active
        or global_expected != global_active
        or minimum_step != float(expected_step)
        or maximum_step != float(expected_step)
        or not math.isfinite(exp_avg_l2)
        or not math.isfinite(exp_avg_sq_l2)
        or exp_avg_l2 <= 0.0
        or exp_avg_sq_l2 <= 0.0
        or not cumulative_changed
    ):
        raise RuntimeError("final AdamW optimizer witness is incomplete or unchanged")
    return {
        "capture_point": "after_timed_window_and_peak_reads",
        "method": "trusted_AdamW_final_state_and_compact_parameter_signature",
        "expected_step": expected_step,
        "global_active_parameter_tensors": global_active,
        "global_state_entries": global_entries,
        "global_entries_at_expected_step": global_expected,
        "minimum_step": minimum_step,
        "maximum_step": maximum_step,
        "global_exp_avg_l2_norm": exp_avg_l2,
        "global_exp_avg_sq_l2_norm": exp_avg_sq_l2,
        "parameter_at_capture_after": dict(capture_parameter_after),
        "parameter_final": final_signature,
        "cumulative_parameter_global_sum_delta": cumulative_sum_delta,
        "cumulative_parameter_global_l2_sq_delta": cumulative_l2_sq_delta,
        "cumulative_parameter_signature_changed": cumulative_changed,
        "finite": True,
    }


def _one_step(
    *,
    fsdp_module: FSDP,
    optimizer: torch.optim.Optimizer,
    batch,
    device: torch.device,
    measured: bool,
    capture_correctness: bool = False,
    expected_optimizer_step: int | None = None,
) -> tuple[float, float | None, int, int, dict[str, Any] | None]:
    # Both counters are recomputed by task-owned batch construction.  Scored
    # predicted completion tokens are the primary throughput denominator;
    # non-padding prompt+completion tokens are diagnostic telemetry only.
    global_scored_tokens = _global_integer(batch.scored_tokens, device)
    global_non_padding_tokens = _global_integer(batch.non_padding_tokens, device)
    optimizer.zero_grad(set_to_none=True)
    _TRUSTED_DIST_BARRIER()
    _TRUSTED_CUDA_SYNCHRONIZE(device)
    started = _TRUSTED_PERF_COUNTER() if measured else 0.0
    with _TRUSTED_AUTOCAST(device_type="cuda", dtype=torch.bfloat16):
        result = fsdp_module(batch.input_ids, batch.attention_mask, batch.labels)
    if result.loss.dtype != torch.float32 or not bool(_TRUSTED_ISFINITE(result.loss.detach())):
        raise RuntimeError("training loss must be a finite FP32 scalar")
    result.loss.backward()

    tied_gradient = fsdp_module.module.model.lm_head.weight.grad
    if tied_gradient is None:
        raise RuntimeError("trainable tied lm_head received no gradient")
    if not bool(_TRUSTED_ISFINITE(tied_gradient).all()):
        raise RuntimeError("trainable tied lm_head gradient is non-finite")
    correctness = None
    if capture_correctness:
        if measured or expected_optimizer_step is None or expected_optimizer_step <= 0:
            raise RuntimeError("correctness capture is valid only on a warm-up optimizer step")
        correctness = _capture_post_backward_correctness(
            fsdp_module=fsdp_module,
            optimizer=optimizer,
            aux=result.aux,
            device=device,
        )
    optimizer.step()
    if correctness is not None:
        correctness["optimizer_update"] = _capture_post_step_update_evidence(
            fsdp_module=fsdp_module,
            optimizer=optimizer,
            device=device,
            before=correctness["parameter_before_step"],
            expected_step=expected_optimizer_step,
        )
        # Avoid duplicate state in output: the post-step witness nests both
        # sides of the compact parameter signature.
        del correctness["parameter_before_step"]
    _TRUSTED_CUDA_SYNCHRONIZE(device)
    local_elapsed = _TRUSTED_PERF_COUNTER() - started if measured else None

    # Telemetry collectives are outside the measured boundary.  The elapsed
    # cohort uses the slowest rank, while loss is reported as the rank mean.
    elapsed = (
        _global_max(local_elapsed, device, torch.float64)
        if local_elapsed is not None
        else None
    )
    mean_loss = _global_mean(float(result.loss.detach()), device)
    if not math.isfinite(mean_loss):
        raise RuntimeError("rank-mean loss is non-finite")
    return mean_loss, elapsed, global_scored_tokens, global_non_padding_tokens, correctness


def run(args: argparse.Namespace) -> dict[str, Any] | None:
    rank, local_rank, device, identities = _validate_distributed_environment()
    _TRUSTED_MANUAL_SEED(args.seed)
    _TRUSTED_CUDA_MANUAL_SEED_ALL(args.seed)

    contract = ObjectiveContract(beta=args.beta, ignore_index=-100)
    total_steps = args.warmup_steps + args.timed_steps
    frozen_dataset = None
    if args.data_path is not None:
        frozen_dataset = load_frozen_preference_dataset(
            args.data_path,
            expected_partition=args.expected_data_partition,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_tensors_sha256=args.expected_tensors_sha256,
            ignore_index=contract.ignore_index,
        )
        frozen_dataset.require_capacity(
            total_steps=total_steps,
            pair_batch=args.pair_batch,
            world_size=WORLD_SIZE_REQUIRED,
        )
        cpu_batches = [
            frozen_dataset.batch_for(
                rank=rank,
                global_step=step,
                pair_batch=args.pair_batch,
                world_size=WORLD_SIZE_REQUIRED,
                ignore_index=contract.ignore_index,
            )
            for step in range(total_steps)
        ]
    else:
        cpu_batches = [
            make_synthetic_preference_batch(
                seed=args.seed,
                rank=rank,
                step=step,
                pair_batch=args.pair_batch,
                sequence_length=args.sequence_length,
                prompt_length=args.prompt_length,
                ignore_index=contract.ignore_index,
            )
            for step in range(total_steps)
        ]

    # Construct the trusted model before importing participant code.  The
    # optional model hook runs only against this validated instance and before
    # FSDP; task-owned invariant checks reject state-topology changes.
    model, artifact = load_pinned_model(args.model_path)
    objective, candidate_version, model_optimizer, model_patch = load_probe_objective(
        args.kind,
        args.lab_root,
        contract,
        requested_version=args.candidate_version,
    )
    model_optimization = apply_model_optimizer(
        model,
        model_optimizer,
        contract,
        model_patch=model_patch,
    )
    step_module = TrainStepModule(model, objective)
    fsdp_module = _build_fsdp_module(step_module, device)
    optimizer = _TRUSTED_ADAMW(
        fsdp_module.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        foreach=False,
        fused=False,
    )

    batches = [batch.to(device) for batch in cpu_batches]
    local_fingerprints = [batch.fingerprint for batch in batches]
    gathered_fingerprints = _all_gather_objects(local_fingerprints, WORLD_SIZE_REQUIRED)
    digest = hashlib.sha256()
    for per_rank in gathered_fingerprints:
        for fingerprint in per_rank:
            digest.update(fingerprint.encode("ascii"))
    batch_fingerprint = digest.hexdigest()

    gathered_record_indices: list[Any] | None = None
    if frozen_dataset is not None:
        local_record_indices = [list(batch.record_indices) for batch in batches]
        gathered_record_indices = _all_gather_objects(
            local_record_indices, WORLD_SIZE_REQUIRED
        )
        flattened = [
            record_index
            for per_rank in gathered_record_indices
            for per_step in per_rank
            for record_index in per_step
        ]
        required_records = frozen_dataset.required_records(
            total_steps=total_steps,
            pair_batch=args.pair_batch,
            world_size=WORLD_SIZE_REQUIRED,
        )
        if len(flattened) != required_records or len(set(flattened)) != required_records:
            raise RuntimeError("frozen global-step/rank record selections are not disjoint")
        if sorted(flattened) != list(range(required_records)):
            raise RuntimeError("frozen record selection differs from the task-owned schedule")

    warmup_losses: list[float] = []
    correctness_telemetry: dict[str, Any] | None = None
    # Qualification reductions can reserve temporary allocator blocks.  For a
    # formal-sized warmup cohort, capture one step before the end, release only
    # those free blocks, then run a plain final warmup step to re-settle the
    # allocator before peak reset/timing.  The one-warmup public quick probe
    # has no spare re-settle step and is explicitly diagnostic-only.
    capture_warmup_step = args.warmup_steps - 2 if args.warmup_steps >= 2 else 0
    memory_qualification = (
        "settled_after_final_plain_warmup"
        if args.warmup_steps >= 2
        else "diagnostic_capture_on_only_warmup"
    )
    for step in range(args.warmup_steps):
        loss, _, _, _, step_correctness = _one_step(
            fsdp_module=fsdp_module,
            optimizer=optimizer,
            batch=batches[step],
            device=device,
            measured=False,
            capture_correctness=(step == capture_warmup_step),
            expected_optimizer_step=step + 1,
        )
        warmup_losses.append(loss)
        if step_correctness is not None:
            step_correctness["capture_warmup_step_index"] = step
            step_correctness["capture_expected_optimizer_step"] = step + 1
            step_correctness["memory_qualification"] = memory_qualification
            step_correctness["post_capture_allocator_settled"] = args.warmup_steps >= 2
            correctness_telemetry = step_correctness
            if args.warmup_steps >= 2:
                # `_one_step` synchronized after its trusted optimizer step.
                # Releasing only currently-free capture scratch here keeps it
                # out of allocator high-water telemetry; the next plain
                # warmup repopulates normal training allocations before reset.
                _TRUSTED_CUDA_EMPTY_CACHE()
                _TRUSTED_CUDA_SYNCHRONIZE(device)
                _TRUSTED_DIST_BARRIER()
    if correctness_telemetry is None:
        raise RuntimeError("warm-up correctness telemetry was not captured")

    _TRUSTED_DIST_BARRIER()
    _TRUSTED_CUDA_SYNCHRONIZE(device)
    # Do not empty the allocator after warm-up.  The final plain warm-up has
    # restored the normal training allocation state, so resetting only the
    # peak counters preserves a representative warmed allocator while still
    # measuring absolute allocated/reserved high-water marks.
    _TRUSTED_CUDA_RESET_PEAK_MEMORY_STATS(device)

    losses: list[float] = []
    rank_max_seconds: list[float] = []
    scored_tokens: list[int] = []
    non_padding_tokens: list[int] = []
    for offset in range(args.timed_steps):
        loss, elapsed, scored, non_padding, step_correctness = _one_step(
            fsdp_module=fsdp_module,
            optimizer=optimizer,
            batch=batches[args.warmup_steps + offset],
            device=device,
            measured=True,
        )
        if step_correctness is not None:
            raise RuntimeError("timed step unexpectedly captured correctness telemetry")
        if elapsed is None or elapsed <= 0.0:
            raise RuntimeError("invalid measured step duration")
        losses.append(loss)
        rank_max_seconds.append(elapsed)
        scored_tokens.append(scored)
        non_padding_tokens.append(non_padding)

    local_allocated = _TRUSTED_CUDA_MAX_MEMORY_ALLOCATED(device)
    local_reserved = _TRUSTED_CUDA_MAX_MEMORY_RESERVED(device)
    peak_allocated = int(_global_max(local_allocated, device, torch.int64))
    peak_reserved = int(_global_max(local_reserved, device, torch.int64))
    rank_memory = _all_gather_objects(
        {
            "rank": rank,
            "peak_allocated_bytes": local_allocated,
            "peak_reserved_bytes": local_reserved,
        },
        WORLD_SIZE_REQUIRED,
    )
    capture_update = correctness_telemetry.get("optimizer_update")
    if not isinstance(capture_update, dict):
        raise RuntimeError("warm-up correctness telemetry lacks optimizer update evidence")
    capture_after = capture_update.get("parameter_after")
    active_tensors = capture_update.get("global_active_parameter_tensors")
    if not isinstance(capture_after, dict) or isinstance(active_tensors, bool) or not isinstance(
        active_tensors, int
    ):
        raise RuntimeError("warm-up optimizer evidence has invalid topology")
    # This comes strictly after both the timed loop and CUDA peak snapshots.
    # It is a correctness witness only, and cannot affect throughput or peaks.
    correctness_telemetry["final_optimizer_update"] = _capture_final_optimizer_update_witness(
        fsdp_module=fsdp_module,
        optimizer=optimizer,
        device=device,
        capture_parameter_after=capture_after,
        expected_step=total_steps,
        expected_active_parameter_tensors=active_tensors,
    )
    completion = _all_gather_objects(
        {"rank": rank, "pid": os.getpid(), "status": "PASS"}, WORLD_SIZE_REQUIRED
    )
    if sorted(item["rank"] for item in completion) != [0, 1] or any(
        item["status"] != "PASS" for item in completion
    ):
        raise RuntimeError("not every rank completed successfully")

    total_seconds = sum(rank_max_seconds)
    total_scored_tokens = sum(scored_tokens)
    total_non_padding_tokens = sum(non_padding_tokens)
    if frozen_dataset is not None:
        batch_metadata = {
            "source": "frozen_tldr",
            "partition": frozen_dataset.partition,
            "manifest_sha256": frozen_dataset.manifest_sha256,
            "tensors_sha256": frozen_dataset.tensors_sha256,
            "partition_record_count": frozen_dataset.record_count,
            "chosen_first": True,
            "row_order_on_disk": "all_chosen_then_all_rejected",
            "selection": "disjoint_contiguous_records_by_global_step_then_rank",
            "or_labels_contract": "completion_only",
            "nll_target_contract": "separate_full_sequence_tensor",
            "pair_batch_per_rank": args.pair_batch,
            "sequence_length": frozen_dataset.sequence_length,
            "selected_unique_record_count": total_steps
            * WORLD_SIZE_REQUIRED
            * args.pair_batch,
            "record_indices_by_rank_and_step": gathered_record_indices,
            "fingerprint_all_ranks": batch_fingerprint,
            "task_owned_token_counting": {
                "primary": "labels[:, 1:] != -100",
                "diagnostic": "attention_mask == 1",
            },
        }
    else:
        batch_metadata = {
            "source": "deterministic_synthetic_pretokenized_preference",
            "chosen_first": True,
            "or_labels_contract": "completion_only",
            "nll_target_contract": "separate_full_sequence_tensor",
            "pair_batch_per_rank": args.pair_batch,
            "sequence_length": args.sequence_length,
            "prompt_length": args.prompt_length,
            "seed": args.seed,
            "fingerprint_all_ranks": batch_fingerprint,
            "task_owned_token_counting": {
                "primary": "labels[:, 1:] != -100",
                "diagnostic": "attention_mask == 1",
            },
        }
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "classification": "PUBLIC_FULL_MODEL_SMOKE_DIAGNOSTIC_ONLY",
        "formal": False,
        "run_id": str(uuid.uuid4()),
        "objective_kind": args.kind,
        "candidate_version": candidate_version,
        "model_optimization": model_optimization,
        "world_size": WORLD_SIZE_REQUIRED,
        "fsdp": {
            "api": "FSDP1",
            "sharding_strategy": "FULL_SHARD",
            "use_orig_params": True,
            "parameter_dtype": "bfloat16",
            "reduce_dtype": "bfloat16",
        },
        "model": artifact,
        "batch": batch_metadata,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "warmup_rank_mean_losses": warmup_losses,
        "correctness_telemetry": correctness_telemetry,
        "timed_rank_mean_losses": losses,
        "rank_max_step_seconds": rank_max_seconds,
        "global_scored_tokens_per_step": scored_tokens,
        "global_non_padding_tokens_per_step": non_padding_tokens,
        "total_global_scored_tokens": total_scored_tokens,
        "total_global_non_padding_tokens": total_non_padding_tokens,
        "total_rank_max_seconds": total_seconds,
        "throughput_scored_tokens_per_second": total_scored_tokens / total_seconds,
        "throughput_non_padding_tokens_per_second_diagnostic": (
            total_non_padding_tokens / total_seconds
        ),
        "peak_allocated_bytes_max_rank": peak_allocated,
        "peak_reserved_bytes_max_rank": peak_reserved,
        "per_rank_memory": rank_memory,
        "rank_identities": identities,
        "rank_completion": completion,
    }
    _TRUSTED_DIST_BARRIER()
    if rank == 0:
        _write_json_atomic(args.output, payload)
        return payload
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("reference", "candidate"))
    parser.add_argument(
        "--model-path", type=Path, default=Path(os.environ.get("ORPO_MODEL_PATH", "/models/Qwen3-1.7B"))
    )
    parser.add_argument("--lab-root", type=Path, default=Path("/app/workdir/orpo_lab"))
    parser.add_argument("--candidate-version", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        help="partition directory containing manifest.json and tensors.safetensors",
    )
    parser.add_argument(
        "--expected-data-partition",
        "--expected-partition",
        dest="expected_data_partition",
        choices=("public", "private"),
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        "--expected-data-manifest-sha256",
        dest="expected_manifest_sha256",
    )
    parser.add_argument(
        "--expected-tensors-sha256",
        "--expected-data-sha256",
        "--expected-data-tensors-sha256",
        dest="expected_tensors_sha256",
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--pair-batch", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--prompt-length", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--timed-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()
    if args.warmup_steps < 1 or args.timed_steps < 1:
        parser.error("the public smoke requires at least one warm-up and timed step")
    if args.timed_steps > 20 or args.warmup_steps > 10:
        parser.error("public smoke step counts exceed the diagnostic safety bound")
    if args.pair_batch < 1:
        parser.error("pair batch must be positive")
    if args.candidate_version is not None and not 1 <= args.candidate_version <= 999:
        parser.error("candidate version must be in [1, 999]")
    if args.sequence_length > 2048:
        parser.error("public smoke sequence length exceeds 2048")
    if not 0.0 < args.learning_rate <= 1e-3:
        parser.error("learning rate must be in (0, 1e-3]")
    expected_data = (
        args.expected_data_partition,
        args.expected_manifest_sha256,
        args.expected_tensors_sha256,
    )
    if args.data_path is not None and any(value is None for value in expected_data):
        parser.error(
            "--data-path requires expected partition, manifest sha256, and tensors sha256"
        )
    if args.data_path is None and any(value is not None for value in expected_data):
        parser.error("expected frozen-data fields require --data-path")
    return args


def main() -> int:
    args = parse_args()
    initialized = False
    try:
        payload = run(args)
        initialized = _TRUSTED_DIST_IS_INITIALIZED()
        if payload is not None:
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        initialized = _TRUSTED_DIST_IS_INITIALIZED()
        rank = _TRUSTED_DIST_GET_RANK() if initialized else os.environ.get("RANK", "unknown")
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "classification": "PUBLIC_FULL_MODEL_SMOKE_DIAGNOSTIC_ONLY",
                    "formal": False,
                    "rank": rank,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        if initialized and _TRUSTED_DIST_IS_INITIALIZED():
            _TRUSTED_DIST_DESTROY_PROCESS_GROUP()


if __name__ == "__main__":
    raise SystemExit(main())
