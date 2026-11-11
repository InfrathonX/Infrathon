#!/usr/bin/env python3
"""Gold-free candidate worker for the private LLaDA decoder Judge.

This process owns the model and imports untrusted candidate code, but it never
opens or receives hidden answers, tests, benchmark labels, row identifiers, or
the quality reference.  The root controller retains those values in a separate
process and communicates only one prompt plus its public generation shape.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pwd
import sys
from pathlib import Path
from typing import Any, TextIO

import torch
from transformers import AutoTokenizer


# Bind protocol and CUDA primitives before importing any candidate module.
_JSON_DUMPS = json.dumps
_JSON_LOADS = json.loads
_CUDA_SYNCHRONIZE = torch.cuda.synchronize
_RESET_PEAK = torch.cuda.reset_peak_memory_stats
_MAX_ALLOCATED = torch.cuda.max_memory_allocated
_MAX_RESERVED = torch.cuda.max_memory_reserved
_TORCH_EQUAL = torch.equal


def _sanitize_environment() -> None:
    allowed = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "CUDA_VISIBLE_DEVICES",
            "HF_DATASETS_OFFLINE",
            "HF_HUB_DISABLE_TELEMETRY",
            "HF_HUB_OFFLINE",
            "LD_LIBRARY_PATH",
            "PATH",
            "TOKENIZERS_PARALLELISM",
            "TRANSFORMERS_OFFLINE",
            "TRITON_CACHE_DIR",
            "TRITON_LIBCUDA_PATH",
        }
    }
    allowed.update({"HOME": "/tmp", "PYTHONDONTWRITEBYTECODE": "1"})
    os.environ.clear()
    os.environ.update(allowed)


def _protocol_writer() -> TextIO:
    protocol_fd = os.dup(1)
    sink = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(sink, 1)
        os.dup2(sink, 2)
    finally:
        os.close(sink)
    return os.fdopen(protocol_fd, "w", encoding="utf-8", buffering=1)


def _emit(writer: TextIO, payload: dict[str, Any]) -> None:
    writer.write(_JSON_DUMPS(payload, ensure_ascii=True, sort_keys=True, allow_nan=False) + "\n")
    writer.flush()


def _drop_to_agent() -> None:
    if os.geteuid() == 0:
        account = pwd.getpwnam("agent")
        os.setgroups([])
        os.setresgid(account.pw_gid, account.pw_gid, account.pw_gid)
        os.setresuid(account.pw_uid, account.pw_uid, account.pw_uid)
    libc = ctypes.CDLL(None, use_errno=True)
    # PR_SET_DUMPABLE=0 blocks ptrace/core access; PR_SET_NO_NEW_PRIVS=1
    # prevents regaining privilege through any executable path.
    if libc.prctl(4, 0, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_DUMPABLE failed")
    if libc.prctl(38, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_NO_NEW_PRIVS failed")
    os.umask(0o077)


class ForwardAudit:
    """Count the supplied model and every transformer block before build()."""

    def __init__(self, model: Any) -> None:
        self.forward_calls = 0
        self.layer_calls = [0] * int(model.config.n_layers)
        original_forward = model.forward

        def counted(*args: Any, **kwargs: Any):
            self.forward_calls += 1
            return original_forward(*args, **kwargs)

        model.forward = counted
        blocks = list(model.model.transformer.blocks)
        if len(blocks) != len(self.layer_calls):
            raise ValueError("unexpected backbone block count")
        self._handles = [
            block.register_forward_hook(self._layer_hook(index))
            for index, block in enumerate(blocks)
        ]

    def _layer_hook(self, index: int):
        def hook(_module: Any, _inputs: Any, _output: Any) -> None:
            self.layer_calls[index] += 1

        return hook

    def snapshot(self) -> tuple[int, list[int]]:
        return self.forward_calls, list(self.layer_calls)

    def delta(self, before: tuple[int, list[int]]) -> tuple[int, list[int]]:
        return (
            self.forward_calls - before[0],
            [now - old for now, old in zip(self.layer_calls, before[1])],
        )


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _make_nvml_used_probe():
    """Bind NVML before importing candidate-controlled Python."""
    library = ctypes.CDLL("libnvidia-ml.so.1")
    library.nvmlInit_v2.restype = ctypes.c_int
    library.nvmlDeviceGetCount_v2.argtypes = [ctypes.POINTER(ctypes.c_uint)]
    library.nvmlDeviceGetCount_v2.restype = ctypes.c_int
    library.nvmlDeviceGetHandleByIndex_v2.argtypes = [
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
    library.nvmlDeviceGetMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_NvmlMemory),
    ]
    library.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
    if library.nvmlInit_v2() != 0:
        raise RuntimeError("NVML initialization failed")
    count = ctypes.c_uint()
    if library.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0 or count.value != 1:
        raise RuntimeError("Judge must expose exactly one NVML device")
    handle = ctypes.c_void_p()
    if library.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != 0:
        raise RuntimeError("NVML could not resolve the sole Judge device")

    def used_bytes() -> int:
        memory = _NvmlMemory()
        if library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) != 0:
            raise RuntimeError("NVML memory query failed")
        return int(memory.used)

    return used_bytes


def _validate_model_state(model: Any) -> None:
    if torch.cuda.device_count() != 1:
        raise ValueError("Judge worker must see exactly one CUDA device")
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("model must remain inference-only and untrained")
    if any(parameter.dtype != torch.bfloat16 for parameter in model.parameters()):
        raise ValueError("model parameters must remain BF16")
    if bool(getattr(model, "is_quantized", False)) or getattr(
        model.config, "quantization_config", None
    ) is not None:
        raise ValueError("quantized model weights are outside this task")


def _load(lab_root: Path, model_path: Path):
    sys.path.insert(0, str(lab_root))
    from runtime.configuration_llada import LLaDAConfig
    from runtime.loader import import_candidate, sampled_weight_fingerprint
    from runtime.modeling_llada import LLaDAModelLM

    config = LLaDAConfig.from_pretrained(str(model_path), local_files_only=True)
    config.flash_attention = False
    config.use_cache = False
    model = LLaDAModelLM.from_pretrained(
        str(model_path),
        config=config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).eval().to(torch.device("cuda"))
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    baseline_fingerprint = sampled_weight_fingerprint(model)
    _validate_model_state(model)
    audit = ForwardAudit(model)

    # No candidate-controlled Python executes while this process is root.
    _sanitize_environment()
    _drop_to_agent()
    version, module = import_candidate(lab_root)
    engine = module.build(model, tokenizer)
    if not callable(getattr(engine, "generate", None)):
        raise TypeError("build() must return an object with generate(prompt_ids, spec)")
    if sampled_weight_fingerprint(model) != baseline_fingerprint:
        raise ValueError("candidate mutated sampled model weights during build")
    _validate_model_state(model)
    return version, model, tokenizer, engine, audit, baseline_fingerprint, sampled_weight_fingerprint


def _prepare(request: dict[str, Any]):
    from runtime.api import GenerationSpec

    allowed = {"op", "prompt_ids", "gen_length", "block_length", "mask_id"}
    if set(request) != allowed:
        raise ValueError("prepare command has unexpected fields")
    prompt_ids = request["prompt_ids"]
    if (
        not isinstance(prompt_ids, list)
        or not prompt_ids
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token >= 126464
            for token in prompt_ids
        )
    ):
        raise ValueError("prompt_ids must be a nonempty vocabulary-id list")
    prompt = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
    spec = GenerationSpec(
        gen_length=int(request["gen_length"]),
        block_length=int(request["block_length"]),
        mask_id=int(request["mask_id"]),
    )
    spec.validate(int(prompt.shape[1]))
    _CUDA_SYNCHRONIZE()
    return prompt, spec


def _run(
    engine: Any,
    audit: ForwardAudit,
    prompt: torch.Tensor,
    spec: Any,
    nvml_used_bytes: Any,
) -> dict[str, Any]:
    from runtime.api import VOCAB_SIZE, validate_result

    before = audit.snapshot()
    _RESET_PEAK()
    _CUDA_SYNCHRONIZE()
    result = engine.generate(prompt, spec)
    _CUDA_SYNCHRONIZE()
    validate_result(result, prompt, spec)
    nfe, layer_delta = audit.delta(before)
    if nfe <= 0 or any(count != nfe for count in layer_delta):
        raise ValueError("model/layer forward coverage mismatch")
    tokens = result.token_ids
    if not _TORCH_EQUAL(tokens[:, : prompt.shape[1]], prompt):
        raise ValueError("candidate modified prompt tokens")
    generated = tokens[0, prompt.shape[1] :]
    generated_ids = generated.detach().to(device="cpu", dtype=torch.long).tolist()
    if len(generated_ids) != int(spec.gen_length):
        raise ValueError("worker generated length mismatch")
    if any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or token < 0
        or token >= VOCAB_SIZE
        or token == int(spec.mask_id)
        for token in generated_ids
    ):
        raise ValueError("worker generated invalid token ids")
    allocated_bytes = int(_MAX_ALLOCATED())
    reserved_bytes = int(_MAX_RESERVED())
    device_used_bytes = int(nvml_used_bytes())
    if not 0 < allocated_bytes <= reserved_bytes <= device_used_bytes:
        raise ValueError("allocated/reserved/NVML memory ordering is invalid")
    return {
        "ok": True,
        "generated_ids": generated_ids,
        "nfe": nfe,
        "layer_calls": layer_delta,
        "torch_peak_allocated_bytes": allocated_bytes,
        "torch_peak_reserved_bytes": reserved_bytes,
        # Informational post-run point only. The trusted controller samples the
        # complete run window concurrently and supplies the authoritative peak.
        "nvml_device_used_bytes_postrun": device_used_bytes,
        "peak_allocated_gib": allocated_bytes / 2**30,
        "peak_reserved_gib": reserved_bytes / 2**30,
        "nvml_device_used_gib": device_used_bytes / 2**30,
    }


def serve(lab_root: Path, model_path: Path) -> int:
    writer = _protocol_writer()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Judge worker must see exactly one CUDA device")
        nvml_used_bytes = _make_nvml_used_probe()
        (
            version,
            model,
            tokenizer,
            engine,
            audit,
            baseline_fingerprint,
            fingerprint,
        ) = _load(lab_root.resolve(), model_path.resolve())
        _emit(
            writer,
            {
                "ok": True,
                "event": "ready",
                "version": version,
                "layers": len(audit.layer_calls),
                "weight_fingerprint": baseline_fingerprint,
            },
        )
        prepared = None
        for line in sys.stdin:
            request = _JSON_LOADS(line)
            if not isinstance(request, dict) or not isinstance(request.get("op"), str):
                raise ValueError("invalid worker command")
            operation = request["op"]
            if operation == "prepare":
                prepared = _prepare(request)
                _emit(
                    writer,
                    {
                        "ok": True,
                        "event": "prepared",
                        "prompt_length": int(prepared[0].shape[1]),
                    },
                )
            elif operation == "run":
                if set(request) != {"op"} or prepared is None:
                    raise ValueError("run requires exactly one prepared request")
                prompt, spec = prepared
                prepared = None
                _validate_model_state(model)
                _emit(writer, _run(engine, audit, prompt, spec, nvml_used_bytes))
                _validate_model_state(model)
            elif operation == "integrity":
                if set(request) != {"op"}:
                    raise ValueError("integrity command has unexpected fields")
                _validate_model_state(model)
                _emit(
                    writer,
                    {
                        "ok": True,
                        "event": "integrity",
                        "weight_fingerprint": fingerprint(model),
                    },
                )
            elif operation == "close":
                if set(request) != {"op"}:
                    raise ValueError("close command has unexpected fields")
                _emit(writer, {"ok": True, "event": "closed"})
                return 0
            else:
                raise ValueError("unknown worker command")
    except BaseException as exc:
        _emit(
            writer,
            {
                "ok": False,
                "event": "fatal",
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            },
        )
        return 1
    finally:
        writer.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    return serve(args.lab_root, args.model_path)


if __name__ == "__main__":
    raise SystemExit(main())
