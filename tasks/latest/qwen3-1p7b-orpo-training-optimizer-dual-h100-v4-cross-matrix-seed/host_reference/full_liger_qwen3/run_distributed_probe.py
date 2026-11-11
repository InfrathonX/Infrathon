#!/usr/bin/env python3
"""Host-only full-Liger entry point for the task-owned distributed probe.

This file deliberately does not implement a dataloader, optimizer, training
step, FSDP policy, timer, or metric.  It delegates to
``runtime.distributed_probe`` with a host-only candidate archive whose optional
model hook is invoked by the ordinary task loader after model construction and
before FSDP.  No model patch is applied by this wrapper.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable


sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parents[1]
REPOSITORY_ROOT = TASK_ROOT.parents[2]
TASK_LAB = TASK_ROOT / "images/work/workdir/orpo_lab"
TASK_DRIVER = TASK_LAB / "runtime/distributed_probe.py"
TASK_RUNTIME_FILES = (
    TASK_LAB / "runtime/api.py",
    TASK_LAB / "runtime/loader.py",
    TASK_LAB / "runtime/train_step.py",
    TASK_DRIVER,
)
ORACLE_ROOT = HERE
VENDOR_ROOT = REPOSITORY_ROOT / "vendor/Liger-Kernel"
VENDOR_SRC = VENDOR_ROOT / "src"

LIGER_COMMIT = "780e76b8115bfd7b78592569853b245a33cc7bf0"
ORACLE_VERSION = 1
PATCH_OPTIONS = {
    "rope": True,
    "rms_norm": True,
    "swiglu": True,
    # The task's external ORPO objective owns projection and loss.  Enabling
    # either model CE path would be redundant treatment contamination.
    "cross_entropy": False,
    "fused_linear_cross_entropy": False,
}
PINNED_SOURCE_SHA256 = {
    "src/liger_kernel/chunked_loss/__init__.py": (
        "279fe3367cd9e20666037f16e5fff8a1a6fbc4ca0d9355f2fb28775ff5e57feb"
    ),
    "src/liger_kernel/chunked_loss/orpo_loss.py": (
        "9eef54606d7a75c330f7796f1e2e3f858b3743414ad4a9e598c463ecea5853cb"
    ),
    "src/liger_kernel/transformers/__init__.py": (
        "15c32485311d0dc66d94e5137119506888ba49592f2d27fdafdfe3d2070264fb"
    ),
    "src/liger_kernel/transformers/monkey_patch.py": (
        "d8be36002c40a57392ceb26222997452fe1abaf5d77672cd641580248007a11b"
    ),
    "src/liger_kernel/transformers/rms_norm.py": (
        "f38c41bbaaca9fc437a88e111269cfcc9d0ac546103e4c19e5fe0757b5419e36"
    ),
    "src/liger_kernel/transformers/rope.py": (
        "f96f5a60b6b684c3a6986e7278772f3ec388e544dcf79ca563151d7719309a40"
    ),
    "src/liger_kernel/transformers/swiglu.py": (
        "b6ebc4b57c6592bf28aa38ff7f67865c702b65920a1491f9cc6c8e87f2011557"
    ),
}
ORACLE_ADAPTER_SHA256 = "490cbcee7a396c1c0b92da132b0cc7af2ddd4ea221fa796ef4348013ad891375"
OWNED_DRIVER_FLAGS = {"--kind", "--lab-root", "--candidate-version"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repository: Path) -> str:
    git_dir = repository / ".git"
    if git_dir.is_file():
        marker = git_dir.read_text(encoding="utf-8").strip()
        if not marker.startswith("gitdir: "):
            raise RuntimeError("unrecognized vendored Liger .git file")
        git_dir = (repository / marker.removeprefix("gitdir: ")).resolve()
    head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    if not head.startswith("ref: "):
        return head
    reference = head.removeprefix("ref: ")
    loose_ref = git_dir / reference
    if loose_ref.is_file():
        return loose_ref.read_text(encoding="ascii").strip()
    packed = git_dir / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="ascii").splitlines():
            if line and not line.startswith(("#", "^")):
                commit, name = line.split(" ", 1)
                if name == reference:
                    return commit
    raise RuntimeError("cannot resolve vendored Liger HEAD")


def verify_host_sources() -> dict[str, Any]:
    """Fail closed if the treatment is not backed by the pinned host sources."""

    if _git_head(VENDOR_ROOT) != LIGER_COMMIT:
        raise RuntimeError("vendored Liger checkout is not the pinned commit")
    observed: dict[str, str] = {}
    for relative, expected in PINNED_SOURCE_SHA256.items():
        actual = _sha256(VENDOR_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"pinned Liger source hash mismatch: {relative}")
        observed[relative] = actual
    adapter = ORACLE_ROOT / "solutions/objective_v001/candidate.py"
    if _sha256(adapter) != ORACLE_ADAPTER_SHA256:
        raise RuntimeError("full-Liger candidate adapter has changed")
    if not TASK_DRIVER.is_file():
        raise RuntimeError("task-owned distributed training driver is missing")
    runtime_hashes = {
        str(path.relative_to(TASK_LAB)): _sha256(path) for path in TASK_RUNTIME_FILES
    }
    return {
        "liger_commit": LIGER_COMMIT,
        "liger_source_sha256": observed,
        "full_candidate_adapter_sha256": ORACLE_ADAPTER_SHA256,
        "task_driver": str(TASK_DRIVER),
        "task_driver_sha256": _sha256(TASK_DRIVER),
        "task_runtime_sha256": runtime_hashes,
    }


def _prepend_import_paths() -> None:
    for path in (VENDOR_SRC, TASK_LAB):
        rendered = str(path)
        if rendered in sys.path:
            sys.path.remove(rendered)
        sys.path.insert(0, rendered)


def build_driver_argv(arguments: list[str]) -> list[str]:
    """Inject the immutable objective arm and reject conflicting overrides."""

    for argument in arguments:
        flag = argument.split("=", 1)[0]
        if flag in OWNED_DRIVER_FLAGS:
            raise ValueError(f"full-Liger launcher owns {flag}")
    return [
        "--kind",
        "candidate",
        "--lab-root",
        str(ORACLE_ROOT),
        "--candidate-version",
        str(ORACLE_VERSION),
        *arguments,
    ]


def load_task_driver() -> Callable[[], int]:
    """Import, but never copy, the task-owned training driver."""

    _prepend_import_paths()
    from runtime import distributed_probe

    if Path(distributed_probe.__file__).resolve() != TASK_DRIVER.resolve():
        raise RuntimeError("runtime.distributed_probe resolved outside the task lab")
    return distributed_probe.main


def _argument_value(arguments: list[str], flag: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == flag and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith(flag + "="):
            return argument.split("=", 1)[1]
    return None


def _annotate_rank_zero_output(arguments: list[str], sources: dict[str, Any]) -> None:
    if os.environ.get("RANK") != "0":
        return
    output = _argument_value(arguments, "--output")
    if output is None:
        return
    path = Path(output)
    payload = json.loads(path.read_text(encoding="utf-8"))
    optimization = payload.get("model_optimization", {})
    if not (
        optimization.get("exported") is True
        and optimization.get("invoked") is True
        and optimization.get("invariants_validated") is True
    ):
        raise RuntimeError("task driver did not attest the full-Liger model hook")
    attestation = {
        "treatment": "full_liger_qwen3_plus_fused_orpo",
        "model_patch": dict(PATCH_OPTIONS),
        "model_hook_execution": optimization,
        "model_forward_replaced": False,
        "objective": "stock_host_only_liger_fused_orpo_v001",
        "formal": False,
        "liger_commit": sources["liger_commit"],
        "liger_source_sha256": sources["liger_source_sha256"],
        "full_candidate_adapter_sha256": sources[
            "full_candidate_adapter_sha256"
        ],
        "task_driver_sha256": sources["task_driver_sha256"],
        "task_runtime_sha256": sources["task_runtime_sha256"],
    }
    payload["host_oracle_treatment"] = attestation
    temporary = path.with_name(path.name + f".full-liger-{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_probe(arguments: list[str]) -> int:
    sources = verify_host_sources()
    driver_arguments = build_driver_argv(arguments)
    driver_main = load_task_driver()
    original_argv = sys.argv
    try:
        sys.argv = [str(TASK_DRIVER), *driver_arguments]
        returncode = int(driver_main())
    finally:
        sys.argv = original_argv
    if returncode == 0:
        _annotate_rank_zero_output(driver_arguments, sources)
    return returncode


def validate_cpu_only() -> dict[str, Any]:
    """Exercise the ordinary loader/hook path on a tiny CPU Qwen3 model."""

    sources = verify_host_sources()
    _prepend_import_paths()
    import torch
    import transformers
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3 import modeling_qwen3

    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
    from runtime.api import ObjectiveContract
    from runtime.loader import load_objective
    from runtime.train_step import apply_model_optimizer

    if torch.cuda.is_initialized():
        raise RuntimeError("CPU validation must run before CUDA initialization")
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        tie_word_embeddings=True,
        use_cache=False,
    )
    torch.manual_seed(17)
    baseline = Qwen3ForCausalLM(config)
    baseline_schema = {name: tuple(value.shape) for name, value in baseline.state_dict().items()}
    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    torch.manual_seed(17)
    treatment = Qwen3ForCausalLM(config)
    original_causal_lm_forward = type(treatment).forward
    contract = ObjectiveContract(beta=0.1, ignore_index=-100)
    loaded = load_objective(ORACLE_ROOT, contract, requested_version=ORACLE_VERSION)
    if loaded.model_patch is None:
        raise RuntimeError("full-Liger candidate did not export build_model_patch")
    optimization = apply_model_optimizer(
        treatment,
        loaded.optimize_model,
        contract,
        model_patch=loaded.model_patch,
    )
    treatment_schema = {name: tuple(value.shape) for name, value in treatment.state_dict().items()}
    treatment_parameters = sum(parameter.numel() for parameter in treatment.parameters())
    if treatment_schema != baseline_schema or treatment_parameters != baseline_parameters:
        raise RuntimeError("Liger Qwen3 patch changed parameter names, shapes, or count")
    if treatment.lm_head.weight is not treatment.model.embed_tokens.weight:
        raise RuntimeError("Liger Qwen3 patch broke tied embedding/lm_head identity")
    if modeling_qwen3.apply_rotary_pos_emb is not liger_rotary_pos_emb:
        raise RuntimeError("Qwen3 RoPE patch did not take effect")
    norm_modules = [treatment.model.norm]
    mlp_modules = []
    for layer in treatment.model.layers:
        norm_modules.extend(
            (
                layer.input_layernorm,
                layer.post_attention_layernorm,
                layer.self_attn.q_norm,
                layer.self_attn.k_norm,
            )
        )
        mlp_modules.append(layer.mlp)
    if any(module.forward.__func__ is not LigerRMSNorm.forward for module in norm_modules):
        raise RuntimeError("Qwen3 RMSNorm instance patch did not take effect")
    if any(module.forward.__func__ is not LigerSwiGLUMLP.forward for module in mlp_modules):
        raise RuntimeError("Qwen3 SwiGLU instance patch did not take effect")
    if type(treatment).forward is not original_causal_lm_forward or "forward" in treatment.__dict__:
        raise RuntimeError("model fused-linear-CE forward must remain disabled for ORPO")
    if loaded.module.implementation.__class__.__name__ != "LigerFusedLinearORPOLoss":
        raise RuntimeError("full-Liger candidate did not construct stock fused ORPO")
    if torch.cuda.is_initialized():
        raise RuntimeError("static CPU validation unexpectedly initialized CUDA")
    patch = {
        "treatment": "full_liger_qwen3_plus_fused_orpo",
        "model_patch": dict(PATCH_OPTIONS),
        "patched_symbols": {
            "rope": f"{liger_rotary_pos_emb.__module__}.{liger_rotary_pos_emb.__name__}",
            "rms_norm": f"{LigerRMSNorm.__module__}.{LigerRMSNorm.__name__}",
            "swiglu": f"{LigerSwiGLUMLP.__module__}.{LigerSwiGLUMLP.__name__}",
        },
        "model_forward_replaced": False,
        "objective": "stock_host_only_liger_fused_orpo_v001",
        "runtime_model_optimization": optimization,
        "formal": False,
    }
    return {
        "schema_version": 1,
        "status": "PASS",
        "classification": "HOST_ONLY_STATIC_CPU_ORACLE_VALIDATION",
        "formal": False,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "tiny_unique_parameters": treatment_parameters,
        "state_schema_equal": True,
        "tied_lm_head": True,
        "cuda_initialized": False,
        "patch": patch,
        "sources": sources,
    }


def main() -> int:
    arguments = sys.argv[1:]
    if arguments == ["--validate-only"]:
        print(json.dumps(validate_cpu_only(), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    if "--validate-only" in arguments:
        raise SystemExit("--validate-only cannot be combined with training-driver arguments")
    return run_probe(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
