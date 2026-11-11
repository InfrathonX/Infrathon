#!/usr/bin/env python3
"""Static and CPU-only validation for host-only Oracle treatment arms."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path


sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parent
FULL_ENTRY = HERE / "full_liger_qwen3/run_distributed_probe.py"
FULL_ADAPTER = HERE / "full_liger_qwen3/solutions/objective_v001/candidate.py"
STOCK_ADAPTER = HERE / "liger_oracle/solutions/objective_v001/candidate.py"


def _load_full_entry():
    spec = importlib.util.spec_from_file_location("host_full_liger_qwen3", FULL_ENTRY)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load full-Liger host entry")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_static_boundaries(module) -> dict[str, object]:
    stock_source = STOCK_ADAPTER.read_text(encoding="utf-8")
    stock_tree = ast.parse(stock_source)
    imported_names = {
        alias.name
        for node in ast.walk(stock_tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    if "LigerFusedLinearORPOLoss" not in imported_names:
        raise RuntimeError("stock objective-only arm no longer imports fused ORPO")
    if "apply_liger_kernel_to_qwen3" in stock_source:
        raise RuntimeError("stock objective-only arm unexpectedly patches the model")

    full_source = FULL_ADAPTER.read_text(encoding="utf-8")
    full_candidate_tree = ast.parse(full_source)
    full_imported_names = {
        alias.name
        for node in ast.walk(full_candidate_tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    if not {
        "LigerFusedLinearORPOLoss",
        "LigerRMSNorm",
        "liger_rotary_pos_emb",
        "LigerSwiGLUMLP",
        "Qwen3ModelPatch",
    }.issubset(full_imported_names):
        raise RuntimeError("full candidate lacks the pinned objective or PatchSpec symbols")
    patch_calls = [
        node
        for node in ast.walk(full_candidate_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Qwen3ModelPatch"
    ]
    if len(patch_calls) != 1:
        raise RuntimeError("full candidate must construct exactly one Qwen3ModelPatch")
    patch_keywords = {keyword.arg: keyword.value for keyword in patch_calls[0].keywords}
    expected_names = {
        "rope": ("name", "liger_rotary_pos_emb"),
        "rms_norm_forward": ("attribute", "LigerRMSNorm", "forward"),
        "swiglu_forward": ("attribute", "LigerSwiGLUMLP", "forward"),
    }
    for name, expected in expected_names.items():
        value = patch_keywords.get(name)
        if expected[0] == "name":
            valid = isinstance(value, ast.Name) and value.id == expected[1]
        else:
            valid = (
                isinstance(value, ast.Attribute)
                and value.attr == expected[2]
                and isinstance(value.value, ast.Name)
                and value.value.id == expected[1]
            )
        if not valid:
            raise RuntimeError(f"full candidate has unexpected {name} PatchSpec field")
    function_names = {
        node.name for node in full_candidate_tree.body if isinstance(node, ast.FunctionDef)
    }
    if "build_model_patch" not in function_names or "optimize_model" in function_names:
        raise RuntimeError("full candidate must use PatchSpec as its primary model path")

    full_entry_source = FULL_ENTRY.read_text(encoding="utf-8")
    if "apply_liger_kernel_to_qwen3" in full_entry_source:
        raise RuntimeError("full wrapper must not patch Qwen3 before the task loader")
    full_tree = ast.parse(full_entry_source)
    forbidden_training_owners = {
        "AdamW",
        "FullyShardedDataParallel",
        "FSDP",
        "load_frozen_preference_dataset",
        "make_synthetic_preference_batch",
        "optimizer.step",
    }
    observed_names = {
        node.id for node in ast.walk(full_tree) if isinstance(node, ast.Name)
    }
    observed_attributes = {
        node.attr for node in ast.walk(full_tree) if isinstance(node, ast.Attribute)
    }
    if forbidden_training_owners & (observed_names | observed_attributes):
        raise RuntimeError("full-Liger adapter duplicates task-owned training machinery")

    injected = module.build_driver_argv(["--output", "/tmp/prototype.json"])
    expected_prefix = [
        "--kind",
        "candidate",
        "--lab-root",
        str(module.ORACLE_ROOT),
        "--candidate-version",
        "1",
    ]
    if injected[: len(expected_prefix)] != expected_prefix:
        raise RuntimeError("full-Liger arm did not select its hook-enabled archive")
    for forbidden in ("--kind", "--lab-root", "--candidate-version"):
        try:
            module.build_driver_argv([forbidden, "unexpected"])
        except ValueError:
            pass
        else:
            raise RuntimeError(f"full-Liger arm accepted an override for {forbidden}")

    if (TASK_ROOT / "images/work/host_reference").exists() or (
        TASK_ROOT / "images/judge/host_reference"
    ).exists():
        raise RuntimeError("host-only references leaked into an image build context")
    return {
        "stock_objective_only_model_patch": False,
        "full_model_patch_declared_by_candidate_patch_spec": True,
        "full_model_attribute_writes_owned_by_task_runtime": True,
        "wrapper_preimport_model_patch": False,
        "full_arm_duplicates_training_driver": False,
        "wrapper_owned_driver_flags": sorted(module.OWNED_DRIVER_FLAGS),
        "participant_image_context_contains_host_reference": False,
    }


def _validate_stock_objective_cpu(module) -> dict[str, object]:
    module._prepend_import_paths()
    import torch
    from runtime.api import ObjectiveContract

    spec = importlib.util.spec_from_file_location("host_stock_liger_orpo", STOCK_ADAPTER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load stock fused-ORPO adapter")
    adapter = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = adapter
    spec.loader.exec_module(adapter)
    objective = adapter.build_objective(ObjectiveContract(beta=0.1, ignore_index=-100))
    if objective.implementation.__class__.__name__ != "LigerFusedLinearORPOLoss":
        raise RuntimeError("stock arm did not construct LigerFusedLinearORPOLoss")
    if sum(parameter.numel() for parameter in objective.parameters()) != 0:
        raise RuntimeError("stock objective adapter unexpectedly owns parameters")
    if torch.cuda.is_initialized():
        raise RuntimeError("constructing the stock objective unexpectedly initialized CUDA")
    return {
        "implementation": "LigerFusedLinearORPOLoss",
        "owned_parameters": 0,
        "cuda_initialized": False,
        "forward_executed": False,
    }


def main() -> int:
    module = _load_full_entry()
    static = _validate_static_boundaries(module)
    objective_cpu = _validate_stock_objective_cpu(module)
    cpu = module.validate_cpu_only()
    report = {
        "schema_version": 1,
        "status": "PASS",
        "classification": "HOST_ONLY_ORACLE_TREATMENT_VALIDATION",
        "formal": False,
        "static": static,
        "objective_cpu": objective_cpu,
        "cpu": cpu,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
