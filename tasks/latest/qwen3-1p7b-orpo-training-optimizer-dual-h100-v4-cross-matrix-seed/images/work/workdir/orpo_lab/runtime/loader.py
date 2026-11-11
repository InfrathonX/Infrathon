"""Versioned candidate loader owned by the task runtime."""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from .api import ModelOptimizer, ObjectiveContract, Qwen3ModelPatch


VERSION_RE = re.compile(r"objective_v([0-9]{3})")

# Resolve trusted loader/type handles before executing candidate module code.
# The source policy also forbids external attribute writes; these aliases keep
# the post-import validation independent of later module attribute mutation.
_TRUSTED_SPEC_FROM_FILE_LOCATION = importlib.util.spec_from_file_location
_TRUSTED_MODULE_FROM_SPEC = importlib.util.module_from_spec
_TRUSTED_MODULE_TYPE = torch.nn.Module
_TRUSTED_MODEL_PATCH_TYPE = Qwen3ModelPatch


@dataclass(frozen=True)
class LoadedObjective:
    version: int
    module: torch.nn.Module
    source_path: Path
    # Reserved for historical host-only calibration archives. Participant
    # admission rejects this export before the loader runs.
    optimize_model: ModelOptimizer | None
    model_patch: Qwen3ModelPatch | None


def selected_version(lab_root: Path) -> int:
    marker = lab_root / "FINAL_VERSION"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("FINAL_VERSION must be a regular file")
    raw = marker.read_text(encoding="utf-8").strip()
    if not raw.isascii() or not raw.isdigit():
        raise ValueError("FINAL_VERSION must contain one positive integer")
    version = int(raw)
    if version <= 0 or version > 999:
        raise ValueError("FINAL_VERSION is outside [1, 999]")
    return version


def load_objective(
    lab_root: Path,
    contract: ObjectiveContract,
    requested_version: int | None = None,
) -> LoadedObjective:
    lab_root = lab_root.resolve()
    version = selected_version(lab_root) if requested_version is None else requested_version
    source_path = lab_root / "solutions" / f"objective_v{version:03d}" / "candidate.py"
    if source_path.is_symlink() or not source_path.is_file():
        raise ValueError(f"missing candidate source for version {version}")
    spec = _TRUSTED_SPEC_FROM_FILE_LOCATION(
        f"mlsysforge_orpo_candidate_v{version:03d}", source_path
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot construct candidate module spec")
    module = _TRUSTED_MODULE_FROM_SPEC(spec)
    spec.loader.exec_module(module)
    build = getattr(module, "build_objective", None)
    if not callable(build):
        raise TypeError("candidate.py must export build_objective(contract)")
    objective = build(contract)
    if not isinstance(objective, _TRUSTED_MODULE_TYPE):
        raise TypeError("build_objective must return torch.nn.Module")
    if sum(parameter.numel() for parameter in objective.parameters()) != 0:
        raise ValueError("candidate objective may not own trainable parameters")
    optimize_model = getattr(module, "optimize_model", None)
    if optimize_model is not None and not callable(optimize_model):
        raise TypeError("candidate optimize_model export must be callable")
    build_model_patch = getattr(module, "build_model_patch", None)
    if build_model_patch is not None and not callable(build_model_patch):
        raise TypeError("candidate build_model_patch export must be callable")
    model_patch = None if build_model_patch is None else build_model_patch(contract)
    if model_patch is not None and not isinstance(model_patch, _TRUSTED_MODEL_PATCH_TYPE):
        raise TypeError("build_model_patch must return Qwen3ModelPatch or None")
    return LoadedObjective(version, objective, source_path, optimize_model, model_patch)
