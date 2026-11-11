"""Stable loader for the supplied model and versioned candidate modules."""

from __future__ import annotations

import importlib.util
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from transformers import AutoTokenizer

from .configuration_llada import LLaDAConfig
from .modeling_llada import LLaDAModelLM


VERSION_RE = re.compile(r"[1-9][0-9]{0,2}")


@dataclass(slots=True)
class LoadedCandidate:
    version: int
    module: ModuleType
    model: LLaDAModelLM
    tokenizer: Any
    engine: Any
    weight_fingerprint: str


def sampled_weight_fingerprint(model: LLaDAModelLM) -> str:
    digest = hashlib.sha256()
    parameters = list(model.named_parameters())
    selected = parameters[:: max(1, len(parameters) // 16)][:16]
    with torch.no_grad():
        for name, parameter in selected:
            flat = parameter.detach().reshape(-1)
            indices = torch.tensor(
                sorted({0, int(flat.numel() // 2), int(flat.numel() - 1)}),
                device=flat.device,
            )
            digest.update(name.encode("utf-8"))
            digest.update(flat.index_select(0, indices).float().cpu().numpy().tobytes())
    return digest.hexdigest()


def selected_version(lab_root: Path, version_override: int | None = None) -> tuple[int, Path]:
    text = (
        str(version_override)
        if version_override is not None
        else (lab_root / "FINAL_VERSION").read_text(encoding="utf-8").strip()
    )
    if VERSION_RE.fullmatch(text) is None:
        raise ValueError("FINAL_VERSION must contain one integer from 1 through 999")
    version = int(text)
    package = lab_root / "solutions" / f"decoder_v{version:03d}"
    if not package.is_dir() or package.is_symlink():
        raise ValueError(f"selected solution directory is missing: {package}")
    candidate = package / "candidate.py"
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError(f"selected candidate is missing: {candidate}")
    return version, candidate


def import_candidate(
    lab_root: Path, version_override: int | None = None
) -> tuple[int, ModuleType]:
    version, candidate_path = selected_version(lab_root, version_override)
    module_name = f"mlsysforge_decoder_v{version:03d}"
    spec = importlib.util.spec_from_file_location(module_name, candidate_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {candidate_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "build", None)):
        raise TypeError("candidate.py must export callable build(model, tokenizer)")
    return version, module


def load_model_and_candidate(
    lab_root: Path,
    model_path: Path,
    device: torch.device,
    version_override: int | None = None,
) -> LoadedCandidate:
    version, module = import_candidate(lab_root, version_override)
    config = LLaDAConfig.from_pretrained(str(model_path), local_files_only=True)
    config.flash_attention = False
    config.use_cache = False
    model = LLaDAModelLM.from_pretrained(
        str(model_path),
        config=config,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).eval().to(device)
    model.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=False,
    )
    weight_fingerprint = sampled_weight_fingerprint(model)
    engine = module.build(model, tokenizer)
    if not callable(getattr(engine, "generate", None)):
        raise TypeError("build() must return an object with generate(prompt_ids, spec)")
    return LoadedCandidate(
        version, module, model, tokenizer, engine, weight_fingerprint
    )
