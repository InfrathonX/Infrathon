"""Offline loaders for the frozen Qwen3 target and EAGLE-3 draft head."""
from __future__ import annotations

import hashlib
import json
import os
import weakref
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3ForCausalLM

from .api import DraftHead, EngineContract


TARGET_MODEL_ID = "Qwen/Qwen3-8B"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DRAFT_MODEL_ID = "Tengyunw/qwen3_8b_eagle3"
TARGET_REPO_DEFAULT = Path("/opt/models/qwen3-8b-hf")
HEAD_DIR_DEFAULT = Path("/opt/models/eagle3-head")
_LAST_TARGET: weakref.ReferenceType[Qwen3ForCausalLM] | None = None


def _policy() -> dict[str, Any]:
    path = Path(__file__).with_name("policy_public.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_contract(
    draft_revision: str,
    *,
    target_model_id: str = TARGET_MODEL_ID,
    draft_model_id: str = DRAFT_MODEL_ID,
    device: str = "cuda:0",
) -> EngineContract:
    if not draft_revision or not isinstance(draft_revision, str):
        raise ValueError("a pinned draft_revision is required")
    return EngineContract(
        target_model_id=target_model_id,
        target_revision=TARGET_REVISION,
        draft_model_id=draft_model_id,
        draft_revision=draft_revision,
        device=device,
        max_sequence_length=4096,
        max_request_tokens=3968,
    )


def load_target(contract: EngineContract) -> Qwen3ForCausalLM:
    """Load the exact target snapshot offline as the standard HF Qwen3 class."""
    global _LAST_TARGET
    if contract.target_model_id != TARGET_MODEL_ID or contract.target_revision != TARGET_REVISION:
        raise ValueError("contract does not name the frozen Qwen3-8B target revision")
    snapshot = _target_snapshot(contract)
    kwargs = dict(
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=contract.dtype,
        attn_implementation=_policy().get("attn_implementation", "eager"),
        device_map=contract.device,
    )
    model = Qwen3ForCausalLM.from_pretrained(snapshot, **kwargs).eval()
    model.requires_grad_(False)
    if model.config.vocab_size != contract.vocab_size or model.config.hidden_size != contract.hidden_size:
        raise ValueError("loaded target architecture disagrees with EngineContract")
    model._target_revision = contract.target_revision
    _LAST_TARGET = weakref.ref(model)
    return model


def load_tokenizer(contract: EngineContract):
    if contract.target_model_id != TARGET_MODEL_ID or contract.target_revision != TARGET_REVISION:
        raise ValueError("contract does not name the frozen Qwen3-8B target revision")
    snapshot = _target_snapshot(contract)
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)


def _target_snapshot(contract: EngineContract) -> Path:
    """Resolve only the frozen read-only repo mount (or an explicit smoke path)."""
    root = Path(os.environ.get("EAGLE3_TARGET_REPO", str(TARGET_REPO_DEFAULT))).resolve()
    snapshot = root if root.name == contract.target_revision else root / "snapshots" / contract.target_revision
    if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
        raise FileNotFoundError(f"frozen target snapshot is unavailable: {snapshot}")
    return snapshot


def default_head_dir() -> Path:
    root = Path(os.environ.get("EAGLE3_HEAD_DIR", str(HEAD_DIR_DEFAULT))).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"frozen draft-head mount is unavailable: {root}")
    return root


def _expected_head_shapes(config: dict[str, Any], contract: EngineContract) -> dict[str, tuple[int, ...]]:
    h = int(config["hidden_size"])
    target_h = int(config.get("target_hidden_size", contract.hidden_size))
    inter = int(config["intermediate_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config.get("num_key_value_heads", heads))
    head_dim = int(config.get("head_dim", h // heads))
    vocab = int(config["vocab_size"])
    draft_vocab = int(config["draft_vocab_size"])
    return {
        "d2t": (draft_vocab,),
        "fc.weight": (h, target_h * 3),
        "lm_head.weight": (draft_vocab, h),
        "midlayer.hidden_norm.weight": (h,),
        "midlayer.input_layernorm.weight": (h,),
        "midlayer.mlp.down_proj.weight": (h, inter),
        "midlayer.mlp.gate_proj.weight": (inter, h),
        "midlayer.mlp.up_proj.weight": (inter, h),
        "midlayer.post_attention_layernorm.weight": (h,),
        "midlayer.self_attn.k_proj.weight": (kv_heads * head_dim, h * 2),
        "midlayer.self_attn.o_proj.weight": (h, heads * head_dim),
        "midlayer.self_attn.q_proj.weight": (heads * head_dim, h * 2),
        "midlayer.self_attn.v_proj.weight": (kv_heads * head_dim, h * 2),
        "norm.weight": (h,),
        "t2d": (vocab,),
    }


def load_draft_head(path: str | Path, contract: EngineContract) -> DraftHead:
    """Load and validate a safetensors-only head, adding the target embedding.

    `load_target` must precede this function.  This preserves the owner ABI, whose
    draft loader signature does not itself carry the target module.
    """
    root = Path(path)
    weights_path = root / "model.safetensors"
    config_path = root / "config.json"
    if not weights_path.is_file() or not config_path.is_file():
        raise ValueError("draft directory must contain config.json and model.safetensors")
    for manifest_name in ("MANIFEST.json", "sha256s.json"):
        manifest_path = root / manifest_name
        if manifest_path.is_file():
            pinned = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("revision", ""))
            if pinned != contract.draft_revision:
                raise ValueError(f"draft revision {pinned!r} disagrees with contract")
            break
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("vocab_size", -1)) != contract.vocab_size:
        raise ValueError("draft config vocab_size disagrees with contract")
    if int(config.get("draft_vocab_size", -1)) != contract.draft_vocab_size:
        raise ValueError("draft config draft_vocab_size disagrees with contract")
    if int(config.get("hidden_size", -1)) != contract.hidden_size:
        raise ValueError("draft config hidden_size disagrees with contract")

    expected = _expected_head_shapes(config, contract)
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        actual = set(handle.keys())
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        if missing:
            raise ValueError(f"draft safetensors missing required keys: {missing}")
        if extra:
            raise ValueError(f"draft safetensors has unexpected keys: {extra}")
        for name, shape in expected.items():
            got = tuple(handle.get_slice(name).get_shape())
            if got != shape:
                raise ValueError(f"draft tensor {name} shape {got} != {shape}")
        tensors = {name: handle.get_tensor(name) for name in expected}

    for name, tensor in tensors.items():
        expected_dtype = torch.int64 if name == "d2t" else torch.bool if name == "t2d" else contract.dtype
        if tensor.dtype != expected_dtype:
            raise ValueError(f"draft tensor {name} dtype {tensor.dtype} != {expected_dtype}")
    target = _LAST_TARGET() if _LAST_TARGET is not None else None
    if target is None:
        raise RuntimeError("load_target(contract) must be called before load_draft_head")
    embedding = target.get_input_embeddings().weight.detach().clone()
    if tuple(embedding.shape) != (contract.vocab_size, contract.hidden_size):
        raise ValueError("target embedding shape disagrees with contract")
    tensors["embed_tokens.weight"] = embedding
    frozen = {}
    for name, tensor in tensors.items():
        moved = tensor.detach().to(contract.device)
        moved.requires_grad_(False)
        frozen[name] = moved
    runtime_config = dict(config)
    runtime_config["_runtime_head_sha256"] = _sha256(weights_path)
    runtime_config["_runtime_target_revision_for_embedding"] = contract.target_revision
    runtime_config["_runtime_source_revision"] = contract.draft_revision
    return DraftHead(MappingProxyType(runtime_config), MappingProxyType(frozen))
