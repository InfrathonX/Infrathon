from __future__ import annotations

import shutil
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from conftest import LAB_ROOT
from runtime.api import ObjectiveContract, Qwen3ModelPatch
from runtime.loader import load_objective
from runtime.train_step import apply_model_optimizer


class TinyTiedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.embedding = torch.nn.Embedding(7, 3)
        self.block = torch.nn.Linear(3, 3, bias=False)
        self.lm_head = torch.nn.Linear(3, 7, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.register_buffer("scale", torch.ones(1))
        self.train()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head


class TinyNorm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3))
        self.variance_epsilon = 1e-6

    def forward(self, value):
        return value


class TinyLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = TinyNorm()
        self.post_attention_layernorm = TinyNorm()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_norm = TinyNorm()
        self.self_attn.k_norm = TinyNorm()
        self.mlp = torch.nn.Linear(3, 3, bias=False)


class TinyQwenLike(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_cache=False)
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(7, 3)
        self.model.layers = torch.nn.ModuleList([TinyLayer(), TinyLayer()])
        self.model.norm = TinyNorm()
        self.lm_head = torch.nn.Linear(3, 7, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.train()

    def get_input_embeddings(self) -> torch.nn.Module:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head


def _contract() -> ObjectiveContract:
    return ObjectiveContract(beta=0.1, ignore_index=-100)


def test_seed_candidate_has_no_imperative_model_hook() -> None:
    loaded = load_objective(LAB_ROOT, _contract())
    assert loaded.optimize_model is None
    assert loaded.model_patch is None
    model = TinyTiedModel()
    report = apply_model_optimizer(model, loaded.optimize_model, _contract())
    assert report["exported"] is False
    assert report["invoked"] is False
    assert report["return"] == "not_exported"
    assert report["invariants_validated"] is False


def test_absent_model_hook_is_an_explicit_noop() -> None:
    model = TinyTiedModel()
    report = apply_model_optimizer(model, None, _contract())
    assert report == {
        "abi": "Qwen3ModelPatch",
        "exported": False,
        "invoked": False,
        "return": "not_exported",
        "invariants_validated": False,
        "patch_spec": {"exported": False, "applied": False},
    }


def test_task_runtime_installs_declarative_qwen3_patch_spec(monkeypatch) -> None:
    from transformers.models.qwen3 import modeling_qwen3

    model = TinyQwenLike()
    original_rope = modeling_qwen3.apply_rotary_pos_emb
    monkeypatch.setattr(modeling_qwen3, "apply_rotary_pos_emb", original_rope)

    def patched_rope(*args, **kwargs):
        return args, kwargs

    def patched_norm(module, value):
        return value * module.weight

    def patched_mlp(module, value):
        return torch.nn.functional.linear(value, module.weight)

    patch = Qwen3ModelPatch(
        rope=patched_rope,
        rms_norm_forward=patched_norm,
        swiglu_forward=patched_mlp,
    )
    report = apply_model_optimizer(
        model,
        None,
        _contract(),
        model_patch=patch,
    )

    assert modeling_qwen3.apply_rotary_pos_emb is patched_rope
    norms = [model.model.norm]
    for layer in model.model.layers:
        norms.extend(
            (
                layer.input_layernorm,
                layer.post_attention_layernorm,
                layer.self_attn.q_norm,
                layer.self_attn.k_norm,
            )
        )
        assert layer.mlp.forward.__func__ is patched_mlp
    for norm in norms:
        assert norm.forward.__func__ is patched_norm
        assert norm.offset == 0.0
        assert norm.casting_mode == "llama"
        assert norm.in_place is True
        assert norm.row_mode is None
    assert report["return"] == "patch_spec_only"
    assert report["patch_spec"]["fields"] == [
        "rope",
        "rms_norm_forward",
        "swiglu_forward",
    ]
    assert report["patch_spec"]["attribute_writes_owned_by"] == "task_runtime"
    assert report["patch_spec"]["norm_module_count"] == 9
    assert report["patch_spec"]["mlp_module_count"] == 2
    assert report["invariants_validated"] is True


@pytest.mark.parametrize("return_same_model", [False, True])
def test_method_only_in_place_patch_preserves_model_topology(return_same_model: bool) -> None:
    model = TinyTiedModel()
    original_forward = model.block.forward

    def optimized_forward(block, value):
        return torch.nn.functional.linear(value, block.weight)

    def hook(candidate, contract):
        del contract
        candidate.block.forward = types.MethodType(optimized_forward, candidate.block)
        return candidate if return_same_model else None

    report = apply_model_optimizer(model, hook, _contract())
    assert model.block.forward != original_forward
    assert report["return"] == ("same_model" if return_same_model else "none")
    assert report["invariants_validated"] is True


def test_hook_cannot_replace_root_model() -> None:
    model = TinyTiedModel()

    with pytest.raises(TypeError, match="same model object or None"):
        apply_model_optimizer(model, lambda *_: TinyTiedModel(), _contract())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda model: setattr(
                model.block,
                "weight",
                torch.nn.Parameter(model.block.weight.detach().clone()),
            ),
            "replaced parameter",
        ),
        (
            lambda model: model.block.weight.requires_grad_(False),
            "requires_grad",
        ),
        (
            lambda model: setattr(
                model.block.weight,
                "data",
                model.block.weight.detach().double(),
            ),
            "dtype",
        ),
        (
            lambda model: setattr(
                model.block.weight,
                "data",
                model.block.weight.detach()[:2],
            ),
            "shape",
        ),
        (
            lambda model: model.register_buffer("scale", model.scale.clone()),
            "replaced buffer",
        ),
        (
            lambda model: model.register_buffer("extra", torch.zeros(1)),
            "buffer names",
        ),
        (lambda model: model.scale.add_(1), "buffer value"),
        (lambda model: model.eval(), "training mode"),
        (lambda model: setattr(model.config, "use_cache", True), "use_cache"),
        (
            lambda model: setattr(
                model.lm_head,
                "weight",
                torch.nn.Parameter(model.lm_head.weight.detach().clone()),
            ),
            "parameter|tied",
        ),
    ],
)
def test_hook_rejects_model_state_mutations(mutation, message: str) -> None:
    model = TinyTiedModel()

    def hook(candidate, contract):
        del contract
        mutation(candidate)
        return None

    with pytest.raises(ValueError, match=message):
        apply_model_optimizer(model, hook, _contract())


def test_loader_rejects_non_callable_optional_export(tmp_path: Path) -> None:
    lab = tmp_path / "orpo_lab"
    shutil.copytree(LAB_ROOT, lab)
    candidate = lab / "solutions/objective_v001/candidate.py"
    source = candidate.read_text(encoding="utf-8")
    candidate.write_text(source + "\noptimize_model = 17\n", encoding="utf-8")
    with pytest.raises(TypeError, match="optimize_model export must be callable"):
        load_objective(lab, _contract())
