"""v3 scored real layers: source table, recipe binding, verifier, wiring."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from conftest import TASK_ROOT, load_module

ROOT = TASK_ROOT
RECIPES = ROOT / "fixtures/recipe/real_layers"
runtime = load_module("qattn_v3_runtime_real_layers", "runtime/quant_attention_runtime.py")
verifier = load_module("qattn_v3_verifier_real_layers", "images/judge/verify_quant_attention.py")
worker = load_module("qattn_v3_worker_real_layers", "images/judge/candidate_worker.py")
adapter = load_module("qattn_v3_adapter_real_layers", "images/judge/run_judge.py")
generator = load_module("qattn_v3_generator_real_layers", "scripts/generate_final_manifest.py")
entry_text = (ROOT / "scripts/sforge_judge_entry.py").read_text()


def test_source_tables_agree() -> None:
    assert set(runtime.REAL_SOURCES) == set(verifier.REAL_SOURCES) == set(generator.REAL_SOURCES)
    for source, (_name, _rid, _dir, geometry) in runtime.REAL_SOURCES.items():
        v_geometry, v_layers = verifier.REAL_SOURCES[source]
        g = generator.REAL_SOURCES[source]
        assert geometry == v_geometry == g[:6]
        assert v_layers == g[6]
    assert sum(len(layers) for _g, layers in verifier.REAL_SOURCES.values()) == 139
    assert runtime.REAL_LAYER_P999_MULT == verifier.REAL_LAYER_P999_MULT == 2.5
    assert runtime.P999_MULT == 2.0  # synthetic and edge cases keep the v1 multiplier


def test_real_layer_limits_only_relax_row_p999() -> None:
    ref = {"rel_l1": 0.01, "worst_head_rel_l1": 0.02, "row_rel_l2_p999": 0.04, "row_rel_l2_max": 0.1}
    v1 = runtime.realised_limits(ref)
    v2 = runtime.realised_limits(ref, runtime.REAL_LAYER_P999_MULT)
    assert v2["row_rel_l2_p999_limit"] == pytest.approx(0.1) and v1["row_rel_l2_p999_limit"] == pytest.approx(0.08)
    for key in ("rel_l1_limit", "worst_head_rel_l1_limit", "row_rel_l2_max_limit"):
        assert v1[key] == v2[key]


@pytest.mark.parametrize("source", sorted(runtime.REAL_SOURCES))
def test_recipes_pin_models_shapes_and_every_captured_layer_band(source: str) -> None:
    name, recipe_id, capture_dir, (b, hq, hkv, d, l, causal) = runtime.REAL_SOURCES[source]
    recipe, digest = runtime.load_real_recipe(source)
    assert digest == hashlib.sha256((RECIPES / name).read_bytes()).hexdigest()
    assert recipe["recipe_id"] == recipe_id and recipe["capture_dir"] == capture_dir
    assert recipe["tensors"]["q"] == [b, l, hq, d] and recipe["tensors"]["k"] == [b, l, hkv, d]
    assert recipe["attention"]["is_causal"] is causal
    assert recipe["attention"]["softmax_scale"] == pytest.approx(d ** -0.5)
    assert recipe["model"]["files"] and all(re.fullmatch(r"[0-9a-f]{64}", v) for v in recipe["model"]["files"].values())
    layers = runtime.captured_layers(recipe)
    assert set(verifier.REAL_SOURCES[source][1]) <= set(layers)
    assert sorted(recipe["sanity_bands"]["layers"]) == [f"{i:02d}" for i in layers]
    for band in recipe["sanity_bands"]["layers"].values():
        for key in ("q", "k", "v"):
            assert 0 < band[key]["rms_min"] < band[key]["rms_max"]
            assert 0 < band[key]["abs_max_min"] < band[key]["abs_max_max"]
    assert recipe["tensors"]["layout"] == "B,L,H,D" and recipe["tensors"]["dtype"] == "torch.bfloat16"
    assert "hf_" not in (RECIPES / name).read_text()


def test_new_recipes_follow_their_sources() -> None:
    wan, _ = runtime.load_real_recipe("wan14b-720p")
    assert wan["generation"]["call"]["height"] == 720 and wan["generation"]["call"]["num_frames"] == 81
    assert wan["capture"]["calls_per_step"] == 2  # conditional pass only
    long, _ = runtime.load_real_recipe("llama3-262k")
    assert long["text"]["tokens"] == 262144 and long["attention"]["is_causal"] is True
    sd, _ = runtime.load_real_recipe("sd35m-1024")
    assert sd["tensors"]["q"] == [2, 4429, 24, 64]  # 64x64 image tokens + 333 text tokens, CFG batch


def _capture(tmp_path: Path, source: str, *, recipe_sha: str | None = None, **overrides) -> Path:
    recipe, digest = runtime.load_real_recipe(source)
    directory = tmp_path / "real_layers" / recipe["capture_dir"]
    directory.mkdir(parents=True)
    manifest = {"schema_version": 1, "recipe_id": recipe["recipe_id"], "recipe_sha256": recipe_sha or digest,
                "capture_dir": recipe["capture_dir"], "files": {}}
    manifest.update(overrides)
    (directory / "capture.manifest.json").write_text(json.dumps(manifest))
    return directory


def test_capture_binding_accepts_the_shipped_recipe(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SAGE_QATTN_FIXTURE_DIR", str(tmp_path))
    directory = _capture(tmp_path, "llama31-16k")
    recipe, digest = runtime.load_real_recipe("llama31-16k")
    assert runtime._real_capture("llama31-16k", recipe, digest)[0] == directory


@pytest.mark.parametrize("mutation", ["recipe_sha", "recipe_id", "capture_dir", "missing"])
def test_capture_binding_fails_closed(tmp_path, monkeypatch, mutation: str) -> None:
    monkeypatch.setenv("SAGE_QATTN_FIXTURE_DIR", str(tmp_path))
    if mutation == "recipe_sha":
        _capture(tmp_path, "wan14b-720p", recipe_sha="0" * 64)
    elif mutation == "recipe_id":
        _capture(tmp_path, "wan14b-720p", recipe_id="rl-other")
    elif mutation == "capture_dir":
        _capture(tmp_path, "wan14b-720p", capture_dir="elsewhere")
    recipe, digest = runtime.load_real_recipe("wan14b-720p")
    with pytest.raises(RuntimeError):
        runtime._real_capture("wan14b-720p", recipe, digest)


def test_a_recipe_with_other_geometry_is_rejected(tmp_path, monkeypatch) -> None:
    for name, *_ in runtime.REAL_SOURCES.values():
        data = json.loads((RECIPES / name).read_text())
        data["tensors"]["q"][1] -= 1
        (tmp_path / name).write_text(json.dumps(data))
    monkeypatch.setenv("SAGE_QATTN_REAL_RECIPE_DIR", str(tmp_path))
    runtime._REAL_RECIPE_CACHE.clear()
    try:
        with pytest.raises(RuntimeError):
            runtime.load_real_recipe("sd35m-1024")
    finally:
        runtime._REAL_RECIPE_CACHE.clear()


@pytest.mark.parametrize("layer", ["sd35m-1024/24", "sd35m-1024/7", "wan14b-720p/00", "llama31-16k/00"])
def test_real_case_identity_must_match_its_source(layer: str) -> None:
    source = layer.partition("/")[0]
    b, hq, hkv, d, l, causal = runtime.REAL_SOURCES[source][3]
    case = {"case_id": "x", "family": "real", "tier": "real", "model": source, "b": b, "h_q": hq, "h_kv": hkv,
            "d": d, "l": l, "is_causal": causal, "regime": "real", "seed": 1, "gated": True, "layer": layer}
    if layer == "sd35m-1024/7":  # single-digit layer numbers are not an identity
        with pytest.raises(ValueError):
            runtime.validate_case(case)
        return
    runtime.validate_case(case)
    if layer == "sd35m-1024/24":  # not a captured layer: the verifier rejects it
        with pytest.raises(RuntimeError):
            verifier.validate_case(case)
    else:
        verifier.validate_case(case)
    with pytest.raises(ValueError):
        runtime.validate_case({**case, "l": l + 1})


@pytest.mark.parametrize(("message", "stage"), [
    ("edge case edge-e1-small failed: x", "edge"),
    ("real layer final-0003 failed: x", "real"),
    ("case final-0400 failed: x", "scored"),
    ("runtime requires exactly one visible CUDA device", "setup"),
])
def test_worker_names_the_failing_stage(message: str, stage: str) -> None:
    assert worker.failure_stage(message) == stage


def test_bf16_reference_is_private_and_never_affects_validity() -> None:
    fields = verifier.bf16_reference_fields({"bf16_reference": {"ok": True, "family_tflops": {"video": 380.0, "llm": 421.0}}}, {})
    assert fields == {"bf16_reference_video_tflops": 380.0, "bf16_reference_llm_tflops": 421.0}
    assert verifier.bf16_reference_fields({"bf16_reference": {"ok": False, "error": "x"}}, {}) == {}
    assert verifier.bf16_reference_fields({}, {}) == {}
    assert not any(key.startswith("bf16") or "speedup" in key for key in adapter.SAFE_KEYS)


def test_images_and_build_wire_the_captures() -> None:
    judge = (ROOT / "images/judge/Dockerfile").read_text()
    judge_private = (ROOT / "images/judge/Dockerfile.private").read_text()
    work = (ROOT / "images/work/Dockerfile").read_text()
    work_local = (ROOT / "images/work/Dockerfile.local").read_text()
    build = (ROOT / "scripts/build_images.sh").read_text()
    # No image carries a Judge real layer; the Judge gets them read-only at run time.
    assert "--from=real_layers " not in judge + judge_private + work + work_local
    assert "COPY --from=fixtures recipe/real_layers/RECIPE_*.json " in judge and "RECIPE_" not in work
    assert '--build-context "real_layers=' not in build
    assert 'REAL_LAYERS_MOUNT = "/opt/sageattention_quant_attention_judge/fixtures/real_layers"' in entry_text
    for script in ("run_fixed_budget.sh", "eval_archive.sh", "replay_candidates.sh"):
        assert '"${JUDGE_ENTRY}"' in (ROOT / "scripts" / script).read_text()
    # Work has its own two layers, from models no Judge case captures.
    assert "FROM ${WORK_PUBLIC_IMAGE}" in work_local
    assert build.count('--build-context "real_layers_public=${real_public_context}"') == 1
    assert "for capture_dir in public-wan22 public-lm; do" in build


def test_no_hugging_face_token_is_stored_in_the_task_tree() -> None:
    pattern = re.compile(r"hf_[A-Za-z0-9]{30,}")
    captures = ROOT / "fixtures/real_layers"
    for path in ROOT.rglob("*"):
        if path.is_file() and not path.is_relative_to(captures) and path.suffix in {".py", ".sh", ".json", ".md", ".txt", ".yaml", ".toml", ""}:
            assert not pattern.search(path.read_text(errors="ignore")), path
