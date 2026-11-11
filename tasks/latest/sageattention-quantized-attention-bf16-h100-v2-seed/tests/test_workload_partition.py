from __future__ import annotations

import random

import pytest
from conftest import load_module

runtime=load_module("qattn_runtime_partition","runtime/quant_attention_runtime.py")
verifier=load_module("qattn_verifier_partition","images/judge/verify_quant_attention.py")
generator=load_module("qattn_manifest_partition","scripts/generate_final_manifest.py")
public_runner=load_module("qattn_public_runner_partition","images/work/benchmark/public_runner.py")
public=load_module("qattn_public_runtime_partition","images/work/benchmark/quant_attention_runtime.py")

MANIFEST=generator.generate_manifest()


def test_manifest_has_the_declared_shape_and_validates_on_both_sides() -> None:
    cases=verifier.validate_final_manifest(MANIFEST)
    assert len(cases)==verifier.EXPECTED_CASE_COUNT==774
    rows=[runtime.validate_case(c) for c in cases]
    assert max(r["l"] for r in rows)<=runtime.MAX_L and max(r["input_bytes"] for r in rows)<=runtime.MAX_CASE_INPUT_BYTES
    # Reproducible from the suite identity alone.
    assert generator.generate_manifest()==MANIFEST


def test_case_rejects_contract_violations() -> None:
    base={"case_id":"a","family":"llm","tier":"sweep","model":"m","b":1,"h_q":32,"h_kv":8,"d":128,"l":12345,
          "is_causal":True,"regime":"raw","seed":1,"gated":False,"layer":None}
    runtime.validate_case(base)
    for change in ({"d":96},{"h_kv":5},{"is_causal":False},{"l":runtime.MAX_L+1},{"tier":"real"},
                   {"regime":"drift"},{"layer":"llama31-16k/00"},{"tier":"recipe","gated":False}):
        with pytest.raises(ValueError):
            runtime.validate_case({**base,**change})


def test_edge_fixture_set_is_frozen() -> None:
    assert len(runtime.EDGE_CASES)==8
    assert runtime.EDGE_CASES[-1]["fixture"] is True


def test_manifest_rejects_a_changed_shape() -> None:
    bad=dict(MANIFEST,cases=MANIFEST["cases"][:-1])
    with pytest.raises(RuntimeError):
        verifier.validate_final_manifest(bad)
    cases=[dict(c) for c in MANIFEST["cases"]]
    sweeps=[c for c in cases if c["tier"]=="sweep"]
    for c in sweeps:
        c["gated"]=False
    with pytest.raises(RuntimeError,match="gates too few"):
        verifier.validate_final_manifest(dict(MANIFEST,cases=cases))
    redrawn=generator.generate_manifest(random.Random(7))
    assert verifier.validate_final_manifest(redrawn) and redrawn!=MANIFEST


def test_public_suite_shares_no_geometry_length_or_data_with_the_judge() -> None:
    judge=[c for c in MANIFEST["cases"] if c["family"]!="real"]
    geometry=lambda c:(c["h_q"],c["h_kv"],c["d"],c["is_causal"])
    judge_geometries={geometry(c) for c in judge}
    judge_lengths={c["l"] for c in MANIFEST["cases"]}
    for case in public_runner.PUBLIC_CASES:
        public.validate_case(case)
        assert geometry(case) not in judge_geometries,case["case_id"]
        assert case["l"] not in judge_lengths,case["case_id"]
    assert {geometry(c) for c in public_runner.PUBLIC_CASES}==verifier.PUBLIC_GEOMETRIES==generator.PUBLIC_GEOMETRIES
    assert {c["l"] for c in public_runner.PUBLIC_CASES}==verifier.PUBLIC_LENGTHS==generator.PUBLIC_LENGTHS
    edge_key=lambda c:(c["b"],c["l"],c["h_q"],c["h_kv"],c["d"],c["is_causal"])
    assert not {edge_key(c) for c in public.EDGE_CASES}&{edge_key(c) for c in runtime.EDGE_CASES}
    assert not {c["seed"] for c in public.EDGE_CASES}&{c["seed"] for c in runtime.EDGE_CASES}
    assert not any(c.get("fixture") for c in public.EDGE_CASES)
    # Public real layers come from models no Judge case captures.
    assert {c["capture_dir"] for c in public.PUBLIC_REAL_LAYER_CASES}=={"public-wan22","public-lm"}
    # Distinct input generator and no Judge constants in the Work runtime.
    assert not set(public.PROFILES)&{"norm","raw"} and not hasattr(public,"RANGES")
    for name in ("REAL_SOURCES","RECIPE_ID","SYNTHETIC_FAMILIES","TIERS"):
        assert not hasattr(public,name), name


def test_public_work_is_a_small_fraction_of_the_judge() -> None:
    work=lambda c:4*c["b"]*c["h_q"]*c["d"]*runtime.pairs_for_length(c["l"],c["is_causal"])
    assert sum(work(c) for c in public_runner.PUBLIC_CASES)*5<sum(work(c) for c in MANIFEST["cases"])


def test_agent_visible_files_say_nothing_about_hidden_cases() -> None:
    from conftest import TASK_ROOT
    work=TASK_ROOT/"images/work"
    hidden_lengths={str(c["l"]) for c in MANIFEST["cases"]}
    for path in (*work.rglob("*.md"),*work.rglob("public_runner.py"),*work.rglob("quant_attention_runtime.py"),*work.rglob("submission.py")):
        text=path.read_text(encoding="utf-8")
        for token in ("final-holdout","SageAttention","sageattn","Llama","CogVideo","Wan2","Hunyuan","Mochi","FLUX","Qwen",
                      "Mistral","RANGES","holdout","decoy","Decoy","fresh_input","PyroMind","HAMi","prepare_submit",
                      "recipe","sweep"):
            assert token not in text,(path.name,token)
        for length in hidden_lengths:
            assert f" {length} " not in text and f"={length}" not in text,(path.name,length)
