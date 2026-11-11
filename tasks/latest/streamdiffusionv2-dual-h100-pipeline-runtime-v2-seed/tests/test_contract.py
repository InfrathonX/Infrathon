from __future__ import annotations
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def test_task_metadata():
    spec=json.loads((ROOT/"tasks/streamdiffusionv2_dual_h100_pipeline_runtime_seed_v2.json").read_text())
    assert spec["task_id"]=="streamdiffusionv2_dual_h100_pipeline_runtime_seed_v2"
    assert spec["cwd"]=="/app/workdir/sdv2_pipeline_lab"
    assert spec["version_enumeration"]=={"root":"candidates","directory_pattern":"runtime_v([0-9]{3})","selector_file":"FINAL_VERSION"}
    assert spec["submit_paths"]==["candidates","FINAL_VERSION","notes.md"]

def test_media_generator_is_exactly_deterministic(load_module):
    media=load_module("runtime/media_gen.py")
    a=media.generate(19,9,32,48,cut_at=5); b=media.generate(19,9,32,48,cut_at=5)
    assert a.shape==(9,32,48,3) and a.dtype==np.uint8
    assert np.array_equal(a,b)
    assert not np.array_equal(a,media.generate(20,9,32,48,cut_at=5))

def test_media_motion_and_cut_are_observable(load_module):
    media=load_module("runtime/media_gen.py")
    smooth=media.generate(5,17,48,64)
    cut=media.generate(5,17,48,64,cut_at=9)
    delta=np.abs(cut[1:].astype(np.int16)-cut[:-1].astype(np.int16)).mean(axis=(1,2,3))
    assert delta[8] > np.median(delta)*3
    assert np.any(smooth[1:] != smooth[:-1])

def test_participant_brief_and_readme_contract():
    agent=(ROOT/"images/work/AGENT_TASK.md").read_text(); readme=(ROOT/"README.md").read_text()
    for name in ("start", "push", "flush", "close", "FINAL_VERSION"): assert name in agent
    # The maintainer README carries the submission-policy section; the public README does not.
    if "## Evaluation submission policy" in readme:
        for text in ("final_archive.tar.gz","submission.tar.gz"): assert text in readme

def test_case_tiers_are_canonical_and_disjoint(load_module):
    cases=load_module("runtime/cases.py")
    cases.validate_cases()
    assert {tier:len(rows) for tier,rows in cases.CASES_BY_TIER.items()}=={"public":6,"final":12}
    identities=[case.canonical_identity() for rows in cases.CASES_BY_TIER.values() for case in rows]
    assert len(identities)==len(set(identities))
    assert all(case.frames%4==1 for rows in cases.CASES_BY_TIER.values() for case in rows)
    assert all(case.prompt_id.startswith("hidden_") for case in cases.FINAL_CASES)
    assert all(case.prompt_id.startswith("public_") for case in cases.PUBLIC_CASES)

def test_final_manifest_is_private_and_matches_cases(load_module, final_cases_path):
    cases=load_module("runtime/cases.py")
    path=final_cases_path
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())==cases.manifest("final")

def test_judge_uses_sforge_private_candidate_profile():
    verifier=(ROOT/"images/judge/verify_sdv2_pipeline.py").read_text()
    archive_eval=(ROOT/"scripts/eval_archive.sh").read_text()
    adapter=(ROOT/"scripts/sforge_dual_h100_entry.py").read_text()
    assert '/tmp/.sforge-evaluation-context.json' in verifier
    assert 'app_dir / "evaluation_profile.json"' not in verifier
    assert 'SDV2_EVAL_ROUND_ID' in archive_eval
    assert 'kwargs["round_id"]=requested' in adapter
