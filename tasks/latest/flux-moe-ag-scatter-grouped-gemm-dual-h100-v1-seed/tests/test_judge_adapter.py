from __future__ import annotations
from pathlib import Path
import pytest
from conftest import TASK_ROOT,load_module

adapter=load_module("moeag_adapter","images/judge/run_judge.py")

def records(profile="final"):
    families={key:float(50+10*i) for i,key in enumerate(adapter.FAMILY_KEYS)}
    import math
    score=math.exp(sum(math.log(v) for v in families.values())/len(families))
    row={"schema_version":2.0,"case_count":adapter.EXPECTED_CASE_COUNT,"edge_case_count":adapter.EXPECTED_EDGE_CASE_COUNT,"reward":score,"score_tflops":score,"hidden_comm":0.75,**families,"profile_ok":1.0,"suite_ok":1.0,"source_ok":1.0,"worker_ok":1.0,"hardware_ok":1.0,"correctness_ok":1.0,"edge_ok":1.0,"fresh_input_ok":1.0,"rank_agreement_ok":1.0,"profile_code":2.0,"final_version":1.0,"num_candidate_versions":1.0}
    detail=dict(row); detail.update(evaluation_policy="full-final-shared-v3",selection_protocol_id="flux-moe-ag-shared-final-v3",evaluation_profile=profile,suite_id="final-holdout-v2",manifest_sha256="a"*64,private_runtime_result={"secret":True}); return row,detail

def build(r,d,completed=True,returncode=0): return adapter.build_structured_result(r,d,verifier_completed=completed,verifier_returncode=returncode)

def test_valid_result_exposes_sanitized_keys_only():
    r,d=records(); result=build(r,d); assert result["valid"] and abs(result["score"]-r["score_tflops"])<1e-9
    assert adapter.FAMILY_KEYS==("tp8_tflops","tp4_tflops","tp2_tflops","ep_tflops","decode_tflops","sequence_tflops")
    assert result["metrics"]["source"]=="flux_moe_ag_v2_verifier_adapter"
    assert result["metrics"]["hidden_comm"]==0.75
    for key in adapter.FAMILY_KEYS: assert key in result["metrics"]
    assert "private_runtime_result" not in result["metrics"] and "manifest_sha256" not in result["metrics"]

@pytest.mark.parametrize("gate",[x for x,_ in adapter.GATES])
def test_every_gate_required(gate):
    r,d=records(); r[gate]=d[gate]=0.0; assert not build(r,d)["valid"]

def test_score_must_be_the_family_geomean():
    r,d=records(); r["score_tflops"]=d["score_tflops"]=r["reward"]=d["reward"]=r["score_tflops"]*1.01; assert not build(r,d)["valid"]

def test_case_counts_must_match_the_frozen_suite():
    r,d=records(); r["case_count"]=d["case_count"]=5.0; assert not build(r,d)["valid"]

def test_unknown_profile_fails_closed():
    r,d=records(); r["profile_code"]=d["profile_code"]=9.0; assert not build(r,d)["valid"]

def test_trusted_failure_is_failed():
    r,d=records()
    for row in (r,d):
        row["reward"]=row["score_tflops"]=0.0
        for key in adapter.FAMILY_KEYS: row[key]=0.0
        row["edge_ok"]=0.0
    got=build(r,d,returncode=1); assert not got["valid"] and any(x["status"]=="FAILED" for x in got["details"])

def test_worker_ok_gate_explains_missing_result_and_skips_result_gates():
    r,d=records()
    derived={"hardware_ok","correctness_ok","edge_ok","fresh_input_ok","rank_agreement_ok"}
    for row in (r,d):
        row["reward"]=row["score_tflops"]=0.0
        for key in adapter.FAMILY_KEYS: row[key]=0.0
        row["worker_ok"]=0.0
        for gate in derived: row[gate]=0.0
    got=build(r,d,returncode=1)
    details={row["name"]:row for row in got["details"]}
    assert details["worker_ok"]["message"]=="candidate worker produced no result (the candidate crashed or timed out)"
    assert details["worker_ok"]["score"]==0.0
    for gate in derived:
        assert details[gate]["message"]=="not evaluated: the candidate worker produced no result"
        assert details[gate]["score"]==0.0

@pytest.mark.parametrize("payload",('{"x":NaN}','{"x":Infinity}','{"x":1e400}','{'))
def test_bad_json_rejected(tmp_path: Path,payload):
    path=tmp_path/"x"; path.write_text(payload); assert adapter.read_json_object(path)=={}

def test_host_and_image_adapter_identical():
    assert (TASK_ROOT/"images/judge/run_judge.py").read_bytes()==(TASK_ROOT/"judge/run_judge.py").read_bytes()
