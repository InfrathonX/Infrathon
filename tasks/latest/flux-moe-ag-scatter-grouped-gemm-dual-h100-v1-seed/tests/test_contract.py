from __future__ import annotations
import hashlib
import pytest
from conftest import TASK_ROOT,load_module

public=load_module("moeag_public_contract","images/work/benchmark/public_runner.py")
admission=load_module("moeag_admission_contract","images/judge/source_admission.py")

# sha256 of the eight-rank lab seed this task ships (v2, W-rank ABI). The lab tree
# itself is an internal artifact and is not part of this repository, so the seed's
# identity is pinned here instead of compared against that tree.
SEED_SHA256 = "091ccd2ba631a53900a6e6273af2c6f9f13e50e73564d898af16f62a09ffbeee"


def test_seed_byte_identity_and_admission():
    work=TASK_ROOT/"images/work/workdir/moe_ag_lab"; oracle=TASK_ROOT/"oracle"
    data=(work/"candidates/kernel_v001/submission.py").read_bytes()
    assert data==(oracle/"candidates/kernel_v001/submission.py").read_bytes()
    assert hashlib.sha256(data).hexdigest()==SEED_SHA256
    got=admission.validate_submission(work,1,workspace_mode=True); assert got["ok"] and got["selected_sha256"]==hashlib.sha256(data).hexdigest()

def test_public_selection():
    cases=public.load_public_cases()
    edges=[x for x in cases if x["family"]=="edge"]; scored=[x for x in cases if x["family"]!="edge"]
    assert len(cases)==131 and len(edges)==19 and len(scored)==112
    assert {x["family"] for x in scored}==set(public.FAMILIES)
    one=public.cases_for_suite("one",cases); assert len(one)==1 and one[0]["family"]!="edge"
    quick=public.cases_for_suite("quick",cases)
    assert quick[:4]==edges[:4] and [x["family"] for x in quick[4:]]==list(dict.fromkeys(x["family"] for x in scored))
    assert public.cases_for_suite("full",cases)==edges+scored
    tp2=public.cases_for_suite("full",cases,family="tp2"); assert tp2 and all(x["family"]=="tp2" for x in tp2)
    picked=public.cases_for_suite("select",cases,case_ids=[scored[3]["case_id"],edges[0]["case_id"]])
    assert {x["case_id"] for x in picked}=={scored[3]["case_id"],edges[0]["case_id"]}
    with pytest.raises(RuntimeError): public.cases_for_suite("select",cases,case_ids=["nope"])
    with pytest.raises(RuntimeError): public.cases_for_suite("select",cases,case_ids=[edges[0]["case_id"]])
    with pytest.raises(ValueError): public.cases_for_suite("bogus",cases)
    assert public.SUITE_IDS=={"one":"public-one-v2","quick":"public-quick-v2","full":"public-v2","select":"public-select-v2"}

def test_public_payload_discloses_diagnostics_and_trial_spread():
    case=next(x for x in public.load_public_cases() if x["family"]=="tp8")
    metrics={name:0.0 for name in public.METRICS}
    metrics["cossim"]=1.0
    row={key:case.get(key) for key in public.IDENTITY}
    row.update(E_rank=case["E"]//case["EP"],split_counts=[1]*case["W"],effective_flops=10**12,
               median_ms=2.0,tflops=500.0,regime={"kind":"uniform","rows":4},hidden_comm=0.4,
               trial_seconds=[0.001,0.002,0.003],
               rank_records=[{"rank":r,"rows_per_trial":[5,6,7+r]} for r in range(case["W"])],**metrics)
    result={"cases":[row],"edge_cases":[],"correctness_ok":True,"edge_ok":True,"rank_agreement_ok":True,
            "trials_per_case":3,"family_tflops":{"tp8":500.0},"score_tflops":500.0,"hidden_comm":0.4}
    payload=public.public_payload(result,"one",1,1)
    disclosed=payload["cases"][0]
    for key in ("W","TP","EP","split_counts","effective_flops","hidden_comm","trial_min_ms","trial_max_ms","per_rank_rows","freivalds_rel_l2"):
        assert key in disclosed
    assert disclosed["trial_min_ms"]==1.0 and disclosed["trial_max_ms"]==3.0
    assert disclosed["per_rank_rows"]==[7+r for r in range(case["W"])]
    assert payload["hidden_comm"]==0.4 and payload["schema_version"]==2 and payload["hidden_cases_run"] is False
    for secret in ("seed","source","model","routing"):
        assert secret not in disclosed
    broken=dict(row); broken["trial_seconds"]=[0.0]
    with pytest.raises(RuntimeError): public.public_payload({**result,"cases":[broken]},"one",1,1)

def test_agent_contract_discloses_required_surface():
    text=(TASK_ROOT/"images/work/AGENT_TASK.md").read_text()
    for fragment in ("moe_ag_scatter_gemm_bf16","peer_ptrs","workspace_ptrs","world_size","token_offsets",
                     "2 * M * topk * N * K / EP","max(1.25 * reference, 1.0e-3)","single-current-stream","4 GiB","G7 random projection"):
        assert fragment in text, fragment

def test_work_image_absence_assertions_and_no_fixture():
    text=(TASK_ROOT/"images/work/Dockerfile").read_text()
    for name in ("flux","flux_triton","nvshmem","deep_ep","deep_gemm","triton_dist","flashinfer","vllm","sglang"): assert f'"{name}"' in text
    assert "--from=fixtures" not in text

def test_protected_hashes_match():
    mapping={"judge:candidate_worker.py":"images/judge/candidate_worker.py","judge:run_judge.py":"images/judge/run_judge.py","judge:source_admission.py":"images/judge/source_admission.py","judge:verify_moe_ag.py":"images/judge/verify_moe_ag.py","worker:moe_ag_runtime.py":"runtime/moe_ag_runtime.py"}
    observed={line.split()[1]:line.split()[0] for line in (TASK_ROOT/"images/judge/protected_hashes.txt").read_text().splitlines() if line.strip()}
    assert set(observed)==set(mapping)
    for label,relative in mapping.items(): assert observed[label]==hashlib.sha256((TASK_ROOT/relative).read_bytes()).hexdigest()


def test_oracle_final_selection_and_admissible_fallback():
    # oracle/FINAL_VERSION names the strongest paper candidate, which may be a
    # forced (non-admissible) one; smoke_test.sh and anything else that uses the
    # normal admission-enforcing Judge selects ORACLE_ADMISSIBLE_VERSION instead.
    import re
    oracle=TASK_ROOT/"oracle"
    final=int((oracle/"FINAL_VERSION").read_text().strip())
    assert (oracle/f"candidates/kernel_v{final:03d}/submission.py").is_file()
    smoke=(TASK_ROOT/"scripts/smoke_test.sh").read_text()
    match=re.search(r'ORACLE_ADMISSIBLE_VERSION:-([0-9]+)',smoke); assert match
    admissible=int(match.group(1))
    got=admission.validate_submission(oracle,admissible,workspace_mode=True)
    assert got["ok"],got["errors"]
    assert got["selected_relative_path"]==f"candidates/kernel_v{admissible:03d}/submission.py"


def test_launcher_does_not_reprobe_its_own_gpu_locks():
    # The eight-GPU launcher takes every GPU's /tmp/sforge-gpu-<uuid>.lock itself; its
    # race-closing second idle probe must not flock them again (that collides with its
    # own lock and the run never starts, as the first v2 launch did).
    text=(TASK_ROOT/"scripts/launch_fixed_budget_after_gpu_pair_idle.sh").read_text()
    assert 'exec {single_lock_fd}>"/tmp/sforge-gpu-${uuid}.lock"' in text
    assert "  if (( locks_held )); then\n    return 0\n  fi" in text
    assert '[[ "${JUDGE_MODE}" == separate ]] && (( locks_held ))' not in text


def test_brief_states_no_judge_duration_and_preflight_accepts_pcie_peers():
    # The brief does not quote Judge times (they depend on the card and the kernel), and
    # the preflight requires peer access and symmetric memory between all eight GPUs, not
    # NVLink on every pair: H100 NVL hosts bridge their GPUs in pairs and use PCIe between.
    text=(TASK_ROOT/"images/work/AGENT_TASK.md").read_text()
    assert "minutes" not in text and "readable over NVLink" not in text
    preflight=(TASK_ROOT/"scripts/preflight.py").read_text()
    assert "can_device_access_peer" in preflight and "symm.rendezvous" in preflight
    assert "not NVLink-connected" not in preflight and "allow-non-nvlink" not in preflight


def test_every_launch_path_uses_the_task_json_resources():
    # Interactive Judges, the final Judge and replays must measure under the same limits:
    # case time includes host wall time, so a CPU limit can move scores.
    import json,re
    task=json.loads(next((TASK_ROOT/"tasks").glob("*.json")).read_text())
    want={"cpu":str(task["work"]["cpu_limit"]),"mem":task["work"]["mem_limit"]}
    assert (task["judge"]["cpu_limit"],task["judge"]["mem_limit"])==(task["work"]["cpu_limit"],task["work"]["mem_limit"])
    seen=0
    for name in ("launch_fixed_budget_after_gpu_pair_idle.sh","eval_archive.sh","replay_candidates.sh"):
        text=(TASK_ROOT/"scripts"/name).read_text()
        for kind,value in re.findall(r"--(?:work|judge)-(cpu|mem)-limit\s+([0-9]+g?)",text):
            assert value==want[kind],(name,kind,value); seen+=1
    assert seen>=8
