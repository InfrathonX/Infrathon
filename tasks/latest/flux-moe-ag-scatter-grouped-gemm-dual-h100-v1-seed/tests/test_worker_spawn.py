from __future__ import annotations
import subprocess
from pathlib import Path
import pytest
from conftest import load_module

worker=load_module("moeag_worker_spawn","images/judge/candidate_worker.py")
public_worker=load_module("moeag_public_worker_spawn","images/work/benchmark/world_worker.py")

def manifest(count=5,suite="final-holdout-v2"): return {"schema_version":2,"suite_id":suite,"cases":[{"case_id":f"x{i}"} for i in range(count)]}

def test_manifest_validation():
    assert len(worker.validate_manifest(manifest(),expected_count=5))==5
    with pytest.raises(RuntimeError): worker.validate_manifest(manifest(4),expected_count=5)
    with pytest.raises(RuntimeError): worker.validate_manifest({**manifest(),"schema_version":1},expected_count=5)

def test_worker_suite_identities_and_world():
    assert worker.WORLD==public_worker.WORLD==8
    assert worker.SUITE_IDS=={"final-holdout-v2"}
    assert public_worker.SUITE_IDS=={"public-v2","public-quick-v2","public-one-v2","public-select-v2","host-smoke-v2"}

@pytest.mark.parametrize("suite_id",("public-v2","public-quick-v2","host-smoke-v2","final-holdout-v1"))
def test_judge_worker_rejects_non_judge_suite_ids(suite_id):
    with pytest.raises(RuntimeError,match="identity"):
        worker.validate_manifest(manifest(suite=suite_id),expected_count=5)

def test_port_is_loopback_bindable():
    assert 0<worker.free_loopback_port()<65536

def test_child_environment_strips_proxy_and_builds_rank(tmp_path: Path):
    env=worker.child_environment(3,12345,tmp_path,{"PATH":"/bin","HTTP_PROXY":"secret","CUDA_VISIBLE_DEVICES":"0,1,2,3,4,5,6,7"})
    assert env["RANK"]==env["LOCAL_RANK"]=="3" and env["WORLD_SIZE"]=="8" and env["CUDA_VISIBLE_DEVICES"]=="0,1,2,3,4,5,6,7"
    assert "HTTP_PROXY" not in env and env["MASTER_PORT"]=="12345" and env["MASTER_ADDR"]=="127.0.0.1"

def test_child_environment_defaults_to_all_eight_gpus(tmp_path: Path):
    assert worker.child_environment(0,12345,tmp_path,{"PATH":"/bin"})["CUDA_VISIBLE_DEVICES"]=="0,1,2,3,4,5,6,7"

def test_cpu_slices_are_disjoint_and_cover_ranks():
    slices=worker.cpu_slices()
    assert len(slices)==8 and all(slices)

def test_process_group_kill_path():
    process=subprocess.Popen(["bash","-c","sleep 30"],start_new_session=True)
    worker.kill_process_group(process)
    assert process.poll() is not None
