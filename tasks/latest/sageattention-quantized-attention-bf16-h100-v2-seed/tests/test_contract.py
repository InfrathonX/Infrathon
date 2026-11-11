from __future__ import annotations
import hashlib,re,shlex
from pathlib import Path
from conftest import TASK_ROOT,load_module

public=load_module("qattn_public_contract","images/work/benchmark/public_runner.py")
admission=load_module("qattn_admission_contract","images/judge/source_admission.py")

def commands(text: str):
    blocks=re.findall(r"```(?:bash|sh)\s*\n(.*?)```",text,re.S)
    return [shlex.split(line) for block in blocks for line in block.splitlines() if line.strip() and not line.lstrip().startswith("#")]

def test_seed_is_byte_identical_and_selected() -> None:
    work=TASK_ROOT/"images/work/workdir/quant_attention_lab"; oracle=TASK_ROOT/"oracle"
    assert (work/"candidates/kernel_v001/submission.py").read_bytes()==(oracle/"candidates/kernel_v001/submission.py").read_bytes()
    assert (work/"FINAL_VERSION").read_text().strip()=="1"
    result=admission.validate_submission(work,1,workspace_mode=True)
    assert result["ok"] is True and result["selected_sha256"]==hashlib.sha256((oracle/"candidates/kernel_v001/submission.py").read_bytes()).hexdigest()

def test_public_partition_and_commands() -> None:
    assert len(public.cases_for_suite("one"))==1
    assert [x["case_id"] for x in public.cases_for_suite("quick")]==["public-p1","public-p4",public.QUICK_PERF_ONLY_ID]
    assert [x["gated"] for x in public.cases_for_suite("quick")]==[True,True,False]
    assert len(public.cases_for_suite("full"))==19 and all(x["gated"] for x in public.cases_for_suite("full"))
    assert not any(x["gated"] for x in public.cases_for_suite("full",perf_only=True))
    assert [x["case_id"] for x in public.cases_for_suite("full",case_id="public-p3")]==["public-p3"]
    text=(TASK_ROOT/"images/work/AGENT_TASK.md").read_text()
    parsed=commands(text)
    assert ["./tools/run_one.sh","--version","1"] in parsed
    assert ["./tools/run_public_tests.sh","--quick","--candidate-version","1"] in parsed
    assert ["./tools/run_public_tests.sh","--candidate-version","1"] in parsed

def test_agent_contract_discloses_exact_gates_and_host_delta() -> None:
    text=(TASK_ROOT/"images/work/AGENT_TASK.md").read_text()
    for fragment in ("max(3 * reference, 0.025)","max(3 * reference, 0.04)","max(2 * reference, 0.02)","max(2.5 * reference, 0.05)","FP8 e4m3","including `k.mean(...)`","hidden test cases"):
        assert fragment in text

def test_work_image_has_its_own_runtime_and_absence_asserts() -> None:
    text=(TASK_ROOT/"images/work/Dockerfile").read_text().lower()
    for name in ("sageattention","flash_attn","flashinfer","xformers","flash_mla","deep_gemm"): assert name in text
    assert "--from=fixtures" not in text and "task_runtime" not in text

def test_judge_image_carries_the_capture_manifest_and_recipe() -> None:
    text=(TASK_ROOT/"images/judge/Dockerfile").read_text()
    for fragment in (
        "COPY --from=fixtures recipe/RECIPE.json "
        "/opt/sageattention_quant_attention_worker/RECIPE.json",
        "COPY --from=fixtures e8_wan13b_p0_b15_s00_cond_8192.manifest.json "
        "/opt/sageattention_quant_attention_judge/fixtures/"
        "e8_wan13b_p0_b15_s00_cond_8192.manifest.json",
        "/opt/sageattention_quant_attention_worker/RECIPE.json \\",
    ):
        assert fragment in text

def test_protected_hashes_match() -> None:
    mapping={"judge:candidate_worker.py":"images/judge/candidate_worker.py","judge:run_judge.py":"images/judge/run_judge.py","judge:source_admission.py":"images/judge/source_admission.py","judge:verify_quant_attention.py":"images/judge/verify_quant_attention.py","worker:quant_attention_runtime.py":"runtime/quant_attention_runtime.py"}
    observed={line.split()[1]:line.split()[0] for line in (TASK_ROOT/"images/judge/protected_hashes.txt").read_text().splitlines() if line.strip()}
    assert set(observed)==set(mapping)
    for label,relative in mapping.items(): assert observed[label]==hashlib.sha256((TASK_ROOT/relative).read_bytes()).hexdigest()
